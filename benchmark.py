import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing


class SeparateSwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_val = nn.Linear(in_features, hidden_features, bias=False)
    #@torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w_gate(x)) * self.w_val(x)


class FusedSwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w_proj = nn.Linear(in_features, 2 * hidden_features, bias=False)
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w_proj(x).chunk(2, dim=-1)
        return F.silu(gate) * val


@triton.jit
def swiglu_fwd_kernel(
    out_ptr,
    x_ptr,
    stride_out_m, stride_out_n,
    stride_xm, stride_xn,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_out_ptr = out_ptr + row_idx * stride_out_m
    row_x_ptr = x_ptr + row_idx * stride_xm

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_dim

    gate_ptrs = row_x_ptr + cols * stride_xn
    val_ptrs = row_x_ptr + (cols + hidden_dim) * stride_xn

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    val = tl.load(val_ptrs, mask=mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    res = silu_gate * val

    tl.store(row_out_ptr + cols * stride_out_n, res.to(out_ptr.dtype.element_ty), mask=mask)


class TritonSwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        # Weight shape [in_features, 2 * hidden_features] for direct matmul(x, w)
        self.w_fused = nn.Parameter(torch.empty(in_features, 2 * hidden_features))
        nn.init.normal_(self.w_fused, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        x_2d = x.view(-1, x.shape[-1])

        # Standard high-throughput cuBLAS GEMM
        projected = torch.matmul(x_2d, self.w_fused)

        rows, total_cols = projected.shape
        hidden_dim = total_cols // 2

        out = torch.empty((rows, hidden_dim), device=x.device, dtype=x.dtype)

        block_size = triton.next_power_of_2(hidden_dim)
        num_warps = 4
        if block_size > 2048:
            num_warps = 8
        if block_size > 4096:
            num_warps = 16

        grid = (rows,)

        swiglu_fwd_kernel[grid](
            out,
            projected,
            out.stride(0), out.stride(1),
            projected.stride(0), projected.stride(1),
            hidden_dim,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        return out.view(*orig_shape, hidden_dim)


def run_benchmark():
    assert torch.cuda.is_available(), "CUDA is required for triton benchmarks."
    device = torch.device("cuda")
    dtype = torch.bfloat16

    in_dim = 4096
    hidden_dim = 11008

    sep_model = SeparateSwiGLU(in_dim, hidden_dim).to(device=device, dtype=dtype).eval()
    fused_model = FusedSwiGLU(in_dim, hidden_dim).to(device=device, dtype=dtype).eval()
    triton_model = TritonSwiGLU(in_dim, hidden_dim).to(device=device, dtype=dtype).eval()

    # Align weights between fused_model and triton_model for numerical consistency
    with torch.no_grad():
        triton_model.w_fused.copy_(fused_model.w_proj.weight.t())

    test_shapes = [
    (1, 512),
    (8, 1),       
    (32, 1),      
    (64, 1),      
    (2, 512),     
    (4, 1024),    
    (8, 2048),    
    (16, 1024),   
    (32, 512),    
    (1, 4096),    
    (2, 4096),    
]

    header = f"{'Shape (B, S)':<14} | {'Separate (ms)':<14} | {'Fused (ms)':<14} | {'Triton (ms)':<14} | {'Speedup vs Sep':<15}| {'Speedup vs Fused':<15}"
    print(header)
    print("-" * len(header))

    with torch.inference_mode():
        for bsz, seq_len in test_shapes:
            x = torch.randn(bsz, seq_len, in_dim, device=device, dtype=dtype)

            # Measure median latencies
            ms_sep = triton.testing.do_bench(lambda: sep_model(x), return_mode="median")
            ms_fused = triton.testing.do_bench(lambda: fused_model(x), return_mode="median")
            ms_triton = triton.testing.do_bench(lambda: triton_model(x), return_mode="median")

            speedup = ms_sep / ms_triton
            speedup_fused = ms_fused / ms_triton
            shape_str = f"({bsz}, {seq_len})"
            print(f"{shape_str:<14} | {ms_sep:<14.4f} | {ms_fused:<14.4f} | {ms_triton:<14.4f} | x{speedup:<15.2f}| x{speedup_fused:<15.2f}")


if __name__ == "__main__":
    run_benchmark()