import torch
import triton 
import triton.language as tl

'''
a 2D-tiled grid or fixed block-size iteration is recommended to avoid
hardware register pressure and shared memory limits for intermediate dimensions h > 4096

This implementation maps one block per row for clarity and simplicity
'''

@triton.jit
def swiglu_fwd_kernel(
    out_ptr,
    x_ptr,
    stride_out_m, stride_out_n,
    stride_xm, stride_xn,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    # one row
    row_idx = tl.program_id(0)

    # set pointers to the start of this row
    row_out_ptr = out_ptr + row_idx * stride_out_m
    row_x_ptr = x_ptr + row_idx * stride_xm

    # column indices for [0, hidden_dim) to make mask
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_dim

    # Load gate from [0, hidden_dim) and value from [hidden_dim, 2 * hidden_dim)
    gate_ptrs = row_x_ptr + cols * stride_xn
    val_ptrs = row_x_ptr + (cols + hidden_dim) * stride_xn

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    val = tl.load(val_ptrs, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    silu_gate = gate * tl.sigmoid(gate)
    res = silu_gate * val

    # Store back to output buffer
    tl.store(row_out_ptr + cols * stride_out_n, res.to(out_ptr.dtype.element_ty), mask=mask)

def swiglu_triton(x: torch.Tensor, w_fused: torch.Tensor) -> torch.Tensor:

    # shape: [rows, 2*h_dim]
    projected = torch.matmul(x, w_fused)

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
    return out
