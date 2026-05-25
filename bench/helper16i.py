import triton
import triton.language as tl

from helper16 import (
    _fp32x16_to_e2m1_roundtrip_fp32x16,
    _pack_final_code_16_cols,
    _max_abs_16,
)


@triton.jit
def _weighted_mse_after_e2m1_roundtrip_16_cols(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    iw0, iw1, iw2, iw3,
    iw4, iw5, iw6, iw7,
    iw8, iw9, iw10, iw11,
    iw12, iw13, iw14, iw15,
    inv_scale,
    scale,
):
    """
    v0..v15: each [OUTS_PER_PROGRAM], already global-scale normalized.
    iw0..iw15: scalars from imp[0, in_base + 0..15].
    inv_scale: [OUTS_PER_PROGRAM]
    scale:     [OUTS_PER_PROGRAM]

    Computes weighted candidate error:
      x = vals / scale
      q = e2m1_round(x)
      weighted_mse = sum_j imp[j] * (q_j - x_j)^2 * scale^2

    This form avoids q * scale - vals inside each term and matches helper16.py's
    scale-space MSE implementation.
    """
    x0 = v0 * inv_scale
    x1 = v1 * inv_scale
    x2 = v2 * inv_scale
    x3 = v3 * inv_scale
    x4 = v4 * inv_scale
    x5 = v5 * inv_scale
    x6 = v6 * inv_scale
    x7 = v7 * inv_scale
    x8 = v8 * inv_scale
    x9 = v9 * inv_scale
    x10 = v10 * inv_scale
    x11 = v11 * inv_scale
    x12 = v12 * inv_scale
    x13 = v13 * inv_scale
    x14 = v14 * inv_scale
    x15 = v15 * inv_scale

    (
        q0, q1, q2, q3,
        q4, q5, q6, q7,
        q8, q9, q10, q11,
        q12, q13, q14, q15,
    ) = _fp32x16_to_e2m1_roundtrip_fp32x16(
        x0, x1, x2, x3,
        x4, x5, x6, x7,
        x8, x9, x10, x11,
        x12, x13, x14, x15,
    )

    e0 = q0 - x0
    e1 = q1 - x1
    e2 = q2 - x2
    e3 = q3 - x3
    e4 = q4 - x4
    e5 = q5 - x5
    e6 = q6 - x6
    e7 = q7 - x7
    e8 = q8 - x8
    e9 = q9 - x9
    e10 = q10 - x10
    e11 = q11 - x11
    e12 = q12 - x12
    e13 = q13 - x13
    e14 = q14 - x14
    e15 = q15 - x15

    weighted_err = (
        ((e0 * e0 * iw0 + e1 * e1 * iw1)
        + (e2 * e2 * iw2 + e3 * e3 * iw3)
        + (e4 * e4 * iw4 + e5 * e5 * iw5)
        + (e6 * e6 * iw6 + e7 * e7 * iw7))
        + ((e8 * e8 * iw8 + e9 * e9 * iw9)
        + (e10 * e10 * iw10 + e11 * e11 * iw11)
        + (e12 * e12 * iw12 + e13 * e13 * iw13)
        + (e14 * e14 * iw14 + e15 * e15 * iw15))
    )

    return weighted_err * (scale * scale)


@triton.jit
def _load_imp_16_1din(
    imp_ptr,
    in_base,
):
    """
    imp: [1, In], contiguous.
    in_base is the first column of the current 16-element in-block.

    Returns 16 scalar importance values. A caller can reuse these scalars across
    all out channels handled by the same program.
    """
    iw0 = tl.load(imp_ptr + in_base + 0).to(tl.float32)
    iw1 = tl.load(imp_ptr + in_base + 1).to(tl.float32)
    iw2 = tl.load(imp_ptr + in_base + 2).to(tl.float32)
    iw3 = tl.load(imp_ptr + in_base + 3).to(tl.float32)
    iw4 = tl.load(imp_ptr + in_base + 4).to(tl.float32)
    iw5 = tl.load(imp_ptr + in_base + 5).to(tl.float32)
    iw6 = tl.load(imp_ptr + in_base + 6).to(tl.float32)
    iw7 = tl.load(imp_ptr + in_base + 7).to(tl.float32)
    iw8 = tl.load(imp_ptr + in_base + 8).to(tl.float32)
    iw9 = tl.load(imp_ptr + in_base + 9).to(tl.float32)
    iw10 = tl.load(imp_ptr + in_base + 10).to(tl.float32)
    iw11 = tl.load(imp_ptr + in_base + 11).to(tl.float32)
    iw12 = tl.load(imp_ptr + in_base + 12).to(tl.float32)
    iw13 = tl.load(imp_ptr + in_base + 13).to(tl.float32)
    iw14 = tl.load(imp_ptr + in_base + 14).to(tl.float32)
    iw15 = tl.load(imp_ptr + in_base + 15).to(tl.float32)

    return (
        iw0, iw1, iw2, iw3,
        iw4, iw5, iw6, iw7,
        iw8, iw9, iw10, iw11,
        iw12, iw13, iw14, iw15,
    )


@triton.jit
def _load_16_cols_2d_outtile(
    ptr,
    out_offsets,
    out_mask,
    in_base,
    IN_FEATURES: tl.constexpr,
    global_scale_inv,
):
    """
    weight: [Out, In], contiguous.

    A program handles several out channels at a fixed 16-column in-block:
      out_offsets: [OUTS_PER_PROGRAM]
      in_base: scalar, block column start in [0, In)

    Returns v0..v15, each [OUTS_PER_PROGRAM].
    """
    base_elem = out_offsets * IN_FEATURES + in_base

    v0 = tl.load(ptr + base_elem + 0, mask=out_mask, other=0.0)
    v1 = tl.load(ptr + base_elem + 1, mask=out_mask, other=0.0)
    v2 = tl.load(ptr + base_elem + 2, mask=out_mask, other=0.0)
    v3 = tl.load(ptr + base_elem + 3, mask=out_mask, other=0.0)
    v4 = tl.load(ptr + base_elem + 4, mask=out_mask, other=0.0)
    v5 = tl.load(ptr + base_elem + 5, mask=out_mask, other=0.0)
    v6 = tl.load(ptr + base_elem + 6, mask=out_mask, other=0.0)
    v7 = tl.load(ptr + base_elem + 7, mask=out_mask, other=0.0)
    v8 = tl.load(ptr + base_elem + 8, mask=out_mask, other=0.0)
    v9 = tl.load(ptr + base_elem + 9, mask=out_mask, other=0.0)
    v10 = tl.load(ptr + base_elem + 10, mask=out_mask, other=0.0)
    v11 = tl.load(ptr + base_elem + 11, mask=out_mask, other=0.0)
    v12 = tl.load(ptr + base_elem + 12, mask=out_mask, other=0.0)
    v13 = tl.load(ptr + base_elem + 13, mask=out_mask, other=0.0)
    v14 = tl.load(ptr + base_elem + 14, mask=out_mask, other=0.0)
    v15 = tl.load(ptr + base_elem + 15, mask=out_mask, other=0.0)

    v0 = v0.to(tl.float32) * global_scale_inv
    v1 = v1.to(tl.float32) * global_scale_inv
    v2 = v2.to(tl.float32) * global_scale_inv
    v3 = v3.to(tl.float32) * global_scale_inv
    v4 = v4.to(tl.float32) * global_scale_inv
    v5 = v5.to(tl.float32) * global_scale_inv
    v6 = v6.to(tl.float32) * global_scale_inv
    v7 = v7.to(tl.float32) * global_scale_inv
    v8 = v8.to(tl.float32) * global_scale_inv
    v9 = v9.to(tl.float32) * global_scale_inv
    v10 = v10.to(tl.float32) * global_scale_inv
    v11 = v11.to(tl.float32) * global_scale_inv
    v12 = v12.to(tl.float32) * global_scale_inv
    v13 = v13.to(tl.float32) * global_scale_inv
    v14 = v14.to(tl.float32) * global_scale_inv
    v15 = v15.to(tl.float32) * global_scale_inv

    return (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    )


__all__ = [
    "_max_abs_16",
    "_pack_final_code_16_cols",
    "_weighted_mse_after_e2m1_roundtrip_16_cols",
    "_load_imp_16_1din",
    "_load_16_cols_2d_outtile",
]
