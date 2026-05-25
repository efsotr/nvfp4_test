import triton
import triton.language as tl

@triton.jit
def _f16x2_u32_to_fp32_pair(h):
    lo_f32, hi_f32 = tl.inline_asm_elementwise(
        asm=r"""
        {
        .reg .b16 lo;
        .reg .b16 hi;
        mov.b32 {lo, hi}, $2;
        cvt.f32.f16 $0, lo;
        cvt.f32.f16 $1, hi;
        }
        """,
        constraints="=f,=f,r",
        args=[h],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )
    return lo_f32, hi_f32


@triton.jit
def _fp32x16_to_e2m1_roundtrip_fp32x16(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    """
    Performance path for candidate MSE.

    16 fp32 -> 8 packed e2m1x2 bytes -> 8 f16x2 -> 16 fp32.

    No fallback. Assumes target supports:
      cvt.rn.satfinite.e2m1x2.f32
      cvt.rn.f16x2.e2m1x2
    """
    h0, h1, h2, h3, h4, h5, h6, h7 = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 b0;
          .reg .b8 b1;
          .reg .b8 b2;
          .reg .b8 b3;
          .reg .b8 b4;
          .reg .b8 b5;
          .reg .b8 b6;
          .reg .b8 b7;

          cvt.rn.satfinite.e2m1x2.f32 b0,  $9,  $8;
          cvt.rn.f16x2.e2m1x2 $0, b0;

          cvt.rn.satfinite.e2m1x2.f32 b1,  $11, $10;
          cvt.rn.f16x2.e2m1x2 $1, b1;

          cvt.rn.satfinite.e2m1x2.f32 b2,  $13, $12;
          cvt.rn.f16x2.e2m1x2 $2, b2;

          cvt.rn.satfinite.e2m1x2.f32 b3,  $15, $14;
          cvt.rn.f16x2.e2m1x2 $3, b3;

          cvt.rn.satfinite.e2m1x2.f32 b4,  $17, $16;
          cvt.rn.f16x2.e2m1x2 $4, b4;

          cvt.rn.satfinite.e2m1x2.f32 b5,  $19, $18;
          cvt.rn.f16x2.e2m1x2 $5, b5;

          cvt.rn.satfinite.e2m1x2.f32 b6,  $21, $20;
          cvt.rn.f16x2.e2m1x2 $6, b6;

          cvt.rn.satfinite.e2m1x2.f32 b7,  $23, $22;
          cvt.rn.f16x2.e2m1x2 $7, b7;
        }
        """,
        constraints=(
            "=r,=r,=r,=r,=r,=r,=r,=r,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f"
        ),
        args=[
            x0, x1, x2, x3,
            x4, x5, x6, x7,
            x8, x9, x10, x11,
            x12, x13, x14, x15,
        ],
        dtype=(
            tl.uint32, tl.uint32, tl.uint32, tl.uint32,
            tl.uint32, tl.uint32, tl.uint32, tl.uint32,
        ),
        is_pure=True,
        pack=1,
    )

    q0, q1 = _f16x2_u32_to_fp32_pair(h0)
    q2, q3 = _f16x2_u32_to_fp32_pair(h1)
    q4, q5 = _f16x2_u32_to_fp32_pair(h2)
    q6, q7 = _f16x2_u32_to_fp32_pair(h3)
    q8, q9 = _f16x2_u32_to_fp32_pair(h4)
    q10, q11 = _f16x2_u32_to_fp32_pair(h5)
    q12, q13 = _f16x2_u32_to_fp32_pair(h6)
    q14, q15 = _f16x2_u32_to_fp32_pair(h7)

    return (
        q0, q1, q2, q3,
        q4, q5, q6, q7,
        q8, q9, q10, q11,
        q12, q13, q14, q15,
    )


@triton.jit
def _fp32x16_to_e2m1_u32x2(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    """
    Performance path for final FP4 code.

    16 fp32 -> 16 e2m1 -> two uint32:
      lo = bytes for x0..x7
      hi = bytes for x8..x15

    Final code layout:
      uint8 view of [lo, hi] gives 8 packed bytes.
    """
    lo, hi = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 b0;
          .reg .b8 b1;
          .reg .b8 b2;
          .reg .b8 b3;
          .reg .b8 b4;
          .reg .b8 b5;
          .reg .b8 b6;
          .reg .b8 b7;

          cvt.rn.satfinite.e2m1x2.f32 b0,  $3,  $2;
          cvt.rn.satfinite.e2m1x2.f32 b1,  $5,  $4;
          cvt.rn.satfinite.e2m1x2.f32 b2,  $7,  $6;
          cvt.rn.satfinite.e2m1x2.f32 b3,  $9,  $8;
          cvt.rn.satfinite.e2m1x2.f32 b4,  $11, $10;
          cvt.rn.satfinite.e2m1x2.f32 b5,  $13, $12;
          cvt.rn.satfinite.e2m1x2.f32 b6,  $15, $14;
          cvt.rn.satfinite.e2m1x2.f32 b7,  $17, $16;

          mov.b32 $0, {b0, b1, b2, b3};
          mov.b32 $1, {b4, b5, b6, b7};
        }
        """,
        constraints=(
            "=r,=r,"
            "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f"
        ),
        args=[
            x0, x1, x2, x3,
            x4, x5, x6, x7,
            x8, x9, x10, x11,
            x12, x13, x14, x15,
        ],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )

    return lo, hi


@triton.jit
def _mse_after_e2m1_roundtrip_16_cols(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
    scale,
):
    """
    v0..v15:   each [BLOCKS_PER_PROGRAM], already global-scale normalized.
    inv_scale: [BLOCKS_PER_PROGRAM]
    scale:     [BLOCKS_PER_PROGRAM]

    Computes:
      x = vals / scale
      q = e2m1_round(x)
      mse = sum((q - x)^2) * scale^2
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

    return (
        ((e0 * e0 + e1 * e1) + (e2 * e2 + e3 * e3)
        + (e4 * e4 + e5 * e5) + (e6 * e6 + e7 * e7))
        + ((e8 * e8 + e9 * e9) + (e10 * e10 + e11 * e11)
        + (e12 * e12 + e13 * e13) + (e14 * e14 + e15 * e15))
    ) * (scale * scale)


@triton.jit
def _pack_final_code_16_cols(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
):
    """
    v0..v15:   each [BLOCKS_PER_PROGRAM], already global-scale normalized.
    inv_scale: [BLOCKS_PER_PROGRAM]

    Returns:
      lo, hi uint32 vectors, each [BLOCKS_PER_PROGRAM].
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

    return _fp32x16_to_e2m1_u32x2(
        x0, x1, x2, x3,
        x4, x5, x6, x7,
        x8, x9, x10, x11,
        x12, x13, x14, x15,
    )


@triton.jit
def _max_abs_16(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
):
    m0 = tl.maximum(tl.abs(v0), tl.abs(v1))
    m1 = tl.maximum(tl.abs(v2), tl.abs(v3))
    m2 = tl.maximum(tl.abs(v4), tl.abs(v5))
    m3 = tl.maximum(tl.abs(v6), tl.abs(v7))
    m4 = tl.maximum(tl.abs(v8), tl.abs(v9))
    m5 = tl.maximum(tl.abs(v10), tl.abs(v11))
    m6 = tl.maximum(tl.abs(v12), tl.abs(v13))
    m7 = tl.maximum(tl.abs(v14), tl.abs(v15))

    m01 = tl.maximum(m0, m1)
    m23 = tl.maximum(m2, m3)
    m45 = tl.maximum(m4, m5)
    m67 = tl.maximum(m6, m7)

    m0123 = tl.maximum(m01, m23)
    m4567 = tl.maximum(m45, m67)

    return tl.maximum(m0123, m4567)

@triton.jit
def _load_16_cols_2d_seperate(
    ptr,
    block_offsets,
    block_mask,
    global_scale_inv,
):
    base_elem = block_offsets * 16

    v0 = tl.load(ptr + base_elem + 0, mask=block_mask, other=0.0)
    v1 = tl.load(ptr + base_elem + 1, mask=block_mask, other=0.0)
    v2 = tl.load(ptr + base_elem + 2, mask=block_mask, other=0.0)
    v3 = tl.load(ptr + base_elem + 3, mask=block_mask, other=0.0)
    v4 = tl.load(ptr + base_elem + 4, mask=block_mask, other=0.0)
    v5 = tl.load(ptr + base_elem + 5, mask=block_mask, other=0.0)
    v6 = tl.load(ptr + base_elem + 6, mask=block_mask, other=0.0)
    v7 = tl.load(ptr + base_elem + 7, mask=block_mask, other=0.0)
    v8 = tl.load(ptr + base_elem + 8, mask=block_mask, other=0.0)
    v9 = tl.load(ptr + base_elem + 9, mask=block_mask, other=0.0)
    v10 = tl.load(ptr + base_elem + 10, mask=block_mask, other=0.0)
    v11 = tl.load(ptr + base_elem + 11, mask=block_mask, other=0.0)
    v12 = tl.load(ptr + base_elem + 12, mask=block_mask, other=0.0)
    v13 = tl.load(ptr + base_elem + 13, mask=block_mask, other=0.0)
    v14 = tl.load(ptr + base_elem + 14, mask=block_mask, other=0.0)
    v15 = tl.load(ptr + base_elem + 15, mask=block_mask, other=0.0)

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

@triton.jit
def _load_16_cols_2d(
    ptr,
    block_offsets,
    block_mask,
    global_scale_inv,
):
    base_elem = block_offsets * 16                         # [B]
    cols = tl.arange(0, 16)                                # [16]

    offs = base_elem[:, None] + cols[None, :]              # [B, 16]
    mask = block_mask[:, None]                             # [B, 1]

    x = tl.load(ptr + offs, mask=mask, other=0.0)          # [B, 16]
    x = x.to(tl.float32) * global_scale_inv

    return (
        x[:, 0],  x[:, 1],  x[:, 2],  x[:, 3],
        x[:, 4],  x[:, 5],  x[:, 6],  x[:, 7],
        x[:, 8],  x[:, 9],  x[:, 10], x[:, 11],
        x[:, 12], x[:, 13], x[:, 14], x[:, 15],
    )


@triton.jit
def _load_16_cols_2d_trans(
    ptr,
    block_offsets,
    block_mask,
    global_scale_inv,
):
    base_elem = block_offsets * 16                         # [B]
    cols = tl.arange(0, 16)                                # [16]

    offs = base_elem[:, None] + cols[None, :]              # [B, 16]
    mask = block_mask[:, None]                             # [B, 1]

    x = tl.load(ptr + offs, mask=mask, other=0.0)          # [B, 16]
    x = x.to(tl.float32) * global_scale_inv
    xt = tl.trans(x)

    return (
        xt[0], xt[1], xt[2], xt[3],
        xt[4], xt[5], xt[6], xt[7],
        xt[8], xt[9], xt[10], xt[11],
        xt[12], xt[13], xt[14], xt[15],
    )