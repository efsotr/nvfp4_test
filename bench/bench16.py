from re import S

import torch
import triton
import triton.language as tl

from helper import (
    check_sm100,
    dequantize,
    error_stats,
    get_nvfp4_global_scales,
    make_w,
    print_result,
    time_cuda,
)


BLOCK_SIZE = 16
LOWER_BOUND = -8
UPPER_BOUND = 7
PERSISTENT_LAUNCH_BLOCKS_CAP = 4
MAX_BLOCKS_PER_WARP = 16

SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32}, num_warps=1, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=2, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=16, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 512}, num_warps=16, num_stages=3),
]


@triton.jit
def _f16x2_u32_to_fp32_pair(h):
    lo = (h & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    hi = (h >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    return lo, hi


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
        e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3
        + e4 * e4 + e5 * e5 + e6 * e6 + e7 * e7
        + e8 * e8 + e9 * e9 + e10 * e10 + e11 * e11
        + e12 * e12 + e13 * e13 + e14 * e14 + e15 * e15
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


def _runtime_blocks_per_sm(block_threads, device=None):
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    max_threads_per_sm = getattr(props, "max_threads_per_multi_processor", 2048)
    blocks = max_threads_per_sm // block_threads if block_threads > 0 else 1
    return max(1, min(blocks, PERSISTENT_LAUNCH_BLOCKS_CAP))


def persistent_grid(num_blocks, device, config):
    blocks_per_program = config.kwargs["BLOCKS_PER_PROGRAM"]

    if blocks_per_program > config.num_warps * MAX_BLOCKS_PER_WARP:
        raise ValueError(
            "ScaleSweep configurations require BLOCKS_PER_PROGRAM <= "
            f"num_warps * {MAX_BLOCKS_PER_WARP}; got blocks={blocks_per_program}, "
            f"warps={config.num_warps}"
        )

    block_threads = config.num_warps * 32
    num_programs = triton.cdiv(num_blocks, blocks_per_program)
    blocks_per_sm = _runtime_blocks_per_sm(block_threads, device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    return min(num_programs, max(1, sm_count * blocks_per_sm))

@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=["num_blocks", "grid_size", "LOWER_BOUND", "NUM_CANDIDATES"],
)
@triton.jit
def scalesweep_quantize_kernel(
    weight_ptr,
    scale_ptr,
    code_i32_ptr,
    global_scale_inv_ptr,
    num_blocks: tl.constexpr,
    grid_size: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)

    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM

    while block_start < num_blocks:
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        block_mask = block_offsets < num_blocks

        base_elem = block_offsets * 16

        v0 = (
            tl.load(weight_ptr + base_elem + 0, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v1 = (
            tl.load(weight_ptr + base_elem + 1, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v2 = (
            tl.load(weight_ptr + base_elem + 2, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v3 = (
            tl.load(weight_ptr + base_elem + 3, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v4 = (
            tl.load(weight_ptr + base_elem + 4, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v5 = (
            tl.load(weight_ptr + base_elem + 5, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v6 = (
            tl.load(weight_ptr + base_elem + 6, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v7 = (
            tl.load(weight_ptr + base_elem + 7, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v8 = (
            tl.load(weight_ptr + base_elem + 8, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v9 = (
            tl.load(weight_ptr + base_elem + 9, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v10 = (
            tl.load(weight_ptr + base_elem + 10, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v11 = (
            tl.load(weight_ptr + base_elem + 11, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v12 = (
            tl.load(weight_ptr + base_elem + 12, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v13 = (
            tl.load(weight_ptr + base_elem + 13, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v14 = (
            tl.load(weight_ptr + base_elem + 14, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )
        v15 = (
            tl.load(weight_ptr + base_elem + 15, mask=block_mask, other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )

        abs_max = _max_abs_16(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
        )

        base_fp8 = (abs_max * (1.0 / 6.0)).to(tl.float8e4nv)
        base_raw = base_fp8.to(tl.uint8, bitcast=True).to(tl.int32)

        for i in tl.static_range(0, NUM_CANDIDATES):
            raw_i = tl.minimum(
                tl.maximum(base_raw + LOWER_BOUND + i, 1),
                126,
            ).to(tl.uint8)

            scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
            scale_i = scale_fp8.to(tl.float32)
            inv_scale_i = 1.0 / scale_i

            mse_i = _mse_after_e2m1_roundtrip_16_cols(
                v0, v1, v2, v3,
                v4, v5, v6, v7,
                v8, v9, v10, v11,
                v12, v13, v14, v15,
                inv_scale_i,
                scale_i,
            )

            if i > 0:
                better = mse_i < best_mse
                best_mse = tl.where(better, mse_i, best_mse)
                best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)
            else:
                best_mse = mse_i
                best_scale_fp8 = scale_fp8

        tl.store(
            scale_ptr + block_offsets,
            best_scale_fp8,
            mask=block_mask,
        )

        best_scale_inv = 1.0 / best_scale_fp8.to(tl.float32)

        lo, hi = _pack_final_code_16_cols(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            best_scale_inv,
        )

        code_i32_offsets = block_offsets * 2

        tl.store(
            code_i32_ptr + code_i32_offsets + 0,
            lo.to(tl.int32),
            mask=block_mask,
        )
        tl.store(
            code_i32_ptr + code_i32_offsets + 1,
            hi.to(tl.int32),
            mask=block_mask,
        )

        block_start += grid_size * BLOCKS_PER_PROGRAM


def scalesweep_quantize(
    weight,
    global_scale_inv,
    block_size,
    lower_bound,
    upper_bound,
    config,
):
    if block_size != 16:
        raise ValueError("optimized kernel is specialized for block_size == 16")
    if weight.numel() % 16 != 0:
        raise ValueError("weight.numel() must be divisible by 16")

    num_blocks = weight.numel() // 16

    scale = torch.empty(
        num_blocks,
        device=weight.device,
        dtype=torch.float8_e4m3fn,
    )

    # 16 FP4 values = 8 bytes = 2 int32.
    # Store as int32 for fast vectorized writes, return uint8 view.
    code_i32 = torch.empty(
        num_blocks * 2,
        device=weight.device,
        dtype=torch.int32,
    )

    num_programs = persistent_grid(num_blocks, weight.device, config)

    scalesweep_quantize_kernel[(num_programs,)](
        weight,
        scale,
        code_i32,
        global_scale_inv,
        num_blocks,
        num_programs,
        LOWER_BOUND=lower_bound,
        NUM_CANDIDATES=upper_bound - lower_bound + 1,
    )

    code = code_i32.view(torch.uint8)

    scale_shape = (*weight.shape[:-1], weight.shape[-1] // 16)
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)

    return scale.view(scale_shape), code.view(code_shape)


def main():
    check_sm100()
    sm_count = torch.cuda.get_device_properties(weight.device).multi_processor_count
    print(f"[triton.ScaleSweep [{LOWER_BOUND}, {UPPER_BOUND}]] [SM {sm_count}]")
    for bsz in [1, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192]:
        weight = make_w(bsz, 8192)
        global_scale, global_scale_inv = get_nvfp4_global_scales(weight, FP8_MAX=256)

        for config in SCALESWEEP_CONFIGS:
            (scale, code), ms = time_cuda(
                lambda: scalesweep_quantize(
                    weight,
                    global_scale_inv,
                    BLOCK_SIZE,
                    LOWER_BOUND,
                    UPPER_BOUND,
                    config,
                )
            )

            reconstructed = dequantize("base", code, scale, global_scale, high_first=False)
            mse, max_abs_error = error_stats(weight, reconstructed)

            print(f"bsz = {bsz}, dim = {weight.shape[1]}")
            print(f"latency_ms    = {ms:.6f}")
            print(f"mse           = {mse:.8e}")
            print(f"max_abs_error = {max_abs_error:.8e}")
            print()

if __name__ == "__main__":
    main()