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
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=16, num_stages=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=16, num_stages=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=16, num_stages=3),
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

    e0 = q0 * scale - v0
    e1 = q1 * scale - v1
    e2 = q2 * scale - v2
    e3 = q3 * scale - v3
    e4 = q4 * scale - v4
    e5 = q5 * scale - v5
    e6 = q6 * scale - v6
    e7 = q7 * scale - v7
    e8 = q8 * scale - v8
    e9 = q9 * scale - v9
    e10 = q10 * scale - v10
    e11 = q11 * scale - v11
    e12 = q12 * scale - v12
    e13 = q13 * scale - v13
    e14 = q14 * scale - v14
    e15 = q15 * scale - v15

    return (
        e0 * e0 * iw0
        + e1 * e1 * iw1
        + e2 * e2 * iw2
        + e3 * e3 * iw3
        + e4 * e4 * iw4
        + e5 * e5 * iw5
        + e6 * e6 * iw6
        + e7 * e7 * iw7
        + e8 * e8 * iw8
        + e9 * e9 * iw9
        + e10 * e10 * iw10
        + e11 * e11 * iw11
        + e12 * e12 * iw12
        + e13 * e13 * iw13
        + e14 * e14 * iw14
        + e15 * e15 * iw15
    )


@triton.jit
def _pack_final_code_16_cols(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
):
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


@triton.jit
def scalesweep_quantize_kernel(
    weight_ptr,
    imp_ptr,
    scale_ptr,
    code_i32_ptr,
    global_scale_inv_ptr,
    num_blocks: tl.constexpr,
    grid_size: tl.constexpr,
    BLOCKS_PER_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)

    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM

    block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
    block_mask = block_offsets < num_blocks
    block_in_offsets = block_offsets - (block_offsets // BLOCKS_PER_OUT) * BLOCKS_PER_OUT
    imp_base = block_in_offsets * 16

    iw0 = tl.load(imp_ptr + imp_base + 0, mask=block_mask, other=0.0).to(tl.float32)
    iw1 = tl.load(imp_ptr + imp_base + 1, mask=block_mask, other=0.0).to(tl.float32)
    iw2 = tl.load(imp_ptr + imp_base + 2, mask=block_mask, other=0.0).to(tl.float32)
    iw3 = tl.load(imp_ptr + imp_base + 3, mask=block_mask, other=0.0).to(tl.float32)
    iw4 = tl.load(imp_ptr + imp_base + 4, mask=block_mask, other=0.0).to(tl.float32)
    iw5 = tl.load(imp_ptr + imp_base + 5, mask=block_mask, other=0.0).to(tl.float32)
    iw6 = tl.load(imp_ptr + imp_base + 6, mask=block_mask, other=0.0).to(tl.float32)
    iw7 = tl.load(imp_ptr + imp_base + 7, mask=block_mask, other=0.0).to(tl.float32)
    iw8 = tl.load(imp_ptr + imp_base + 8, mask=block_mask, other=0.0).to(tl.float32)
    iw9 = tl.load(imp_ptr + imp_base + 9, mask=block_mask, other=0.0).to(tl.float32)
    iw10 = tl.load(imp_ptr + imp_base + 10, mask=block_mask, other=0.0).to(tl.float32)
    iw11 = tl.load(imp_ptr + imp_base + 11, mask=block_mask, other=0.0).to(tl.float32)
    iw12 = tl.load(imp_ptr + imp_base + 12, mask=block_mask, other=0.0).to(tl.float32)
    iw13 = tl.load(imp_ptr + imp_base + 13, mask=block_mask, other=0.0).to(tl.float32)
    iw14 = tl.load(imp_ptr + imp_base + 14, mask=block_mask, other=0.0).to(tl.float32)
    iw15 = tl.load(imp_ptr + imp_base + 15, mask=block_mask, other=0.0).to(tl.float32)

    while block_start < num_blocks:
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        block_mask = block_offsets < num_blocks

        # weight layout:
        #   weight: [Out, In], contiguous
        #   flattened blocks are ordered row-major by 16-element chunks.
        #
        # imp layout:
        #   imp: [1, In], contiguous
        #   same column weights are shared across all Out rows.
        

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

        base_fp8 = (abs_max / 6.0).to(tl.float8e4nv)
        base_raw = base_fp8.to(tl.uint8, bitcast=True).to(tl.int32)

        for i in tl.static_range(0, NUM_CANDIDATES):
            raw_i = tl.minimum(
                tl.maximum(base_raw + LOWER_BOUND + i, 1),
                126,
            ).to(tl.uint8)

            scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
            scale_i = scale_fp8.to(tl.float32)
            inv_scale_i = 1.0 / scale_i

            mse_i = _weighted_mse_after_e2m1_roundtrip_16_cols(
                v0, v1, v2, v3,
                v4, v5, v6, v7,
                v8, v9, v10, v11,
                v12, v13, v14, v15,
                iw0, iw1, iw2, iw3,
                iw4, iw5, iw6, iw7,
                iw8, iw9, iw10, iw11,
                iw12, iw13, iw14, iw15,
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
    imp,
    global_scale_inv,
    block_size,
    lower_bound,
    upper_bound,
    config,
):
    if block_size != 16:
        raise ValueError("optimized kernel is specialized for block_size == 16")

    if weight.ndim != 2:
        raise ValueError(
            f"this weighted kernel expects weight.shape == [Out, In], "
            f"got {tuple(weight.shape)}"
        )

    if weight.shape[-1] % 16 != 0:
        raise ValueError("weight.shape[-1] must be divisible by 16")

    if imp.shape != (1, weight.shape[1]):
        raise ValueError(
            f"imp must have shape [1, In], got {tuple(imp.shape)}, "
            f"expected {(1, weight.shape[1])}"
        )

    if weight.device != imp.device:
        raise ValueError(
            f"weight and imp must be on the same device, "
            f"got weight.device={weight.device}, imp.device={imp.device}"
        )

    if not weight.is_contiguous():
        weight = weight.contiguous()

    if not imp.is_contiguous():
        imp = imp.contiguous()

    out_features = weight.shape[0]
    in_features = weight.shape[1]

    blocks_per_out = in_features // 16
    num_blocks = out_features * blocks_per_out

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
        imp,
        scale,
        code_i32,
        global_scale_inv,
        num_blocks,
        num_programs,
        BLOCKS_PER_OUT=blocks_per_out,
        LOWER_BOUND=lower_bound,
        NUM_CANDIDATES=upper_bound - lower_bound + 1,
        BLOCKS_PER_PROGRAM=config.kwargs["BLOCKS_PER_PROGRAM"],
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    code = code_i32.view(torch.uint8)

    scale_shape = (out_features, blocks_per_out)
    code_shape = (out_features, in_features // 2)

    return scale.view(scale_shape), code.view(code_shape)


def weighted_error_stats(weight, reconstructed, imp):
    """
    Optional CPU/PyTorch-side validation metric.

    weight:        [Out, In]
    reconstructed: [Out, In]
    imp:           [1, In]
    """
    err = reconstructed - weight
    weighted_mse = torch.mean(err.float() * err.float() * imp.float()).item()
    max_abs_error = torch.max(torch.abs(err)).item()
    return weighted_mse, max_abs_error


def main():
    check_sm100()

    weight = make_w()
    global_scale, global_scale_inv = get_nvfp4_global_scales(weight, FP8_MAX=256)

    if weight.ndim != 2:
        raise ValueError(
            "This weighted version assumes weight is 2D: [Out, In]. "
            f"Got weight.shape={tuple(weight.shape)}"
        )

    # Replace this with your real importance tensor.
    # Expected shape: [1, In].
    imp = torch.ones(
        (1, weight.shape[1]),
        device=weight.device,
        dtype=torch.float32,
    )

    sm_count = torch.cuda.get_device_properties(weight.device).multi_processor_count
    num_blocks = weight.numel() // BLOCK_SIZE

    print(f"sm_count         = {sm_count}")

    for config in SCALESWEEP_CONFIGS:
        num_programs = persistent_grid(num_blocks, weight.device, config)

        (scale, code), ms = time_cuda(
            lambda: scalesweep_quantize(
                weight,
                imp,
                global_scale_inv,
                BLOCK_SIZE,
                LOWER_BOUND,
                UPPER_BOUND,
                config,
            )
        )

        reconstructed = dequantize(
            "base",
            code,
            scale,
            global_scale,
            high_first=False,
        )

        mse, max_abs_error = error_stats(weight, reconstructed)
        weighted_mse, _ = weighted_error_stats(weight, reconstructed, imp)

        blocks_per_program = config.kwargs["BLOCKS_PER_PROGRAM"]
        name = (
            f"triton.ScaleSweep.PTX.WeightedMSE[blocks={blocks_per_program},"
            f"warps={config.num_warps},stages={config.num_stages},"
            f"programs={num_programs}]"
        )

        print_result(name, ms, mse, max_abs_error)
        print(f"  weighted_mse   = {weighted_mse:.8e}")

    print(f"q.shape          = {tuple(code.shape)}, {code.dtype}")
    print(f"scale.shape      = {tuple(scale.shape)}, {scale.dtype}")
    print(f"global_scale     = {global_scale.item():.8e}")
    print(f"global_scale_inv = {global_scale_inv.item():.8e}")
    print(f"candidate_range  = [{LOWER_BOUND}, {UPPER_BOUND}]")
    print(f"imp.shape        = {tuple(imp.shape)}, {imp.dtype}")


if __name__ == "__main__":
    main()