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
def _mse_after_e2m1_roundtrip_16(vals, inv_scale, scale):
    """
    vals:      [BLOCKS_PER_PROGRAM, 16], already global-scale normalized.
    inv_scale: [BLOCKS_PER_PROGRAM]
    scale:     [BLOCKS_PER_PROGRAM]

    Computes:
      q = e2m1_round(vals / scale)
      mse = sum((q * scale - vals)^2)
    """
    x = vals * inv_scale[:, None]

    (
        q0, q1, q2, q3,
        q4, q5, q6, q7,
        q8, q9, q10, q11,
        q12, q13, q14, q15,
    ) = _fp32x16_to_e2m1_roundtrip_fp32x16(
        x[:, 0], x[:, 1], x[:, 2], x[:, 3],
        x[:, 4], x[:, 5], x[:, 6], x[:, 7],
        x[:, 8], x[:, 9], x[:, 10], x[:, 11],
        x[:, 12], x[:, 13], x[:, 14], x[:, 15],
    )

    e0 = q0 * scale - vals[:, 0]
    e1 = q1 * scale - vals[:, 1]
    e2 = q2 * scale - vals[:, 2]
    e3 = q3 * scale - vals[:, 3]
    e4 = q4 * scale - vals[:, 4]
    e5 = q5 * scale - vals[:, 5]
    e6 = q6 * scale - vals[:, 6]
    e7 = q7 * scale - vals[:, 7]
    e8 = q8 * scale - vals[:, 8]
    e9 = q9 * scale - vals[:, 9]
    e10 = q10 * scale - vals[:, 10]
    e11 = q11 * scale - vals[:, 11]
    e12 = q12 * scale - vals[:, 12]
    e13 = q13 * scale - vals[:, 13]
    e14 = q14 * scale - vals[:, 14]
    e15 = q15 * scale - vals[:, 15]

    return (
        e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3
        + e4 * e4 + e5 * e5 + e6 * e6 + e7 * e7
        + e8 * e8 + e9 * e9 + e10 * e10 + e11 * e11
        + e12 * e12 + e13 * e13 + e14 * e14 + e15 * e15
    )


@triton.jit
def _pack_final_code_16(vals, inv_scale):
    """
    vals:      [BLOCKS_PER_PROGRAM, 16]
    inv_scale: [BLOCKS_PER_PROGRAM]

    Returns:
      lo, hi uint32 vectors, each [BLOCKS_PER_PROGRAM].
    """
    x = vals * inv_scale[:, None]

    return _fp32x16_to_e2m1_u32x2(
        x[:, 0], x[:, 1], x[:, 2], x[:, 3],
        x[:, 4], x[:, 5], x[:, 6], x[:, 7],
        x[:, 8], x[:, 9], x[:, 10], x[:, 11],
        x[:, 12], x[:, 13], x[:, 14], x[:, 15],
    )


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

    elem_offsets = tl.arange(0, 16)

    while block_start < num_blocks:
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        block_mask = block_offsets < num_blocks

        offsets = block_offsets[:, None] * 16 + elem_offsets[None, :]

        vals = (
            tl.load(weight_ptr + offsets, mask=block_mask[:, None], other=0.0)
            .to(tl.float32)
            * global_scale_inv
        )

        base_fp8 = (tl.max(tl.abs(vals), axis=1) / 6.0).to(tl.float8e4nv)
        base_raw = base_fp8.to(tl.uint8, bitcast=True).to(tl.int32)

        for i in tl.static_range(0, NUM_CANDIDATES):
            raw_i = tl.minimum(
                tl.maximum(base_raw + LOWER_BOUND + i, 1),
                126,
            ).to(tl.uint8)

            scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
            scale_i = scale_fp8.to(tl.float32)
            inv_scale_i = 1.0 / scale_i

            mse_i = _mse_after_e2m1_roundtrip_16(vals, inv_scale_i, scale_i)

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

        lo, hi = _pack_final_code_16(vals, best_scale_inv)

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
        BLOCKS_PER_PROGRAM=config.kwargs["BLOCKS_PER_PROGRAM"],
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )

    code = code_i32.view(torch.uint8)

    scale_shape = (*weight.shape[:-1], weight.shape[-1] // 16)
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)

    return scale.view(scale_shape), code.view(code_shape)


def main():
    check_sm100()

    weight = make_w()
    global_scale, global_scale_inv = get_nvfp4_global_scales(weight, FP8_MAX=256)

    sm_count = torch.cuda.get_device_properties(weight.device).multi_processor_count
    num_blocks = weight.numel() // BLOCK_SIZE

    print(f"sm_count         = {sm_count}")

    for config in SCALESWEEP_CONFIGS:
        num_programs = persistent_grid(num_blocks, weight.device, config)

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

        reconstructed = dequantize(
            "base",
            code,
            scale,
            global_scale,
            high_first=False,
        )

        mse, max_abs_error = error_stats(weight, reconstructed)

        blocks_per_program = config.kwargs["BLOCKS_PER_PROGRAM"]
        name = (
            f"triton.ScaleSweep.PTX[blocks={blocks_per_program},"
            f"warps={config.num_warps},stages={config.num_stages},"
            f"programs={num_programs}]"
        )

        print_result(name, ms, mse, max_abs_error)

    print(f"q.shape          = {tuple(code.shape)}, {code.dtype}")
    print(f"scale.shape      = {tuple(scale.shape)}, {scale.dtype}")
    print(f"global_scale     = {global_scale.item():.8e}")
    print(f"global_scale_inv = {global_scale_inv.item():.8e}")
    print(f"candidate_range  = [{LOWER_BOUND}, {UPPER_BOUND}]")


if __name__ == "__main__":
    main()