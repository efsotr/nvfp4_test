import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
def fp32_round_to_fp4_code(x):
    ax = tl.abs(x)
    le2 = ax <= 2.0
    le4 = ax <= 4.0
    exp = tl.where(le2, 0.5, tl.where(le4, 1.0, 2.0))
    r = libdevice.round(ax / exp)
    mag = tl.where(le2, r, tl.where(le4, r + 2.0, tl.minimum(r + 4.0, 7.0))).to(tl.uint8)
    sign = (x < 0.0).to(tl.uint8) << 3
    return mag | sign


@triton.jit
def fp4_pair_code_sim(x0, x1):
    return fp32_round_to_fp4_code(x0) | (fp32_round_to_fp4_code(x1) << 4)


@triton.jit
def fp4_block_code_sim(x, BLOCK_SIZE: tl.constexpr):
    x_pair = x.reshape((x.shape[0], BLOCK_SIZE // 2, 2))
    x0, x1 = x_pair.split()
    return fp4_pair_code_sim(x0, x1)


@triton.jit
def fp32_round_to_fp4_value(x):
    ax = tl.abs(x)
    exp = tl.where(ax <= 2.0, 0.5, tl.where(ax <= 4.0, 1.0, 2.0))
    q = libdevice.round(x / exp) * exp
    return tl.minimum(tl.maximum(q, -6.0), 6.0)


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
    code_ptr,
    global_scale_inv_ptr,
    num_blocks: tl.constexpr,
    grid_size: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)
    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM
    elem_offsets = tl.arange(0, BLOCK_SIZE)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    while block_start < num_blocks:
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        offsets = block_offsets[:, None] * BLOCK_SIZE + elem_offsets[None, :]
        code_offsets = block_offsets[:, None] * (BLOCK_SIZE // 2) + pair_offsets[None, :]
        mask = block_offsets[:, None] < num_blocks
        vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * global_scale_inv
        base_fp8 = (tl.max(tl.abs(vals), axis=1) / 6.0).to(tl.float8e4nv)
        base_raw = base_fp8.to(tl.uint8, bitcast=True).to(tl.int32)
        best_mse = tl.full((BLOCKS_PER_PROGRAM,), float("inf"), tl.float32)
        best_scale_fp8 = tl.full((BLOCKS_PER_PROGRAM,), 0, tl.float8e4nv)
        for i in tl.static_range(0, NUM_CANDIDATES):
            raw_i = tl.minimum(tl.maximum(base_raw + LOWER_BOUND + i, 1), 126).to(tl.uint8)
            scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
            scale_i = scale_fp8.to(tl.float32)
            q_i = fp32_round_to_fp4_value(vals * (1.0 / scale_i)[:, None])
            err_i = q_i * scale_i[:, None] - vals
            mse_i = tl.sum(err_i * err_i, axis=1)
            better = mse_i < best_mse
            best_mse = tl.where(better, mse_i, best_mse)
            best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)
        tl.store(scale_ptr + block_offsets, best_scale_fp8, mask=block_offsets < num_blocks)
        best_scale_inv = 1.0 / best_scale_fp8.to(tl.float32)
        best_code = fp4_block_code_sim(vals * best_scale_inv[:, None], BLOCK_SIZE)
        tl.store(code_ptr + code_offsets, best_code, mask=mask)
        block_start += grid_size * BLOCKS_PER_PROGRAM


def scalesweep_quantize(weight, global_scale_inv, block_size, lower_bound, upper_bound, config):
    if weight.numel() % block_size != 0:
        raise ValueError("weight.numel() must be divisible by block_size")
    if block_size % 2 != 0:
        raise ValueError("block_size must be even for packed FP4 codes")

    num_blocks = weight.numel() // block_size
    scale = torch.empty(num_blocks, device=weight.device, dtype=torch.float8_e4m3fn)
    code = torch.empty(weight.numel() // 2, device=weight.device, dtype=torch.uint8)
    num_programs = persistent_grid(num_blocks, weight.device, config)
    scalesweep_quantize_kernel[(num_programs,)](
        weight,
        scale,
        code,
        global_scale_inv,
        num_blocks,
        num_programs,
        LOWER_BOUND=lower_bound,
        NUM_CANDIDATES=upper_bound - lower_bound + 1,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=config.kwargs["BLOCKS_PER_PROGRAM"],
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    scale_shape = (*weight.shape[:-1], weight.shape[-1] // block_size)
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)
    return scale.view(scale_shape), code.view(code_shape)


def main():
    # check_sm100()
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
            f"triton.ScaleSweep[blocks={blocks_per_program},"
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
