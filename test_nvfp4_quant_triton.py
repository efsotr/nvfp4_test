import argparse

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer

from helper import (
    fp32_round_to_fp4_value,
    fp4_block_code_sim,
    fp4_pair_values_sim_f32,
    quantize_error,
    timed,
    timed_cast_fp4,
)

FP4_MAX = 6.0
VLLM_LAUNCH_BLOCKS_CAP = 4
SCALESWEEP_PROGRAMS_CAP = 2048

PERSISTENT_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=8, num_stages=1),
]

SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=16, num_stages=2),
]

HEAVY_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCKS_PER_PROGRAM": 256}, num_warps=8, num_stages=3),
]


def _runtime_blocks_per_sm(block_threads, device=None):
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    max_threads_per_sm = getattr(props, "max_threads_per_multi_processor", 2048)
    blocks = max_threads_per_sm // block_threads if block_threads > 0 else 1
    return max(1, min(blocks, VLLM_LAUNCH_BLOCKS_CAP))


def persistent_grid(num_blocks, device):
    block_threads = 256
    num_programs = triton.cdiv(num_blocks, block_threads)
    blocks_per_sm = _runtime_blocks_per_sm(block_threads, device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return min(num_programs, max(1, sm_count * blocks_per_sm))


def sm_multiplier_grid(device, sm_multiplier):
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    return max(1, sm_count * sm_multiplier)


def scalesweep_grid(num_blocks):
    blocks_per_program = SCALESWEEP_CONFIGS[0].kwargs["BLOCKS_PER_PROGRAM"]
    return min(triton.cdiv(num_blocks, blocks_per_program), SCALESWEEP_PROGRAMS_CAP)


@triton.autotune(configs=HEAVY_CONFIGS, key=[])
@triton.jit
def cast_fp4_kernel(
    weight_ptr,
    code_ptr,
    duration_ns_ptr,
    num_blocks: tl.constexpr,
    grid_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM
    elem_offsets = tl.arange(0, BLOCK_SIZE)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    while block_start < num_blocks:
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        offsets = block_offsets[:, None] * BLOCK_SIZE + elem_offsets[None, :]
        code_offsets = block_offsets[:, None] * (BLOCK_SIZE // 2) + pair_offsets[None, :]
        mask = block_offsets[:, None] < num_blocks
        vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        start_ns = globaltimer()
        code = fp4_block_code_sim(vals, BLOCK_SIZE)
        end_ns = globaltimer()
        tl.store(code_ptr + code_offsets, code, mask=mask)
        tl.atomic_add(duration_ns_ptr + pid, end_ns - start_ns)
        block_start += grid_size * BLOCKS_PER_PROGRAM


@triton.autotune(configs=HEAVY_CONFIGS, key=["BLOCK_SIZE"])
@triton.jit
def absmax_quantize_kernel(
    weight_ptr,
    scale_ptr,
    code_ptr,
    global_scale_inv_ptr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    block_offsets = pid * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    elem_offsets = tl.arange(0, BLOCK_SIZE)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    offsets = block_offsets[:, None] * BLOCK_SIZE + elem_offsets[None, :]
    code_offsets = block_offsets[:, None] * (BLOCK_SIZE // 2) + pair_offsets[None, :]
    mask = block_offsets[:, None] < num_blocks
    vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    global_scale_inv = tl.load(global_scale_inv_ptr)

    vec_max = tl.max(tl.abs(vals), axis=1)
    sf_value = global_scale_inv * (vec_max * (1.0 / 6.0))
    sf_fp8 = sf_value.to(tl.float8e4nv)
    tl.store(scale_ptr + block_offsets, sf_fp8, mask=block_offsets < num_blocks)
    sf_value = sf_fp8.to(tl.float32)
    output_scale = tl.where(sf_value != 0.0, 1.0 / (sf_value * (1.0 / global_scale_inv)), 0.0)
    code = fp4_block_code_sim(vals * output_scale[:, None], BLOCK_SIZE)
    tl.store(code_ptr + code_offsets, code, mask=mask)


@triton.autotune(configs=HEAVY_CONFIGS, key=[])
@triton.jit
def absmax_persistent_quantize_kernel(
    weight_ptr,
    scale_ptr,
    code_ptr,
    global_scale_inv_ptr,
    num_blocks: tl.constexpr,
    grid_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM
    elem_offsets = tl.arange(0, BLOCK_SIZE)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    global_scale_inv = tl.load(global_scale_inv_ptr)
    while block_start < num_blocks:
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        offsets = block_offsets[:, None] * BLOCK_SIZE + elem_offsets[None, :]
        code_offsets = block_offsets[:, None] * (BLOCK_SIZE // 2) + pair_offsets[None, :]
        mask = block_offsets[:, None] < num_blocks
        vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        vec_max = tl.max(tl.abs(vals), axis=1)
        sf_value = global_scale_inv * (vec_max * (1.0 / 6.0))
        sf_fp8 = sf_value.to(tl.float8e4nv)
        tl.store(scale_ptr + block_offsets, sf_fp8, mask=block_offsets < num_blocks)
        sf_value = sf_fp8.to(tl.float32)
        output_scale = tl.where(sf_value != 0.0, 1.0 / (sf_value * (1.0 / global_scale_inv)), 0.0)
        code = fp4_block_code_sim(vals * output_scale[:, None], BLOCK_SIZE)
        tl.store(code_ptr + code_offsets, code, mask=mask)
        block_start += grid_size * BLOCKS_PER_PROGRAM


@triton.autotune(configs=HEAVY_CONFIGS, key=[])
@triton.jit
def four_over_six_quantize_kernel(
    weight_ptr,
    scale_ptr,
    code_ptr,
    global_scale_ptr,
    num_blocks: tl.constexpr,
    grid_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM
    elem_offsets = tl.arange(0, BLOCK_SIZE)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    global_scale = tl.load(global_scale_ptr)
    while block_start < num_blocks:
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        offsets = block_offsets[:, None] * BLOCK_SIZE + elem_offsets[None, :]
        code_offsets = block_offsets[:, None] * (BLOCK_SIZE // 2) + pair_offsets[None, :]
        mask = block_offsets[:, None] < num_blocks
        vals = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        vec_max = tl.max(tl.abs(vals), axis=1)
        vec_max = vec_max / 6.0 * (1.0 / global_scale)
        scale6_fp8 = vec_max.to(tl.float8e4nv)
        scale4_fp8 = (vec_max * 1.5).to(tl.float8e4nv)
        scale6 = scale6_fp8.to(tl.float32)
        scale4 = scale4_fp8.to(tl.float32)
        code6 = fp4_block_code_sim(vals * (1.0 / (scale6 * global_scale))[:, None], BLOCK_SIZE)
        q6 = fp4_pair_values_sim_f32(code6, BLOCK_SIZE, BLOCKS_PER_PROGRAM)
        err6 = q6 * scale6[:, None] * global_scale - vals
        code4 = fp4_block_code_sim(vals * (1.0 / (scale4 * global_scale))[:, None], BLOCK_SIZE)
        q4 = fp4_pair_values_sim_f32(code4, BLOCK_SIZE, BLOCKS_PER_PROGRAM)
        err4 = q4 * scale4[:, None] * global_scale - vals
        use6 = tl.sum(err6 * err6, axis=1) < tl.sum(err4 * err4, axis=1)
        scale = tl.where(use6, scale6_fp8, scale4_fp8)
        code = tl.where(use6[:, None], code6, code4)
        tl.store(scale_ptr + block_offsets, scale, mask=block_offsets < num_blocks)
        tl.store(code_ptr + code_offsets, code, mask=mask)
        block_start += grid_size * BLOCKS_PER_PROGRAM


@triton.autotune(configs=SCALESWEEP_CONFIGS, key=["BLOCK_SIZE", "LOWER_BOUND", "NUM_CANDIDATES"])
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
        best_scale_inv = 1.0 / best_scale_fp8.to(tl.float32)
        best_code = fp4_block_code_sim(vals * best_scale_inv[:, None], BLOCK_SIZE)
        tl.store(scale_ptr + block_offsets, best_scale_fp8, mask=block_offsets < num_blocks)
        tl.store(code_ptr + code_offsets, best_code, mask=mask)
        block_start += grid_size * BLOCKS_PER_PROGRAM


def check_shape(weight, block_size):
    if weight.numel() % block_size != 0:
        raise ValueError("weight.numel() must be divisible by block_size")
    if block_size % 2 != 0:
        raise ValueError("block_size must be even for packed FP4 codes")


def empty_outputs(weight, block_size):
    num_blocks = weight.numel() // block_size
    scale = torch.empty(num_blocks, device=weight.device, dtype=torch.float8_e4m3fn)
    code = torch.empty(weight.numel() // 2, device=weight.device, dtype=torch.uint8)
    return scale, code


def reshape_outputs(weight, scale, code, block_size):
    scale_shape = (*weight.shape[:-1], weight.shape[-1] // block_size)
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)
    return scale.view(scale_shape), code.view(code_shape)


def cast_fp4(weight, block_size):
    check_shape(weight, block_size)
    code = torch.empty(weight.numel() // 2, device=weight.device, dtype=torch.uint8)
    num_blocks = weight.numel() // block_size
    num_programs = persistent_grid(num_blocks, weight.device)
    duration_ns = torch.zeros(num_programs, device=weight.device, dtype=torch.int64)
    cast_fp4_kernel[(num_programs,)](
        weight,
        code,
        duration_ns,
        num_blocks,
        num_programs,
        BLOCK_SIZE=block_size,
    )
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)
    return code.view(code_shape), duration_ns


def absmax_quantize(weight, global_scale_inv, block_size):
    check_shape(weight, block_size)
    scale, code = empty_outputs(weight, block_size)
    num_blocks = weight.numel() // block_size
    grid = lambda meta: (triton.cdiv(num_blocks, meta["BLOCKS_PER_PROGRAM"]),)
    absmax_quantize_kernel[grid](
        weight,
        scale,
        code,
        global_scale_inv,
        num_blocks,
        BLOCK_SIZE=block_size,
    )
    return reshape_outputs(weight, scale, code, block_size)


def absmax_persistent_quantize(weight, global_scale_inv, block_size, sm_multiplier):
    check_shape(weight, block_size)
    scale, code = empty_outputs(weight, block_size)
    num_blocks = weight.numel() // block_size
    num_programs = sm_multiplier_grid(weight.device, sm_multiplier)
    absmax_persistent_quantize_kernel[(num_programs,)](
        weight,
        scale,
        code,
        global_scale_inv,
        num_blocks,
        num_programs,
        BLOCK_SIZE=block_size,
    )
    return reshape_outputs(weight, scale, code, block_size)


def four_over_six_quantize(weight, global_scale, block_size):
    check_shape(weight, block_size)
    scale, code = empty_outputs(weight, block_size)
    num_blocks = weight.numel() // block_size
    num_programs = persistent_grid(num_blocks, weight.device)
    four_over_six_quantize_kernel[(num_programs,)](
        weight,
        scale,
        code,
        global_scale,
        num_blocks,
        num_programs,
        BLOCK_SIZE=block_size,
    )
    return reshape_outputs(weight, scale, code, block_size)


def scalesweep_quantize(weight, global_scale_inv, block_size, lower_bound, upper_bound):
    check_shape(weight, block_size)
    scale, code = empty_outputs(weight, block_size)
    num_blocks = weight.numel() // block_size
    num_programs = scalesweep_grid(num_blocks)
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
    )
    return reshape_outputs(weight, scale, code, block_size)


def prepare_method_case(method, weight, global_scale, block_size, lower_bound, upper_bound):
    if method == "cast_fp4":
        inputs = (weight, block_size)
        return inputs, lambda: cast_fp4(*inputs)
    if method == "AbsMax":
        global_scale_inv = torch.reciprocal(global_scale)
        inputs = (weight, global_scale_inv, block_size)
        return inputs, lambda: absmax_quantize(*inputs)
    if method.startswith("AbsMaxPersistent") and method.endswith("x"):
        sm_multiplier = int(method.removeprefix("AbsMaxPersistent").removesuffix("x"))
        if sm_multiplier not in (1, 2, 4):
            raise ValueError(f"unsupported AbsMax persistent SM multiplier: {sm_multiplier}")
        global_scale_inv = torch.reciprocal(global_scale)
        inputs = (weight, global_scale_inv, block_size, sm_multiplier)
        return inputs, lambda: absmax_persistent_quantize(*inputs)
    if method == "4over6":
        inputs = (weight, global_scale, block_size)
        return inputs, lambda: four_over_six_quantize(*inputs)
    if method == "ScaleSweep":
        global_scale_inv = torch.reciprocal(global_scale)
        inputs = (weight, global_scale_inv, block_size, lower_bound, upper_bound)
        return inputs, lambda: scalesweep_quantize(*inputs)
    raise ValueError(f"unknown method: {method}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=2, default=(4096, 4096))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-scale", type=float, default=None)
    parser.add_argument("--methods", nargs="+", default=["AbsMax", "ScaleSweep"])
    parser.add_argument("--lower-bound", type=int, default=-8)
    parser.add_argument("--upper-bound", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("This benchmark only runs on CUDA GPUs.")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    weight = torch.randn(*args.shape, device=args.device, dtype=dtype).contiguous()
    if weight.shape[-1] % args.block_size != 0:
        raise ValueError("last dimension must be divisible by block_size")

    if args.global_scale is None:
        global_scale = (weight.abs().amax() / (256.0 * FP4_MAX)).float().reshape(())
    else:
        global_scale = torch.tensor(args.global_scale, device=args.device, dtype=torch.float32)

    torch.cuda.synchronize()
    sm_count = torch.cuda.get_device_properties(weight.device).multi_processor_count
    print(
        f"shape={tuple(weight.shape)} block_size={args.block_size} "
        f"warmup={args.warmup} repeat={args.repeat} sm_count={sm_count} "
        f"global_scale={global_scale.item():.8g}"
    )
    for method in args.methods:
        inputs, run_quantize = prepare_method_case(
            method,
            weight,
            global_scale,
            args.block_size,
            args.lower_bound,
            args.upper_bound,
        )
        if method == "cast_fp4":
            code, t_total = timed_cast_fp4(run_quantize, args.warmup, args.repeat)
            print(
                f"{method:10s} code={tuple(code.shape)} "
                f"input={[tuple(x.shape) if isinstance(x, torch.Tensor) else x for x in inputs]} "
                f"fp4_block_code_sim_time={t_total:.9f}s"
            )
            continue
        output, t_total = timed(run_quantize, args.warmup, args.repeat)
        scale, code = output
        mse, max_abs = quantize_error(weight, scale, code, global_scale, args.block_size)
        print(
            f"{method:10s} scale={tuple(scale.shape)} code={tuple(code.shape)} "
            f"input={[tuple(x.shape) if isinstance(x, torch.Tensor) else x for x in inputs]} "
            f"time={t_total:.6f}s mse={mse.item():.8g} max_abs={max_abs.item():.8g}"
        )


if __name__ == "__main__":
    main()
