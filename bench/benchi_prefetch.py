import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=8192)
parser.add_argument("--load", type=str, choices=["default", "sep", "trans"], default="sep")
parser.add_argument("--mse", type=str, choices=["direct", "default"], default="default")
parser.add_argument("--imp", type=str, choices=["ones", "ramp", "random"], default="ones")
args = parser.parse_args()
print(args)

from helper import (
    check_sm100,
    dequantize,
    error_stats,
    get_nvfp4_global_scales,
    make_w,
    time_cuda,
)
from helper16 import (
    _load_16_cols_2d_seperate,
    _load_16_cols_2d,
    _load_16_cols_2d_trans,
    _max_abs_16,
    _pack_final_code_16_cols,
    _fp32x16_to_e2m1_roundtrip_fp32x16,
)
from benchi_global import (
    _weighted_mse_after_e2m1_roundtrip_16_cols,
    _weighted_mse_after_e2m1_roundtrip_16_cols_direct,
    _load_imp_16_cols_global,
    _load_imp_16_cols_global_trans,
)

LOAD_FN = None
if args.load == "default":
    LOAD_FN = _load_16_cols_2d
elif args.load == "sep":
    LOAD_FN = _load_16_cols_2d_seperate
elif args.load == "trans":
    LOAD_FN = _load_16_cols_2d_trans
else:
    raise NotImplementedError(f"unsupported --load {args.load}")

import torch
sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
NUM_PROGRAMS_LIST = [sm_count, sm_count * 2, sm_count * 4]
bsz_list = [1, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192]

import triton
import triton.language as tl

BLOCK_SIZE = 16
LOWER_BOUND = -8
UPPER_BOUND = 7

SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32, "NUM_STAGES": 2}, num_warps=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 64, "NUM_STAGES": 2}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 128, "NUM_STAGES": 2}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 256, "NUM_STAGES": 2}, num_warps=8),
    triton.Config({"BLOCKS_PER_PROGRAM": 512, "NUM_STAGES": 2}, num_warps=16),
    triton.Config({"BLOCKS_PER_PROGRAM": 1024, "NUM_STAGES": 2}, num_warps=32),
]


MSE_FN = None
if args.mse == "default":
    MSE_FN = _weighted_mse_after_e2m1_roundtrip_16_cols
elif args.mse == "direct":
    MSE_FN = _weighted_mse_after_e2m1_roundtrip_16_cols_direct
else:
    raise NotImplementedError(f"unsupported --mse {args.mse}")


@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=["NUM_BLOCKS", "NUM_PROGRAMS", "LOWER_BOUND", "NUM_CANDIDATES", "BLOCKS_PER_OUT"],
)
@triton.jit
def scalesweep_quantize_kernel(
    weight_ptr,
    imp_ptr,
    scale_ptr,
    code_i32_ptr,
    global_scale_inv_ptr,
    NUM_BLOCKS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    BLOCKS_PER_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    imp_local_ptr = tl.load(imp_ptr + tl.arange(0, BLOCKS_PER_OUT * 16), eviction_policy="evict_last").to(tl.pointer_type(tl.bfloat16))
    global_scale_inv = tl.load(global_scale_inv_ptr)

    pid = tl.program_id(0)

    for block_start in tl.range(
        pid * BLOCKS_PER_PROGRAM,
        NUM_BLOCKS,
        NUM_PROGRAMS * BLOCKS_PER_PROGRAM,
        num_stages=NUM_STAGES,
    ):
        block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
        block_mask = block_offsets < NUM_BLOCKS

        (
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
        ) = LOAD_FN(
            weight_ptr,
            block_offsets,
            block_mask,
            global_scale_inv,
        )

        (
            iw0, iw1, iw2, iw3,
            iw4, iw5, iw6, iw7,
            iw8, iw9, iw10, iw11,
            iw12, iw13, iw14, iw15,
        ) = _load_imp_16_cols_global(
            imp_local_ptr,
            block_offsets,
            block_mask,
            BLOCKS_PER_OUT,
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

            mse_i = MSE_FN(
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


def make_imp(kind, dim, device):
    if kind == "ones":
        return torch.ones((1, dim), device=device, dtype=torch.float32)
    if kind == "ramp":
        return torch.linspace(0.25, 2.0, dim, device=device, dtype=torch.float32).view(1, dim)
    if kind == "random":
        g = torch.Generator(device=device)
        g.manual_seed(123)
        return (0.25 + 1.75 * torch.rand((1, dim), device=device, dtype=torch.float32, generator=g))
    raise NotImplementedError(f"unsupported --imp {kind}")


def weighted_error_stats(weight, reconstructed, imp):
    err = reconstructed.float() - weight.float()
    weighted_mse = torch.mean(err * err * imp.float()).item()
    max_abs_error = torch.max(torch.abs(err)).item()
    return weighted_mse, max_abs_error


def scalesweep_quantize(
    weight,
    imp,
    global_scale_inv,
    block_size,
    lower_bound,
    upper_bound,
    NUM_PROGRAMS,
):
    if block_size != 16:
        raise ValueError("optimized kernel is specialized for block_size == 16")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [bsz, dim], got {tuple(weight.shape)}")
    if weight.shape[-1] % 16 != 0:
        raise ValueError("weight.shape[-1] must be divisible by 16")
    if imp.shape != (1, weight.shape[-1]):
        raise ValueError(
            f"imp must have shape [1, d_in], got {tuple(imp.shape)}, expected {(1, weight.shape[-1])}"
        )
    if weight.device != imp.device:
        raise ValueError(f"weight and imp must be on the same device: {weight.device} vs {imp.device}")

    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not imp.is_contiguous():
        imp = imp.contiguous()

    num_blocks = weight.numel() // 16
    blocks_per_out = weight.shape[-1] // 16

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

    scalesweep_quantize_kernel[(NUM_PROGRAMS,)](
        weight,
        imp,
        scale,
        code_i32,
        global_scale_inv,
        num_blocks,
        NUM_PROGRAMS,
        BLOCKS_PER_OUT=blocks_per_out,
        LOWER_BOUND=lower_bound,
        NUM_CANDIDATES=upper_bound - lower_bound + 1,
    )

    code = code_i32.view(torch.uint8)

    scale_shape = (*weight.shape[:-1], weight.shape[-1] // 16)
    code_shape = (*weight.shape[:-1], weight.shape[-1] // 2)

    return scale.view(scale_shape), code.view(code_shape)


def main():
    check_sm100()
    print(f"[triton.ScaleSweep.i.prefetch [{LOWER_BOUND}, {UPPER_BOUND}]] [SM {sm_count}]")

    for NUM_PROGRAMS in NUM_PROGRAMS_LIST:
        print(f"NUM_PROGRAMS = {NUM_PROGRAMS}")
        for bsz in bsz_list:
            weight = make_w(bsz, args.dim)
            imp = make_imp(args.imp, weight.shape[1], weight.device)
            global_scale, global_scale_inv = get_nvfp4_global_scales(weight, FP8_MAX=256)

            (scale, code), ms = time_cuda(
                lambda: scalesweep_quantize(
                    weight,
                    imp,
                    global_scale_inv,
                    BLOCK_SIZE,
                    LOWER_BOUND,
                    UPPER_BOUND,
                    NUM_PROGRAMS,
                )
            )

            reconstructed = dequantize("base", code, scale, global_scale, high_first=False)
            mse, max_abs_error = error_stats(weight, reconstructed)
            weighted_mse, _ = weighted_error_stats(weight, reconstructed, imp)

            print(f"bsz = {bsz}, dim = {weight.shape[1]}")
            print(f"latency_ms    = {ms:.6f}")
            print(f"mse           = {mse:.8e}")
            print(f"weighted_mse  = {weighted_mse:.8e}")
            print(f"max_abs_error = {max_abs_error:.8e}")
            print(flush=True)


if __name__ == "__main__":
    main()
