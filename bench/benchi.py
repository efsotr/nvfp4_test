import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imp", type=str, choices=["ones", "ramp", "random"], default="ones")
parser.add_argument("--dim", type=int, default=8192)
args = parser.parse_args()
print(args)

import torch
import triton
import triton.language as tl

from helper import (
    check_sm100,
    dequantize,
    error_stats,
    get_nvfp4_global_scales,
    make_w,
    time_cuda,
)
from helper16i import (
    _load_16_cols_2d_outtile,
    _load_imp_16_1din,
    _max_abs_16,
    _pack_final_code_16_cols,
    _weighted_mse_after_e2m1_roundtrip_16_cols,
)


BLOCK_SIZE = 16
LOWER_BOUND = -8
UPPER_BOUND = 7

SCALESWEEP_CONFIGS = [
    triton.Config({"OUTS_PER_PROGRAM": 32, "NUM_STAGES": 2}, num_warps=1, num_stages=2),
    triton.Config({"OUTS_PER_PROGRAM": 64, "NUM_STAGES": 2}, num_warps=2, num_stages=2),
    triton.Config({"OUTS_PER_PROGRAM": 128, "NUM_STAGES": 2}, num_warps=4, num_stages=2),
    triton.Config({"OUTS_PER_PROGRAM": 256, "NUM_STAGES": 2}, num_warps=8, num_stages=2),
    triton.Config({"OUTS_PER_PROGRAM": 512, "NUM_STAGES": 2}, num_warps=16, num_stages=2),
    triton.Config({"OUTS_PER_PROGRAM": 1024, "NUM_STAGES": 2}, num_warps=32, num_stages=2),
]


@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=[
        "NUM_PROGRAMS",
        "OUT_FEATURES",
        "IN_FEATURES",
        "BLOCKS_PER_OUT",
        "LOWER_BOUND",
        "NUM_CANDIDATES",
    ],
)
@triton.jit
def scalesweep_quantize_kernel(
    weight_ptr,
    imp_ptr,
    scale_ptr,
    code_i32_ptr,
    global_scale_inv_ptr,
    NUM_PROGRAMS: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    IN_FEATURES: tl.constexpr,
    BLOCKS_PER_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    OUTS_PER_PROGRAM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)

    # 2D launch:
    #   grid[0] sweeps out-channel tiles persistently.
    #   grid[1] is the fixed 16-column in-block.
    # This keeps in_base constant for the whole program, so imp[0, in_base:in_base+16]
    # is loaded once and reused across the out channels handled by that program.
    pid_out = tl.program_id(0)
    in_block = tl.program_id(1)
    in_base = in_block * 16

    (
        iw0, iw1, iw2, iw3,
        iw4, iw5, iw6, iw7,
        iw8, iw9, iw10, iw11,
        iw12, iw13, iw14, iw15,
    ) = _load_imp_16_1din(
        imp_ptr,
        in_base,
    )

    for out_start in tl.range(
        pid_out * OUTS_PER_PROGRAM,
        OUT_FEATURES,
        NUM_PROGRAMS * OUTS_PER_PROGRAM,
        num_stages=NUM_STAGES,
    ):
        out_offsets = out_start + tl.arange(0, OUTS_PER_PROGRAM)
        out_mask = out_offsets < OUT_FEATURES

        (
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
        ) = _load_16_cols_2d_outtile(
            weight_ptr,
            out_offsets,
            out_mask,
            in_base,
            IN_FEATURES,
            global_scale_inv,
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

        # scale is viewed as [Out, In // 16].
        block_offsets = out_offsets * BLOCKS_PER_OUT + in_block

        tl.store(
            scale_ptr + block_offsets,
            best_scale_fp8,
            mask=out_mask,
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
            mask=out_mask,
        )
        tl.store(
            code_i32_ptr + code_i32_offsets + 1,
            hi.to(tl.int32),
            mask=out_mask,
        )


def _make_imp(mode, dim, device):
    if mode == "ones":
        return torch.ones((1, dim), device=device, dtype=torch.float32)
    if mode == "ramp":
        return torch.linspace(0.25, 1.75, dim, device=device, dtype=torch.float32).view(1, dim)
    if mode == "random":
        return 0.25 + torch.rand((1, dim), device=device, dtype=torch.float32) * 1.5
    raise NotImplementedError(f"unsupported --imp {mode}")


def weighted_error_stats(weight, reconstructed, imp):
    err = reconstructed - weight
    weighted_mse = torch.mean(err.float() * err.float() * imp.float()).item()
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
        raise ValueError(f"weight must be 2D [Out, In], got {tuple(weight.shape)}")
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

    scalesweep_quantize_kernel[(NUM_PROGRAMS, blocks_per_out)](
        weight,
        imp,
        scale,
        code_i32,
        global_scale_inv,
        NUM_PROGRAMS,
        OUT_FEATURES=out_features,
        IN_FEATURES=in_features,
        BLOCKS_PER_OUT=blocks_per_out,
        LOWER_BOUND=lower_bound,
        NUM_CANDIDATES=upper_bound - lower_bound + 1,
    )

    code = code_i32.view(torch.uint8)

    scale_shape = (out_features, blocks_per_out)
    code_shape = (out_features, in_features // 2)

    return scale.view(scale_shape), code.view(code_shape)


def main():
    check_sm100()

    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"[triton.ScaleSweep.Imp[1,d_in] [{LOWER_BOUND}, {UPPER_BOUND}]] [SM {sm_count}]")

    for NUM_PROGRAMS in [sm_count, sm_count * 2, sm_count * 4]:
        print(f"NUM_PROGRAMS = {NUM_PROGRAMS}")
        for bsz in [1, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192]:
            weight = make_w(bsz, args.dim)
            imp = _make_imp(args.imp, weight.shape[1], weight.device)
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

            print(f"bsz = {bsz}, dim = {weight.shape[1]}, imp = {args.imp}")
            print(f"latency_ms    = {ms:.6f}")
            print(f"mse           = {mse:.8e}")
            print(f"weighted_mse  = {weighted_mse:.8e}")
            print(f"max_abs_error = {max_abs_error:.8e}")
            print()


if __name__ == "__main__":
    main()
