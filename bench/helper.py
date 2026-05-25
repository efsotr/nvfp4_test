import torch

M = 4096
K = 4096
SEED = 0
DTYPE = torch.bfloat16
DEVICE = "cuda"

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX_NVFP4 = 448.0
FP8_E4M3_MAX_4OVER6 = 256.0


def check_sm100():
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability()
    assert major >= 10, f"need SM >= 100, got sm_{major}{minor}"


def make_w():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return torch.randn((M, K), device=DEVICE, dtype=DTYPE).contiguous()


def get_nvfp4_global_scales(x):
    amax = x.abs().max().float()
    global_scale = amax / (FP4_E2M1_MAX * FP8_E4M3_MAX_NVFP4)
    global_scale_inv = global_scale.reciprocal()
    return global_scale, global_scale_inv


def time_cuda(fn, warmup=5, iters=20):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    end.record()

    return out, start.elapsed_time(end) / iters


def error_stats(ref, pred):
    diff = pred.float() - ref.float()
    mse = torch.mean(diff * diff).item()
    max_abs_error = torch.max(torch.abs(diff)).item()
    return mse, max_abs_error


def print_result(name, ms, mse, max_abs_error):
    print(f"[{name}]")
    print(f"latency_ms    = {ms:.6f}")
    print(f"mse           = {mse:.8e}")
    print(f"max_abs_error = {max_abs_error:.8e}")


_FP4_E2M1_LUT = None


def fp4_e2m1_lut(device):
    global _FP4_E2M1_LUT
    if _FP4_E2M1_LUT is None or _FP4_E2M1_LUT.device != device:
        _FP4_E2M1_LUT = torch.tensor(
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            ],
            device=device,
            dtype=torch.float32,
        )
    return _FP4_E2M1_LUT


def unpack_fp4_e2m1(packed, high_first=False):
    assert packed.dtype == torch.uint8

    lut = fp4_e2m1_lut(packed.device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    if high_first:
        out = torch.stack((lut[high.long()], lut[low.long()]), dim=-1)
    else:
        out = torch.stack((lut[low.long()], lut[high.long()]), dim=-1)

    return out.reshape(packed.shape[0], packed.shape[1] * 2)

def dequantize_base(q, scale_fp8, global_scale, high_first=False):
    values = unpack_fp4_e2m1(q, high_first=high_first)
    scales = scale_fp8.to(torch.float32).repeat_interleave(16, dim=1)
    return values * scales * global_scale.float()

def dequantize_4over6_raw(
    values,
    scale_factors,
    amax,
    *,
    original_shape=(M, K),
    padded_shape=(M, K),
    dtype=None,
    scale_rule=None,
    round_style=None,
    scale_factors_are_in_blackwell_layout=True,
    out_dtype=torch.float32,
    intermediate_dtype=torch.float16,
):
    from fouroversix.quantize.dequantize_utils import from_blocked, unpack_packed_fp4
    from fouroversix.utils import DataType, RoundStyle, ScaleRule

    dtype = DataType.nvfp4 if dtype is None else dtype
    scale_rule = ScaleRule.mse if scale_rule is None else scale_rule
    round_style = RoundStyle.nearest if round_style is None else round_style

    # This mirrors QuantizeBackendBase.dequantize(), but without QuantizedTensor.
    x = unpack_packed_fp4(values).to(intermediate_dtype)

    if scale_factors_are_in_blackwell_layout:
        scales = from_blocked(
            scale_factors,
            (
                padded_shape[0],
                padded_shape[1] // dtype.block_size,
            ),
        )
    else:
        scales = scale_factors

    scales = scales.to(intermediate_dtype).repeat_interleave(dtype.block_size, -1)

    x = x * scales

    x = (
        x.to(torch.float32)
        * amax.float()
        / (
            dtype.quantized_value_type.get_maximum_value(scale_rule)
            * dtype.scale_type.get_maximum_value(scale_rule)
            * round_style.adjustment_factor
        )
    ).to(out_dtype)

    if x.shape != original_shape:
        x = x[: original_shape[0], : original_shape[1]]

    return x


def dequantize(kind, *args, **kwargs):
    if kind == "base":
        return dequantize_base(*args, **kwargs)
    if kind in ("4over6", "fouroversix"):
        return dequantize_4over6_raw(*args, **kwargs)

    raise ValueError(f"unknown dequant kind: {kind}")
