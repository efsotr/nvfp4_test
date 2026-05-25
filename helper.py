import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


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
def fp4_nibble_value_sim_f16(code):
    mag = code & 0x07
    fmag = mag.to(tl.float16)
    val = tl.where(mag <= 4, fmag * 0.5, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))).to(tl.float16)
    return tl.where((code & 0x08) != 0, -val, val)


@triton.jit
def fp4_pair_value_sim_f16(code):
    return fp4_nibble_value_sim_f16(code & 0x0F), fp4_nibble_value_sim_f16((code >> 4) & 0x0F)


@triton.jit
def fp4_pair_values_sim_f32(code, BLOCK_SIZE: tl.constexpr, BLOCKS_PER_PROGRAM: tl.constexpr):
    q0, q1 = fp4_pair_value_sim_f16(code)
    return tl.join(q0, q1).reshape((BLOCKS_PER_PROGRAM, BLOCK_SIZE)).to(tl.float32)


@triton.jit
def fp32_round_to_fp4_value(x):
    ax = tl.abs(x)
    exp = tl.where(ax <= 2.0, 0.5, tl.where(ax <= 4.0, 1.0, 2.0))
    q = libdevice.round(x / exp) * exp
    return tl.minimum(tl.maximum(q, -6.0), 6.0)


def timed(fn, warmup, repeat):
    ret = None
    for _ in range(warmup):
        ret = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        ret = fn()
    end.record()
    torch.cuda.synchronize()
    return ret, start.elapsed_time(end) / repeat / 1000.0


def timed_cast_fp4(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    durations = []
    code = None
    for _ in range(repeat):
        code, duration_ns = fn()
        durations.append(duration_ns)
    torch.cuda.synchronize()
    compute_time = torch.stack(durations).to(torch.float32).mean().item() * 1.0e-9
    return code, compute_time


@torch.compile
def quantize_error(weight, scale, code, global_scale, block_size):
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=code.device,
        dtype=torch.float32,
    )
    low = code & 0x0F
    high = (code >> 4) & 0x0F
    packed = torch.stack((low, high), dim=-1).flatten(-2)
    value = fp4_values[packed.long() & 0x07]
    value = torch.where((packed & 0x08) != 0, -value, value)
    expanded_scale = scale.to(torch.float32).repeat_interleave(block_size, dim=-1)
    reconstructed = value * expanded_scale * global_scale
    error = reconstructed - weight.to(torch.float32)
    return (error * error).mean(), error.abs().amax()
