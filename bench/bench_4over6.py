from fouroversix.quantize.cuda.ops import quantize_to_fp4 as cuda_quantize

from helper import (
    check_sm100,
    make_w,
    time_cuda,
    error_stats,
    print_result,
    dequantize,
)


# FourOverSix CUDA selection_rule:
# static_6 = 0
# static_4 = 1
# mae      = 2
# mse      = 3
# abs_max  = 4
SCALE_RULE = "mse"
SELECTION_RULE = 3


def quant_4over6_raw(W):
    q, s, amax = cuda_quantize(
        W,
        True,            # is_nvfp4
        True,            # is_rtn / nearest
        False,           # is_rht
        False,           # is_2d
        False,           # is_transpose
        SELECTION_RULE,  # mse = 3
        -1,              # rbits
    )
    return q, s, amax


def main():
    check_sm100()
    W = make_w()

    (q, s, amax), ms = time_cuda(lambda: quant_4over6_raw(W))

    W_hat = dequantize(
        "4over6",
        q,
        s,
        amax,
        original_shape=W.shape,
        scale_rule=SCALE_RULE,
    )

    mse, max_abs_error = error_stats(W, W_hat)

    print_result("fouroversix.cuda.raw.4over6", ms, mse, max_abs_error)
    print(f"q.shape    = {tuple(q.shape)}, {q.dtype}")
    print(f"s.shape    = {tuple(s.shape)}, {s.dtype}")
    print(f"amax       = {amax.item():.8e}")
    print(f"scale_rule = {SCALE_RULE}")


if __name__ == "__main__":
    main()