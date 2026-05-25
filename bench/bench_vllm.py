from vllm._custom_ops import scaled_fp4_quant

from helper import (
    check_sm100,
    make_w,
    get_nvfp4_global_scales,
    time_cuda,
    error_stats,
    print_result,
    dequantize,
)


def main():
    check_sm100()
    W = make_w()

    global_scale, global_scale_inv = get_nvfp4_global_scales(W)

    (q, s), ms = time_cuda(lambda: scaled_fp4_quant(W, global_scale_inv))

    W_hat = dequantize(
        "base",
        q,
        s,
        global_scale,
        high_first=False,
    )

    mse, max_abs_error = error_stats(W, W_hat)

    print_result("vllm._custom_ops.scaled_fp4_quant", ms, mse, max_abs_error)
    print(f"q.shape          = {tuple(q.shape)}, {q.dtype}")
    print(f"scale.shape      = {tuple(s.shape)}, {s.dtype}")
    print(f"global_scale     = {global_scale.item():.8e}")
    print(f"global_scale_inv = {global_scale_inv.item():.8e}")


if __name__ == "__main__":
    main()