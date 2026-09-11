"""Two algebraically equivalent execution strategies, no imported GAN ops."""

import torch
from torch.nn import functional as F
from .conv_grad import conv2d


def modulated_conv2d(
    x,
    weight,
    style,
    demodulate=True,
    backend="factorized",
    eps=1e-8,
    conv_backend="native",
):
    """weight: [O,I,K,K], style: [N,I]. Weight is already equalised.

    Factorization avoids allocating [N,O,I,K,K] and lets a normal minibatch
    convolution use cuDNN/Tensor Cores. Speed versus groups is shape-dependent.
    FP64 is preserved for numerical differentiation; norms otherwise use FP32.
    """
    n, cin, h, w = x.shape
    cout, _, kh, kw = weight.shape
    with torch.autocast(
        device_type="cpu" if x.device.type == "meta" else x.device.type, enabled=False
    ):
        calc_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
        k = weight.to(calc_dtype)
        s = style.to(calc_dtype)
        if demodulate:
            # d[n,o]^-2 = sum_i style[n,i]^2 * sum_uv weight[o,i,u,v]^2
            inv_norm = (s.square() @ k.square().sum((2, 3)).t() + eps).rsqrt()
    if backend == "factorized":
        y = conv2d(
            x * style.to(x.dtype)[:, :, None, None],
            weight,
            padding=kh // 2,
            backend=conv_backend,
            weight_independent=True,
        )
        return y * inv_norm.to(y.dtype)[:, :, None, None] if demodulate else y
    if backend == "grouped":
        wk = k[None] * s[:, None, :, None, None]
        if demodulate:
            wk = wk * inv_norm[:, :, None, None, None]
        # This kernel depends on style: its direct weight derivative is needed
        # even when differentiating only wrt ws. Never suppress that branch.
        y = conv2d(
            x.reshape(1, n * cin, h, w),
            wk.to(x.dtype).reshape(n * cout, cin, kh, kw),
            padding=kh // 2,
            groups=n,
            backend=conv_backend,
        )
        return y.reshape(n, cout, y.shape[2], y.shape[3])
    raise ValueError(f"Unknown modulation backend: {backend}")
