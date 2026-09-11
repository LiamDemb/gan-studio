"""Fixed separable [1,3,3,1] FIR. All coordinates and adjoints are ours.

Contract: zero-insert INCLUDING trailing zeros, pad, correlate, decimate.
Output size = floor((input_size * up + left + right - 4) / down) + 1.
The symmetric filter makes correlation and convolution equivalent.
"""

import torch
from torch.nn import functional as F
from .conv_grad import conv2d


def resample(x, up=1, down=1, left=1, right=2, backend="torch", conv_backend="native"):
    if x.ndim != 4 or up not in (1, 2) or down not in (1, 2):
        raise ValueError("Expected NCHW and up/down in {1,2}")
    if left < 0 or right < 0:
        raise ValueError("Negative padding is not supported")
    if backend == "triton":
        if not x.is_cuda:
            raise ValueError("The Triton backend requires CUDA")
        from .triton_fir import fir

        return fir(x, up, down, left, right)
    if backend != "torch":
        raise ValueError(f"Unknown FIR backend: {backend}")
    n, c, h, w = x.shape
    if up > 1:
        # Padding a singleton dimension inserts zeros without assignment into
        # a differentiable view. Works for first and higher derivatives.
        x = F.pad(x.reshape(n, c, h, 1, w, 1), (0, up - 1, 0, 0, 0, up - 1))
        x = x.reshape(n, c, h * up, w * up)
    taps = x.new_tensor([1.0, 3.0, 3.0, 1.0])
    kernel = (taps[:, None] * taps[None, :]) * (up * up / 64)
    kernel = kernel.expand(c, 1, 4, 4)
    x = F.pad(x, (left, right, left, right))
    # Preserve input dtype even inside autocast. FIR is linear and its adjoint
    # must have the same well-defined numerical contract in both backends.
    with torch.autocast(
        device_type="cpu" if x.device.type == "meta" else x.device.type, enabled=False
    ):
        return conv2d(x, kernel, stride=down, groups=c, backend=conv_backend)


def upsample(x, backend="torch", conv_backend="native"):
    return resample(
        x, up=2, left=2, right=1, backend=backend, conv_backend=conv_backend
    )


def downsample(x, backend="torch", conv_backend="native"):
    return resample(
        x, down=2, left=1, right=1, backend=backend, conv_backend=conv_backend
    )


def blur(x, backend="torch", conv_backend="native"):
    return resample(x, left=1, right=2, backend=backend, conv_backend=conv_backend)
