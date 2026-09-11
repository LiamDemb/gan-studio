"""Independent convolution adjoints for eager FP32/FP64 regularisation.

For y=C(x,w), <u,C(v,w)> gives the input adjoint; <u,C(x,v)>
gives the weight adjoint. Differentiating these bilinear identities yields
the backward methods below. No third-party GAN implementation is used.
Ordinary/compiled training still uses native autograd. GPU speed is unverified.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

_skip_direct_weight = ContextVar("sg2_skip_direct_weight", default=False)


@contextmanager
def input_derivative_only():
    """Only around autograd.grad wrt real images or factorized style inputs.

    Never use this for a requested derivative through a dynamic/grouped kernel.
    It suppresses the direct weight branch, NOT the weight dependence of dx.
    The outer regulariser backward runs with the context restored.
    """
    token = _skip_direct_weight.set(True)
    try:
        yield
    finally:
        _skip_direct_weight.reset(token)


def conv2d(
    x,
    weight,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    backend="native",
    weight_independent=False,
):
    if backend == "native":
        return F.conv2d(
            x, weight, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    if backend != "analytic":
        raise ValueError(f"Unknown convolution derivative backend: {backend}")
    if x.dtype not in (torch.float32, torch.float64) or weight.dtype != x.dtype:
        raise ValueError(
            "Analytic convolution is an eager FP32/FP64 regularisation path"
        )
    args = (_pair(stride), _pair(padding), _pair(dilation), groups)
    return _Convolution.apply(x, weight, args, weight_independent)


class _Convolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, args, weight_independent):
        ctx.save_for_backward(x, w)
        ctx.args, ctx.weight_independent = args, weight_independent
        stride, padding, dilation, groups = args
        return F.conv2d(
            x, w, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    @staticmethod
    def backward(ctx, u):
        x, w = ctx.saved_tensors
        dx = (
            _InputAdjoint.apply(u, w, x.shape, ctx.args)
            if ctx.needs_input_grad[0]
            else None
        )
        skip = ctx.weight_independent and _skip_direct_weight.get()
        dw = (
            _WeightAdjoint.apply(x, u, w.shape, ctx.args)
            if ctx.needs_input_grad[1] and not skip
            else None
        )
        return dx, dw, None, None


class _InputAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, w, shape, args):
        ctx.save_for_backward(u, w)
        ctx.args = args
        stride, padding, dilation, groups = args
        # conv_transpose2d is the input adjoint; output_padding resolves the
        # discarded remainder of a strided forward, including odd dimensions.
        output_padding = tuple(
            shape[2 + i]
            - (
                (u.shape[2 + i] - 1) * stride[i]
                - 2 * padding[i]
                + dilation[i] * (w.shape[2 + i] - 1)
                + 1
            )
            for i in range(2)
        )
        return F.conv_transpose2d(
            u,
            w,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

    @staticmethod
    def backward(ctx, v):
        u, w = ctx.saved_tensors
        du = (
            _Convolution.apply(v, w, ctx.args, False)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = (
            _WeightAdjoint.apply(v, u, w.shape, ctx.args)
            if ctx.needs_input_grad[1]
            else None
        )
        return du, dw, None, None


class _WeightAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, shape, args):
        ctx.save_for_backward(x, u)
        ctx.args = args
        stride, padding, dilation, groups = args
        return torch.nn.grad.conv2d_weight(
            x, shape, u, stride, padding, dilation, groups
        )

    @staticmethod
    def backward(ctx, v):
        x, u = ctx.saved_tensors
        dx = (
            _InputAdjoint.apply(u, v, x.shape, ctx.args)
            if ctx.needs_input_grad[0]
            else None
        )
        du = (
            _Convolution.apply(x, v, ctx.args, False)
            if ctx.needs_input_grad[1]
            else None
        )
        return dx, du, None, None
