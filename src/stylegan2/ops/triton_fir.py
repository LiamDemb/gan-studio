"""Experimental fused FIR forward and gather-adjoint kernels.

Independently written from the discrete resampling equation. No CUDA/StyleGAN
source was consulted. CUDA validation is mandatory before production use.
Registration of the adjoint's adjoint preserves differentiability for R1/PL.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    X,
    Y,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    UP: tl.constexpr,
    DOWN: tl.constexpr,
    LEFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Consecutive lanes traverse channels, the physical output is NHWC.
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = q % C
    ox = (q // C) % OW
    oy = (q // (C * OW)) % OH
    n = q // (C * OW * OH)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for ky in tl.static_range(4):
        ey = oy * DOWN + ky - LEFT
        iy = ey // UP
        fy = 1.0 if ky == 0 or ky == 3 else 3.0
        for kx in tl.static_range(4):
            ex = ox * DOWN + kx - LEFT
            ix = ex // UP
            fx = 1.0 if kx == 0 or kx == 3 else 3.0
            valid = (n < N) & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W)
            valid = valid & (ey % UP == 0) & (ex % UP == 0)
            v = tl.load(X + n * S0 + c * S1 + iy * S2 + ix * S3, valid, other=0).to(
                tl.float32
            )
            acc += v * (fy * fx * UP * UP / 64.0)
    tl.store(Y + q, acc, q < N * C * OH * OW)


@triton.jit
def _adjoint(
    Y,
    X,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    UP: tl.constexpr,
    DOWN: tl.constexpr,
    LEFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = q % C
    ix = (q // C) % W
    iy = (q // (C * W)) % H
    n = q // (C * W * H)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for ky in tl.static_range(4):
        ey = iy * UP + LEFT - ky
        oy = ey // DOWN
        fy = 1.0 if ky == 0 or ky == 3 else 3.0
        for kx in tl.static_range(4):
            ex = ix * UP + LEFT - kx
            ox = ex // DOWN
            fx = 1.0 if kx == 0 or kx == 3 else 3.0
            valid = (n < N) & (oy >= 0) & (oy < OH) & (ox >= 0) & (ox < OW)
            valid = valid & (ey % DOWN == 0) & (ex % DOWN == 0)
            v = tl.load(Y + n * S0 + c * S1 + oy * S2 + ox * S3, valid, other=0).to(
                tl.float32
            )
            acc += v * (fy * fx * UP * UP / 64.0)
    tl.store(X + q, acc, q < N * C * H * W)


def _output(x, up, down, left, right):
    n, c, h, w = x.shape
    oh = (h * up + left + right - 4) // down + 1
    ow = (w * up + left + right - 4) // down + 1
    if min(oh, ow) < 1:
        raise ValueError("Filter produces an empty output")
    return torch.empty(
        (n, c, oh, ow),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last,
    )


@torch.library.custom_op("independent_sg2::fir", mutates_args=())
def fir(x: torch.Tensor, up: int, down: int, left: int, right: int) -> torch.Tensor:
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton FIR requires CUDA FP16/BF16/FP32")
    y = _output(x, up, down, left, right)
    n, c, h, w = x.shape
    with torch.cuda.device(x.device):
        _forward[(triton.cdiv(y.numel(), 256),)](
            x, y, n, c, h, w, y.shape[2], y.shape[3], *x.stride(), up, down, left, 256
        )
    return y


@fir.register_fake
def _fir_fake(x, up, down, left, right):
    return _output(x, up, down, left, right)


@torch.library.custom_op("independent_sg2::fir_adjoint", mutates_args=())
def fir_adjoint(
    y: torch.Tensor, h: int, w: int, up: int, down: int, left: int, right: int
) -> torch.Tensor:
    n, c, oh, ow = y.shape
    x = torch.empty(
        (n, c, h, w), device=y.device, dtype=y.dtype, memory_format=torch.channels_last
    )
    with torch.cuda.device(y.device):
        _adjoint[(triton.cdiv(x.numel(), 256),)](
            y, x, n, c, h, w, oh, ow, *y.stride(), up, down, left, 256
        )
    return x


@fir_adjoint.register_fake
def _adjoint_fake(y, h, w, up, down, left, right):
    return torch.empty(
        (y.shape[0], y.shape[1], h, w),
        device=y.device,
        dtype=y.dtype,
        memory_format=torch.channels_last,
    )


def _setup_forward(ctx, inputs, output):
    x, up, down, left, right = inputs
    ctx.params = (x.shape[2], x.shape[3], up, down, left, right)


def _backward_forward(ctx, grad):
    return (fir_adjoint(grad, *ctx.params), None, None, None, None)


def _setup_adjoint(ctx, inputs, output):
    _, _, _, up, down, left, right = inputs
    ctx.params = (up, down, left, right)


def _backward_adjoint(ctx, grad):
    return (fir(grad, *ctx.params), None, None, None, None, None, None)


fir.register_autograd(_backward_forward, setup_context=_setup_forward)
fir_adjoint.register_autograd(_backward_adjoint, setup_context=_setup_adjoint)
