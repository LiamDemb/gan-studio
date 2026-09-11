import numpy as np
import pytest
import torch
from torch.nn import functional as F

from stylegan2.ops import modulated_conv2d, resample, upsample, downsample


def literal_modconv(x, weight, style, demodulate):
    outputs = []
    for image, scales in zip(x, style):
        kernel = weight * scales[None, :, None, None]
        if demodulate:
            kernel = (
                kernel / (kernel.square().sum((1, 2, 3), keepdim=True) + 1e-8).sqrt()
            )
        outputs.append(F.conv2d(image[None], kernel, padding=weight.shape[2] // 2))
    return torch.cat(outputs)


@pytest.mark.parametrize("backend", ["factorized", "grouped"])
@pytest.mark.parametrize("demodulate,k", [(True, 3), (False, 1)])
def test_modconv_matches_literal_and_all_gradients(backend, demodulate, k):
    tensors = [
        torch.randn(2, 2, 4, 4, dtype=torch.double, requires_grad=True),
        torch.randn(3, 2, k, k, dtype=torch.double, requires_grad=True),
        torch.randn(2, 2, dtype=torch.double, requires_grad=True),
    ]
    actual = modulated_conv2d(*tensors, demodulate, backend)
    reference = literal_modconv(*tensors, demodulate)
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    direction = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * direction).sum(), tensors)
    gr = torch.autograd.grad((reference * direction).sum(), tensors)
    for a, b in zip(ga, gr):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("backend", ["factorized", "grouped"])
def test_modconv_numerical_first_and_second_derivatives(backend):
    tensors = (
        torch.randn(1, 2, 3, 3, dtype=torch.double, requires_grad=True),
        torch.randn(2, 2, 3, 3, dtype=torch.double, requires_grad=True),
        torch.randn(1, 2, dtype=torch.double, requires_grad=True),
    )
    fn = lambda x, k, s: modulated_conv2d(x, k, s, backend=backend)
    assert torch.autograd.gradcheck(fn, tensors, fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, tensors, fast_mode=True)


def test_modconv_batch_independence_and_zero_style():
    x, k, s = torch.randn(3, 2, 5, 5), torch.randn(4, 2, 3, 3), torch.randn(3, 2)
    output = modulated_conv2d(x, k, s)
    torch.testing.assert_close(output[0], modulated_conv2d(x[:1], k, s[:1])[0])
    assert modulated_conv2d(x, k, torch.zeros_like(s)).eq(0).all()


def literal_resample(x, up, down, left, right):
    # Slow NumPy spatial oracle, deliberately independent of pad/conv ops.
    n, c, h, w = x.shape
    oh = (h * up + left + right - 4) // down + 1
    ow = (w * up + left + right - 4) // down + 1
    y = np.zeros((n, c, oh, ow))
    taps = [1, 3, 3, 1]
    for oy in range(oh):
        for ox in range(ow):
            for ky in range(4):
                for kx in range(4):
                    iy, ix = oy * down + ky - left, ox * down + kx - left
                    if (
                        iy % up == 0
                        and ix % up == 0
                        and 0 <= iy < h * up
                        and 0 <= ix < w * up
                    ):
                        y[:, :, oy, ox] += (
                            x[:, :, iy // up, ix // up]
                            * taps[ky]
                            * taps[kx]
                            * up
                            * up
                            / 64
                        )
    return y


@pytest.mark.parametrize(
    "up,down,left,right", [(2, 1, 2, 1), (1, 2, 1, 1), (1, 1, 1, 2), (2, 2, 0, 3)]
)
def test_fir_coordinate_oracle_and_higher_derivatives(up, down, left, right):
    x = torch.randn(1, 2, 5, 7, dtype=torch.double, requires_grad=True)
    fn = lambda t: resample(t, up, down, left, right)
    result = fn(x)
    expected = torch.tensor(literal_resample(x.detach().numpy(), up, down, left, right))
    torch.testing.assert_close(result, expected)
    assert torch.autograd.gradcheck(fn, (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (x,), fast_mode=True)


def test_fir_interior_gain_and_output_dimensions():
    x = torch.ones(1, 2, 16, 16)
    assert upsample(x).shape == (1, 2, 32, 32)
    assert downsample(x).shape == (1, 2, 8, 8)
    torch.testing.assert_close(upsample(x)[:, :, 3:-3, 3:-3], torch.ones(1, 2, 26, 26))
    torch.testing.assert_close(downsample(x)[:, :, 2:-2, 2:-2], torch.ones(1, 2, 4, 4))


def test_fir_adjoint_inner_product():
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    y = upsample(x)
    v = torch.randn_like(y)
    (adjoint,) = torch.autograd.grad(y, x, v)
    torch.testing.assert_close((y * v).sum(), (x * adjoint).sum())
