import pytest
import torch

from stylegan2.ops import resample

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA hardware unavailable"
    ),
]


@pytest.mark.parametrize("params", [(2, 1, 2, 1), (1, 2, 1, 1), (1, 1, 1, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_fir_forward_adjoint_and_second_derivative(params, dtype):
    pytest.importorskip("triton")
    x = (
        torch.randn(2, 7, 9, 11, device="cuda", dtype=dtype)
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_()
    )
    ref = resample(x, *params, backend="torch")
    fused = resample(x, *params, backend="triton")
    tol = 1e-5 if dtype == torch.float32 else (0.01 if dtype == torch.float16 else 0.05)
    torch.testing.assert_close(fused, ref, atol=tol, rtol=tol)
    # A nonlinear enclosing loss gives a NONZERO second derivative of the
    # linear FIR's adjoint, and exercises the custom adjoint registration.
    (a,) = torch.autograd.grad(fused.float().square().sum(), x, create_graph=True)
    (b,) = torch.autograd.grad(ref.float().square().sum(), x, create_graph=True)
    torch.testing.assert_close(a, b, atol=tol * 4, rtol=tol * 4)
    direction = torch.randn_like(x)
    (aa,) = torch.autograd.grad((a * direction).sum(), x)
    (bb,) = torch.autograd.grad((b * direction).sum(), x)
    torch.testing.assert_close(aa, bb, atol=tol * 4, rtol=tol * 4)


def test_fused_fir_custom_op_contract():
    pytest.importorskip("triton")
    from stylegan2.ops.triton_fir import fir

    torch.library.opcheck(
        fir, (torch.randn(2, 5, 8, 8, device="cuda", requires_grad=True), 2, 1, 2, 1)
    )


def test_compiled_main_with_fused_fir_and_eager_regularisation():
    pytest.importorskip("triton")
    from stylegan2 import ModelConfig
    from stylegan2.training import Trainer, TrainConfig

    trainer = Trainer(
        ModelConfig(
            resolution=16,
            z_dim=16,
            w_dim=16,
            mapping_layers=2,
            channel_base=256,
            channel_max=32,
            resample="triton",
        ),
        TrainConfig(precision="bf16", compile_main=True, r1_interval=2, pl_interval=2),
        "cuda",
    )
    for _ in range(2):
        logs = trainer.step(torch.randn(4, 3, 16, 16, device="cuda"))
        assert all(torch.isfinite(logs[k]) for k in ("d_loss", "g_loss", "r1", "pl"))
