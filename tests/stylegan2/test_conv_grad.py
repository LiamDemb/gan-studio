import copy
import random
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from stylegan2.models import ModelConfig, Generator, Discriminator
from stylegan2.ops.conv_grad import conv2d, input_derivative_only
from stylegan2.training import Trainer, TrainConfig, r1_penalty, path_lengths


@pytest.mark.parametrize(
    "stride,padding,dilation,groups",
    [(1, 1, 1, 1), (2, 1, 1, 1), (2, 2, 2, 2), (1, 0, 1, 2)],
)
@pytest.mark.parametrize("channels_last", [False, True])
def test_adjoint_gradcheck(stride, padding, dilation, groups, channels_last):
    torch.manual_seed(8)
    x = torch.randn(2, 2, 5, 6, dtype=torch.double)
    w = torch.randn(2, 2 // groups, 3, 3, dtype=torch.double)
    if channels_last:
        x, w = [t.contiguous(memory_format=torch.channels_last) for t in (x, w)]
    x.requires_grad_()
    w.requires_grad_()
    fn = lambda x, w: conv2d(x, w, stride, padding, dilation, groups, "analytic")
    torch.testing.assert_close(
        fn(x, w),
        F.conv2d(
            x, w, stride=stride, padding=padding, dilation=dilation, groups=groups
        ),
    )
    assert torch.autograd.gradcheck(fn, (x, w), fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, (x, w), fast_mode=True)


@pytest.mark.parametrize("modconv", ["factorized", "grouped"])
@pytest.mark.parametrize("penalty", ["r1", "pl"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA hardware unavailable"
                ),
            ],
        ),
    ],
)
def test_regularizer_every_parameter_gradient_matches_native(modconv, penalty, device):
    torch.manual_seed(19)
    cfg = ModelConfig(
        resolution=8,
        z_dim=8,
        w_dim=8,
        mapping_layers=2,
        channel_base=64,
        channel_max=8,
        modconv=modconv,
    )
    net = Discriminator(cfg) if penalty == "r1" else Generator(cfg)
    net = net.to(device).to(memory_format=torch.channels_last)
    inputs = (torch.randn(4, 3, 8, 8) if penalty == "r1" else torch.randn(4, 8)).to(
        device
    )
    noise = torch.randn(4, 3, 8, 8, device=device)
    if device == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for backend in ("native", "analytic"):
        model = copy.deepcopy(net)
        if penalty == "r1":
            x = inputs.clone().requires_grad_()
            loss = r1_penalty(model(x, conv_backend=backend), x)
        else:
            w = model.mapping(inputs)
            ws = w[:, None].expand(-1, model.num_ws, -1).contiguous()
            image = model.synthesis(ws, "const", conv_backend=backend)
            loss = (path_lengths(image, ws, noise) - 0.2).square().mean()
        loss.backward()
        results.append(
            (
                loss.detach(),
                {
                    k: None if p.grad is None else p.grad.clone()
                    for k, p in model.named_parameters()
                },
            )
        )
    torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-5, atol=2e-6)
    for name, expected in results[0][1].items():
        actual = results[1][1][name]
        assert (actual is None) == (expected is None), name
        if actual is not None:
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-5, msg=name)


def test_skip_only_removes_direct_weight_branch_and_restores_on_exception():
    x = torch.randn(2, 2, 4, 4, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, requires_grad=True)
    y = conv2d(x, w, padding=1, backend="analytic", weight_independent=True)
    with input_derivative_only():
        gx, gw = torch.autograd.grad(
            y.sum(), (x, w), create_graph=True, allow_unused=True
        )
    assert gw is None
    assert torch.autograd.grad(gx.square().sum(), w)[0].abs().sum() > 0
    with pytest.raises(RuntimeError), input_derivative_only():
        raise RuntimeError("test restoration")
    conv2d(
        x, w, padding=1, backend="analytic", weight_independent=True
    ).sum().backward()
    assert w.grad is not None


def test_full_cycle_parameter_and_ema_parity():
    cfg = ModelConfig(
        resolution=8, z_dim=8, w_dim=8, mapping_layers=2, channel_base=64, channel_max=8
    )
    tc = TrainConfig(r1_interval=2, pl_interval=2, microbatch=2)
    states = []
    for backend in ("native", "analytic"):
        torch.manual_seed(37)
        random.seed(37)
        trainer = Trainer(cfg, replace(tc, regularizer_conv=backend), "cpu")
        real = torch.randn(4, 3, 8, 8)
        for _ in range(4):
            trainer.step(real)
        states.append(
            {
                name: copy.deepcopy(getattr(trainer, name).state_dict())
                for name in ("G", "D", "G_ema")
            }
        )
    for name in states[0]:
        for key in states[0][name]:
            torch.testing.assert_close(
                states[0][name][key], states[1][name][key], rtol=2e-4, atol=2e-5
            )


def test_legacy_checkpoint_defaults_and_changed_backend_rejected(tmp_path):
    cfg = ModelConfig(
        resolution=8, z_dim=8, w_dim=8, mapping_layers=2, channel_base=64, channel_max=8
    )
    tc = TrainConfig(r1_interval=2, pl_interval=2)
    trainer = Trainer(cfg, tc, "cpu")
    path = tmp_path / "legacy.pt"
    trainer.save(path)
    payload = torch.load(path, weights_only=True)
    del payload["train_config"]["regularizer_conv"]
    torch.save(payload, path)
    trainer.load(path)
    changed = Trainer(cfg, replace(tc, regularizer_conv="analytic"), "cpu")
    with pytest.raises(ValueError, match="identical"):
        changed.load(path)
