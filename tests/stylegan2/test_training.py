import copy

import pytest
import torch

from stylegan2 import Generator, Discriminator, ModelConfig
from stylegan2.models import minibatch_std
from stylegan2.training import Trainer, TrainConfig, r1_penalty, path_lengths


def small(resolution=8):
    return ModelConfig(
        resolution=resolution,
        z_dim=8,
        w_dim=8,
        mapping_layers=2,
        channel_base=128,
        channel_max=16,
    )


@pytest.mark.parametrize("resolution", [4, 8, 16, 32])
def test_network_shapes_style_slots_and_noise(resolution):
    cfg = small(resolution)
    g, d = Generator(cfg), Discriminator(cfg)
    z = torch.randn(2, cfg.z_dim)
    images = g(z)
    assert images.shape == (2, 3, resolution, resolution)
    assert g.num_ws == 2 * (resolution.bit_length() - 1) - 2
    assert d(images).shape == (2,)
    torch.testing.assert_close(images, g(z), rtol=0, atol=0)
    # Noise defaults to zero strength; deliberately enable it for this test.
    with torch.no_grad():
        g.synthesis.first.noise_strength.fill_(1)
    assert not torch.equal(g(z, noise_mode="random"), g(z, noise_mode="random"))


@pytest.mark.parametrize("batch", [1, 2, 3, 4, 6, 8])
def test_minibatch_std_handles_arbitrary_batch(batch):
    x = torch.randn(batch, 5, 4, 4, requires_grad=True)
    y = minibatch_std(x)
    assert y.shape == (batch, 6, 4, 4)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_r1_analytical_linear_discriminator():
    x = torch.randn(3, 2, 4, 4, requires_grad=True)
    weight = torch.randn(2, 4, 4, requires_grad=True)
    score = (x * weight).flatten(1).sum(1)
    penalty = r1_penalty(score, x)
    torch.testing.assert_close(penalty, weight.square().sum())
    (grad,) = torch.autograd.grad(penalty, weight)
    torch.testing.assert_close(grad, 2 * weight)


def test_path_length_against_explicit_linear_jacobian():
    ws = torch.randn(2, 3, 4, requires_grad=True)
    matrix = torch.randn(12, 12, requires_grad=True)
    images = (ws.flatten(1) @ matrix.T).reshape(2, 3, 2, 2)
    noise = torch.randn_like(images)
    result = path_lengths(images, ws, noise)
    expected = (
        (noise.flatten(1) @ matrix).reshape(2, 3, 4).square().sum(2).mean(1) + 1e-8
    ).sqrt()
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert matrix.grad is not None and torch.isfinite(matrix.grad).all()


def test_complete_training_updates_and_lazy_adam_counts():
    trainer = Trainer(
        small(), TrainConfig(r1_interval=2, pl_interval=2, microbatch=2), "cpu"
    )
    old_g = trainer.G.synthesis.first.weight.detach().clone()
    old_d = trainer.D.from_rgb.weight.detach().clone()
    for step in range(2):
        log = trainer.step(torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8))
        assert log["did_r1"] == log["did_pl"] == (step == 1)
        assert all(torch.isfinite(log[k]) for k in ("d_loss", "g_loss", "r1", "pl"))
    assert not torch.equal(old_g, trainer.G.synthesis.first.weight)
    assert not torch.equal(old_d, trainer.D.from_rgb.weight)
    assert trainer.pl_mean > 0
    assert trainer.images_seen == 8
    assert trainer.g_opt.state[trainer.G.synthesis.first.weight]["step"].item() == 3
    assert trainer.d_opt.state[trainer.D.from_rgb.weight]["step"].item() == 3
    assert trainer.g_opt.param_groups[0]["lr"] == pytest.approx(0.002 * 2 / 3)
    assert trainer.G.mapping.w_avg.abs().sum() > 0
    assert all(not p.requires_grad for p in trainer.G_ema.parameters())


def test_checkpoint_exact_next_regularised_step(tmp_path):
    cfg, tc = small(), TrainConfig(r1_interval=2, pl_interval=2, microbatch=2)
    trainer = Trainer(cfg, tc, "cpu")
    real = torch.randn(4, 3, 8, 8)
    trainer.step(real)
    file = tmp_path / "checkpoint.pt"
    trainer.save(file, {"tag": "resume-test"})
    expected_log = trainer.step(real)
    expected_g, expected_d = copy.deepcopy(trainer.G.state_dict()), copy.deepcopy(
        trainer.D.state_dict()
    )
    expected_ema = copy.deepcopy(trainer.G_ema.state_dict())
    restored = Trainer(cfg, tc, "cpu")
    assert restored.load(file) == {"tag": "resume-test"}
    actual_log = restored.step(real)
    for actual, expected in (
        (restored.G.state_dict(), expected_g),
        (restored.D.state_dict(), expected_d),
        (restored.G_ema.state_dict(), expected_ema),
    ):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key in ("d_loss", "g_loss", "r1", "pl"):
        torch.testing.assert_close(actual_log[key], expected_log[key], rtol=0, atol=0)
    torch.testing.assert_close(restored.pl_mean, trainer.pl_mean, rtol=0, atol=0)


def test_disabled_regularisers_do_not_adjust_optimizer():
    trainer = Trainer(small(), TrainConfig(pl_weight=0, r1_gamma=0), "cpu")
    assert trainer.g_opt.param_groups[0]["lr"] == 0.002
    assert trainer.d_opt.param_groups[0]["betas"] == (0.0, 0.99)


def test_bf16_main_with_fp32_regularisation():
    trainer = Trainer(
        small(), TrainConfig(precision="bf16", r1_interval=2, pl_interval=2), "cpu"
    )
    for _ in range(2):
        log = trainer.step(torch.randn(4, 3, 8, 8))
        assert all(torch.isfinite(log[k]) for k in ("d_loss", "g_loss", "r1", "pl"))
    assert trainer.pl_mean.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in trainer.G.parameters())


def test_gradient_accumulation_equals_full_batch_without_mbstd_coupling():
    cfg = ModelConfig(**{**small().__dict__, "mbstd_group": 1})
    d = Discriminator(cfg)
    real = torch.randn(4, 3, 8, 8)
    torch.nn.functional.softplus(-d(real)).mean().backward()
    full = [p.grad.clone() for p in d.parameters()]
    d.zero_grad(set_to_none=True)
    for chunk in real.split(2):
        (torch.nn.functional.softplus(-d(chunk)).mean() / 2).backward()
    for expected, p in zip(full, d.parameters()):
        torch.testing.assert_close(p.grad, expected, atol=1e-6, rtol=1e-5)


def test_bad_config_and_batch_are_rejected():
    with pytest.raises(ValueError):
        ModelConfig(resolution=17)
    with pytest.raises(ValueError):
        TrainConfig(pl_interval=0)
    trainer = Trainer(small(), TrainConfig(microbatch=3), "cpu")
    with pytest.raises(ValueError):
        trainer.step(torch.randn(4, 3, 8, 8))


@pytest.mark.parametrize("resolution", [512, 1024])
def test_full_width_target_shapes_on_meta_device(resolution):
    with torch.device("meta"):
        cfg = ModelConfig(resolution=resolution)
        generator, discriminator = Generator(cfg), Discriminator(cfg)
        images = generator(torch.randn(2, 512))
        assert images.shape == (2, 3, resolution, resolution)
        assert discriminator(images).shape == (2,)
