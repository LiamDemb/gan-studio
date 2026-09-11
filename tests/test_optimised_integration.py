import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
import torch
import yaml
from PIL import Image

from gan_studio.generator import ArtGenerator
from gan_studio.training import build_model_config, build_train_config, train
from stylegan2 import ModelConfig, TrainConfig, Trainer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return {
        "project_name": "tiny",
        "data": {"target_size": 8, "build_id": "001"},
        "training": {
            "model_id": "fast",
            "device": "cpu",
            "batch_size": 4,
            "microbatch": 2,
            "num_train_steps": 2,
            "num_workers": 0,
            "threads": 2,
            "precision": "fp32",
            "regularizer_conv": "analytic",
            "z_dim": 8,
            "w_dim": 8,
            "mapping_layers": 2,
            "channel_base": 64,
            "channel_max": 8,
            "r1_interval": 2,
            "pl_interval": 2,
            "log_every": 1,
            "save_every": 2,
            "progress": False,
        },
    }


def dataset(tmp_path):
    folder = tmp_path / "projects/tiny/data/processed/001/images"
    folder.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (8, 8), (i * 50, 100, 200 - i * 30)).save(folder / f"{i}.png")


def test_4090_profile_resolves_measured_policy():
    config = yaml.safe_load((ROOT / "configs/rtx4090-512.yaml").read_text())
    mc, tc = build_model_config(config), build_train_config(config)
    assert (mc.resolution, mc.resample, mc.modconv) == (512, "triton", "factorized")
    assert (tc.precision, tc.regularizer_conv, tc.microbatch) == ("bf16", "analytic", 4)
    assert tc.compile_main and tc.tf32 and tc.channels_last
    assert config["training"]["batch_size"] == 32
    assert tc.lr == 0.002 and tc.r1_interval == 16 and tc.pl_interval == 8
    assert not tc.mirror


@pytest.mark.parametrize("batch,micro", [(0, 2), (4, -1), (4, 3), (4, 1)])
def test_invalid_batch_configs_fail_before_model_build(batch, micro):
    config = tiny_config()
    config["training"].update(batch_size=batch, microbatch=micro)
    with pytest.raises(ValueError):
        build_train_config(config)


def test_microbatch_zero_means_full_batch():
    config = tiny_config()
    config["training"]["microbatch"] = 0
    assert build_train_config(config).microbatch == 0


@pytest.mark.parametrize("cache", [False, True])
def test_actual_studio_train_resume_and_generate(tmp_path, monkeypatch, cache):
    monkeypatch.chdir(tmp_path)
    dataset(tmp_path)
    config = tiny_config()
    config["training"]["cache_npy"] = cache
    path = train(config)
    state = torch.load(path, weights_only=True)
    assert state["steps"] == 2 and state["images_seen"] == 8
    assert state["train_config"]["regularizer_conv"] == "analytic"
    config["training"]["num_train_steps"] = 4
    train(config)
    resumed = torch.load(path, weights_only=True)
    assert resumed["steps"] == 4 and resumed["images_seen"] == 16
    config["training"]["model_id"] = "uninterrupted"
    uninterrupted = torch.load(train(config), weights_only=True)
    for model in ("G", "D", "G_ema"):
        for name in resumed[model]:
            torch.testing.assert_close(
                resumed[model][name], uninterrupted[model][name], atol=0, rtol=0
            )
    assert (path.parent / "samples-0000004.png").is_file()
    art = ArtGenerator(tmp_path / "projects/tiny", "fast", "cpu")
    a, z = art.sample(seed=17)
    b, _ = art.sample(seed=17)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    w = art.latent_to_style(z)
    assert len(list(art.interpolate_styles([w, w * 0.9], 2))) == 3


def test_explicit_backend_resume_preserves_training_state(tmp_path):
    config = tiny_config()
    mc = build_model_config(config)
    tc = replace(build_train_config(config), regularizer_conv="native")
    old = Trainer(mc, tc, "cpu")
    old.step(torch.randn(4, 3, 8, 8))
    path = tmp_path / "old.pt"
    old.save(path)
    # Simulate the original Studio schema, which omitted regularizer_conv.
    payload = torch.load(path, weights_only=True)
    del payload["train_config"]["regularizer_conv"]
    torch.save(payload, path)
    new = Trainer(mc, replace(tc, regularizer_conv="analytic"), "cpu")
    with pytest.raises(ValueError, match="identical"):
        new.load(path)
    new.load(path, allow_execution_changes=True)
    assert new.steps == old.steps and new.images_seen == old.images_seen
    assert new.last_load_changes == {
        "training.regularizer_conv": {"old": "native", "new": "analytic"}
    }
    assert len(new.g_opt.state) == len(old.g_opt.state) > 0
    for a, b in zip(new.g_opt.state.values(), old.g_opt.state.values()):
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    new.step(torch.randn(4, 3, 8, 8))
    assert new.steps == 2
    for change in (
        {"lr": 0.004},
        {"microbatch": 4},
        {"precision": "bf16"},
        {"r1_interval": 4},
    ):
        incompatible = Trainer(mc, replace(tc, **change), "cpu")
        with pytest.raises(ValueError, match="identical"):
            incompatible.load(path, allow_execution_changes=True)


def test_triton_checkpoint_portable_inference_without_triton(tmp_path):
    config = tiny_config()
    trainer = Trainer(build_model_config(config), build_train_config(config), "cpu")
    path = tmp_path / "models/portable/latest.pt"
    trainer.save(path)
    payload = torch.load(path, weights_only=True)
    # Backend has no weight tensors or checkpoint-specific tensor names.
    payload["model_config"]["resample"] = "triton"
    torch.save(payload, path)
    art = ArtGenerator(tmp_path, "portable", "cpu")
    assert art.model.cfg.resample == "torch"
    assert art.sample(seed=0)[0].shape == (1, 3, 8, 8)


def test_original_catalog_interpolation_and_video_scripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset(tmp_path)
    config = tiny_config()
    train(config)
    config["generate"] = {
        "catalog_id": "001",
        "catalog_size": 2,
        "trunc_psi": 0.7,
        "interp_id": "walk",
        "keyframes": [0, 1],
        "frames_per_leg": 2,
        "fps": 24,
        "video_name": "walk",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    for script in ("generate.py", "catalog.py", "interpolate.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--config", str(path)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert (
        len(list((tmp_path / "projects/tiny/results/interp/walk").glob("*.png"))) == 3
    )
    # Test video handoff without requiring ffmpeg or encoding a video here.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "studio_video", ROOT / "scripts/video.py"
    )
    video = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(video)
    monkeypatch.setattr(sys, "argv", ["video.py", "--config", str(path)])
    calls = []
    monkeypatch.setattr(
        video.subprocess, "run", lambda command, check: calls.append(command)
    )
    video.main()
    assert calls[0][0] == "ffmpeg" and calls[0][-1].endswith("walk.mp4")
