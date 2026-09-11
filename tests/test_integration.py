from pathlib import Path

import pytest
import torch
from PIL import Image

from gan_studio.generator import ArtGenerator
from gan_studio.training import build_model_config, build_train_config, checkpoint_path
from gan_studio.utils import open_config
from stylegan2 import ModelConfig, Trainer, TrainConfig
from stylegan2.data import ProcessedPNGDataset, cache_processed_pngs


def test_yaml_maps_to_stylegan_configs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "dev.yaml"
    config_path.write_text("""
project_name: walkers
data:
  target_size: 64
  format: png
  build_id: "001"
  dedup:
    exact: true
    perceptual_threshold: 4
training:
  model_id: tiny
  batch_size: 4
  microbatch: 2
  num_train_steps: 10
  num_workers: 0
  precision: fp32
  mirror: true
  compile_main: false
  cache_npy: false
""")
    config = open_config(config_path)
    model_cfg = build_model_config(config)
    train_cfg = build_train_config(config)
    assert model_cfg.resolution == 64
    assert train_cfg.microbatch == 2
    assert train_cfg.mirror is True
    assert checkpoint_path(config) == Path("projects/walkers/models/tiny/latest.pt")


def test_processed_png_dataset_reads_ingest_output(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(images_dir / "000001.png")
    Image.new("RGB", (32, 32), (40, 50, 60)).save(images_dir / "000002.png")
    dataset = ProcessedPNGDataset(images_dir, resolution=32)
    assert len(dataset) == 2
    assert dataset[0].shape == (3, 32, 32)
    assert int(dataset[0][0, 0, 0]) == 10


def test_cache_processed_pngs_builds_npy(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(images_dir / "000001.png")
    npy_path = tmp_path / "images.npy"
    metadata = cache_processed_pngs(images_dir, npy_path)
    assert npy_path.is_file()
    assert metadata["count"] == 1
    assert metadata["resolution"] == 16
    second = cache_processed_pngs(images_dir, npy_path)
    assert second["count"] == 1


def test_art_generator_roundtrip(tmp_path):
    model_cfg = ModelConfig(
        resolution=8,
        z_dim=8,
        w_dim=8,
        mapping_layers=2,
        channel_base=128,
        channel_max=16,
    )
    trainer = Trainer(model_cfg, TrainConfig(microbatch=2), "cpu")
    model_dir = tmp_path / "models" / "demo"
    model_dir.mkdir(parents=True)
    ckpt = model_dir / "latest.pt"
    trainer.save(
        ckpt,
        {
            "data_identity": {
                "images_dir": str(tmp_path),
                "count": 4,
                "seed": 1,
                "batch": 2,
                "cache_npy": False,
            }
        },
    )
    generator = ArtGenerator(base_dir=tmp_path, name="demo", device="cpu")
    image, latent = generator.sample(trunc_psi=0.7, seed=0)
    assert image.shape == (1, 3, 8, 8)
    assert latent.shape == (1, 8)
    assert image.min() >= 0
    assert image.max() <= 1
    style = generator.latent_to_style(latent, trunc_psi=0.7)
    assert style.shape == (1, 8)
    frames = list(generator.interpolate_styles([style, style * 0], frames_per_leg=2))
    assert len(frames) == 3
