from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stylegan2 import ModelConfig, TrainConfig, Trainer
from stylegan2.data import (
    ImageArray,
    InfiniteBatches,
    ProcessedPNGDataset,
    cache_processed_pngs,
)
from stylegan2.sample import save_grid


def project_dir(config: dict) -> Path:
    return Path("projects") / config["project_name"]


def processed_images_dir(config: dict) -> Path:
    return (
        project_dir(config)
        / "data"
        / "processed"
        / str(config["data"]["build_id"])
        / "images"
    )


def npy_cache_path(config: dict) -> Path:
    return processed_images_dir(config).parent / "images.npy"


def model_dir(config: dict) -> Path:
    return project_dir(config) / "models" / str(config["training"]["model_id"])


def checkpoint_path(config: dict) -> Path:
    return model_dir(config) / "latest.pt"


def build_model_config(config: dict) -> ModelConfig:
    training = config["training"]
    return ModelConfig(
        resolution=int(config["data"]["target_size"]),
        modconv=str(training.get("modconv", "factorized")),
        resample=str(training.get("resample", "torch")),
    )


def build_train_config(config: dict) -> TrainConfig:
    training = config["training"]
    precision = str(training.get("precision", "bf16"))
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"Unsupported training.precision: {precision}")
    microbatch = int(training.get("microbatch", training["batch_size"]))
    batch_size = int(training["batch_size"])
    if batch_size % microbatch != 0:
        raise ValueError("training.batch_size must be divisible by training.microbatch")
    return TrainConfig(
        lr=float(
            training.get(
                "lr", 0.0025 if config["data"]["target_size"] >= 512 else 0.002
            )
        ),
        precision=precision,
        microbatch=microbatch,
        mirror=bool(training.get("mirror", False)),
        compile_main=bool(training.get("compile_main", False)),
        tf32=bool(training.get("tf32", False)),
    )


def resolve_device(config: dict) -> str:
    requested = str(config["training"].get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but no GPU is available")
    return requested


def build_dataset(config: dict, model_cfg: ModelConfig):
    images_dir = processed_images_dir(config)
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Processed images not found: {images_dir}. Run ingest first."
        )
    if config["training"].get("cache_npy", False):
        npy_path = npy_cache_path(config)
        cache_processed_pngs(images_dir, npy_path)
        return ImageArray(npy_path, model_cfg.resolution)
    return ProcessedPNGDataset(images_dir, model_cfg.resolution)


def data_identity(config: dict, dataset, batch: int, seed: int) -> dict:
    images_dir = processed_images_dir(config)
    identity = {
        "images_dir": str(images_dir.resolve()),
        "count": len(dataset),
        "seed": seed,
        "batch": batch,
        "cache_npy": bool(config["training"].get("cache_npy", False)),
    }
    if config["training"].get("cache_npy", False):
        npy_path = npy_cache_path(config)
        identity["npy_path"] = str(npy_path.resolve())
        identity["npy_size"] = npy_path.stat().st_size
        identity["npy_mtime_ns"] = npy_path.stat().st_mtime_ns
    return identity


def train(config: dict) -> Path:
    training_cfg = config["training"]
    model_cfg = build_model_config(config)
    train_cfg = build_train_config(config)
    device = resolve_device(config)
    batch_size = int(training_cfg["batch_size"])
    num_steps = int(training_cfg["num_train_steps"])
    num_workers = int(training_cfg.get("num_workers", 4))
    seed = int(training_cfg.get("seed", 42))
    log_every = int(training_cfg.get("log_every", 50))
    save_every = int(training_cfg.get("save_every", 1000))

    torch.manual_seed(seed)
    random.seed(seed)

    out = model_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path(config)
    trainer = Trainer(model_cfg, train_cfg, device)
    dataset = build_dataset(config, model_cfg)
    identity = data_identity(config, dataset, batch_size, seed)

    resume = ckpt.is_file()
    if resume:
        extra = trainer.load(str(ckpt))
        if extra.get("data_identity") != identity:
            raise ValueError("Dataset configuration changed; refusing to resume")

    if not resume:
        resolved = {
            "model": asdict(model_cfg),
            "training": asdict(train_cfg),
            "batch": batch_size,
            "seed": seed,
        }
        (out / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    sampler = InfiniteBatches(len(dataset), batch_size, seed, trainer.steps)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    iterator = iter(loader)
    start = time.perf_counter()
    start_images = trainer.images_seen

    while trainer.steps < num_steps:
        log = trainer.step(next(iterator))
        if trainer.steps % log_every == 0:
            elapsed = time.perf_counter() - start
            images_per_sec = (trainer.images_seen - start_images) / max(elapsed, 1e-8)
            values = {
                key: float(value) if isinstance(value, torch.Tensor) else value
                for key, value in log.items()
            }
            values.update(
                step=trainer.steps,
                kimg=trainer.images_seen / 1000,
                images_per_sec=images_per_sec,
            )
            print(json.dumps(values), flush=True)
            with (out / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(values) + "\n")
        if trainer.steps % save_every == 0 or trainer.steps == num_steps:
            trainer.save(str(ckpt), {"data_identity": identity})
            sample_batch = train_cfg.microbatch or min(4, batch_size)
            save_grid(
                trainer.G_ema,
                out / f"samples-{trainer.steps:07d}.png",
                batch=sample_batch,
            )

    return ckpt
