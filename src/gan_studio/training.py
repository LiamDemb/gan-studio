from __future__ import annotations

import json
import math
import random
import time
import warnings
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
from stylegan2.diagnostics import Progress


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
    dimensions = {
        key: int(training[key])
        for key in (
            "z_dim",
            "w_dim",
            "mapping_layers",
            "channel_base",
            "channel_max",
            "mbstd_group",
        )
        if key in training
    }
    return ModelConfig(
        resolution=int(config["data"]["target_size"]),
        modconv=str(training.get("modconv", "factorized")),
        resample=str(training.get("resample", "torch")),
        **dimensions,
    )


def build_train_config(config: dict) -> TrainConfig:
    training = config["training"]
    precision = str(training.get("precision", "bf16"))
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"Unsupported training.precision: {precision}")
    microbatch = int(training.get("microbatch", training["batch_size"]))
    batch_size = int(training["batch_size"])
    if batch_size < 1 or microbatch < 0:
        raise ValueError("batch_size must be positive and microbatch nonnegative")
    effective_micro = microbatch or batch_size
    if batch_size % effective_micro != 0:
        raise ValueError("training.batch_size must be divisible by training.microbatch")
    regularisation = {
        key: float(training[key])
        for key in (
            "r1_gamma",
            "pl_weight",
            "pl_decay",
            "ema_kimg",
            "ema_rampup",
            "w_avg_beta",
            "style_mixing",
        )
        if key in training
    }
    regularisation.update(
        {
            key: int(training[key])
            for key in ("r1_interval", "pl_interval", "pl_batch_shrink")
            if key in training
        }
    )
    result = TrainConfig(
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
        channels_last=bool(training.get("channels_last", True)),
        regularizer_conv=str(training.get("regularizer_conv", "native")),
        **regularisation,
    )
    if result.pl_weight and effective_micro % result.pl_batch_shrink:
        raise ValueError(
            "training.microbatch (or full batch) must be divisible by pl_batch_shrink"
        )
    return result


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
    threads = int(training_cfg.get("threads", 4))
    loader_timeout = float(training_cfg.get("loader_timeout", 60))
    if (
        min(num_steps, log_every, save_every, threads) < 1
        or num_workers < 0
        or loader_timeout < 0
    ):
        raise ValueError(
            "Steps/log/save/threads must be positive; workers/timeout nonnegative"
        )
    torch.set_num_threads(threads)

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
        extra = trainer.load(
            str(ckpt),
            allow_execution_changes=bool(
                training_cfg.get("allow_execution_changes", False)
            ),
        )
        if extra.get("data_identity") != identity:
            raise ValueError("Dataset configuration changed; refusing to resume")

    if not resume or trainer.last_load_changes:
        resolved = {
            "model": asdict(model_cfg),
            "training": asdict(train_cfg),
            "batch": batch_size,
            "seed": seed,
            "runtime": {"threads": threads, "num_workers": num_workers},
        }
        if trainer.last_load_changes:
            resolved["resume_execution_changes"] = trainer.last_load_changes
            warnings.warn(
                "Resuming with explicitly changed execution settings; this is not bitwise continuation"
            )
        (out / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")

    sampler = InfiniteBatches(len(dataset), batch_size, seed, trainer.steps)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        timeout=loader_timeout if num_workers > 0 else 0,
        # Worker startup must not consume the restored model RNG on resume.
        generator=torch.Generator().manual_seed(seed + 1000),
    )
    iterator = iter(loader)
    start = time.perf_counter()
    start_images = trainer.images_seen

    with Progress(bool(training_cfg.get("progress", True)), label="training") as status:
        while trainer.steps < num_steps:
            status.state = f"step {trainer.steps+1}/{num_steps}: loading data"
            batch = next(iterator)
            observer = (
                (
                    lambda name: setattr(
                        status,
                        "state",
                        f"step {trainer.steps+1}/{num_steps}: {name or 'submitted'}",
                    )
                )
                if status.enabled
                else None
            )
            log = trainer.step(batch, observer)
            if trainer.steps % log_every == 0 or trainer.steps == num_steps:
                # Scalar conversion waits for the device before timing is read.
                values = {
                    key: float(value) if isinstance(value, torch.Tensor) else value
                    for key, value in log.items()
                }
                if not all(
                    math.isfinite(values[k]) for k in ("d_loss", "g_loss", "r1", "pl")
                ):
                    raise FloatingPointError(
                        "Non-finite training loss; refusing to save this step"
                    )
                elapsed = time.perf_counter() - start
                values.update(
                    step=trainer.steps,
                    kimg=trainer.images_seen / 1000,
                    images_per_sec=(trainer.images_seen - start_images)
                    / max(elapsed, 1e-8),
                )
                print(json.dumps(values), flush=True)
                with (out / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(values) + "\n")
            if trainer.steps % save_every == 0 or trainer.steps == num_steps:
                status.state = f"step {trainer.steps}: saving checkpoint and samples"
                trainer.save(str(ckpt), {"data_identity": identity})
                sample_batch = train_cfg.microbatch or min(4, batch_size)
                save_grid(
                    trainer.G_ema,
                    out / f"samples-{trainer.steps:07d}.png",
                    batch=sample_batch,
                )

    return ckpt
