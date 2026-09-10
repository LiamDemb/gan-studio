import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .data import ImageArray, InfiniteBatches
from .models import ModelConfig
from .training import Trainer, TrainConfig
from .sample import save_grid


def main():
    parser = argparse.ArgumentParser(
        description="Train independent StyleGAN2 on a prepared uint8 .npy dataset"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--loader-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for workers; use 0 to disable",
    )
    args = parser.parse_args()
    if (
        min(args.steps, args.log_every, args.save_every, args.threads) < 1
        or args.workers < 0
        or args.loader_timeout < 0
    ):
        parser.error("Counts must be positive; workers may be zero")
    torch.set_num_threads(args.threads)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        if args.device.startswith("cuda"):
            torch.cuda.set_device(local)
            args.device = f"cuda:{local}"
        dist.init_process_group("nccl" if args.device.startswith("cuda") else "gloo")
    config = json.loads(Path(args.config).read_text())
    seed, batch = config.get("seed", 42), config.get("batch", 32)
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)
    trainer = Trainer(
        ModelConfig(**config["model"]), TrainConfig(**config["training"]), args.device
    )
    data = ImageArray(args.data, trainer.model_cfg.resolution)
    data_identity = {
        "path": str(Path(args.data).resolve()),
        "size": Path(args.data).stat().st_size,
        "mtime_ns": Path(args.data).stat().st_mtime_ns,
        "count": len(data),
        "seed": seed,
        "batch_per_rank": batch,
        "world": world,
    }
    if args.resume:
        extra = trainer.load(args.resume)
        if extra.get("data_identity") != data_identity:
            raise ValueError(
                "Dataset/order configuration changed; refusing to claim exact resume"
            )
    sampler = InfiniteBatches(len(data), batch, seed, trainer.steps, rank, world)
    # Dedicated loader RNG prevents worker initialisation from perturbing the
    # model RNG on resume. Worker transforms are deterministic.
    loader = DataLoader(
        data,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
        multiprocessing_context="spawn" if args.workers > 0 else None,
        timeout=args.loader_timeout if args.workers > 0 else 0,
        generator=torch.Generator().manual_seed(seed + 1000 + rank),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / "latest.pt").exists():
        raise FileExistsError(
            "Output already contains a checkpoint; use --resume or a new directory"
        )
    if rank == 0:
        (out / "config.json").write_text(
            json.dumps(
                {
                    "model": asdict(trainer.model_cfg),
                    "training": asdict(trainer.cfg),
                    "batch": batch,
                    "seed": seed,
                },
                indent=2,
            )
            + "\n"
        )
    iterator = iter(loader)
    start, start_images = time.perf_counter(), trainer.images_seen
    try:
        while trainer.steps < args.steps:
            log = trainer.step(next(iterator))
            if trainer.steps % args.log_every == 0 and rank == 0:
                values = {
                    k: float(v) if isinstance(v, torch.Tensor) else v
                    for k, v in log.items()
                }
                elapsed = time.perf_counter() - start
                values.update(
                    step=trainer.steps,
                    kimg=trainer.images_seen / 1000,
                    images_per_sec=(trainer.images_seen - start_images) / elapsed,
                )
                if not all(
                    torch.isfinite(v).all()
                    for v in (log["d_loss"], log["g_loss"], log["r1"], log["pl"])
                ):
                    raise FloatingPointError("Non-finite training loss")
                line = json.dumps(values)
                print(line, flush=True)
                with (out / "training.jsonl").open("a") as f:
                    f.write(line + "\n")
            if trainer.steps % args.save_every == 0 or trainer.steps == args.steps:
                trainer.save(out / "latest.pt", {"data_identity": data_identity})
                if rank == 0:
                    save_grid(
                        trainer.G_ema,
                        out / f"samples-{trainer.steps:07d}.png",
                        batch=trainer.cfg.microbatch or 4,
                    )
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
