"""Measure complete training cycles. No image/s versus NVIDIA is invented."""

import argparse
import json
import math
import platform
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .models import ModelConfig
from .training import Trainer, TrainConfig


def environment(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else None
        ),
        "threads": torch.get_num_threads(),
    }


def compiler_graph_count():
    # Diagnostic only: PyTorch does not expose this counter as a stable API.
    # Absence must never prevent a benchmark or be interpreted as zero.
    try:
        from torch._dynamo.utils import counters

        return int(counters["stats"]["unique_graphs"])
    except (ImportError, AttributeError, KeyError):
        return None


def benchmark(model, training, batch, device="cuda", warmup=16, steps=64, trace=None):
    cycle = math.lcm(
        training.r1_interval if training.r1_gamma else 1,
        training.pl_interval if training.pl_weight else 1,
    )
    if warmup < cycle or warmup % cycle or steps < cycle or steps % cycle:
        raise ValueError(
            f"warmup and steps must be positive multiples of regularisation cycle ({cycle})"
        )
    torch.manual_seed(42)
    random.seed(42)
    trainer = Trainer(model, training, device)
    device = trainer.device
    # Data already on device: deliberately a compute benchmark. Report this
    # explicitly. Training CLI's throughput includes the actual data pipeline.
    real = torch.randint(
        0,
        256,
        (batch, 3, model.resolution, model.resolution),
        device=device,
        dtype=torch.uint8,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    begin = time.perf_counter()
    for _ in range(warmup):
        trainer.step(real)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    warmup_seconds = time.perf_counter() - begin
    graphs_before = compiler_graph_count() if training.compile_main else 0

    gpu_events, cpu_times, kinds, losses = [], [], [], []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    context = (
        torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True
        )
        if trace
        else nullcontext()
    )
    begin = time.perf_counter()
    with context as profiler:
        for _ in range(steps):
            if device.type == "cuda":
                start_event, end_event = torch.cuda.Event(
                    enable_timing=True
                ), torch.cuda.Event(enable_timing=True)
                start_event.record()
            t0 = time.perf_counter()
            result = trainer.step(real)
            cpu_times.append((time.perf_counter() - t0) * 1000)
            if device.type == "cuda":
                end_event.record()
                gpu_events.append((start_event, end_event))
            kinds.append((result["did_r1"], result["did_pl"]))
            losses.append(
                torch.stack([result[k] for k in ("d_loss", "g_loss", "r1", "pl")])
            )
            if trace:
                profiler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - begin
    graphs_after = compiler_graph_count() if training.compile_main else 0
    recompiled = (
        None
        if graphs_before is None or graphs_after is None
        else graphs_after > graphs_before
    )
    if not torch.isfinite(torch.stack(losses)).all():
        raise FloatingPointError("Nonfinite loss encountered; timing result is invalid")
    timings = [a.elapsed_time(b) for a, b in gpu_events] if gpu_events else cpu_times
    phase_classes = {}
    for kind in sorted(set(kinds)):
        values = [value for value, label in zip(timings, kinds) if label == kind]
        phase_classes[f"r1={kind[0]},pl={kind[1]}"] = {
            "steps": len(values),
            "median_ms": float(np.median(values)),
        }
    if trace:
        Path(trace).parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace))
    return {
        "schema": 1,
        "benchmark": "full-training-cycle-device-resident-synthetic-data",
        "environment": environment(device),
        "model": asdict(model),
        "training": asdict(training),
        "batch_per_gpu": batch,
        "world_size": 1,
        "warmup_steps": warmup,
        "warmup_seconds": warmup_seconds,
        "measured_steps": steps,
        "measured_seconds": elapsed,
        "images_per_second": batch * steps / elapsed,
        "seconds_per_kimg": elapsed * 1000 / (batch * steps),
        "step_median_ms": float(np.median(timings)),
        "step_p95_ms": float(np.percentile(timings, 95)),
        "regularisation_step_classes": phase_classes,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "profiled": bool(trace),
        "quality_validated": False,
        "compiler_graphs_during_measurement": (
            None if recompiled is None else graphs_after - graphs_before
        ),
        "steady_state_valid": not trace and recompiled is False,
        "comparison_to_nvidia": "not measured",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--modconv", choices=["factorized", "grouped"])
    parser.add_argument("--resample", choices=["torch", "triton"])
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--microbatch", type=int)
    parser.add_argument(
        "--trace",
        help="Optional Chrome trace; profiling distorts the measured throughput",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = json.loads(Path(args.config).read_text())
    for key in ("modconv", "resample"):
        if getattr(args, key) is not None:
            config["model"][key] = getattr(args, key)
    for key in ("precision", "microbatch"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    if args.compile is not None:
        config["training"]["compile_main"] = args.compile
    result = benchmark(
        ModelConfig(**config["model"]),
        TrainConfig(**config["training"]),
        config["batch"],
        args.device,
        args.warmup,
        args.steps,
        args.trace,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
