"""Exercise Studio's actual YAML adapter using synthetic device-resident data."""

import argparse
import json
from pathlib import Path

import torch

from gan_studio.training import build_model_config, build_train_config, resolve_device
from gan_studio.utils import open_config
from stylegan2.benchmark import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new benchmark filename")
    config = open_config(args.config)
    threads = int(config["training"].get("threads", 4))
    if threads < 1:
        parser.error("training.threads must be positive")
    torch.set_num_threads(threads)
    result = benchmark(
        build_model_config(config),
        build_train_config(config),
        int(config["training"]["batch_size"]),
        resolve_device(config),
        warmup=args.warmup,
        steps=args.steps,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
