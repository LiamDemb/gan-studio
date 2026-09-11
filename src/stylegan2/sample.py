import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .models import Generator, ModelConfig


@torch.no_grad()
def save_grid(generator, path, count=16, seed=123, psi=1.0, batch=4):
    if min(count, batch) < 1:
        raise ValueError("count and batch must be positive")
    device = next(generator.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(count, generator.cfg.z_dim, generator=rng, device=device)
    pixels = []
    for chunk in z.split(batch):
        images = generator(chunk, truncation_psi=psi, noise_mode="const")
        images = (images.clamp(-1, 1) + 1).mul(127.5).round().byte()
        pixels.extend(images.permute(0, 2, 3, 1).cpu().numpy())
    side = math.ceil(math.sqrt(count))
    r = generator.cfg.resolution
    grid = np.zeros((math.ceil(count / side) * r, side * r, 3), dtype=np.uint8)
    for i, image in enumerate(pixels):
        row, col = divmod(i, side)
        grid[row * r : (row + 1) * r, col * r : (col + 1) * r] = image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Sample only this project's checkpoints; no external model loading"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--psi", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = dict(state["model_config"])
    # A torch FIR can sample an experimental-kernel checkpoint on CPU.
    config["resample"] = "torch"
    model = (
        Generator(ModelConfig(**config)).to(args.device).eval().requires_grad_(False)
    )
    model.load_state_dict(state["G_ema"])
    save_grid(model, args.output, args.count, args.seed, args.psi, args.batch)


if __name__ == "__main__":
    main()
