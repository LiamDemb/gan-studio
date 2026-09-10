import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

from gan_studio import ArtGenerator
from gan_studio.utils import open_config, parse_config_arg


def main():
    args = parse_config_arg()
    config = open_config(args.config)
    generate_cfg = config.get("generate") or {}

    keyframes = _keyframes(generate_cfg)
    frames_per_leg = _frames_per_leg(generate_cfg)

    project_dir = Path("projects") / config["project_name"]
    catalog_dir = project_dir / "results" / "catalog"
    interp_dir = project_dir / "results" / "interp"
    manifest_path = catalog_dir / "catalog.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Catalog not found at {manifest_path}. Run make catalog first."
        )

    manifest = json.loads(manifest_path.read_text())
    trunc_psi = float(manifest.get("trunc_psi", generate_cfg.get("trunc_psi", 0.7)))
    known_ids = {int(item["id"]) for item in manifest.get("items", [])}

    generator = ArtGenerator(
        base_dir=project_dir,
        name=str(config["training"]["model_id"]),
    )

    styles = []
    for keyframe_id in keyframes:
        if known_ids and keyframe_id not in known_ids:
            raise SystemExit(
                f"Keyframe {keyframe_id} is not in the catalog. "
                f"Choose ids from 0 to {max(known_ids)}"
            )
        latent_path = catalog_dir / f"{keyframe_id:04d}.pt"
        if not latent_path.is_file():
            raise SystemExit(f"Missing catalog latent: {latent_path}")
        latent = torch.load(latent_path, map_location="cuda", weights_only=True)
        styles.append(generator.latent_to_style(latent, trunc_psi=trunc_psi))

    interp_dir.mkdir(parents=True, exist_ok=True)
    for path in interp_dir.iterdir():
        if path.is_file():
            path.unlink()

    print(
        f"Interpolating keyframes {keyframes} "
        f"({frames_per_leg} frames per leg) -> {interp_dir}"
    )
    frame_count = 0
    for image in generator.interpolate_styles(styles, frames_per_leg):
        output_path = interp_dir / f"{frame_count:04d}.png"
        save_image(image, output_path)
        frame_count += 1

    print(f"Interpolate complete. {frame_count} frames in {interp_dir}")


def _keyframes(generate_cfg):
    raw = generate_cfg.get("keyframes")
    if not isinstance(raw, list) or len(raw) < 2:
        raise SystemExit(
            "config generate.keyframes must be a list of at least two catalog ids"
        )
    try:
        return [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise SystemExit("generate.keyframes must be integers") from exc


def _frames_per_leg(generate_cfg):
    try:
        frames_per_leg = int(generate_cfg["frames_per_leg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            "config generate.frames_per_leg must be a positive integer"
        ) from exc
    if frames_per_leg < 1:
        raise SystemExit("generate.frames_per_leg must be at least 1")
    return frames_per_leg


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
