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
    try:
        catalog_size = int(generate_cfg["catalog_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            "config generate.catalog_size must be a positive integer"
        ) from exc
    if catalog_size < 1:
        raise SystemExit("generate.catalog_size must be at least 1")
    trunc_psi = float(generate_cfg.get("trunc_psi", 0.7))

    project_dir = Path("projects") / config["project_name"]
    catalog_dir = project_dir / "results" / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    for path in catalog_dir.iterdir():
        if path.is_file():
            path.unlink()

    generator = ArtGenerator(
        base_dir=project_dir,
        name=str(config["training"]["model_id"]),
    )

    items = []
    print(f"Writing {catalog_size} catalog samples to {catalog_dir}")
    for sample_id in range(catalog_size):
        image, latent = generator.sample(trunc_psi=trunc_psi, seed=sample_id)
        png_name = f"{sample_id:04d}.png"
        latent_name = f"{sample_id:04d}.pt"
        save_image(image, catalog_dir / png_name)
        torch.save(latent.detach().cpu(), catalog_dir / latent_name)
        items.append(
            {
                "id": sample_id,
                "png": png_name,
                "latent": latent_name,
                "seed": sample_id,
            }
        )
        print(f"  {png_name}")

    manifest = {
        "model_id": str(config["training"]["model_id"]),
        "trunc_psi": trunc_psi,
        "count": catalog_size,
        "items": items,
    }
    manifest_path = catalog_dir / "catalog.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Catalog complete. {catalog_size} samples. Manifest {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
