import sys
from pathlib import Path

from torchvision.utils import save_image

from gan_studio import ArtGenerator
from gan_studio.utils import open_config, parse_config_arg


def main():
    args = parse_config_arg()
    config = open_config(args.config)
    project_dir = Path("projects") / config["project_name"]
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "random.png"

    generator = ArtGenerator(
        base_dir=project_dir,
        name=str(config["training"]["model_id"]),
    )
    image = generator.generate_random()
    save_image(image, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
