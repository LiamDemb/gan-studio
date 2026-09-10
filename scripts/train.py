import subprocess
import sys
from pathlib import Path

from gan_studio.utils import open_config, parse_config_arg


def main():
    args = parse_config_arg()
    config = open_config(args.config)

    args = {
        "--data": Path("projects")
        / config["project_name"]
        / "data"
        / "processed"
        / config["data"]["build_id"]
        / "images",
        "--name": config["training"]["model_id"],
        "--image-size": config["data"]["target_size"],
        "--batch-size": config["training"]["batch_size"],
        "--aug-prob": config["training"]["aug_prob"],
        "--network-capacity": config["training"]["network_capacity"],
        "--num-train-steps": config["training"]["num_train_steps"],
        "--gradient_accumulate_every": config["training"]["gradient_accumulate_every"],
        "--num-workers": config["training"]["num_workers"],
        "--models_dir": Path("projects") / config["project_name"] / "models",
        "--results_dir": Path("projects") / config["project_name"] / "results",
    }

    command = ["stylegan2_pytorch"]

    for flag, value in args.items():
        command.extend([flag, str(value)])

    print(" ".join(command))
    print("Training...")
    subprocess.run(command, check=True)
    print(
        f"Training complete. Written model to {Path('projects') / config['project_name'] / 'models'}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
