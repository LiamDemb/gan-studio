from pathlib import Path

import yaml


def open_config(path):
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


def write_resolved_config(config, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "config_resolved.yaml"

    with resolved_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Written resolved config to {resolved_path}")
    return resolved_path
