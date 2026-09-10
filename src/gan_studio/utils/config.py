from pathlib import Path
import yaml
import argparse


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


def parse_config_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def require_id(mapping, key, *, section="generate"):
    try:
        value = str(mapping[key]).strip()
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"config {section}.{key} is required") from exc
    if not value:
        raise SystemExit(f"config {section}.{key} must be a non-empty id")
    if Path(value).name != value:
        raise SystemExit(f"config {section}.{key} must be a simple id, not a path")
    return value
