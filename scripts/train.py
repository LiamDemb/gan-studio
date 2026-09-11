import sys
from pathlib import Path

from gan_studio.training import train
from gan_studio.utils import open_config, parse_config_arg


def main():
    args = parse_config_arg()
    config = open_config(args.config)
    ckpt = train(config)
    print(f"Training complete. Checkpoint written to {Path(ckpt)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
