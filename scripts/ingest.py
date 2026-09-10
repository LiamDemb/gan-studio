import sys

from gan_studio.data import IngestError, find_images, validate_images
from gan_studio.utils import open_config, write_resolved_config, parse_config_arg


def main():
    args = parse_config_arg()
    config = open_config(args.config)

    print("Ingesting images...")
    image_paths = find_images(config)
    result = validate_images(config, image_paths)
    write_resolved_config(config, result.base_dir)
    print(f"Ingest complete. {result.ingested} ingested, {result.failed} failed.")


if __name__ == "__main__":
    try:
        main()
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
