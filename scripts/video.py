import json
import subprocess
import sys
from pathlib import Path

from gan_studio.utils import open_config, parse_config_arg, require_id


def main():
    args = parse_config_arg()
    config = open_config(args.config)
    generate_cfg = config.get("generate") or {}
    interp_id = require_id(generate_cfg, "interp_id")
    video_name = _video_name(generate_cfg)
    fps = _fps(generate_cfg)

    project_dir = Path("projects") / config["project_name"]
    interp_dir = project_dir / "results" / "interp" / interp_id
    videos_dir = project_dir / "results" / "videos"
    interp_manifest_path = interp_dir / "interp.json"
    if not interp_manifest_path.is_file():
        raise SystemExit(
            f"Interpolation not found at {interp_manifest_path}. "
            "Run make interpolate first."
        )

    frames = sorted(p for p in interp_dir.glob("*.png") if p.is_file())
    if not frames:
        raise SystemExit(f"No interpolation frames in {interp_dir}.")

    interp_manifest = json.loads(interp_manifest_path.read_text())
    videos_dir.mkdir(parents=True, exist_ok=True)
    output_path = videos_dir / f"{video_name}.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(interp_dir / "%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)

    video_manifest = {
        **interp_manifest,
        "video_name": video_name,
        "fps": fps,
        "video": f"{video_name}.mp4",
        "interp_id": interp_id,
        "frame_count": interp_manifest.get("frame_count", len(frames)),
    }
    video_manifest_path = videos_dir / f"{video_name}.json"
    video_manifest_path.write_text(json.dumps(video_manifest, indent=2) + "\n")
    print(f"Wrote {output_path} ({len(frames)} frames at {fps} fps)")
    print(f"Wrote {video_manifest_path}")


def _video_name(generate_cfg):
    name = require_id(generate_cfg, "video_name")
    if name.endswith(".mp4"):
        name = name[: -len(".mp4")]
    return name


def _fps(generate_cfg):
    try:
        fps = int(generate_cfg["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("config generate.fps must be a positive integer") from exc
    if fps < 1:
        raise SystemExit("generate.fps must be at least 1")
    return fps


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
