from __future__ import annotations

import csv
import hashlib
from collections import Counter, namedtuple
from pathlib import Path

import imagehash
from PIL import Image

MANIFEST_FIELDS = [
    "filename",
    "width",
    "height",
    "status",
    "reason",
    "source_path",
]

_JPEG_SUFFIXES = {".jpg", ".jpeg"}


class IngestError(Exception):
    """Raised when ingest cannot produce a usable dataset."""


IngestResult = namedtuple("IngestResult", ["base_dir", "ingested", "failed"])


def find_images(config: dict) -> list[Path]:
    folder = Path("projects") / config["project_name"] / "data" / "raw"
    if not folder.is_dir():
        raise IngestError(f"Raw data folder not found: {folder}")

    suffixes = _suffixes_for(config["data"]["format"])
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not paths:
        raise IngestError(f"No {config['data']['format']} images found in {folder}")
    return paths


def center_crop_to_aspect(
    img: Image.Image, target_w: int, target_h: int
) -> Image.Image:
    src_w, src_h = img.size
    if src_w * target_h > target_w * src_h:
        new_w = max(1, target_w * src_h // target_h)
        left = (src_w - new_w) // 2
        return img.crop((left, 0, left + new_w, src_h))
    if src_w * target_h < target_w * src_h:
        new_h = max(1, target_h * src_w // target_w)
        top = (src_h - new_h) // 2
        return img.crop((0, top, src_w, top + new_h))
    return img


def resize_to_target(
    img: Image.Image, target_w: int, target_h: int
) -> Image.Image | None:
    cropped = center_crop_to_aspect(img, target_w, target_h)
    crop_w, crop_h = cropped.size
    if crop_w < target_w or crop_h < target_h:
        return None
    if (crop_w, crop_h) == (target_w, target_h):
        return cropped
    return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)


def validate_images(config: dict, paths: list[Path]) -> IngestResult:
    target_w, target_h = _target_size(config)
    exact_dedup, perceptual_threshold = _dedup_settings(config)

    base_dir = (
        Path("projects")
        / config["project_name"]
        / "data"
        / "processed"
        / str(config["data"]["build_id"])
    )
    output_dir = base_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_dir / "manifest.csv"

    rows: list[dict] = []
    seen_hashes: set[str] = set()
    seen_phashes: list[imagehash.ImageHash] = []
    image_number = 0

    for path in paths:
        try:
            rgb = _load_rgb(path)
        except Exception:
            rows.append(
                _row(
                    filename=path.name,
                    width="",
                    height="",
                    status="rejected",
                    reason="corrupt",
                    source_path=path,
                )
            )
            continue

        source_w, source_h = rgb.size
        digest = _exact_hash(rgb)
        if exact_dedup and digest in seen_hashes:
            rows.append(
                _row(
                    filename=path.name,
                    width=source_w,
                    height=source_h,
                    status="rejected",
                    reason="duplicate_exact",
                    source_path=path,
                )
            )
            continue

        phash = _phash(rgb) if perceptual_threshold is not None else None
        if phash is not None and _is_near_duplicate(
            phash, seen_phashes, perceptual_threshold
        ):
            rows.append(
                _row(
                    filename=path.name,
                    width=source_w,
                    height=source_h,
                    status="rejected",
                    reason="duplicate_perceptual",
                    source_path=path,
                )
            )
            continue

        fitted = resize_to_target(rgb, target_w, target_h)
        if fitted is None:
            rows.append(
                _row(
                    filename=path.name,
                    width=source_w,
                    height=source_h,
                    status="rejected",
                    reason="too_small",
                    source_path=path,
                )
            )
            continue

        output_name = f"{image_number:04d}.png"
        # PNG is lossless; compress_level only changes file size, not pixels.
        fitted.save(output_dir / output_name, format="PNG")
        rows.append(
            _row(
                filename=output_name,
                width=target_w,
                height=target_h,
                status="valid",
                reason="ok",
                source_path=path,
            )
        )
        seen_hashes.add(digest)
        if phash is not None:
            seen_phashes.append(phash)
        image_number += 1

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _print_summary(len(paths), rows, manifest_path)

    ingested = sum(1 for row in rows if row["status"] == "valid")
    failed = len(rows) - ingested
    if ingested == 0:
        raise IngestError(f"No valid images produced; see manifest at {manifest_path}")

    return IngestResult(base_dir, ingested, failed)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img.verify()

    with Image.open(path) as img:
        img.load()
        return _to_rgb(img)


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def _exact_hash(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def _phash(img: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(img)


def _is_near_duplicate(
    phash: imagehash.ImageHash,
    seen: list[imagehash.ImageHash],
    threshold: int,
) -> bool:
    return any((phash - other) <= threshold for other in seen)


def _suffixes_for(fmt: str) -> set[str]:
    fmt = str(fmt).lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        return _JPEG_SUFFIXES
    return {f".{fmt}"}


def _target_size(config: dict) -> tuple[int, int]:
    data = config["data"]
    try:
        width = int(data["target_width"])
        height = int(data["target_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestError(
            "config data.target_width and data.target_height must be integers"
        ) from exc
    if width < 1 or height < 1:
        raise IngestError("target_width and target_height must be positive")
    return width, height


def _dedup_settings(config: dict) -> tuple[bool, int | None]:
    dedup = config["data"].get("dedup", {})
    if dedup is False:
        return False, None
    if dedup is True:
        return True, 4
    exact = bool(dedup.get("exact", True))
    threshold = dedup.get("perceptual_threshold", 4)
    if threshold is None or threshold is False:
        return exact, None
    return exact, int(threshold)


def _row(
    *,
    filename: str,
    width: int | str,
    height: int | str,
    status: str,
    reason: str,
    source_path: Path,
) -> dict:
    return {
        "filename": filename,
        "width": width,
        "height": height,
        "status": status,
        "reason": reason,
        "source_path": str(source_path),
    }


def _print_summary(found: int, rows: list[dict], manifest_path: Path) -> None:
    counts = Counter(row["reason"] for row in rows)
    print(f"Found {found} images")
    for reason, count in sorted(counts.items()):
        if reason == "ok":
            continue
        print(f"  {reason}: {count}")
    print(f"Wrote manifest to {manifest_path}")
