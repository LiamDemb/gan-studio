"""One-time image decoding, disk-backed uint8 tensors, resumable batch order."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _sorted_image_paths(folder):
    folder = Path(folder)
    paths = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No images found in {folder}")
    return paths


def cache_processed_pngs(png_dir, npy_path):
    """Stack preprocessed square PNGs into uint8 NHWC .npy without recropping."""
    png_dir, npy_path = Path(png_dir), Path(npy_path)
    if npy_path.suffix != ".npy":
        raise ValueError("Prepared dataset filename must end in .npy")
    paths = _sorted_image_paths(png_dir)
    with Image.open(paths[0]) as first:
        first = first.convert("RGB")
        resolution = first.size[0]
        if first.size[0] != first.size[1]:
            raise ValueError(f"Expected square PNGs in {png_dir}")
    if npy_path.exists():
        return {
            "format": "uint8-NHWC-npy-v1",
            "count": len(paths),
            "resolution": resolution,
            "preprocessing": "gan-studio ingest PNGs, no recrop",
        }
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npy_path.with_suffix(".npy.tmp")
    arr = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(len(paths), resolution, resolution, 3),
    )
    try:
        for index, path in enumerate(paths):
            with Image.open(path) as im:
                im = im.convert("RGB")
                if im.size != (resolution, resolution):
                    raise ValueError(
                        f"{path.name} is {im.size}, expected ({resolution}, {resolution})"
                    )
                arr[index] = np.asarray(im)
        arr.flush()
        del arr
        os.replace(temporary, npy_path)
    except Exception:
        del arr
        temporary.unlink(missing_ok=True)
        raise
    metadata = {
        "format": "uint8-NHWC-npy-v1",
        "count": len(paths),
        "resolution": resolution,
        "preprocessing": "gan-studio ingest PNGs, no recrop",
    }
    npy_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare(source, output, resolution):
    if resolution < 4 or resolution & (resolution - 1):
        raise ValueError("resolution must be a power of two, >=4")
    source, output = Path(source), Path(output)
    if output.suffix != ".npy":
        raise ValueError("Prepared dataset filename must end in .npy")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not paths:
        raise ValueError("No images found")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npy.tmp")
    arr = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(len(paths), resolution, resolution, 3),
    )
    try:
        for index, path in enumerate(paths):
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im = ImageOps.fit(
                    im,
                    (resolution, resolution),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                arr[index] = np.asarray(im)
        arr.flush()
        del arr
        os.replace(temporary, output)
    except Exception:
        del arr
        temporary.unlink(missing_ok=True)
        raise
    metadata = {
        "format": "uint8-NHWC-npy-v1",
        "count": len(paths),
        "resolution": resolution,
        "preprocessing": "EXIF transpose, RGB, centre square crop, Lanczos resize",
        "source_files": [str(p.relative_to(source)) for p in paths],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


class ImageArray(Dataset):
    def __init__(self, path, resolution=None):
        self.path = str(Path(path).resolve())
        self._array = None
        array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if (
            array.dtype != np.uint8
            or array.ndim != 4
            or array.shape[-1] != 3
            or array.shape[1] != array.shape[2]
        ):
            raise ValueError("Expected uint8 [N,H,H,3] .npy dataset")
        if resolution is not None and array.shape[1] != resolution:
            raise ValueError("Dataset resolution differs from model")
        self.count, self.resolution = len(array), array.shape[1]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return torch.from_numpy(np.array(self._array[index], copy=True)).permute(
            2, 0, 1
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state


class ProcessedPNGDataset(Dataset):
    """Read square RGB PNGs produced by gan-studio ingest (no resize/crop)."""

    def __init__(self, folder, resolution=None):
        self.folder = Path(folder)
        self.paths = _sorted_image_paths(self.folder)
        with Image.open(self.paths[0]) as first:
            first = first.convert("RGB")
            width, height = first.size
        if width != height:
            raise ValueError(f"Expected square PNGs in {self.folder}")
        self.resolution = width
        if resolution is not None and self.resolution != resolution:
            raise ValueError("Dataset resolution differs from model")
        self.count = len(self.paths)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            im = im.convert("RGB")
            if im.size != (self.resolution, self.resolution):
                raise ValueError(f"{self.paths[index].name} is not {self.resolution}²")
            array = np.asarray(im, dtype=np.uint8)
        return torch.from_numpy(array.copy()).permute(2, 0, 1)


class InfiniteBatches(Sampler):
    def __init__(self, length, batch, seed=0, start_step=0, rank=0, world=1):
        if min(length, batch, world) < 1 or length < batch * world:
            raise ValueError("Dataset must contain at least one global batch")
        if not 0 <= rank < world or start_step < 0:
            raise ValueError("Invalid rank or resume step")
        self.length, self.batch, self.seed = length, batch, seed
        self.start_step, self.rank, self.world = start_step, rank, world

    def __iter__(self):
        global_batch = self.batch * self.world
        per_epoch = self.length // global_batch
        epoch, offset = divmod(self.start_step, per_epoch)
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.length, generator=generator).tolist()
            for position in range(offset, per_epoch):
                start = position * global_batch + self.rank * self.batch
                yield order[start : start + self.batch]
            epoch, offset = epoch + 1, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.resolution)
    print(
        json.dumps({k: v for k, v in result.items() if k != "source_files"}, indent=2)
    )


if __name__ == "__main__":
    main()
