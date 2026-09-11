import itertools

import numpy as np
import pytest
import torch
from PIL import Image

from stylegan2.data import (
    ImageArray,
    InfiniteBatches,
    ProcessedPNGDataset,
    cache_processed_pngs,
    prepare,
)
from stylegan2.metrics import frechet_distance


def test_prepare_roundtrip_and_validation(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for i in range(3):
        Image.new("RGB", (12, 8), (i * 50, 25, 30)).save(source / f"{i}.png")
    output = tmp_path / "data.npy"
    metadata = prepare(source, output, 8)
    dataset = ImageArray(output, 8)
    assert metadata["count"] == len(dataset) == 3
    assert dataset[1].shape == (3, 8, 8)
    assert int(dataset[1][0, 0, 0]) == 50
    with pytest.raises(FileExistsError):
        prepare(source, output, 8)
    with pytest.raises(ValueError):
        ImageArray(output, 16)


def test_sampler_resume_and_rank_disjointness():
    full = list(itertools.islice(InfiniteBatches(27, 4, seed=9), 20))
    resumed = list(itertools.islice(InfiniteBatches(27, 4, seed=9, start_step=7), 13))
    assert full[7:] == resumed
    ranks = [
        list(itertools.islice(InfiniteBatches(32, 4, rank=r, world=2), 4))
        for r in range(2)
    ]
    flat = sum(ranks[0] + ranks[1], [])
    assert len(flat) == len(set(flat)) == 32


def test_cache_processed_pngs_from_square_pngs(tmp_path):
    images_dir = tmp_path / "pngs"
    images_dir.mkdir()
    Image.new("RGB", (8, 8), (5, 6, 7)).save(images_dir / "000001.png")
    npy_path = tmp_path / "cache.npy"
    cache_processed_pngs(images_dir, npy_path)
    dataset = ProcessedPNGDataset(images_dir, 8)
    cached = ImageArray(npy_path, 8)
    assert len(dataset) == len(cached) == 1
    assert torch.equal(dataset[0], cached[0])


def test_distribution_metric_analytic_cases():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(100, 4))
    assert frechet_distance(features, features) == pytest.approx(0, abs=1e-10)
    assert frechet_distance(features, features + 2) == pytest.approx(16, abs=1e-9)
    singular = np.ones((10, 4))
    assert frechet_distance(singular, singular * 2) == pytest.approx(4)
    with pytest.raises(ValueError):
        frechet_distance(np.zeros((1, 4)), features)
