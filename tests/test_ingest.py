import csv

import pytest
from PIL import Image, ImageEnhance

from gan_studio.data import IngestError, find_images, validate_images
from gan_studio.data.ingest import center_crop_to_aspect, resize_to_target
from gan_studio.utils import open_config, write_resolved_config

from helpers import make_raw_dir, sample_config, save_pattern, save_rgb


def _read_manifest(base_dir):
    with (base_dir / "manifest.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_open_config_reads_yaml(tmp_path):
    path = tmp_path / "dev.yaml"
    path.write_text("project_name: walkers\n")
    assert open_config(path)["project_name"] == "walkers"


def test_open_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_config(tmp_path / "missing.yaml")


def test_write_resolved_config(tmp_path):
    out = write_resolved_config({"project_name": "walkers"}, tmp_path / "processed")
    assert out.is_file()
    assert "walkers" in out.read_text()


def test_find_images_is_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    save_rgb(raw / "a.png", (80, 100))
    save_rgb(raw / "b.PNG", (80, 100))
    (raw / "notes.txt").write_text("ignore me")

    paths = find_images(sample_config())
    assert [p.name for p in paths] == ["a.png", "b.PNG"]


def test_find_images_recurses_into_subfolders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    nested = raw / "batch_a"
    nested.mkdir()
    save_rgb(nested / "nested.png", (80, 100))

    paths = find_images(sample_config())
    assert [p.name for p in paths] == ["nested.png"]


def test_find_images_accepts_jpeg_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    Image.new("RGB", (80, 100), (1, 2, 3)).save(raw / "a.jpg")
    Image.new("RGB", (80, 100), (1, 2, 3)).save(raw / "b.jpeg")

    paths = find_images(sample_config(format="jpg"))
    assert [p.name for p in paths] == ["a.jpg", "b.jpeg"]


def test_find_images_missing_folder_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(IngestError, match="Raw data folder not found"):
        find_images(sample_config())


def test_find_images_empty_folder_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_raw_dir(tmp_path)
    with pytest.raises(IngestError, match="No png images found"):
        find_images(sample_config())


def test_center_crop_keeps_the_middle_of_a_wide_image():
    img = Image.new("RGB", (300, 100))
    img.paste((255, 0, 0), (0, 0, 100, 100))
    img.paste((0, 255, 0), (100, 0, 200, 100))
    img.paste((0, 0, 255), (200, 0, 300, 100))

    cropped = center_crop_to_aspect(img, 50, 50)
    assert cropped.size == (100, 100)
    assert cropped.getpixel((50, 50)) == (0, 255, 0)


def test_resize_to_target_rejects_images_that_would_upscale():
    img = Image.new("RGB", (40, 50), (9, 9, 9))
    assert resize_to_target(img, 64, 80) is None


def test_resize_to_target_supports_non_square_output():
    img = Image.new("RGB", (128, 160), (9, 9, 9))
    fitted = resize_to_target(img, 64, 80)
    assert fitted is not None
    assert fitted.size == (64, 80)


def test_validate_crops_resizes_and_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    save_rgb(raw / "wide.png", (200, 100), (10, 20, 30))

    result = validate_images(sample_config(), find_images(sample_config()))
    rows = _read_manifest(result.base_dir)

    assert result.ingested == 1
    assert result.failed == 0
    assert rows[0]["status"] == "valid"
    assert rows[0]["reason"] == "ok"
    assert rows[0]["width"] == "64"
    assert rows[0]["height"] == "80"
    output = Image.open(result.base_dir / "images" / "0000.png")
    assert output.size == (64, 80)
    assert output.mode == "RGB"


def test_validate_rejects_corrupt_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    save_rgb(raw / "good.png", (80, 100))
    (raw / "bad.png").write_bytes(b"not an image")

    result = validate_images(sample_config(), find_images(sample_config()))
    rows = _read_manifest(result.base_dir)
    by_name = {row["source_path"].split("/")[-1]: row for row in rows}
    assert result.ingested == 1
    assert result.failed == 1
    assert by_name["bad.png"]["reason"] == "corrupt"
    assert by_name["good.png"]["status"] == "valid"


def test_validate_rejects_too_small_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    save_rgb(raw / "tiny.png", (20, 20))

    with pytest.raises(IngestError, match="No valid images produced"):
        validate_images(sample_config(), find_images(sample_config()))

    rows = _read_manifest(
        tmp_path / "projects" / "walkers" / "data" / "processed" / "001"
    )
    assert rows[0]["reason"] == "too_small"


def test_validate_rejects_exact_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    save_rgb(raw / "a.png", (80, 100), (7, 8, 9))
    save_rgb(raw / "b.png", (80, 100), (7, 8, 9))

    result = validate_images(sample_config(), find_images(sample_config()))
    rows = _read_manifest(result.base_dir)
    assert result.ingested == 1
    assert result.failed == 1
    assert [row["reason"] for row in rows] == ["ok", "duplicate_exact"]
    assert (tmp_path / "projects/walkers/data/processed/001/images/0000.png").is_file()
    assert not (
        tmp_path / "projects/walkers/data/processed/001/images/0001.png"
    ).is_file()


def test_validate_rejects_near_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    original = save_pattern(raw / "a.png")
    ImageEnhance.Brightness(original).enhance(1.04).save(raw / "b.png")

    result = validate_images(sample_config(), find_images(sample_config()))
    rows = _read_manifest(result.base_dir)
    assert result.ingested == 1
    assert result.failed == 1
    assert rows[0]["reason"] == "ok"
    assert rows[1]["reason"] == "duplicate_perceptual"


def test_validate_can_disable_dedup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    save_rgb(raw / "a.png", (80, 100), (7, 8, 9))
    save_rgb(raw / "b.png", (80, 100), (7, 8, 9))

    config = sample_config(dedup=False)
    result = validate_images(config, find_images(config))
    rows = _read_manifest(result.base_dir)
    assert result.ingested == 2
    assert result.failed == 0
    assert [row["status"] for row in rows] == ["valid", "valid"]


def test_validate_flattens_transparent_png_on_white(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    rgba = Image.new("RGBA", (80, 100), (0, 0, 0, 0))
    rgba.putpixel((40, 50), (255, 0, 0, 255))
    rgba.save(raw / "alpha.png")

    result = validate_images(sample_config(), find_images(sample_config()))
    out = Image.open(result.base_dir / "images" / "0000.png")
    # After crop/resize the red pixel should still sit on a white background.
    assert out.mode == "RGB"
    corners = [out.getpixel((0, 0)), out.getpixel((63, 0)), out.getpixel((0, 79))]
    assert all(pixel == (255, 255, 255) for pixel in corners)


def test_missing_target_size_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = make_raw_dir(tmp_path)
    save_rgb(raw / "a.png", (80, 100))
    config = sample_config()
    del config["data"]["target_width"]

    with pytest.raises(IngestError, match="target_width"):
        validate_images(config, find_images(config))
