from PIL import Image, ImageDraw


def sample_config(**data_overrides):
    data = {
        "target_size": 64,
        "format": "png",
        "build_id": "001",
        "dedup": {"exact": True, "perceptual_threshold": 4},
    }
    data.update(data_overrides)
    return {"project_name": "walkers", "data": data}


def make_raw_dir(root):
    raw = root / "projects" / "walkers" / "data" / "raw"
    raw.mkdir(parents=True)
    return raw


def save_rgb(path, size, color=(32, 64, 128)):
    Image.new("RGB", size, color).save(path)
    return path


def save_pattern(path, size=(128, 160), seed_color=(220, 40, 40)):
    img = Image.new("RGB", size, (12, 24, 48))
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle((w // 8, h // 8, w // 2, h // 2), fill=seed_color)
    draw.ellipse((w // 3, h // 3, w - 8, h - 8), fill=(40, 180, 90))
    draw.line((0, 0, w, h), fill=(240, 240, 40), width=5)
    img.save(path)
    return img
