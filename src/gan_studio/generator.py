from pathlib import Path

import torch

from stylegan2 import Generator, ModelConfig


class ArtGenerator:
    def __init__(self, base_dir=".", name="art", device=None):
        self.base_dir = Path(base_dir)
        self.name = str(name)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = self.base_dir / "models" / self.name / "latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        model_config = dict(state["model_config"])
        model_config["resample"] = "torch"
        self.model = Generator(ModelConfig(**model_config)).to(self.device).eval()
        self.model.load_state_dict(state["G_ema"])
        self.model.requires_grad_(False)

    def random_latent(self, seed=None):
        kwargs = {"device": self.device}
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            kwargs["generator"] = generator
        return torch.randn(1, self.model.cfg.z_dim, **kwargs)

    def latent_to_style(self, z, trunc_psi=0.7):
        z = z.to(self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        with torch.no_grad():
            w = self.model.mapping(z)
            if trunc_psi != 1:
                w = self.model.mapping.w_avg.lerp(w, trunc_psi)
        return w

    def style_to_image(self, style):
        style = style.to(self.device)
        if style.ndim == 1:
            style = style.unsqueeze(0)
        with torch.no_grad():
            ws = style[:, None].expand(-1, self.model.num_ws, -1)
            image = self.model.synthesis(ws, noise_mode="const")
            image = (image.clamp(-1, 1) + 1).mul(0.5)
        return image

    def generate_random(self, trunc_psi=0.7, seed=None):
        image, _z = self.sample(trunc_psi=trunc_psi, seed=seed)
        return image

    def sample(self, trunc_psi=0.7, seed=None):
        z = self.random_latent(seed=seed)
        style = self.latent_to_style(z, trunc_psi)
        return self.style_to_image(style), z

    def interpolate_styles(self, styles, frames_per_leg):
        if len(styles) < 2:
            raise ValueError("Need at least two styles to interpolate")
        if frames_per_leg < 1:
            raise ValueError("frames_per_leg must be at least 1")

        for start, end in zip(styles, styles[1:]):
            for step in range(frames_per_leg):
                t = step / frames_per_leg
                yield self.style_to_image(start * (1 - t) + end * t)
        yield self.style_to_image(styles[-1])
