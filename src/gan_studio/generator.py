import torch
from stylegan2_pytorch import ModelLoader


class ArtGenerator:
    def __init__(self, base_dir=".", name="art"):
        self.loader = ModelLoader(
            base_dir=base_dir,
            name=name,
        )

    def random_latent(self, seed=None):
        kwargs = {"device": "cuda"}
        if seed is not None:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(int(seed))
            kwargs["generator"] = generator
        return torch.randn(1, 512, **kwargs)

    def latent_to_style(self, z, trunc_psi=0.7):
        return self.loader.noise_to_styles(z, trunc_psi=trunc_psi)

    def style_to_image(self, style):
        return self.loader.styles_to_images(style)

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
