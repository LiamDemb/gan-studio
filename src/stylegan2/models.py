import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ops import modulated_conv2d, upsample, downsample, blur
from .ops.conv_grad import conv2d


@dataclass(frozen=True)
class ModelConfig:
    resolution: int = 256
    z_dim: int = 512
    w_dim: int = 512
    mapping_layers: int = 8
    channel_base: int = 32768
    channel_max: int = 512
    modconv: str = "factorized"
    resample: str = "torch"
    mbstd_group: int = 4

    def __post_init__(self):
        if self.resolution < 4 or self.resolution & (self.resolution - 1):
            raise ValueError("resolution must be a power of two, at least 4")
        if (
            min(
                self.z_dim,
                self.w_dim,
                self.mapping_layers,
                self.channel_max,
                self.mbstd_group,
            )
            < 1
        ):
            raise ValueError("Dimensions and layer counts must be positive")
        if self.channel_base < self.resolution:
            raise ValueError("channel_base must be >= resolution")
        if self.modconv not in ("factorized", "grouped") or self.resample not in (
            "torch",
            "triton",
        ):
            raise ValueError("Unknown operation backend")

    def channels(self, resolution):
        return min(self.channel_base // resolution, self.channel_max)


class EqualLinear(nn.Module):
    def __init__(
        self, inputs, outputs, lr_multiplier=1.0, bias_init=0.0, activate=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outputs, inputs) / lr_multiplier)
        self.bias = nn.Parameter(torch.full((outputs,), float(bias_init)))
        self.weight_gain = lr_multiplier / math.sqrt(inputs)
        self.bias_gain = lr_multiplier
        self.activate = activate

    def forward(self, x):
        x = F.linear(x, self.weight * self.weight_gain, self.bias * self.bias_gain)
        return F.leaky_relu(x, 0.2) * math.sqrt(2) if self.activate else x


class Mapping(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                EqualLinear(
                    cfg.z_dim if i == 0 else cfg.w_dim, cfg.w_dim, 0.01, activate=True
                )
                for i in range(cfg.mapping_layers)
            ]
        )
        self.register_buffer("w_avg", torch.zeros(cfg.w_dim))

    def forward(self, z):
        z = z * (z.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        return self.layers(z)


class StyledConv(nn.Module):
    def __init__(self, cin, cout, resolution, cfg, up=False, rgb=False):
        super().__init__()
        k = 1 if rgb else 3
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k))
        self.gain = 1 / math.sqrt(cin * k * k)
        self.affine = EqualLinear(cfg.w_dim, cin, bias_init=1.0)
        self.bias = nn.Parameter(torch.zeros(cout))
        self.up, self.rgb = up, rgb
        self.modconv, self.resample = cfg.modconv, cfg.resample
        if not rgb:
            self.noise_strength = nn.Parameter(torch.zeros(()))
            self.register_buffer(
                "noise_const", torch.randn(1, 1, resolution, resolution)
            )

    def forward(self, x, w, noise_mode="random", conv_backend="native"):
        with torch.autocast(
            device_type="cpu" if w.device.type == "meta" else w.device.type,
            enabled=False,
        ):
            style = self.affine(w)
        if self.up:
            x = upsample(x, self.resample, conv_backend)
        x = modulated_conv2d(
            x,
            self.weight * self.gain,
            style,
            not self.rgb,
            self.modconv,
            conv_backend=conv_backend,
        )
        if not self.rgb:
            if noise_mode == "random":
                noise = torch.randn(
                    x.shape[0],
                    1,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                )
                x = x + noise * self.noise_strength.to(x.dtype)
            elif noise_mode == "const":
                x = x + self.noise_const.to(x.dtype) * self.noise_strength.to(x.dtype)
            elif noise_mode != "none":
                raise ValueError("noise_mode must be random, const or none")
        x = x + self.bias.to(x.dtype)[None, :, None, None]
        return x if self.rgb else F.leaky_relu(x, 0.2) * math.sqrt(2)


class Synthesis(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_ws = 2 * int(math.log2(cfg.resolution)) - 2
        self.constant = nn.Parameter(torch.randn(1, cfg.channels(4), 4, 4))
        self.first = StyledConv(cfg.channels(4), cfg.channels(4), 4, cfg)
        self.first_rgb = StyledConv(cfg.channels(4), 3, 4, cfg, rgb=True)
        self.blocks = nn.ModuleList()
        for exponent in range(3, int(math.log2(cfg.resolution)) + 1):
            r = 2**exponent
            self.blocks.append(
                nn.ModuleList(
                    [
                        StyledConv(
                            cfg.channels(r // 2), cfg.channels(r), r, cfg, up=True
                        ),
                        StyledConv(cfg.channels(r), cfg.channels(r), r, cfg),
                        StyledConv(cfg.channels(r), 3, r, cfg, rgb=True),
                    ]
                )
            )

    def forward(self, ws, noise_mode="random", conv_backend="native"):
        if ws.ndim != 3 or ws.shape[1] != self.num_ws or ws.shape[2] != self.cfg.w_dim:
            raise ValueError("ws must be [batch, num_ws, w_dim]")
        x = self.constant.expand(ws.shape[0], -1, -1, -1)
        x = self.first(x, ws[:, 0], noise_mode, conv_backend)
        rgb = self.first_rgb(x, ws[:, 1], conv_backend=conv_backend).float()
        slot = 1
        for conv_up, conv, to_rgb in self.blocks:
            x = conv_up(x, ws[:, slot], noise_mode, conv_backend)
            x = conv(x, ws[:, slot + 1], noise_mode, conv_backend)
            rgb = (
                upsample(rgb, self.cfg.resample, conv_backend)
                + to_rgb(x, ws[:, slot + 2], conv_backend=conv_backend).float()
            )
            slot += 2
        return rgb


class Generator(nn.Module):
    def __init__(self, cfg=ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.mapping = Mapping(cfg)
        self.synthesis = Synthesis(cfg)

    @property
    def num_ws(self):
        return self.synthesis.num_ws

    def forward(
        self, z, truncation_psi=1.0, truncation_cutoff=None, noise_mode="const"
    ):
        w = self.mapping(z)
        ws = w[:, None].expand(-1, self.num_ws, -1)
        if truncation_psi != 1:
            cutoff = self.num_ws if truncation_cutoff is None else truncation_cutoff
            mask = torch.arange(self.num_ws, device=z.device)[None, :, None] < cutoff
            truncated = self.mapping.w_avg.lerp(ws, truncation_psi)
            ws = torch.where(mask, truncated, ws)
        return self.synthesis(ws, noise_mode)


class EqualConv(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, activate=True, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(cout, cin, k, k))
        self.bias = nn.Parameter(torch.zeros(cout)) if bias else None
        self.gain = 1 / math.sqrt(cin * k * k)
        self.stride, self.pad, self.activate = stride, k // 2, activate

    def forward(self, x, conv_backend="native"):
        x = conv2d(
            x,
            self.weight * self.gain,
            stride=self.stride,
            padding=self.pad,
            backend=conv_backend,
            weight_independent=True,
        )
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)[None, :, None, None]
        return F.leaky_relu(x, 0.2) * math.sqrt(2) if self.activate else x


class DownBlock(nn.Module):
    def __init__(self, cin, cout, backend):
        super().__init__()
        self.conv = EqualConv(cin, cin)
        self.conv_down = EqualConv(cin, cout, stride=2)
        self.skip = EqualConv(cin, cout, k=1, activate=False, bias=False)
        self.backend = backend

    def forward(self, x, conv_backend="native"):
        skip = self.skip(downsample(x, self.backend, conv_backend), conv_backend)
        y = self.conv_down(
            blur(self.conv(x, conv_backend), self.backend, conv_backend), conv_backend
        )
        return (y + skip) / math.sqrt(2)


def minibatch_std(x, max_group=4):
    n, c, h, w = x.shape
    group = min(max_group, n)
    # A batch of 6 still has meaningful groups of 3; no reshape failure.
    while n % group:
        group -= 1
    y = x.float().reshape(group, n // group, c, h, w)
    std = (y.var(0, unbiased=False) + 1e-8).sqrt().mean((1, 2, 3), keepdim=True)
    std = std.repeat(group, 1, h, w).to(x.dtype)
    return torch.cat((x, std), dim=1)


class Discriminator(nn.Module):
    def __init__(self, cfg=ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.from_rgb = EqualConv(3, cfg.channels(cfg.resolution), k=1)
        self.blocks = nn.Sequential(
            *[
                DownBlock(cfg.channels(2**e), cfg.channels(2 ** (e - 1)), cfg.resample)
                for e in range(int(math.log2(cfg.resolution)), 2, -1)
            ]
        )
        c = cfg.channels(4)
        self.final_conv = EqualConv(c + 1, c)
        self.final_hidden = EqualLinear(c * 4 * 4, c, activate=True)
        self.final_score = EqualLinear(c, 1)

    def forward(self, image, conv_backend="native"):
        x = self.from_rgb(image, conv_backend)
        for block in self.blocks:
            x = block(x, conv_backend)
        x = self.final_conv(minibatch_std(x, self.cfg.mbstd_group), conv_backend)
        x = self.final_hidden(x.flatten(1))
        return self.final_score(x).float().flatten()
