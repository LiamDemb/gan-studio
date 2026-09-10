"""Complete training phases; compilation never encloses a regularisation graph."""

import copy
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .models import Generator, Discriminator, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 0.002
    r1_gamma: float = 10.0
    r1_interval: int = 16
    pl_weight: float = 2.0
    pl_interval: int = 8
    pl_batch_shrink: int = 2
    pl_decay: float = 0.01
    style_mixing: float = 0.9
    ema_kimg: float = 10.0
    ema_rampup: float = 0.05
    w_avg_beta: float = 0.995
    precision: str = "fp32"
    compile_main: bool = False
    channels_last: bool = True
    tf32: bool = False
    microbatch: int = 0
    mirror: bool = False

    def __post_init__(self):
        if self.lr <= 0 or min(self.r1_gamma, self.pl_weight) < 0:
            raise ValueError(
                "lr must be positive and regularisation weights nonnegative"
            )
        if min(self.r1_interval, self.pl_interval, self.pl_batch_shrink) < 1:
            raise ValueError("Regularisation intervals and shrink must be positive")
        if not 0 <= self.style_mixing <= 1 or not 0 < self.pl_decay <= 1:
            raise ValueError("Invalid mixing probability or path target decay")
        if not 0 <= self.w_avg_beta < 1 or self.ema_kimg <= 0 or self.ema_rampup < 0:
            raise ValueError("Invalid EMA parameters")
        if self.precision not in ("fp32", "bf16", "fp16") or self.microbatch < 0:
            raise ValueError("Invalid precision or microbatch")


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def average_(tensor):
    if world_size() > 1:
        dist.all_reduce(tensor)
        tensor.div_(world_size())
    return tensor


def sync_gradients(parameters, bucket_elements=4 * 1024 * 1024):
    """Explicit bucketed reduction after backward, including higher derivatives.

    This intentionally avoids putting autograd.grad inside a DDP reducer.
    Communication is not overlapped with backward. Used masks preserve Adam's
    skip semantics for parameters absent from a regularisation graph.
    """
    if world_size() == 1:
        return
    params = list(parameters)
    used = torch.tensor(
        [p.grad is not None for p in params], device=params[0].device, dtype=torch.int32
    )
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = [p for p, flag in zip(params, used.cpu().tolist()) if flag]
    bucket, count = [], 0

    def flush(group):
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in group
            ]
        )
        average_(flat)
        start = 0
        for p in group:
            grad = flat[start : start + p.numel()].reshape(p.shape)
            if p.grad is None:
                p.grad = grad.clone(memory_format=torch.preserve_format)
            else:
                p.grad.copy_(grad)
            start += p.numel()

    for p in active:
        if bucket and count + p.numel() > bucket_elements:
            flush(bucket)
            bucket, count = [], 0
        bucket.append(p)
        count += p.numel()
    if bucket:
        flush(bucket)


def lazy_adam(params, lr, interval, enabled, cuda):
    ratio = interval / (interval + 1) if enabled else 1.0
    return torch.optim.Adam(
        params,
        lr=lr * ratio,
        betas=(0.0, 0.99**ratio),
        eps=1e-8,
        fused=True if cuda else None,
    )


def r1_penalty(scores, real):
    (grad,) = torch.autograd.grad(scores.sum(), real, create_graph=True)
    return grad.square().flatten(1).sum(1).mean()


def path_lengths(images, ws, noise=None):
    if noise is None:
        noise = torch.randn_like(images) / math.sqrt(images.shape[2] * images.shape[3])
    (grad,) = torch.autograd.grad((images * noise).sum(), ws, create_graph=True)
    return (grad.square().sum(2).mean(1) + 1e-8).sqrt()


class Trainer:
    def __init__(self, model=ModelConfig(), train=TrainConfig(), device="cuda"):
        self.model_cfg, self.cfg = model, train
        self.device = torch.device(device)
        cuda = self.device.type == "cuda"
        if train.precision == "fp16" and not cuda:
            raise ValueError("FP16 training is supported only on CUDA")
        if cuda and train.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; use fp16 or fp32")
        if cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = train.tf32
            torch.backends.cudnn.allow_tf32 = train.tf32
        if model.resample == "triton":
            from .ops import triton_fir  # Register custom ops before compiler tracing.
        self.G, self.D = Generator(model).to(self.device), Discriminator(model).to(
            self.device
        )
        if train.channels_last:
            self.G.to(memory_format=torch.channels_last)
            self.D.to(memory_format=torch.channels_last)
        if world_size() > 1:
            for net in (self.G, self.D):
                for tensor in list(net.parameters()) + list(net.buffers()):
                    dist.broadcast(tensor.detach(), 0)
        self.G_ema = copy.deepcopy(self.G).eval().requires_grad_(False)
        self.g_opt = lazy_adam(
            self.G.parameters(), train.lr, train.pl_interval, train.pl_weight > 0, cuda
        )
        self.d_opt = lazy_adam(
            self.D.parameters(), train.lr, train.r1_interval, train.r1_gamma > 0, cuda
        )
        self.g_scaler = torch.amp.GradScaler(
            "cuda", enabled=cuda and train.precision == "fp16"
        )
        self.d_scaler = torch.amp.GradScaler(
            "cuda", enabled=cuda and train.precision == "fp16"
        )
        self.pl_mean = torch.zeros((), device=self.device)
        self.steps, self.images_seen = 0, 0
        self.synth_main = self.G.synthesis
        self.disc_main = self.D
        if train.compile_main:
            # Same parameters, separate callables. R1 and PL below call the
            # ORIGINAL eager modules, including their forward passes.
            self.synth_main = torch.compile(self.G.synthesis, dynamic=False)
            self.disc_main = torch.compile(self.D, dynamic=False)

    def autocast(self):
        if self.cfg.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.cfg.precision == "bf16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def styles(self, batch, mix=True):
        # Mapping and style statistics always run in FP32 in the training API.
        z = torch.randn(batch, self.model_cfg.z_dim, device=self.device)
        w = self.G.mapping(z)
        ws = w[:, None].expand(-1, self.G.num_ws, -1)
        if mix and random.random() < self.cfg.style_mixing:
            cut = random.randrange(1, self.G.num_ws)
            w2 = self.G.mapping(torch.randn_like(z))
            ws = torch.cat(
                (ws[:, :cut], w2[:, None].expand(-1, self.G.num_ws - cut, -1)), 1
            )
        # Mixing and non-mixing must present identical strides to the compiler.
        return ws.contiguous(), w

    def finish_phase(self, opt, scaler, parameters):
        # Reduce scaled gradients before GradScaler's inf check. If any rank
        # overflows, the reduced inf causes every rank to skip the same update.
        sync_gradients(parameters)
        scaler.step(opt)
        scaler.update()

    def step(self, real):
        cfg = self.cfg
        if real.ndim != 4 or tuple(real.shape[1:]) != (
            3,
            self.model_cfg.resolution,
            self.model_cfg.resolution,
        ):
            raise ValueError("real must be [N,3,resolution,resolution]")
        n = real.shape[0]
        if n < 1:
            raise ValueError("Empty training batch")
        micro = cfg.microbatch or n
        if n % micro:
            raise ValueError("Batch must be divisible by microbatch")
        if cfg.pl_weight and micro % cfg.pl_batch_shrink:
            raise ValueError("Microbatch must be divisible by pl_batch_shrink")
        real = real.to(self.device, non_blocking=True)
        if real.dtype == torch.uint8:
            real = real.float().mul_(1 / 127.5).sub_(1.0)
        else:
            real = real.float()
        if cfg.mirror:
            flags = torch.rand(n, 1, 1, 1, device=self.device) < 0.5
            real = torch.where(flags, real.flip(3), real)
        if cfg.channels_last:
            real = real.contiguous(memory_format=torch.channels_last)
        chunks = real.split(micro)
        logs = {
            k: torch.zeros((), device=self.device)
            for k in ("d_loss", "g_loss", "r1", "pl")
        }
        do_r1 = cfg.r1_gamma > 0 and (self.steps + 1) % cfg.r1_interval == 0
        do_pl = cfg.pl_weight > 0 and (self.steps + 1) % cfg.pl_interval == 0

        self.G.requires_grad_(False)
        self.D.requires_grad_(True)
        self.g_opt.zero_grad(set_to_none=True)
        self.d_opt.zero_grad(set_to_none=True)
        for chunk in chunks:
            with torch.no_grad():
                ws, _ = self.styles(micro)
                with self.autocast():
                    fake = self.synth_main(ws, "random")
            with self.autocast():
                loss = (
                    F.softplus(self.disc_main(fake)).mean()
                    + F.softplus(-self.disc_main(chunk)).mean()
                )
            self.d_scaler.scale(loss / len(chunks)).backward()
            logs["d_loss"] += loss.detach() / len(chunks)
        self.finish_phase(self.d_opt, self.d_scaler, self.D.parameters())

        if do_r1:
            self.d_opt.zero_grad(set_to_none=True)
            for chunk in chunks:
                # Both forward and higher derivatives use the eager FP32 path.
                leaf = chunk.detach().requires_grad_(True)
                penalty = r1_penalty(self.D(leaf), leaf)
                loss = penalty * (0.5 * cfg.r1_gamma * cfg.r1_interval / len(chunks))
                self.d_scaler.scale(loss).backward()
                logs["r1"] += penalty.detach() / len(chunks)
            self.finish_phase(self.d_opt, self.d_scaler, self.D.parameters())

        self.D.requires_grad_(False)
        self.G.requires_grad_(True)
        self.d_opt.zero_grad(set_to_none=True)
        self.g_opt.zero_grad(set_to_none=True)
        w_mean = torch.zeros_like(self.G.mapping.w_avg)
        for _ in chunks:
            ws, w = self.styles(micro)
            w_mean += w.detach().mean(0) / len(chunks)
            with self.autocast():
                loss = F.softplus(-self.disc_main(self.synth_main(ws, "random"))).mean()
            self.g_scaler.scale(loss / len(chunks)).backward()
            logs["g_loss"] += loss.detach() / len(chunks)
        self.finish_phase(self.g_opt, self.g_scaler, self.G.parameters())
        with torch.no_grad():
            self.G.mapping.w_avg.lerp_(average_(w_mean), 1 - cfg.w_avg_beta)

        if do_pl:
            self.g_opt.zero_grad(set_to_none=True)
            # Use the previous target throughout accumulation. Update once from
            # all samples/ranks afterwards, independent of microbatch partition.
            length_mean = torch.zeros((), device=self.device)
            for _ in chunks:
                ws, _ = self.styles(micro // cfg.pl_batch_shrink, mix=False)
                images = self.G.synthesis(ws, "random")
                lengths = path_lengths(images, ws)
                penalty = (lengths - self.pl_mean.detach()).square().mean()
                self.g_scaler.scale(
                    penalty * (cfg.pl_weight * cfg.pl_interval / len(chunks))
                ).backward()
                length_mean += lengths.detach().mean() / len(chunks)
                logs["pl"] += penalty.detach() / len(chunks)
            self.finish_phase(self.g_opt, self.g_scaler, self.G.parameters())
            with torch.no_grad():
                self.pl_mean.lerp_(average_(length_mean), cfg.pl_decay)

        self.steps += 1
        batch_global = n * world_size()
        self.images_seen += batch_global
        ema_images = cfg.ema_kimg * 1000
        if cfg.ema_rampup:
            ema_images = min(ema_images, self.images_seen * cfg.ema_rampup)
        beta = 0.5 ** (batch_global / max(ema_images, 1e-8))
        with torch.no_grad():
            for target, source in zip(self.G_ema.parameters(), self.G.parameters()):
                target.lerp_(source, 1 - beta)
            for target, source in zip(self.G_ema.buffers(), self.G.buffers()):
                target.copy_(source)
        logs["did_r1"] = do_r1
        logs["did_pl"] = do_pl
        return logs  # No .item()/synchronisation on the hot path.

    def save(self, path, extra=None):
        """Collective on distributed runs; only rank 0 writes the checkpoint."""
        rng = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "cuda": (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }
        states = [None] * world_size()
        if world_size() > 1:
            dist.all_gather_object(states, rng)
        else:
            states[0] = rng
        if world_size() > 1 and dist.get_rank() != 0:
            return
        payload = {
            "format_version": 1,
            "model_config": asdict(self.model_cfg),
            "train_config": asdict(self.cfg),
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "G_ema": self.G_ema.state_dict(),
            "g_opt": self.g_opt.state_dict(),
            "d_opt": self.d_opt.state_dict(),
            "g_scaler": self.g_scaler.state_dict(),
            "d_scaler": self.d_scaler.state_dict(),
            "pl_mean": self.pl_mean,
            "steps": self.steps,
            "images_seen": self.images_seen,
            "rng": states,
            "extra": extra or {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp)
        os.replace(temp, path)

    def load(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        if payload["model_config"] != asdict(self.model_cfg) or payload[
            "train_config"
        ] != asdict(self.cfg):
            raise ValueError("Resume requires identical model/training configuration")
        if len(payload["rng"]) != world_size():
            raise ValueError("Exact resume requires the same world size")
        for name in ("G", "D", "G_ema", "g_opt", "d_opt", "g_scaler", "d_scaler"):
            getattr(self, name).load_state_dict(payload[name])
        self.pl_mean.copy_(payload["pl_mean"])
        self.steps, self.images_seen = payload["steps"], payload["images_seen"]
        rng = payload["rng"][dist.get_rank() if world_size() > 1 else 0]
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        if self.device.type == "cuda" and rng["cuda"] is not None:
            torch.cuda.set_rng_state(rng["cuda"], self.device)
        return payload["extra"]
