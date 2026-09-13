#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2_botanical.py
===============
Two-phase Latent Diffusion Model for 128x256 (WxH) botanical image synthesis.

Rewrite of `v1_botanical.py`. Main changes vs. the baseline:

  * Custom 5.8M VAE (rFID ~22.0) replaced by frozen `stabilityai/sd-vae-ft-mse`
    (rFID ~1-2 at this resolution). This removes the dominant FID floor.
  * Latents are encoded ONCE into a disk-backed fp16 memmap, then held
    GPU-resident. The training loop touches no DataLoader at all.
  * Horizontal-flip augmentation is baked into the cache (both orientations
    are encoded), because you cannot flip a cached latent safely.
  * ADM-style U-Net (GN -> SiLU -> Conv, FiLM scale-shift time conditioning,
    zero-init residual/output convs, SDPA attention, dropout), scalable to
    ~100M params.
  * v-prediction with a zero-terminal-SNR cosine schedule, DDIM sampler with
    trailing spacing, optional min-SNR-gamma loss weighting.
  * AdamW + warmup + cosine decay + grad clipping. EMA with a warmup ramp
    (the baseline's EMA spent its first ~10k steps averaging random init).
  * FID: Inception statistics for the real set are computed ONCE and reused;
    reals are always the custom botanical images only, never PlantNet.

Pipeline
--------
    # 0. sanity check hardware, schedule, sampler, throughput
    python v2_botanical.py doctor

    # 1. encode latents (once per phase; phase 2 reuses phase 1's custom shard)
    python v2_botanical.py cache --phase 1
    python v2_botanical.py cache --phase 2

    # 2. pre-train on PlantNet + custom (~74k images)
    python v2_botanical.py train --phase 1

    # 3. fine-tune on custom only, half LR, warm-started from phase 1
    python v2_botanical.py train --phase 2 \
        --init-from runs/phase1/best.pt

    # 4. final evaluation / samples
    python v2_botanical.py fid    --ckpt runs/phase2/best.pt --n-fake 10000
    python v2_botanical.py sample --ckpt runs/phase2/best.pt --n 16

Requirements
------------
    torch>=2.1  torchvision  diffusers  torchmetrics  torch-fidelity  scipy
    pillow  tqdm  numpy
    (torch-fidelity is what torchmetrics' FID actually needs; `torchmetrics[image]`
     pulls it in, but installing it explicitly avoids a confusing ImportError.)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# Constants that define cache/checkpoint compatibility.
# Bump CACHE_VERSION whenever the latent encoding procedure changes.
# ----------------------------------------------------------------------------
CACHE_VERSION = "lat-v2"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VAE_DOWNSAMPLE = 8  # sd-vae-ft-mse is an f8 autoencoder


# ============================================================================
# 1. Configuration
# ============================================================================

@dataclass
class Config:
    # --- data ---------------------------------------------------------------
    pretrain_dir: str = "data/pretraining_botanical_128x256"
    custom_dir: str = "data/botanical_garden_512x896"
    cache_dir: str = "data/latent_cache"
    out_dir: str = "runs"

    image_h: int = 256
    image_w: int = 128
    aspect_mode: str = "center_crop"   # center_crop | squash
    flip_cache: bool = True            # encode both orientations

    # --- VAE ----------------------------------------------------------------
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    vae_dtype: str = "fp16"            # fp16 | bf16 | fp32
    vae_batch: int = 32

    # --- U-Net --------------------------------------------------------------
    base: int = 192              # 101.7M params at a 4x32x16 latent
    mults: tuple = (1, 2, 3)
    num_res_blocks: int = 2
    attn_levels: tuple = (1, 2)        # 0-indexed; mid always has attention
    num_heads: int = 8
    dropout: float = 0.10
    latent_ch: int = 4

    # --- diffusion ----------------------------------------------------------
    timesteps: int = 1000
    loss_weight: str = "uniform"       # uniform | min_snr
    min_snr_gamma: float = 5.0
    t_dist: str = "uniform"            # uniform | logit_normal

    # --- optimisation -------------------------------------------------------
    batch: int = 64
    lr: float = 1.5e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.99)
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    lr_min_ratio: float = 0.10
    steps: int = 300_000
    precision: str = "bf16"            # bf16 | fp16 | fp32
    channels_last: bool = True
    compile: bool = False
    latents_device: str = "auto"       # auto | cuda | cpu

    # --- EMA ----------------------------------------------------------------
    ema_decay: float = 0.9995
    ema_warmup: float = 10.0           # d = min(decay, (1+s)/(ema_warmup+s))

    # --- evaluation ---------------------------------------------------------
    eval_every: int = 10_000
    save_every: int = 5_000
    fid_fake: int = 3000
    fid_steps: int = 50
    sample_batch: int = 64
    val_frac: float = 0.0

    # --- misc ---------------------------------------------------------------
    seed: int = 0
    num_workers: int = 4
    max_hours: float = 0.0             # 0 = no wall-clock limit

    @property
    def latent_h(self) -> int:
        return self.image_h // VAE_DOWNSAMPLE

    @property
    def latent_w(self) -> int:
        return self.image_w // VAE_DOWNSAMPLE

    def arch_tag(self) -> str:
        """Identity of the weight-space. Checkpoints only load into a match."""
        m = "-".join(str(x) for x in self.mults)
        a = "-".join(str(x) for x in self.attn_levels) or "none"
        return (f"unet_b{self.base}_m{m}_r{self.num_res_blocks}_a{a}"
                f"_h{self.num_heads}_c{self.latent_ch}"
                f"_{self.latent_h}x{self.latent_w}")

    def cache_tag(self, phase: int) -> str:
        vae_slug = self.vae_id.split("/")[-1]
        return (f"{CACHE_VERSION}_p{phase}_{self.image_h}x{self.image_w}"
                f"_{self.aspect_mode}_{vae_slug}"
                f"_flip{int(self.flip_cache)}")


def _tuple_of_int(s: str) -> tuple:
    s = s.strip()
    if not s:
        return tuple()
    return tuple(int(x) for x in s.replace(" ", "").split(","))


# ============================================================================
# 2. Utilities
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(out_dir: Path, name: str = "run") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / f"{name}.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def dtype_of(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def list_images(root: Path) -> list:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise RuntimeError(f"No images with extensions {IMAGE_EXTS} under {root}")
    return paths

def get_fid_splits(custom_dir: str, seed: int) -> tuple[list, list]:
    """Divide il dataset esattamente a metà in modo deterministico (Split A e Split B)."""
    paths = list_images(Path(custom_dir))
    rng = random.Random(seed)
    rng.shuffle(paths)
    half = len(paths) // 2
    return paths[:half], paths[half:]
# ============================================================================
# 3. Data
# ============================================================================

class AspectImageDataset(Dataset):
    """Reads images from disk and maps them to a fixed (H, W) in [-1, 1].

    `center_crop` first crops to the target aspect ratio, then resizes. This
    matters here: the custom set is 512x896 (ar 0.571) while the target is
    128x256 (ar 0.5), so a plain resize squashes plants horizontally by ~12%.
    `squash` reproduces the baseline behaviour if you want a like-for-like
    comparison against the old FID numbers.
    """

    def __init__(self, paths: Sequence[Path], h: int, w: int, aspect_mode: str = "center_crop"):
        from torchvision import transforms

        self.paths = list(paths)
        self.h, self.w = h, w
        self.aspect_mode = aspect_mode
        self.target_ar = w / h

        ops = []
        if aspect_mode == "center_crop":
            ops.append(transforms.Lambda(self._crop_to_aspect))
        elif aspect_mode != "squash":
            raise ValueError(f"unknown aspect_mode {aspect_mode!r}")
        ops += [
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
        self.tf = transforms.Compose(ops)

    def _crop_to_aspect(self, img):
        W, H = img.size
        ar = W / H
        if abs(ar - self.target_ar) < 1e-6:
            return img
        if ar > self.target_ar:          # too wide -> trim width
            new_w = int(round(H * self.target_ar))
            new_h = H
        else:                            # too tall -> trim height
            new_w = W
            new_h = int(round(W / self.target_ar))
        left = (W - new_w) // 2
        top = (H - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        # Importazione LOCALE: ogni worker su Windows avrà accesso a PIL
        from PIL import Image, ImageOps
        
        img = Image.open(self.paths[i])
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return self.tf(img)


# ============================================================================
# 4. VAE wrapper (frozen, pre-trained)
# ============================================================================

class FrozenVAE:
    """Thin wrapper around a diffusers AutoencoderKL.

    We deliberately ignore `config.scaling_factor`: latents are normalised with
    per-channel statistics measured on *this* dataset, which is strictly better
    than a single LAION-derived scalar for a specialised domain.
    """

    def __init__(self, vae_id: str, device: torch.device, dtype: str = "fp16"):
        from diffusers import AutoencoderKL

        self.device = device
        self.dtype = dtype_of(dtype) if device.type == "cuda" else torch.float32
        logging.info(f"Loading VAE {vae_id} ({self.dtype})")
        self.vae = AutoencoderKL.from_pretrained(vae_id).to(device=device, dtype=self.dtype)
        self.vae.eval()
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [-1,1] -> posterior mean latents (B,4,H/8,W/8)."""
        out = self.vae.encode(x.to(self.device, self.dtype))
        return out.latent_dist.mean.float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: raw (un-normalised) latents -> images in [-1,1]."""
        img = self.vae.decode(z.to(self.device, self.dtype)).sample.float()
        if not torch.isfinite(img).all():
            logging.warning("Non-finite pixels from VAE decode; retrying in fp32.")
            self.vae.to(torch.float32)
            self.dtype = torch.float32
            img = self.vae.decode(z.to(self.device, torch.float32)).sample.float()
        return img

    @torch.no_grad()
    def decode_chunked(self, z: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        return torch.cat([self.decode(z[i:i + chunk]) for i in range(0, z.shape[0], chunk)], 0)


# ============================================================================
# 5. Latent cache
# ============================================================================

class LatentCache:
    """Disk-backed fp16 memmap of VAE latents, optionally GPU-resident.

    Layout: [0, N)   -> original images
            [N, 2N)  -> horizontally flipped images   (if flip_cache)
    Storing both orientations is the correct way to get RandomHorizontalFlip
    with a cached latent pipeline: the VAE is not equivariant to flips, so
    flipping a latent tensor is not the same as flipping the image.
    """

    def __init__(self, path: Path, meta: dict):
        self.path = Path(path)
        self.meta = meta

    # -- construction --------------------------------------------------------
    @staticmethod
    def build(cfg: Config, dirs: Sequence[Path], tag: str, vae: FrozenVAE,
              device: torch.device, force: bool = False) -> "LatentCache":
        from tqdm import tqdm

        cache_dir = Path(cfg.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        npy = cache_dir / f"{tag}.npy"
        jsn = cache_dir / f"{tag}.json"

        paths: list = []
        shard_sizes = []
        for d in dirs:
            p = list_images(Path(d))
            shard_sizes.append(len(p))
            paths.extend(p)
        n_img = len(paths)
        n_out = n_img * (2 if cfg.flip_cache else 1)
        shape = (n_out, cfg.latent_ch, cfg.latent_h, cfg.latent_w)

        if npy.exists() and jsn.exists() and not force:
            meta = json.loads(jsn.read_text())
            if meta.get("shape") == list(shape) and meta.get("complete"):
                logging.info(f"Reusing latent cache {npy.name}  shape={tuple(shape)}")
                return LatentCache(npy, meta)
            logging.warning("Cache present but stale/incomplete; rebuilding.")

        logging.info(f"Encoding {n_img} images -> {n_out} latents  {tuple(shape)}")
        logging.info(f"  sources: " + ", ".join(f"{Path(d).name}={n}"
                                                for d, n in zip(dirs, shard_sizes)))
        mm = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float16, shape=shape)

        ds = AspectImageDataset(paths, cfg.image_h, cfg.image_w, cfg.aspect_mode)
        loader = DataLoader(ds, batch_size=cfg.vae_batch, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))
        cursor = 0
        t0 = time.time()
        for imgs in tqdm(loader, desc="VAE encode", dynamic_ncols=True):
            imgs = imgs.to(device, non_blocking=True)
            b = imgs.shape[0]
            z = vae.encode(imgs)
            if not torch.isfinite(z).all():
                raise RuntimeError("VAE produced non-finite latents; try --vae-dtype fp32.")
            mm[cursor:cursor + b] = z.cpu().numpy().astype(np.float16)
            if cfg.flip_cache:
                zf = vae.encode(torch.flip(imgs, dims=[3]))
                mm[n_img + cursor: n_img + cursor + b] = zf.cpu().numpy().astype(np.float16)
            cursor += b
        mm.flush()
        del mm
        logging.info(f"Encoding finished in {(time.time() - t0) / 60:.1f} min")

        # per-channel statistics over the whole cache (fp32 accumulation)
        arr = np.load(npy, mmap_mode="r")
        mean = np.zeros(cfg.latent_ch, dtype=np.float64)
        sq = np.zeros(cfg.latent_ch, dtype=np.float64)
        count = 0
        for i in range(0, arr.shape[0], 4096):
            block = np.asarray(arr[i:i + 4096], dtype=np.float64)
            mean += block.sum(axis=(0, 2, 3))
            sq += (block ** 2).sum(axis=(0, 2, 3))
            count += block.shape[0] * block.shape[2] * block.shape[3]
        mean /= count
        std = np.sqrt(np.maximum(sq / count - mean ** 2, 1e-12))

        meta = {
            "shape": list(shape),
            "n_images": n_img,
            "flip": bool(cfg.flip_cache),
            "sources": [str(d) for d in dirs],
            "shard_sizes": shard_sizes,
            "image_hw": [cfg.image_h, cfg.image_w],
            "aspect_mode": cfg.aspect_mode,
            "vae_id": cfg.vae_id,
            "version": CACHE_VERSION,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "complete": True,
        }
        jsn.write_text(json.dumps(meta, indent=2))
        logging.info(f"latent per-channel mean={np.round(mean, 4).tolist()}")
        logging.info(f"latent per-channel std ={np.round(std, 4).tolist()}")
        return LatentCache(npy, meta)

    # -- loading -------------------------------------------------------------
    def to_tensor(self, device: torch.device, prefer: str = "auto") -> torch.Tensor:
        arr = np.load(self.path, mmap_mode="r")
        nbytes = arr.size * 2
        target = device
        if prefer == "cpu":
            target = torch.device("cpu")
        elif prefer == "auto" and device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            # keep >=6 GB of headroom for the model, activations and the VAE
            if nbytes > max(0, free - 6 * (1 << 30)):
                logging.warning(f"Latents ({nbytes / 1e9:.2f} GB) do not fit alongside "
                                f"training; keeping them in pinned CPU RAM.")
                target = torch.device("cpu")
        t = torch.from_numpy(np.array(arr, dtype=np.float16, order="C"))  # writable copy
        if target.type == "cpu":
            t = t.pin_memory() if device.type == "cuda" else t
        else:
            t = t.to(target)
        logging.info(f"Latents: {tuple(t.shape)} fp16 = {nbytes / 1e9:.2f} GB on {t.device}")
        return t

    @property
    def stats(self) -> tuple:
        return (torch.tensor(self.meta["mean"], dtype=torch.float32).view(1, -1, 1, 1),
                torch.tensor(self.meta["std"], dtype=torch.float32).view(1, -1, 1, 1))


# ============================================================================
# 6. Diffusion: zero-terminal-SNR cosine schedule, v-prediction
# ============================================================================

class Diffusion:
    """Discrete-time DDPM with the cosine schedule of Nichol & Dhariwal.

    The schedule terminates at alpha_bar = 0, i.e. zero terminal SNR
    (Lin et al., 2024). Combined with v-prediction this removes the
    train/test mismatch that makes samplers start from a state the model
    never saw, and it is why noise-offset hacks are unnecessary here.
    """

    def __init__(self, timesteps: int, device: torch.device, s: float = 0.008):
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        f = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
        acp_full = f / f[0]
        acp_full = acp_full.clamp(0.0, 1.0)
        acp_full[-1] = 0.0                       # exact zero terminal SNR

        self.T = timesteps
        self.device = device
        self.acp = acp_full[1:].float().to(device)          # alpha_bar[t], t=0..T-1
        self.acp_prev = acp_full[:-1].float().to(device)
        self.sqrt_acp = self.acp.sqrt()
        self.sqrt_1macp = (1.0 - self.acp).sqrt()
        self.snr = self.acp / (1.0 - self.acp).clamp(min=1e-12)

    # -- forward process -----------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_acp[t].view(-1, 1, 1, 1)
        b = self.sqrt_1macp[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    def v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_acp[t].view(-1, 1, 1, 1)
        b = self.sqrt_1macp[t].view(-1, 1, 1, 1)
        return a * noise - b * x0

    def x0_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_acp[t].view(-1, 1, 1, 1)
        b = self.sqrt_1macp[t].view(-1, 1, 1, 1)
        return a * xt - b * v

    def eps_from_v(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_acp[t].view(-1, 1, 1, 1)
        b = self.sqrt_1macp[t].view(-1, 1, 1, 1)
        return b * xt + a * v

    # -- loss weights --------------------------------------------------------
    def loss_weights(self, mode: str, gamma: float = 5.0, floor: float = 0.05) -> torch.Tensor:
        if mode == "uniform":
            return torch.ones_like(self.acp)
        if mode == "min_snr":
            # v-prediction form: min(SNR, gamma) / (SNR + 1)  (Hang et al., 2023).
            # A floor is required because zero-terminal-SNR drives the weight of
            # the last timestep to exactly 0, which would silently disable the
            # very timestep the sampler starts from.
            w = torch.clamp(self.snr, max=gamma) / (self.snr + 1.0)
            w = w.clamp(min=floor * w.max())
            return w / w.mean()
        raise ValueError(f"unknown loss weighting {mode!r}")

    # -- timestep sampling ---------------------------------------------------
    def sample_t(self, n: int, mode: str = "uniform") -> torch.Tensor:
        if mode == "uniform":
            return torch.randint(0, self.T, (n,), device=self.device)
        if mode == "logit_normal":
            u = torch.sigmoid(torch.randn(n, device=self.device))
            return (u * self.T).long().clamp(0, self.T - 1)
        raise ValueError(f"unknown t distribution {mode!r}")

    # -- sampler -------------------------------------------------------------
    def timestep_schedule(self, steps: int) -> torch.Tensor:
        """Trailing spacing: always includes t = T-1 (pure noise) and t = 0."""
        idx = torch.linspace(self.T - 1, 0, steps, device=self.device)
        return idx.round().long()

    @torch.no_grad()
    def ddim_sample(self, model, n: int, shape: tuple, steps: int = 50, eta: float = 0.0,
                    generator: Optional[torch.Generator] = None,
                    progress: bool = False, autocast_dtype: Optional[torch.dtype] = None,
                    x_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        C, H, W = shape
        ts = self.timestep_schedule(steps)
        if x_init is not None:
            x = x_init.clone()
        else:
            x = torch.randn(n, C, H, W, device=self.device, generator=generator)
        it = range(len(ts))
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="DDIM", leave=False, dynamic_ncols=True)
        for i in it:
            t = ts[i]
            tb = t.repeat(n)
            if autocast_dtype is not None:
                with torch.amp.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    v = model(x, tb).float()
            else:
                v = model(x, tb).float()
            x0 = self.x0_from_v(x, tb, v)
            eps = self.eps_from_v(x, tb, v)
            a_prev = self.acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            sigma = eta * torch.sqrt(
                ((1 - a_prev) / (1 - self.acp[t]).clamp(min=1e-12)) *
                (1 - self.acp[t] / a_prev.clamp(min=1e-12))
            ) if eta > 0 else torch.zeros((), device=self.device)
            dir_xt = torch.sqrt((1 - a_prev - sigma ** 2).clamp(min=0)) * eps
            x = torch.sqrt(a_prev) * x0 + dir_xt
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn(x.shape, device=x.device, generator=generator)
        return x


# ============================================================================
# 7. U-Net
# ============================================================================

def gn(ch: int, groups: int = 32) -> nn.GroupNorm:
    g = min(groups, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


def zero_module(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    """ADM-style residual block with FiLM (scale-shift) time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, out_ch * 2)
        self.norm2 = gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Multi-head self-attention via scaled_dot_product_attention (flash/mem-efficient)."""

    def __init__(self, ch: int, num_heads: int = 8):
        super().__init__()
        heads = num_heads
        while heads > 1 and ch % heads != 0:
            heads -= 1
        self.heads = heads
        self.norm = gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero_module(nn.Conv2d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)                          # (B, heads, d, HW)
        q, k, v = (z.transpose(-1, -2).contiguous() for z in (q, k, v))   # (B, heads, HW, d)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LatentUNet(nn.Module):
    def __init__(self, latent_ch: int = 4, base: int = 224, mults: Sequence[int] = (1, 2, 3),
                 num_res_blocks: int = 2, attn_levels: Sequence[int] = (1, 2),
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        t_dim = base * 4
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base),
            nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim),
        )
        chans = [base * m for m in mults]
        n_levels = len(chans)
        attn_levels = set(attn_levels)

        self.in_conv = nn.Conv2d(latent_ch, chans[0], 3, padding=1)

        self.down = nn.ModuleList()
        skip_chs = [chans[0]]
        ch = chans[0]
        for lvl, c in enumerate(chans):
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, c, t_dim, dropout)])
                ch = c
                if lvl in attn_levels:
                    block.append(AttnBlock(ch, num_heads))
                self.down.append(block)
                skip_chs.append(ch)
            if lvl != n_levels - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)

        self.mid1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = AttnBlock(ch, num_heads)
        self.mid2 = ResBlock(ch, ch, t_dim, dropout)

        self.up = nn.ModuleList()
        for lvl in reversed(range(n_levels)):
            c = chans[lvl]
            for j in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(ch + skip_chs.pop(), c, t_dim, dropout)])
                ch = c
                if lvl in attn_levels:
                    block.append(AttnBlock(ch, num_heads))
                if lvl != 0 and j == num_res_blocks:
                    block.append(Upsample(ch))
                self.up.append(block)

        self.out = nn.Sequential(gn(ch), nn.SiLU(),
                                 zero_module(nn.Conv2d(ch, latent_ch, 3, padding=1)))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        skips = [h]
        for block in self.down:
            for m in block:
                h = m(h, t_emb) if isinstance(m, ResBlock) else m(h)
            skips.append(h)
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)
        for block in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for m in block:
                h = m(h, t_emb) if isinstance(m, ResBlock) else m(h)
        return self.out(h)


def build_unet(cfg: Config) -> LatentUNet:
    return LatentUNet(
        latent_ch=cfg.latent_ch, base=cfg.base, mults=cfg.mults,
        num_res_blocks=cfg.num_res_blocks, attn_levels=cfg.attn_levels,
        num_heads=cfg.num_heads, dropout=cfg.dropout,
    )


# ============================================================================
# 8. EMA
# ============================================================================

class EMA:
    """fp32 shadow weights with a warmup ramp.

    d(s) = min(decay, (1 + s) / (ema_warmup + s))

    Without the ramp the shadow spends its first ~1/(1-decay) steps averaging
    the random initialisation, which is what made the baseline's early FID
    readings meaningless.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995, warmup: float = 10.0):
        self.decay = decay
        self.warmup = warmup
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone()
                        for k, v in model.state_dict().items() if not v.dtype.is_floating_point}

    def current_decay(self, step: int) -> float:
        return min(self.decay, (1.0 + step) / (self.warmup + step))

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = self.current_decay(step)
        msd = model.state_dict()
        for k, s in self.shadow.items():
            s.mul_(d).add_(msd[k].detach().float(), alpha=1.0 - d)
        for k in self.buffers:
            self.buffers[k] = msd[k].detach().clone()

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        sd = {k: v.clone() for k, v in self.shadow.items()}
        sd.update({k: v.clone() for k, v in self.buffers.items()})
        model.load_state_dict(sd, strict=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup": self.warmup,
                "shadow": self.shadow, "buffers": self.buffers}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd.get("decay", self.decay)
        self.warmup = sd.get("warmup", self.warmup)
        self.shadow = {k: v.clone().float() for k, v in sd["shadow"].items()}
        self.buffers = {k: v.clone() for k, v in sd.get("buffers", {}).items()}


# ============================================================================
# 9. FID
# ============================================================================

class FIDEvaluator:
    """FID against the custom botanical set only.

    Real Inception features are extracted exactly once and their sufficient
    statistics are snapshotted, so every subsequent evaluation only pays for
    generation. The baseline re-encoded the reals on every call and used just
    200 samples, which is why its in-training FID (140-290) bore no relation
    to the final 3000-sample number (78).
    """

    def __init__(self, device: torch.device, feature: int = 2048):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.device = device
        self.fid = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
        self.fid.set_dtype(torch.float64)
        self._snapshot = None
        self.n_real = 0

    def _real_state_keys(self) -> list:
        defaults = getattr(self.fid, "_defaults", None)
        if defaults:
            keys = [k for k in defaults if k.startswith("real_")]
            if keys:
                return keys
        return ["real_features_sum", "real_features_cov_sum", "real_features_num_samples"]

    @torch.no_grad()
    def fit_real(self, loader: Iterable[torch.Tensor], max_images: Optional[int] = None) -> None:
        from tqdm import tqdm
        self.fid.reset()
        seen = 0
        for imgs in tqdm(loader, desc="FID real features", dynamic_ncols=True):
            imgs = ((imgs.to(self.device) + 1) / 2).clamp(0, 1)
            self.fid.update(imgs, real=True)
            seen += imgs.shape[0]
            if max_images is not None and seen >= max_images:
                break
        self.n_real = seen
        self._snapshot = {k: getattr(self.fid, k).clone() for k in self._real_state_keys()}
        logging.info(f"FID reference statistics fitted on {seen} real images.")

    @torch.no_grad()
    def score(self, fake_batches: Iterable[torch.Tensor]) -> float:
        if self._snapshot is None:
            raise RuntimeError("call fit_real() before score()")
        self.fid.reset()
        for k, v in self._snapshot.items():
            setattr(self.fid, k, v.clone())
        n = 0
        for imgs in fake_batches:
            imgs = ((imgs.to(self.device) + 1) / 2).clamp(0, 1)
            self.fid.update(imgs, real=False)
            n += imgs.shape[0]
        val = float(self.fid.compute().item())
        self.fid.reset()
        return val


def generate_batches(model, diffusion: Diffusion, vae: FrozenVAE, cfg: Config,
                     n_total: int, batch: int, steps: int, mean: torch.Tensor,
                     std: torch.Tensor, autocast_dtype, seed: Optional[int] = None):
    """Yields decoded image batches in [-1, 1]. Generator, so FID never holds
    more than one batch of pixels at a time."""
    from tqdm import tqdm
    g = None
    if seed is not None:
        g = torch.Generator(device=diffusion.device).manual_seed(seed)
    shape = (cfg.latent_ch, cfg.latent_h, cfg.latent_w)
    made = 0
    pbar = tqdm(total=n_total, desc="sampling", dynamic_ncols=True)
    while made < n_total:
        b = min(batch, n_total - made)
        z = diffusion.ddim_sample(model, b, shape, steps=steps, generator=g,
                                  autocast_dtype=autocast_dtype)
        z_raw = z * std + mean
        yield vae.decode_chunked(z_raw, chunk=max(8, batch // 4))
        made += b
        pbar.update(b)
    pbar.close()


# ============================================================================
# 10. Checkpointing
# ============================================================================

def save_ckpt(path: Path, model, ema, opt, sched, step: int, cfg: Config,
              stats: dict, best_fid: float, phase: int, include_optim: bool = True) -> None:
    """`include_optim=False` for best.pt: at 100M params the AdamW moments are
    ~800 MB and best.pt is only ever used for evaluation or warm-starting a
    phase, both of which reset the optimiser anyway."""
    atomic_save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optim": opt.state_dict() if include_optim else None,
        "sched": sched.state_dict() if (sched is not None and include_optim) else None,
        "step": step,
        "phase": phase,
        "cfg": asdict(cfg),
        "arch_tag": cfg.arch_tag(),
        "latent_stats": stats,
        "best_fid": best_fid,
    }, path)


def load_ckpt(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def find_latest(out_dir: Path) -> Optional[Path]:
    cands = sorted(out_dir.glob("step_*.pt"))
    last = out_dir / "last.pt"
    if last.exists():
        return last
    return cands[-1] if cands else None


# ============================================================================
# 11. Training
# ============================================================================

def lr_lambda_factory(cfg: Config):
    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        prog = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * cos
    return fn


class IndexSampler:
    """Shuffled-without-replacement stream over an explicit index pool."""

    def __init__(self, pool: torch.Tensor, batch: int, device: torch.device, seed: int = 0):
        self.pool = pool.to(device)
        self.n, self.batch, self.device = self.pool.numel(), batch, device
        if self.n < batch:
            raise RuntimeError(f"batch {batch} exceeds available latents {self.n}")
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self._reshuffle()
        self.epochs = 0

    def _reshuffle(self):
        self.perm = self.pool[torch.randperm(self.n, generator=self.g).to(self.device)]
        self.cursor = 0

    def next(self) -> torch.Tensor:
        if self.cursor + self.batch > self.n:
            self._reshuffle()
            self.epochs += 1
        idx = self.perm[self.cursor:self.cursor + self.batch]
        self.cursor += self.batch
        return idx


def train(cfg: Config, phase: int, init_from: Optional[str], resume: bool) -> None:
    device = pick_device()
    out_dir = Path(cfg.out_dir) / f"phase{phase}"
    setup_logging(out_dir, "train")
    set_seed(cfg.seed + phase)
    logging.info("=" * 78)
    logging.info(f"PHASE {phase} TRAINING | device={device} | arch={cfg.arch_tag()}")
    logging.info(json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2))

    autocast_dtype = None if cfg.precision == "fp32" else dtype_of(cfg.precision)
    if device.type != "cuda":
        autocast_dtype = None

    # -- data ---------------------------------------------------------------
    vae = FrozenVAE(cfg.vae_id, device, cfg.vae_dtype)
    dirs = ([cfg.pretrain_dir, cfg.custom_dir] if phase == 1 else [cfg.custom_dir])
    cache = LatentCache.build(cfg, dirs, cfg.cache_tag(phase), vae, device)
    latents = cache.to_tensor(device, cfg.latents_device)
    mean, std = cache.stats

    # -- phase 2 inherits phase 1's latent normalisation -------------------
    init_ckpt = None
    if init_from:
        init_ckpt = load_ckpt(Path(init_from), device)
        if init_ckpt["arch_tag"] != cfg.arch_tag():
            raise RuntimeError(f"arch mismatch: ckpt={init_ckpt['arch_tag']} cfg={cfg.arch_tag()}")
        stats = init_ckpt["latent_stats"]
        mean = torch.tensor(stats["mean"]).view(1, -1, 1, 1)
        std = torch.tensor(stats["std"]).view(1, -1, 1, 1)
        logging.info(f"Inherited latent normalisation from {init_from} "
                     f"(do NOT recompute it between phases).")
    mean, std = mean.to(device), std.to(device)
    stats = {"mean": mean.flatten().tolist(), "std": std.flatten().tolist()}

    # -- train/val split ----------------------------------------------------
    # The cache stores originals in [0, n_img) and their horizontal flips in
    # [n_img, 2*n_img). A val split must hold out BOTH orientations of an
    # image, otherwise the flipped twin of every "held-out" latent is still in
    # the training pool and the val loss is optimistically biased.
    n_total = latents.shape[0]
    n_img = int(cache.meta["n_images"])
    flipped = bool(cache.meta["flip"])
    n_val_img = int(n_img * cfg.val_frac)
    if n_val_img > 0:
        v = torch.arange(n_img - n_val_img, n_img)
        val_idx = torch.cat([v, v + n_img]) if flipped else v
        keep = torch.ones(n_total, dtype=torch.bool)
        keep[val_idx] = False
        train_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        val_idx = val_idx.to(latents.device)
    else:
        val_idx = None
        train_idx = torch.arange(n_total)
    logging.info(f"latents: {n_total} total ({n_img} images x{2 if flipped else 1} orientation) "
                 f"| train {train_idx.numel()} | val {0 if val_idx is None else val_idx.numel()}")

    # -- model --------------------------------------------------------------
    model = build_unet(cfg).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    logging.info(f"U-Net parameters: {human(param_count(model))}")

    ema = EMA(model, cfg.ema_decay, cfg.ema_warmup)
    raw_model = model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas),
                            weight_decay=cfg.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg))
    scaler = torch.amp.GradScaler(device=device.type, enabled=(cfg.precision == "fp16"
                                                              and device.type == "cuda"))
    diffusion = Diffusion(cfg.timesteps, device)
    w_t = diffusion.loss_weights(cfg.loss_weight, cfg.min_snr_gamma).to(device)

    start_step, best_fid = 0, float("inf")
    if init_ckpt is not None:
        raw_model.load_state_dict(init_ckpt["model"])
        ema.load_state_dict(init_ckpt["ema"])
        logging.info(f"Warm-started from step {init_ckpt['step']} of phase {init_ckpt['phase']}; "
                     f"optimiser and schedule reset, lr={cfg.lr:g}")

    ckpt_path = find_latest(out_dir)
    if resume and ckpt_path is not None and init_ckpt is not None:
        logging.warning(f"Both --init-from and an existing {ckpt_path.name} were found; "
                        f"resuming takes precedence. Pass --no-resume to force a fresh "
                        f"warm start from the phase-1 weights.")
    if resume and ckpt_path is not None:
        ck = load_ckpt(ckpt_path, device)
        if ck["arch_tag"] != cfg.arch_tag():
            raise RuntimeError(f"arch mismatch on resume: {ck['arch_tag']} vs {cfg.arch_tag()}")
        raw_model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if ck.get("optim") is not None:
            opt.load_state_dict(ck["optim"])
        else:
            logging.warning("Checkpoint carries no optimiser state (best.pt); "
                            "AdamW moments restart from zero.")
        if ck.get("sched"):
            sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        best_fid = ck.get("best_fid", float("inf"))
        stats = ck["latent_stats"]
        mean = torch.tensor(stats["mean"], device=device).view(1, -1, 1, 1)
        std = torch.tensor(stats["std"], device=device).view(1, -1, 1, 1)
        logging.info(f"Resumed from {ckpt_path} at step {start_step} (best FID {best_fid:.3f})")

    # torch.compile wraps the module and prefixes state_dict keys with
    # "_orig_mod.". EMA and checkpointing must therefore always go through the
    # uncompiled handle, or the shadow update KeyErrors and the saved weights
    # cannot be loaded back into a plain LatentUNet.
    raw_model = model
    if cfg.compile:
        model = torch.compile(model)
        logging.info("torch.compile enabled (first steps will be slow while it traces).")

    # -- FID reference ------------------------------------------------------
    fid_eval = None
    if cfg.eval_every > 0:
        try:
            fid_eval = FIDEvaluator(device)
            # Fittiamo SOLO sullo Split A anche in fase di addestramento
            paths_A, _ = get_fid_splits(cfg.custom_dir, cfg.seed)
            real_ds = AspectImageDataset(paths_A, cfg.image_h, cfg.image_w, cfg.aspect_mode)
            real_loader = DataLoader(real_ds, batch_size=64, shuffle=False, num_workers=cfg.num_workers)
            fid_eval.fit_real(real_loader)
            
            # Aggiorniamo la config in modo che la generazione intermedia usi il numero corretto
            cfg.fid_fake = len(paths_A)
        except Exception as e:                                       # noqa: BLE001
            logging.warning(f"FID disabled ({type(e).__name__}: {e})")
            fid_eval = None

    eval_model = build_unet(cfg).to(device)

    # -- loop ---------------------------------------------------------------
    sampler = IndexSampler(train_idx, cfg.batch, latents.device, cfg.seed)
    t_start = time.time()
    running, running_n = 0.0, 0
    log_every = 200
    non_blocking = latents.device.type == "cpu"

    logging.info(f"Training {cfg.steps - start_step} steps at batch {cfg.batch} "
                 f"({cfg.steps * cfg.batch / max(1, train_idx.numel()):.1f} epochs over the pool)")

    model.train()
    for step in range(start_step, cfg.steps):
        idx = sampler.next()
        z = latents[idx]
        if non_blocking:
            z = z.to(device, non_blocking=True)
        z = ((z.float() - mean) / std)
        if cfg.channels_last and device.type == "cuda":
            z = z.contiguous(memory_format=torch.channels_last)

        t = diffusion.sample_t(z.shape[0], cfg.t_dist)
        noise = torch.randn_like(z)
        z_t = diffusion.q_sample(z, t, noise)
        v_tgt = diffusion.v_target(z, t, noise)

        opt.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                v_pred = model(z_t, t)
                loss = (F.mse_loss(v_pred.float(), v_tgt, reduction="none")
                        .mean(dim=(1, 2, 3)) * w_t[t]).mean()
        else:
            v_pred = model(z_t, t)
            loss = (F.mse_loss(v_pred, v_tgt, reduction="none")
                    .mean(dim=(1, 2, 3)) * w_t[t]).mean()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        ema.update(raw_model, step)

        running += loss.item()
        running_n += 1

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            done = step + 1 - start_step
            ips = done / max(elapsed, 1e-9)
            eta_h = (cfg.steps - step - 1) / max(ips, 1e-9) / 3600
            logging.info(
                f"step {step + 1}/{cfg.steps} | loss {running / running_n:.5f} "
                f"| lr {sched.get_last_lr()[0]:.2e} | gnorm {float(gnorm):.2f} "
                f"| ema_d {ema.current_decay(step):.5f} | {ips:.2f} it/s | ETA {eta_h:.1f} h"
            )
            running, running_n = 0.0, 0

        # -- periodic checkpoint --------------------------------------------
        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.steps:
            save_ckpt(out_dir / "last.pt", raw_model, ema, opt, sched, step + 1, cfg,
                      stats, best_fid, phase)

        # -- periodic evaluation --------------------------------------------
        if cfg.eval_every > 0 and ((step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.steps):
            ema.copy_to(eval_model)
            eval_model.eval()
            if val_idx is not None:
                vl = validate(eval_model, diffusion, latents, val_idx, mean, std, w_t,
                              cfg, device, autocast_dtype)
                logging.info(f"step {step + 1} | val v-loss {vl:.5f}")
            if fid_eval is not None:
                t_fid = time.time()
                score = fid_eval.score(generate_batches(
                    eval_model, diffusion, vae, cfg, cfg.fid_fake, cfg.sample_batch,
                    cfg.fid_steps, mean, std, autocast_dtype, seed=1234))
                logging.info(f"step {step + 1} | FID({cfg.fid_fake} fake vs "
                             f"{fid_eval.n_real} real) = {score:.3f} "
                             f"| {(time.time() - t_fid) / 60:.1f} min")
                if score < best_fid:
                    best_fid = score
                    save_ckpt(out_dir / "best.pt", raw_model, ema, opt, sched, step + 1, cfg,
                              stats, best_fid, phase, include_optim=False)
                    logging.info(f"new best FID {best_fid:.3f} -> best.pt")
            save_sample_grid(eval_model, diffusion, vae, cfg, mean, std, autocast_dtype,
                             out_dir / f"samples_step{step + 1:07d}.png", n=8, seed=7)
            model.train()
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # -- wall clock guard -----------------------------------------------
        if cfg.max_hours > 0 and (time.time() - t_start) > cfg.max_hours * 3600:
            logging.info(f"Wall-clock budget of {cfg.max_hours} h reached; stopping.")
            save_ckpt(out_dir / "last.pt", raw_model, ema, opt, sched, step + 1, cfg,
                      stats, best_fid, phase)
            break

    logging.info(f"Phase {phase} finished. best FID = {best_fid:.3f}")


@torch.no_grad()
def validate(model, diffusion: Diffusion, latents, val_idx, mean, std, w_t, cfg,
             device, autocast_dtype, max_batches: int = 40) -> float:
    """Deterministic v-loss on held-out latents. Cheap, low-variance, and it
    catches phase-2 overfitting long before a noisy FID reading does."""
    g = torch.Generator(device=device).manual_seed(0)
    total, count = 0.0, 0
    for i in range(0, min(len(val_idx), max_batches * cfg.batch), cfg.batch):
        sel = val_idx[i:i + cfg.batch]
        if sel.numel() == 0:
            break
        z = latents[sel].to(device).float()
        z = (z - mean) / std
        t = torch.randint(0, diffusion.T, (z.shape[0],), device=device, generator=g)
        noise = torch.randn(z.shape, device=device, generator=g)
        z_t = diffusion.q_sample(z, t, noise)
        v_tgt = diffusion.v_target(z, t, noise)
        if autocast_dtype is not None:
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                v = model(z_t, t)
        else:
            v = model(z_t, t)
        loss = (F.mse_loss(v.float(), v_tgt, reduction="none").mean(dim=(1, 2, 3)) * w_t[t]).mean()
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def save_sample_grid(model, diffusion, vae, cfg, mean, std, autocast_dtype,
                     path: Path, n: int = 8, seed: int = 7, steps: Optional[int] = None) -> None:
    import torchvision
    g = torch.Generator(device=diffusion.device).manual_seed(seed)
    z = diffusion.ddim_sample(model, n, (cfg.latent_ch, cfg.latent_h, cfg.latent_w),
                              steps=steps or cfg.fid_steps, generator=g,
                              autocast_dtype=autocast_dtype)
    imgs = vae.decode_chunked(z * std + mean, chunk=8)
    imgs = ((imgs.cpu() + 1) / 2).clamp(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(imgs, path, nrow=min(n, 8))
    logging.info(f"wrote {path}")


# ============================================================================
# 12. Stand-alone stages
# ============================================================================

def stage_cache(cfg: Config, phase: int, force: bool) -> None:
    device = pick_device()
    setup_logging(Path(cfg.out_dir), "cache")
    vae = FrozenVAE(cfg.vae_id, device, cfg.vae_dtype)
    dirs = ([cfg.pretrain_dir, cfg.custom_dir] if phase == 1 else [cfg.custom_dir])
    LatentCache.build(cfg, dirs, cfg.cache_tag(phase), vae, device, force=force)


def _load_for_eval(cfg: Config, ckpt_path: str, device: torch.device, use_ema: bool = True):
    ck = load_ckpt(Path(ckpt_path), device)
    saved = Config(**{k: v for k, v in ck["cfg"].items() if k in Config.__annotations__})
    saved.mults = tuple(saved.mults)
    saved.attn_levels = tuple(saved.attn_levels)
    model = build_unet(saved).to(device)
    if use_ema:
        ema = EMA(model, saved.ema_decay, saved.ema_warmup)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(model)
    else:
        model.load_state_dict(ck["model"])
    model.eval()
    st = ck["latent_stats"]
    mean = torch.tensor(st["mean"], device=device).view(1, -1, 1, 1)
    std = torch.tensor(st["std"], device=device).view(1, -1, 1, 1)
    logging.info(f"Loaded {ckpt_path} (step {ck['step']}, phase {ck['phase']}, "
                 f"{'EMA' if use_ema else 'raw'} weights, best FID {ck.get('best_fid', float('nan')):.3f})")
    return model, saved, mean, std


def stage_fid(cfg: Config, ckpt: str, n_fake: int, steps: int, use_ema: bool) -> None:
    device = pick_device()
    setup_logging(Path(cfg.out_dir), "eval")
    model, saved, mean, std = _load_for_eval(cfg, ckpt, device, use_ema)
    saved.custom_dir = cfg.custom_dir
    saved.sample_batch = cfg.sample_batch
    vae = FrozenVAE(saved.vae_id, device, saved.vae_dtype)
    diffusion = Diffusion(saved.timesteps, device)
    autocast_dtype = None if saved.precision == "fp32" or device.type != "cuda" \
        else dtype_of(saved.precision)

    fid_eval = FIDEvaluator(device)
    
    # Fittiamo le feature sull'INTERO dataset custom
    paths_all = list_images(Path(saved.custom_dir))
    real_ds = AspectImageDataset(paths_all, saved.image_h, saved.image_w, saved.aspect_mode)
    fid_eval.fit_real(DataLoader(real_ds, batch_size=64, num_workers=cfg.num_workers))
    
    # Forziamo il numero di immagini generate ad essere identico all'intero dataset
    actual_n_fake = len(paths_all)
    
    score = fid_eval.score(generate_batches(model, diffusion, vae, saved, actual_n_fake,
                                            saved.sample_batch, steps, mean, std,
                                            autocast_dtype, seed=1234))
    logging.info(f"FID({actual_n_fake} fake vs {fid_eval.n_real} real, {steps} DDIM steps) = {score:.4f}")
    print(f"\nFID = {score:.4f}  [{actual_n_fake} generated vs {fid_eval.n_real} real botanical images]")


def stage_sample(cfg: Config, ckpt: str, n: int, steps: int, seed: int, use_ema: bool) -> None:
    device = pick_device()
    out_dir = Path(cfg.out_dir) / "samples"
    setup_logging(out_dir, "sample")
    model, saved, mean, std = _load_for_eval(cfg, ckpt, device, use_ema)
    vae = FrozenVAE(saved.vae_id, device, saved.vae_dtype)
    diffusion = Diffusion(saved.timesteps, device)
    autocast_dtype = None if saved.precision == "fp32" or device.type != "cuda" \
        else dtype_of(saved.precision)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    save_sample_grid(model, diffusion, vae, saved, mean, std, autocast_dtype,
                     out_dir / f"grid_{stamp}.png", n=n, seed=seed, steps=steps)


def stage_vae_floor(cfg: Config, n: int) -> None:
    """Reconstruction FID of the frozen VAE and Intrinsic Dataset FID on X/2 splits."""
    device = pick_device()
    setup_logging(Path(cfg.out_dir), "eval")
    vae = FrozenVAE(cfg.vae_id, device, cfg.vae_dtype)
    
    # 1. Divisione deterministica in Split A (Ground Truth) e Split B (Target)
    paths_A, paths_B = get_fid_splits(cfg.custom_dir, cfg.seed)
    n_split = len(paths_A)
    
    ds_A = AspectImageDataset(paths_A, cfg.image_h, cfg.image_w, cfg.aspect_mode)
    ds_B = AspectImageDataset(paths_B, cfg.image_h, cfg.image_w, cfg.aspect_mode)
    
    loader_A = DataLoader(ds_A, batch_size=32, shuffle=False, num_workers=cfg.num_workers)
    loader_B = DataLoader(ds_B, batch_size=32, shuffle=False, num_workers=cfg.num_workers)
    
    fid_eval = FIDEvaluator(device)
    fid_eval.fit_real(loader_A) # Fittiamo le feature SOLO sullo Split A
    
    # 2. Calcolo FID Intrinseca (Reale Split B vs Reale Split A)
    def real_b():
        for imgs in loader_B:
            yield imgs.to(device)
            
    score_real = fid_eval.score(real_b())
    logging.info(f"Intrinsic Dataset FID (Split B vs Split A) = {score_real:.4f}")
    
    # 3. Calcolo FID di Ricostruzione (VAE decode(encode(Split B)) vs Reale Split A)
    def recon_b():
        for imgs in loader_B:
            yield vae.decode(vae.encode(imgs.to(device)))

    score_vae = fid_eval.score(recon_b())
    logging.info(f"VAE reconstruction FID ({n_split} images) = {score_vae:.4f}")
    
    print(f"\nFID Intrinseca (Reale vs Reale) = {score_real:.4f} (Il vero asintoto a cui puntare)")
    print(f"VAE reconstruction FID = {score_vae:.4f} (baseline custom VAE era 22.04)")


# ============================================================================
# 13. Doctor
# ============================================================================

def stage_doctor(cfg: Config) -> None:
    device = pick_device()
    setup_logging(Path(cfg.out_dir), "doctor")
    ok = True
    print("=" * 78)
    print(f"torch {torch.__version__} | device {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU {p.name} | {p.total_memory / 1e9:.1f} GB | bf16 "
              f"{'yes' if torch.cuda.is_bf16_supported() else 'no'}")

    # --- 1. schedule -------------------------------------------------------
    print("\n[1] noise schedule")
    d = Diffusion(cfg.timesteps, device)
    print(f"    alpha_bar[0]={d.acp[0]:.6f}  alpha_bar[T-1]={d.acp[-1]:.6e}")
    print(f"    SNR range: {d.snr[-1]:.3e} .. {d.snr[0]:.3e}")
    if float(d.acp[-1]) != 0.0:
        print("    FAIL: terminal SNR is not zero"); ok = False
    else:
        print("    OK: zero terminal SNR")

    # --- 2. parameterisation round trip ------------------------------------
    print("\n[2] v-parameterisation consistency")
    x0 = torch.randn(8, cfg.latent_ch, cfg.latent_h, cfg.latent_w, device=device)
    t = torch.randint(0, cfg.timesteps, (8,), device=device)
    eps = torch.randn_like(x0)
    xt = d.q_sample(x0, t, eps)
    v = d.v_target(x0, t, eps)
    e0 = (d.x0_from_v(xt, t, v) - x0).abs().max().item()
    e1 = (d.eps_from_v(xt, t, v) - eps).abs().max().item()
    print(f"    max|x0_hat - x0| = {e0:.2e}   max|eps_hat - eps| = {e1:.2e}")
    if max(e0, e1) > 1e-3:
        print("    FAIL"); ok = False
    else:
        print("    OK")

    # --- 3. sampler against an oracle denoiser -----------------------------
    print("\n[3] DDIM sampler vs. oracle denoiser")
    target = torch.randn(4, cfg.latent_ch, cfg.latent_h, cfg.latent_w, device=device)

    def oracle(x, tb):
        tb = tb.long()
        a = d.sqrt_acp[tb].view(-1, 1, 1, 1)
        b = d.sqrt_1macp[tb].view(-1, 1, 1, 1)
        e = (x - a * target) / b.clamp(min=1e-8)
        return a * e - b * target

    rec = d.ddim_sample(oracle, 4, (cfg.latent_ch, cfg.latent_h, cfg.latent_w), steps=50)
    err = (rec - target).abs().max().item()
    print(f"    max|sample - target| = {err:.2e}  (50 DDIM steps)")
    if err > 1e-2:
        print("    FAIL: sampler/schedule are inconsistent"); ok = False
    else:
        print("    OK")

    # --- 4. model size and throughput --------------------------------------
    print("\n[4] model + throughput")
    model = build_unet(cfg).to(device)
    print(f"    parameters: {human(param_count(model))}")
    if device.type == "cuda":
        for cl in ([False, True] if cfg.channels_last else [False]):
            m = model.to(memory_format=torch.channels_last if cl else torch.contiguous_format)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            ad = None if cfg.precision == "fp32" else dtype_of(cfg.precision)
            z = torch.randn(cfg.batch, cfg.latent_ch, cfg.latent_h, cfg.latent_w, device=device)
            if cl:
                z = z.contiguous(memory_format=torch.channels_last)
            tt = torch.randint(0, cfg.timesteps, (cfg.batch,), device=device)
            torch.cuda.reset_peak_memory_stats()
            for i in range(12):
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                opt.zero_grad(set_to_none=True)
                if ad is not None:
                    with torch.amp.autocast(device_type="cuda", dtype=ad):
                        loss = F.mse_loss(m(z, tt).float(), torch.zeros_like(z))
                else:
                    loss = F.mse_loss(m(z, tt), torch.zeros_like(z))
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 10
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"    channels_last={cl}: {1 / dt:.2f} it/s @ batch {cfg.batch} "
                  f"({cfg.batch / dt:.0f} img/s) | peak {peak:.2f} GB")
            print(f"      -> {cfg.steps} steps = {cfg.steps * dt / 3600:.1f} h")
            del opt
            torch.cuda.empty_cache()
    else:
        print("    (throughput benchmark needs CUDA)")

    # --- 5. overfit a single batch -----------------------------------------
    print("\n[5] overfit one batch (loss must collapse)")
    model = build_unet(cfg).to(device)
    model.eval()  # Disabilita il dropout per il test di memorizzazione
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)  # LR leggermente più alto per il test
    z = torch.randn(8, cfg.latent_ch, cfg.latent_h, cfg.latent_w, device=device)
    tt = torch.full((8,), cfg.timesteps // 2, device=device, dtype=torch.long)
    eps = torch.randn_like(z)
    zt, vt = d.q_sample(z, tt, eps), d.v_target(z, tt, eps)
    first = last = None
    for i in range(800):  # Aumentato a 800 step per superare l'inizializzazione a zero
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(zt, tt), vt)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    ratio = last / max(first, 1e-12)
    print(f"    loss {first:.4f} -> {last:.4f}  (ratio {ratio:.3f})")
    if ratio < 0.25:
        print("    OK")
    elif ratio < 0.60:
        print("    WARN: converging, but slowly.")
    else:
        print("    FAIL: model is not learning"); ok = False

    # --- 6. caches ---------------------------------------------------------
    print("\n[6] latent caches")
    for ph in (1, 2):
        j = Path(cfg.cache_dir) / f"{cfg.cache_tag(ph)}.json"
        if j.exists():
            m = json.loads(j.read_text())
            print(f"    phase {ph}: {m['shape']} | mean {np.round(m['mean'], 3).tolist()} "
                  f"| std {np.round(m['std'], 3).tolist()}")
        else:
            print(f"    phase {ph}: not built ({j.name})")

    print("\n" + "=" * 78)
    print("DOCTOR: " + ("all checks passed" if ok else "FAILURES ABOVE"))


# ============================================================================
# 14. CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    def common(sp):
        sp.add_argument("--pretrain-dir", default=d.pretrain_dir)
        sp.add_argument("--custom-dir", default=d.custom_dir)
        sp.add_argument("--cache-dir", default=d.cache_dir)
        sp.add_argument("--out-dir", default=d.out_dir)
        sp.add_argument("--image-h", type=int, default=d.image_h)
        sp.add_argument("--image-w", type=int, default=d.image_w)
        sp.add_argument("--aspect-mode", default=d.aspect_mode, choices=["center_crop", "squash"])
        sp.add_argument("--no-flip-cache", action="store_true")
        sp.add_argument("--vae-id", default=d.vae_id)
        sp.add_argument("--vae-dtype", default=d.vae_dtype, choices=["fp16", "bf16", "fp32"])
        sp.add_argument("--vae-batch", type=int, default=d.vae_batch)
        sp.add_argument("--num-workers", type=int, default=d.num_workers)
        sp.add_argument("--seed", type=int, default=d.seed)

    def arch(sp):
        sp.add_argument("--base", type=int, default=d.base)
        sp.add_argument("--mults", type=str, default=",".join(map(str, d.mults)))
        sp.add_argument("--num-res-blocks", type=int, default=d.num_res_blocks)
        sp.add_argument("--attn-levels", type=str, default=",".join(map(str, d.attn_levels)))
        sp.add_argument("--num-heads", type=int, default=d.num_heads)
        sp.add_argument("--dropout", type=float, default=d.dropout)
        sp.add_argument("--timesteps", type=int, default=d.timesteps)
        sp.add_argument("--precision", default=d.precision, choices=["bf16", "fp16", "fp32"])
        sp.add_argument("--batch", type=int, default=d.batch)
        sp.add_argument("--sample-batch", type=int, default=d.sample_batch)

    sp = sub.add_parser("cache", help="encode images into a latent cache")
    common(sp); sp.add_argument("--phase", type=int, required=True, choices=[1, 2])
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("train", help="train phase 1 or 2")
    common(sp); arch(sp)
    sp.add_argument("--phase", type=int, required=True, choices=[1, 2])
    sp.add_argument("--init-from", default=None, help="phase-1 checkpoint to warm-start from")
    sp.add_argument("--no-resume", action="store_true")
    sp.add_argument("--steps", type=int, default=None)
    sp.add_argument("--lr", type=float, default=None)
    sp.add_argument("--ema-decay", type=float, default=None)
    sp.add_argument("--ema-warmup", type=float, default=d.ema_warmup)
    sp.add_argument("--weight-decay", type=float, default=d.weight_decay)
    sp.add_argument("--grad-clip", type=float, default=d.grad_clip)
    sp.add_argument("--warmup-steps", type=int, default=d.warmup_steps)
    sp.add_argument("--lr-min-ratio", type=float, default=None)
    sp.add_argument("--loss-weight", default=d.loss_weight, choices=["uniform", "min_snr"])
    sp.add_argument("--min-snr-gamma", type=float, default=d.min_snr_gamma)
    sp.add_argument("--t-dist", default=d.t_dist, choices=["uniform", "logit_normal"])
    sp.add_argument("--eval-every", type=int, default=None)
    sp.add_argument("--save-every", type=int, default=d.save_every)
    sp.add_argument("--fid-fake", type=int, default=d.fid_fake)
    sp.add_argument("--fid-steps", type=int, default=d.fid_steps)
    sp.add_argument("--val-frac", type=float, default=None)
    sp.add_argument("--max-hours", type=float, default=d.max_hours)
    sp.add_argument("--no-channels-last", action="store_true")
    sp.add_argument("--compile", action="store_true")
    sp.add_argument("--latents-device", default=d.latents_device, choices=["auto", "cuda", "cpu"])

    sp = sub.add_parser("fid", help="evaluate a checkpoint")
    common(sp); arch(sp)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--n-fake", type=int, default=10000)
    sp.add_argument("--steps", type=int, default=50)
    sp.add_argument("--raw-weights", action="store_true", help="use non-EMA weights")

    sp = sub.add_parser("sample", help="write a sample grid")
    common(sp); arch(sp)
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--n", type=int, default=16)
    sp.add_argument("--steps", type=int, default=100)
    sp.add_argument("--raw-weights", action="store_true")

    sp = sub.add_parser("vae-floor", help="reconstruction FID of the frozen VAE")
    common(sp); sp.add_argument("--n", type=int, default=5000)

    sp = sub.add_parser("doctor", help="hardware / schedule / sampler diagnostics")
    common(sp); arch(sp)
    sp.add_argument("--steps", type=int, default=d.steps)

    return p


def cfg_from_args(a: argparse.Namespace) -> Config:
    cfg = Config()
    for k in ("pretrain_dir", "custom_dir", "cache_dir", "out_dir", "image_h", "image_w",
              "aspect_mode", "vae_id", "vae_dtype", "vae_batch", "num_workers", "seed",
              "base", "num_res_blocks", "num_heads", "dropout", "timesteps", "precision",
              "batch", "sample_batch", "weight_decay", "grad_clip", "warmup_steps",
              "loss_weight", "min_snr_gamma", "t_dist", "save_every", "fid_fake",
              "fid_steps", "max_hours", "ema_warmup", "latents_device"):
        if hasattr(a, k) and getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if hasattr(a, "mults"):
        cfg.mults = _tuple_of_int(a.mults)
    if hasattr(a, "attn_levels"):
        cfg.attn_levels = _tuple_of_int(a.attn_levels)
    if getattr(a, "no_flip_cache", False):
        cfg.flip_cache = False
    if getattr(a, "no_channels_last", False):
        cfg.channels_last = False
    if getattr(a, "compile", False):
        cfg.compile = True
    if getattr(a, "stage", None) == "doctor" and getattr(a, "steps", None):
        cfg.steps = a.steps

    # phase-dependent defaults ------------------------------------------------
    phase = getattr(a, "phase", 1)
    if getattr(a, "stage", None) == "train":
        cfg.steps = a.steps if a.steps is not None else (300_000 if phase == 1 else 100_000)
        cfg.lr = a.lr if a.lr is not None else (1.5e-4 if phase == 1 else 0.75e-4)
        cfg.ema_decay = a.ema_decay if a.ema_decay is not None else (0.9995 if phase == 1 else 0.999)
        cfg.eval_every = a.eval_every if a.eval_every is not None else (10_000 if phase == 1 else 5_000)
        cfg.lr_min_ratio = a.lr_min_ratio if a.lr_min_ratio is not None else (0.10 if phase == 1 else 0.02)
        cfg.val_frac = a.val_frac if a.val_frac is not None else (0.0 if phase == 1 else 0.02)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)

    if args.stage == "cache":
        stage_cache(cfg, args.phase, args.force)
    elif args.stage == "train":
        train(cfg, args.phase, args.init_from, resume=not args.no_resume)
    elif args.stage == "fid":
        stage_fid(cfg, args.ckpt, args.n_fake, args.steps, use_ema=not args.raw_weights)
    elif args.stage == "sample":
        stage_sample(cfg, args.ckpt, args.n, args.steps, cfg.seed, use_ema=not args.raw_weights)
    elif args.stage == "vae-floor":
        stage_vae_floor(cfg, args.n)
    elif args.stage == "doctor":
        stage_doctor(cfg)


if __name__ == "__main__":
    main()