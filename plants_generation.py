from __future__ import annotations
import argparse
import contextlib
import dataclasses
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from scipy import linalg as _scipy_linalg
    from scipy.optimize import linear_sum_assignment as _lsa
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    _scipy_linalg = None
    _lsa = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it=None, **kw):
        return it if it is not None else iter(())

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke", "sweet william",
    "carnation", "garden phlox", "love in the mist", "mexican aster",
    "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort",
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily",
    "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
    "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
    "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily",
]

CAPTION_TEMPLATES = [
    "a photo of a {}, a type of flower",
    "a close-up photograph of a {} flower",
    "a {} in bloom",
    "a detailed macro photo of a {}",
    "a beautiful {} flower in natural light",
]


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


class Log:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.t0 = time.time()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str = "", level: str = "INFO") -> None:
        el = time.time() - self.t0
        line = f"[{el/3600:6.2f}h][{level}] {msg}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(line + "\n")


LOG = Log()

def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

@contextlib.contextmanager
def timer(name: str, log: Callable[[str], None] = LOG):
    t = time.time()
    yield
    log(f"{name}: {time.time() - t:.2f}s")


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vram_report(prefix: str = "") -> str:
    if not torch.cuda.is_available():
        return f"{prefix}(cpu)"
    a = torch.cuda.max_memory_allocated() / 2 ** 30
    r = torch.cuda.max_memory_reserved() / 2 ** 30
    return f"{prefix}peak alloc {a:.2f} GiB / reserved {r:.2f} GiB"


@dataclass
class ModelCfg:
    dim: int = 768
    depth: int = 16
    heads: int = 12
    patch_size: int = 2
    in_channels: int = 4
    ctx_dim: int = 768
    pooled_dim: int = 768
    mlp_ratio: float = 4.0
    n_registers: int = 4
    qk_norm: bool = True
    drop_path: float = 0.05
    aug_dim: int = 4
    rope_theta: float = 10000.0
    repa_layer: int = 6
    repa_dim: int = 768
    repa_proj_hidden: int = 2048

    @staticmethod
    def preset(name: str) -> "ModelCfg":
        if name == "small":
            return ModelCfg(dim=512, depth=12, heads=8)
        if name == "base":
            return ModelCfg(dim=768, depth=16, heads=12)
        if name == "large":
            return ModelCfg(dim=1024, depth=20, heads=16, drop_path=0.10)
        raise ValueError(f"unknown preset {name}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.to(dt)
        if self.weight is not None:
            x = x * self.weight.to(dt)
        return x


def build_rope_2d(head_dim: int, h: int, w: int, theta: float,
                  device, n_prefix: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial 2D RoPE"
    half = head_dim // 2
    n_freq = half // 2
    freqs = 1.0 / (theta ** (torch.arange(0, n_freq, device=device, dtype=torch.float32) / n_freq))
    ys = torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2
    xs = torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2
    ay = torch.outer(ys, freqs)[:, None, :].expand(h, w, n_freq)
    ax = torch.outer(xs, freqs)[None, :, :].expand(h, w, n_freq)
    ang = torch.cat([ay, ax], dim=-1).reshape(h * w, half)
    if n_prefix > 0:
        ang = torch.cat([torch.zeros(n_prefix, half, device=device), ang], dim=0)
    return ang.cos()[None, None], ang.sin()[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dt = x.dtype
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xf[..., 0], xf[..., 1]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.flatten(-2).to(dt)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()

    def forward(self, x, cos, sin):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, ctx_dim: int, heads: int, qk_norm: bool = True):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.dh) if qk_norm else nn.Identity()

    def forward(self, x, ctx, ctx_mask=None):
        B, N, C = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, self.dh).transpose(1, 2)
        kv = self.kv(ctx).view(B, L, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask[:, None, None, :].to(torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, ctx_dim, mlp_ratio=4.0, drop_path=0.0, qk_norm=True):
        super().__init__()
        self.norm1 = RMSNorm(dim, affine=False)
        self.attn = SelfAttention(dim, heads, qk_norm)
        self.norm2 = RMSNorm(dim, affine=False)
        self.cross = CrossAttention(dim, ctx_dim, heads, qk_norm)
        self.norm3 = RMSNorm(dim, affine=False)
        self.mlp = SwiGLU(dim, mlp_ratio)
        self.scale_shift = nn.Parameter(torch.zeros(1, 6, dim))
        self.dp = DropPath(drop_path)
        nn.init.zeros_(self.cross.proj.weight)
        nn.init.zeros_(self.cross.proj.bias)

    def forward(self, x, mod, ctx, ctx_mask, cos, sin):
        m = self.scale_shift + mod                       # (B, 6, dim)
        sh1, sc1, g1, sh2, sc2, g2 = m.unbind(1)
        x = x + self.dp(g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, sc1), cos, sin))
        x = x + self.dp(self.cross(self.norm2(x), ctx, ctx_mask))
        x = x + self.dp(g2.unsqueeze(1) * self.mlp(modulate(self.norm3(x), sh2, sc2)))
        return x

def fourier_time_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().reshape(-1, 1) * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AugEmbed(nn.Module):
    def __init__(self, aug_dim: int, out_dim: int, n_freq: int = 8):
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer(
            "freqs", (2.0 ** torch.arange(n_freq, dtype=torch.float32)) * math.pi, persistent=False
        )
        self.proj = nn.Sequential(
            nn.Linear(aug_dim * 2 * n_freq, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        ang = a.float()[..., None] * self.freqs[None, None]     # (B, A, F)
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
        return self.proj(feat)

class FlowDiT(nn.Module):
    def __init__(self, cfg: ModelCfg, latent_size: int = 32):
        super().__init__()
        self.cfg = cfg
        self.latent_size = latent_size
        self.grid = latent_size // cfg.patch_size
        d = cfg.dim

        self.x_embed = nn.Conv2d(cfg.in_channels, d, cfg.patch_size, cfg.patch_size)
        self.registers = nn.Parameter(torch.randn(1, cfg.n_registers, d) * 0.02)

        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.pooled_proj = nn.Sequential(nn.Linear(cfg.pooled_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.aug_embed = AugEmbed(cfg.aug_dim, d)

        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.mod[-1].weight); nn.init.zeros_(self.mod[-1].bias)
        nn.init.zeros_(self.final_mod[-1].weight); nn.init.zeros_(self.final_mod[-1].bias)

        dpr = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList([
            DiTBlock(d, cfg.heads, cfg.ctx_dim, cfg.mlp_ratio, dpr[i], cfg.qk_norm)
            for i in range(cfg.depth)
        ])
        self.norm_out = RMSNorm(d, affine=False)
        self.out = nn.Linear(d, cfg.patch_size ** 2 * cfg.in_channels)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

        self.repa_proj = nn.Sequential(
            nn.Linear(d, cfg.repa_proj_hidden), nn.SiLU(),
            nn.Linear(cfg.repa_proj_hidden, cfg.repa_proj_hidden), nn.SiLU(),
            nn.Linear(cfg.repa_proj_hidden, cfg.repa_dim),
        )

        self.logvar = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.logvar[-1].weight); nn.init.zeros_(self.logvar[-1].bias)

        self.grad_ckpt_every = 0
        self.apply(self._init_weights)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        for b in self.blocks:
            nn.init.zeros_(b.cross.proj.weight); nn.init.zeros_(b.cross.proj.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            m.weight.data = w.view_as(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, c, g = self.cfg.patch_size, self.cfg.in_channels, self.grid
        B = x.shape[0]
        x = x.reshape(B, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, c, g * p, g * p)

    def rope(self, device):
        key = (device, self.grid)
        if getattr(self, "_rope_key", None) != key:
            cos, sin = build_rope_2d(
                self.cfg.dim // self.cfg.heads, self.grid, self.grid,
                self.cfg.rope_theta, device, n_prefix=self.cfg.n_registers,
            )
            self._rope_cache = (cos, sin)
            self._rope_key = key
        return self._rope_cache

    def forward(self, x, t, ctx, ctx_mask=None, pooled=None, aug=None, return_repa=False):
        B = x.shape[0]
        R = self.cfg.n_registers
        h = self.x_embed(x).flatten(2).transpose(1, 2)                 # (B, N, d)
        h = torch.cat([self.registers.expand(B, -1, -1).to(h.dtype), h], dim=1)

        c = self.t_mlp(fourier_time_embedding(t, 256).to(h.dtype))
        if pooled is not None:
            c = c + self.pooled_proj(pooled.to(h.dtype))
        if aug is not None:
            c = c + self.aug_embed(aug).to(h.dtype)
        mod = self.mod(c).view(B, 6, -1)

        cos, sin = self.rope(x.device)
        repa_h = None
        for i, blk in enumerate(self.blocks):
            if self.grad_ckpt_every and self.training and (i % self.grad_ckpt_every == 0):
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, mod, ctx, ctx_mask, cos, sin, use_reentrant=False
                )
            else:
                h = blk(h, mod, ctx, ctx_mask, cos, sin)
            if return_repa and (i + 1) == self.cfg.repa_layer:
                repa_h = h[:, R:]

        fsh, fsc = self.final_mod(c).view(B, 2, -1).unbind(1)
        h = modulate(self.norm_out(h[:, R:]), fsh, fsc)
        v = self.unpatchify(self.out(h))

        if return_repa:
            z = self.repa_proj(repa_h.float()) if repa_h is not None else None
            return v, z
        return v

    def loss_logvar(self, t: torch.Tensor) -> torch.Tensor:
        return self.logvar(fourier_time_embedding(t, 128)).squeeze(-1)

def sample_timesteps(n: int, device, mode: str = "logitnormal",
                     m: float = 0.0, s: float = 1.0, shift: float = 1.0) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand(n, device=device)
    elif mode == "logitnormal":
        t = torch.sigmoid(torch.randn(n, device=device) * s + m)
    elif mode == "cosmap":
        u = torch.rand(n, device=device)
        t = 1.0 - 1.0 / (torch.tan(math.pi / 2.0 * u) ** 2 + 1.0)
    else:
        raise ValueError(mode)
    if shift != 1.0:
        t = shift * t / (1.0 + (shift - 1.0) * t)
    return t.clamp(1e-4, 1.0 - 1e-4)


def ot_couple(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    if not _HAVE_SCIPY or x0.shape[0] < 2:
        return eps
    with torch.no_grad():
        a = x0.flatten(1).float()
        b = eps.flatten(1).float()
        cost = torch.cdist(a, b, p=2) ** 2
        r, c = _lsa(cost.cpu().numpy())
    return eps[torch.as_tensor(c, device=eps.device, dtype=torch.long)]


def flow_loss(model: FlowDiT, z0, ctx, ctx_mask, pooled, aug,
              t_mode="logitnormal", t_m=0.0, t_s=1.0, t_shift=1.0,
              use_ot=True, repa_feat=None, repa_weight=0.5,
              uncertainty_weight=True):
    B = z0.shape[0]
    eps = torch.randn_like(z0)
    if use_ot:
        eps = ot_couple(z0, eps)
    t = sample_timesteps(B, z0.device, t_mode, t_m, t_s, t_shift)
    tb = t.view(B, 1, 1, 1)
    zt = (1.0 - tb) * z0 + tb * eps
    target = eps - z0

    need_repa = repa_feat is not None
    out = model(zt, t, ctx, ctx_mask, pooled, aug, return_repa=need_repa)
    v, zproj = out if need_repa else (out, None)
    mse = (v.float() - target.float()).pow(2).mean(dim=(1, 2, 3))

    if uncertainty_weight:
        lv = model.loss_logvar(t).float()
        loss = (mse / lv.exp() + lv).mean()
    else:
        lv = torch.zeros_like(mse)
        loss = mse.mean()

    logs = {"mse": mse.mean().detach(), "logvar": lv.mean().detach()}

    if repa_feat is not None and zproj is not None:
        a = F.normalize(zproj.float(), dim=-1)
        b = F.normalize(repa_feat.float(), dim=-1)
        repa = -(a * b).sum(-1).mean()
        loss = loss + repa_weight * repa
        logs["repa_cos"] = (-repa).detach()

    logs["loss"] = loss.detach()
    return loss, logs

def exp_to_std(exp: float) -> float:
    exp = np.float64(exp)
    return float(np.sqrt((exp + 1) / ((exp + 2) ** 2 * (exp + 3))))


def std_to_exp(std: float) -> float:
    t = np.float64(std) ** -2
    roots = np.roots([1, 7, 16 - t, 12 - t])
    return float(np.real(roots[np.isreal(roots)]).max())


def _p_dot_p(t_a, gamma_a, t_b, gamma_b):
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den


def solve_posthoc_coefficients(in_t, in_gamma, out_t, out_gamma):
    rv = lambda x: np.float64(x).reshape(-1, 1)
    cv = lambda x: np.float64(x).reshape(1, -1)
    A = _p_dot_p(rv(in_t), rv(in_gamma), cv(in_t), cv(in_gamma))
    B = _p_dot_p(rv(in_t), rv(in_gamma), cv(out_t), cv(out_gamma))
    X = np.linalg.solve(A + np.eye(A.shape[0]) * 1e-12, B)
    return X / np.sum(X, axis=0, keepdims=True)


class PostHocEMA:
    def __init__(self, model: nn.Module, sigma_rels=(0.05, 0.10), device=None):
        self.sigma_rels = list(sigma_rels)
        self.gammas = [std_to_exp(s) for s in self.sigma_rels]
        self.keys = [k for k, v in model.state_dict().items() if v.is_floating_point()]
        self.emas = [
            {k: model.state_dict()[k].detach().clone().float() for k in self.keys}
            for _ in self.sigma_rels
        ]

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        t = max(int(step), 1)
        sd = model.state_dict()
        for gamma, ema in zip(self.gammas, self.emas):
            beta = (1.0 - 1.0 / t) ** (gamma + 1.0)
            w = 1.0 - beta
            if w <= 0:
                continue
            for k in self.keys:
                ema[k].lerp_(sd[k].detach().float(), w)

    def state_dict(self):
        return {"sigma_rels": self.sigma_rels,
                "emas": [{k: v.half().cpu() for k, v in e.items()} for e in self.emas]}

    def load_state_dict(self, sd, device):
        self.sigma_rels = sd["sigma_rels"]
        self.gammas = [std_to_exp(s) for s in self.sigma_rels]
        self.emas = [{k: v.float().to(device) for k, v in e.items()} for e in sd["emas"]]
        self.keys = list(self.emas[0].keys())

    def snapshot(self, path: Path, step: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": int(step), "sigma_rels": self.sigma_rels,
                    "emas": [{k: v.half().cpu() for k, v in e.items()} for e in self.emas]}, path)

    @staticmethod
    def reconstruct(snapshot_paths: Sequence[Path], target_sigma_rel: float,
                    target_step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        metas = []
        for p in snapshot_paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            for j, sr in enumerate(d["sigma_rels"]):
                metas.append((int(d["step"]), std_to_exp(float(sr)), p, j))
        if not metas:
            raise RuntimeError("no post-hoc EMA snapshots found")
        in_t = np.array([m[0] for m in metas], dtype=np.float64)
        in_g = np.array([m[1] for m in metas], dtype=np.float64)
        out_t = float(target_step or in_t.max())
        out_g = std_to_exp(target_sigma_rel)
        coef = solve_posthoc_coefficients(in_t, in_g, out_t, out_g).reshape(-1)

        out: Dict[str, torch.Tensor] = {}
        by_file: Dict[Path, List[Tuple[int, float]]] = {}
        for i, (_, _, p, j) in enumerate(metas):
            by_file.setdefault(p, []).append((j, float(coef[i])))
        for p, items in by_file.items():
            d = torch.load(p, map_location="cpu", weights_only=False)
            for j, c in items:
                if abs(c) < 1e-9:
                    continue
                for k, v in d["emas"][j].items():
                    vf = v.float() * c
                    out[k] = vf if k not in out else out[k] + vf
            del d
        return out

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

@dataclass
class Record:
    path: Optional[str]
    captions: List[str]
    label: int = -1
    hf_index: int = -1


def _load_caption_file(p: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    txt = p.read_text(encoding="utf-8").strip()
    if p.suffix == ".json" or txt.startswith("{"):
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    out[os.path.basename(k)] = [v] if isinstance(v, str) else list(v)
                return out
        except json.JSONDecodeError:
            pass
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        key = d.get("file") or d.get("image") or d.get("file_name") or d.get("path")
        caps = d.get("captions") or d.get("caption") or d.get("text") or d.get("prompt")
        if key is None or caps is None:
            continue
        out[os.path.basename(str(key))] = [caps] if isinstance(caps, str) else list(caps)
    return out


def discover_records(args) -> Tuple[List[Record], Optional[Any]]:
    recs: List[Record] = []
    hf_ds = None

    if args.hf_dataset:
        from datasets import load_dataset
        hf_ds = load_dataset(args.hf_dataset, split=args.hf_split)
        cols = hf_ds.column_names
        img_c = args.hf_image_col if args.hf_image_col in cols else next(
            c for c in ("image", "img", "jpg") if c in cols)
        cap_c = args.hf_caption_col if args.hf_caption_col in cols else next(
            (c for c in ("text", "caption", "captions", "prompt") if c in cols), None)
        lab_c = args.hf_label_col if args.hf_label_col in cols else next(
            (c for c in ("label", "labels", "class") if c in cols), None)
        for i in range(len(hf_ds)):
            row = hf_ds[i] if cap_c or lab_c else None
            caps: List[str] = []
            lab = -1
            if row is not None:
                if cap_c is not None:
                    c = row[cap_c]
                    caps = [c] if isinstance(c, str) else list(c)
                if lab_c is not None:
                    lab = int(row[lab_c])
            if not caps:
                name = FLOWER_CLASSES[lab] if 0 <= lab < len(FLOWER_CLASSES) else "flower"
                caps = [tpl.format(name) for tpl in CAPTION_TEMPLATES]
            recs.append(Record(path=None, captions=caps, label=lab, hf_index=i))
        args._hf_image_col = img_c
        LOG(f"HF dataset {args.hf_dataset}[{args.hf_split}]: {len(recs)} rows, "
            f"image col='{img_c}' caption col='{cap_c}' label col='{lab_c}'")
    else:
        root = Path(args.images_dir)
        files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT])
        if not files:
            raise SystemExit(f"no images under {root}")
        caps_map: Dict[str, List[str]] = {}
        if args.captions:
            caps_map = _load_caption_file(Path(args.captions))
            LOG(f"loaded captions for {len(caps_map)} files")
        labels_map: Dict[str, int] = {}
        if args.labels:
            lp = Path(args.labels)
            if lp.suffix == ".mat":
                from scipy.io import loadmat
                lab = loadmat(str(lp))["labels"].reshape(-1).astype(int) - 1
                labels_map = {f.name: int(lab[i]) for i, f in enumerate(files)}
            else:
                obj = json.loads(lp.read_text())
                labels_map = {os.path.basename(k): int(v) for k, v in obj.items()}
        for f in files:
            lab = labels_map.get(f.name, -1)
            caps = caps_map.get(f.name, [])
            if not caps:
                name = FLOWER_CLASSES[lab] if 0 <= lab < len(FLOWER_CLASSES) else "flower"
                caps = [tpl.format(name) for tpl in CAPTION_TEMPLATES]
            recs.append(Record(path=str(f), captions=caps, label=lab))
        LOG(f"found {len(recs)} images under {root}")

    if args.captions_per_image > 0:
        rng = random.Random(1234)
        for r in recs:
            if len(r.captions) > args.captions_per_image:
                r.captions = rng.sample(r.captions, args.captions_per_image)
    return recs, hf_ds

def make_variant(img, size: int, canonical: bool, rng: random.Random,
                 max_rot_deg: float = 12.0, min_area: float = 0.55):
    from PIL import Image
    W, H = img.size
    if canonical:
        s = size / min(W, H)
        img2 = img.resize((max(size, int(round(W * s))), max(size, int(round(H * s)))),
                          Image.BICUBIC)
        W2, H2 = img2.size
        l, u = (W2 - size) // 2, (H2 - size) // 2
        return img2.crop((l, u, l + size, u + size)), np.zeros(4, dtype=np.float32)

    R = int(round(size * 1.3))
    s = R / min(W, H)
    base = img.resize((max(R, int(round(W * s))), max(R, int(round(H * s)))), Image.BICUBIC)
    W2, H2 = base.size
    theta = rng.uniform(-max_rot_deg, max_rot_deg)
    if abs(theta) > 1e-3:
        base = base.rotate(theta, resample=Image.BICUBIC, expand=False)
    ct = abs(math.cos(math.radians(theta))) + abs(math.sin(math.radians(theta)))
    L_safe = min(W2, H2) / ct
    area = rng.uniform(min_area, 1.0)
    side = L_safe * math.sqrt(area)
    cx0, cy0 = W2 / 2.0, H2 / 2.0
    max_dx = max(0.0, (min(W2, H2) / ct - side) / 2.0)
    dx = rng.uniform(-max_dx, max_dx)
    dy = rng.uniform(-max_dx, max_dx)
    l = int(round(cx0 + dx - side / 2)); u = int(round(cy0 + dy - side / 2))
    side_i = int(round(side))
    l = max(0, min(l, W2 - side_i)); u = max(0, min(u, H2 - side_i))
    crop = base.crop((l, u, l + side_i, u + side_i)).resize((size, size), Image.BICUBIC)

    area_frac = (side / R) ** 2
    aug = np.array([
        1.0 - min(area_frac, 1.0),
        dx / max(max_dx, 1e-6) if max_dx > 0 else 0.0,
        dy / max(max_dx, 1e-6) if max_dx > 0 else 0.0,
        theta / max_rot_deg,
    ], dtype=np.float32)
    return crop, aug


class VariantDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self, recs: List[Record], hf_ds, hf_col: str, size: int,
                 k: int, dino_res: int, seed: int = 0, hflip: bool = True):
        self.recs, self.hf_ds, self.hf_col = recs, hf_ds, hf_col
        self.size, self.k, self.dino_res, self.seed, self.hflip = size, k, dino_res, seed, hflip

    def __len__(self):
        return len(self.recs) * self.k

    def _open(self, i: int):
        from PIL import Image
        r = self.recs[i]
        if r.path is not None:
            im = Image.open(r.path)
        else:
            im = self.hf_ds[r.hf_index][self.hf_col]
        return im.convert("RGB")

    def __getitem__(self, idx: int):
        import torchvision.transforms.functional as TF
        i, k = idx // self.k, idx % self.k
        rng = random.Random(self.seed * 1_000_003 + idx)
        img = self._open(i)
        crop, aug = make_variant(img, self.size, canonical=(k == 0), rng=rng)
        if self.hflip and k > 0 and rng.random() < 0.5:
            from PIL import Image as _I
            crop = crop.transpose(_I.FLIP_LEFT_RIGHT)
        x = TF.to_tensor(crop)
        px = x * 2.0 - 1.0
        dino = torch.zeros(1)
        if self.dino_res > 0:
            d = TF.resize(x, [self.dino_res, self.dino_res], antialias=True)
            dino = TF.normalize(d, self.IMAGENET_MEAN, self.IMAGENET_STD)
        return px, dino, torch.from_numpy(aug), idx

def cache_main(args):
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    dev = device_auto()
    out = Path(args.cache_dir); out.mkdir(parents=True, exist_ok=True)
    global LOG; LOG = Log(out / "cache.log")

    recs, hf_ds = discover_records(args)
    N, K = len(recs), args.aug_variants
    S = args.image_size
    lat = S // 8
    grid = lat // args.patch_size
    dino_res = grid * 14 if args.repa else 0
    LOG(f"N={N} images, K={K} variants, latent {lat}x{lat}, "
        f"tokens={grid*grid}, dino_res={dino_res}")

    uniq: Dict[str, int] = {}
    cap_ids: List[int] = []
    cap_off: List[int] = [0]
    for r in recs:
        for c in r.captions:
            c = c.strip()
            if c not in uniq:
                uniq[c] = len(uniq)
            cap_ids.append(uniq[c])
        cap_off.append(len(cap_ids))
    cap_list = [None] * len(uniq)
    for c, i in uniq.items():
        cap_list[i] = c
    LOG(f"{len(cap_list)} unique captions (of {len(cap_ids)} total)")

    tok = CLIPTokenizer.from_pretrained(args.text_model)
    txt = CLIPTextModel.from_pretrained(args.text_model, torch_dtype=torch.float16).to(dev).eval()
    Dt = txt.config.hidden_size
    L = args.text_len
    emb = np.lib.format.open_memmap(out / "text.npy", mode="w+",
                                    dtype=np.float16, shape=(len(cap_list), L, Dt))
    pool = np.zeros((len(cap_list), Dt), dtype=np.float16)
    tlen = np.zeros((len(cap_list),), dtype=np.int16)
    with torch.no_grad():
        for i in tqdm(range(0, len(cap_list), 256), desc="text"):
            batch = cap_list[i:i + 256]
            b = tok(batch, padding="max_length", max_length=L, truncation=True,
                    return_tensors="pt").to(dev)
            o = txt(**b)
            emb[i:i + len(batch)] = o.last_hidden_state.cpu().numpy().astype(np.float16)
            pool[i:i + len(batch)] = o.pooler_output.cpu().numpy().astype(np.float16)
            tlen[i:i + len(batch)] = b.attention_mask.sum(1).cpu().numpy().astype(np.int16)
        b = tok([""], padding="max_length", max_length=L, truncation=True,
                return_tensors="pt").to(dev)
        o = txt(**b)
        np.savez(out / "text_uncond.npz",
                 emb=o.last_hidden_state[0].cpu().numpy().astype(np.float16),
                 pool=o.pooler_output[0].cpu().numpy().astype(np.float16),
                 len=np.array([int(b.attention_mask.sum())], dtype=np.int16))
    emb.flush(); del emb, txt
    np.save(out / "text_pool.npy", pool)
    np.save(out / "text_len.npy", tlen)
    np.save(out / "cap_ids.npy", np.array(cap_ids, dtype=np.int32))
    np.save(out / "cap_off.npy", np.array(cap_off, dtype=np.int64))
    (out / "captions.json").write_text(json.dumps(cap_list, ensure_ascii=False))
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float16).to(dev).eval()
    sf = float(getattr(vae.config, "scaling_factor", 0.18215))
    Cz = int(vae.config.latent_channels)
    dino = None
    Dd = 0
    if args.repa:
        from transformers import AutoModel
        dino = AutoModel.from_pretrained(args.repa_model, torch_dtype=torch.float16).to(dev).eval()
        Dd = dino.config.hidden_size

    mu = np.lib.format.open_memmap(out / "lat_mu.npy", mode="w+", dtype=np.float16,
                                   shape=(N * K, Cz, lat, lat))
    sd = np.lib.format.open_memmap(out / "lat_sd.npy", mode="w+", dtype=np.float16,
                                   shape=(N * K, Cz, lat, lat))
    aug_arr = np.zeros((N * K, 4), dtype=np.float32)
    dfeat = None
    if args.repa:
        dfeat = np.lib.format.open_memmap(out / "dino.npy", mode="w+", dtype=np.float16,
                                          shape=(N * K, grid * grid, Dd))
        LOG(f"dino.npy will be {human_bytes(N*K*grid*grid*Dd*2)}")

    ds = VariantDataset(recs, hf_ds, getattr(args, "_hf_image_col", "image"),
                        S, K, dino_res, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.cache_batch, num_workers=args.workers,
                    pin_memory=torch.cuda.is_available(), shuffle=False)
    with torch.no_grad():
        for px, dn, ag, idx in tqdm(dl, desc="vae+dino"):
            px = px.to(dev, non_blocking=True).half()
            post = vae.encode(px).latent_dist
            i0 = int(idx[0])
            n = px.shape[0]
            assert int(idx[-1]) == i0 + n - 1, 'cache sampler must stay sequential'
            mu[i0:i0 + n] = post.mean.cpu().numpy().astype(np.float16)
            sd[i0:i0 + n] = post.std.cpu().numpy().astype(np.float16)
            aug_arr[i0:i0 + n] = ag.numpy()
            if dino is not None:
                f = dino(pixel_values=dn.to(dev, non_blocking=True).half()).last_hidden_state
                dfeat[i0:i0 + n] = f[:, 1:, :].cpu().numpy().astype(np.float16)  # drop CLS
    mu.flush(); sd.flush()
    if dfeat is not None:
        dfeat.flush()
    np.save(out / "aug.npy", aug_arr)

    sel = np.arange(0, N * K, max(1, (N * K) // 4096))
    zs = (mu[sel].astype(np.float32)) * sf
    ch_mean = zs.mean(axis=(0, 2, 3)).tolist()
    ch_std = zs.std(axis=(0, 2, 3)).tolist()
    LOG(f"latent per-channel mean={['%.3f'%m for m in ch_mean]} std={['%.3f'%s for s in ch_std]}")

    rng = np.random.RandomState(args.seed)
    val = np.zeros(N, dtype=np.uint8)
    val[rng.permutation(N)[: max(1, int(N * args.val_frac))]] = 1

    meta = dict(
        n_images=N, k_variants=K, image_size=S, latent_size=lat, latent_channels=Cz,
        patch_size=args.patch_size, tokens=grid * grid, vae=args.vae,
        scaling_factor=sf, ch_mean=ch_mean, ch_std=ch_std,
        text_model=args.text_model, text_len=L, text_dim=Dt,
        repa=bool(args.repa), repa_model=args.repa_model if args.repa else None,
        repa_dim=Dd, dino_res=dino_res, aug_dim=4,
        labels=[int(r.label) for r in recs], created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out / "val_mask.npy", val)
    total = sum(f.stat().st_size for f in out.glob("*.npy"))
    LOG(f"cache complete: {human_bytes(total)} in {out}")


class CachedLatents(Dataset):
    def __init__(self, cache_dir: Path, split: str = "train", repa: bool = True,
                 resample_posterior: bool = True):
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.N, self.K = self.meta["n_images"], self.meta["k_variants"]
        self.sf = self.meta["scaling_factor"]
        self.ch_mean = np.array(self.meta["ch_mean"], dtype=np.float32)[:, None, None]
        self.ch_std = np.array(self.meta["ch_std"], dtype=np.float32)[:, None, None]
        self.repa = bool(repa and self.meta.get("repa"))
        self.resample = resample_posterior

        self.mu = np.load(self.dir / "lat_mu.npy", mmap_mode="r")
        self.sd = np.load(self.dir / "lat_sd.npy", mmap_mode="r")
        self.aug = np.load(self.dir / "aug.npy", mmap_mode="r")
        self.text = np.load(self.dir / "text.npy", mmap_mode="r")
        self.pool = np.load(self.dir / "text_pool.npy", mmap_mode="r")
        self.tlen = np.load(self.dir / "text_len.npy", mmap_mode="r")
        self.cap_ids = np.load(self.dir / "cap_ids.npy")
        self.cap_off = np.load(self.dir / "cap_off.npy")
        self.dino = np.load(self.dir / "dino.npy", mmap_mode="r") if self.repa else None

        u = np.load(self.dir / "text_uncond.npz")
        self.unc_emb = torch.from_numpy(u["emb"].astype(np.float32))
        self.unc_pool = torch.from_numpy(u["pool"].astype(np.float32))
        self.unc_len = int(u["len"][0])

        val = np.load(self.dir / "val_mask.npy")
        keep = np.where(val == (1 if split == "val" else 0))[0]
        self.img_idx = keep if split != "all" else np.arange(self.N)
        self.k_range = 1 if split == "val" else self.K

    def __len__(self):
        return len(self.img_idx) * self.k_range

    def n_images(self):
        return len(self.img_idx)

    def caption_of(self, i: int, rng: np.random.RandomState) -> int:
        a, b = self.cap_off[i], self.cap_off[i + 1]
        return int(self.cap_ids[rng.randint(a, b)]) if b > a else 0

    def __getitem__(self, j: int):
        ii = j // self.k_range
        k = j % self.k_range
        i = int(self.img_idx[ii])
        flat = i * self.K + k
        rng = np.random.RandomState((j * 2654435761 + random.randint(0, 2 ** 31)) % (2 ** 32))

        mu = self.mu[flat].astype(np.float32)
        if self.resample:
            mu = mu + self.sd[flat].astype(np.float32) * rng.randn(*mu.shape).astype(np.float32)
        z = (mu * self.sf - self.ch_mean) / self.ch_std

        cid = self.caption_of(i, rng)
        ctx = self.text[cid].astype(np.float32)
        pooled = self.pool[cid].astype(np.float32)
        mask = np.zeros((ctx.shape[0],), dtype=np.float32)
        mask[: int(self.tlen[cid])] = 1.0
        aug = self.aug[flat].astype(np.float32)
        d = self.dino[flat].astype(np.float32) if self.dino is not None else np.zeros(1, np.float32)
        return (torch.from_numpy(z), torch.from_numpy(ctx), torch.from_numpy(mask),
                torch.from_numpy(pooled), torch.from_numpy(aug), torch.from_numpy(d))

    def decode_norm(self, z: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.ch_mean, device=z.device, dtype=z.dtype)
        s = torch.as_tensor(self.ch_std, device=z.device, dtype=z.dtype)
        return (z * s + m) / self.sf


def cfg_dropout(ctx, mask, pooled, unc_emb, unc_pool, unc_len, p: float):
    if p <= 0:
        return ctx, mask, pooled
    B = ctx.shape[0]
    drop = (torch.rand(B, device=ctx.device) < p)
    if drop.any():
        ctx = torch.where(drop[:, None, None], unc_emb.to(ctx.dtype)[None], ctx)
        pooled = torch.where(drop[:, None], unc_pool.to(pooled.dtype)[None], pooled)
        um = torch.zeros_like(mask[0]); um[:unc_len] = 1.0
        mask = torch.where(drop[:, None], um[None], mask)
    return ctx, mask, pooled


def build_model(mcfg: ModelCfg, meta: dict, device) -> FlowDiT:
    mcfg = dataclasses.replace(
        mcfg,
        in_channels=meta["latent_channels"],
        ctx_dim=meta["text_dim"],
        pooled_dim=meta["text_dim"],
        patch_size=meta["patch_size"],
        repa_dim=meta.get("repa_dim") or mcfg.repa_dim,
        aug_dim=meta.get("aug_dim", 4),
    )
    return FlowDiT(mcfg, latent_size=meta["latent_size"]).to(device)


def param_groups(model: nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith("scale_shift") or "registers" in n
         else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, base: float, warmup: int, total: int, final_frac: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def train_main(args):
    dev = device_auto()
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    global LOG; LOG = Log(run / "train.log")
    set_seed(args.seed)

    ds = CachedLatents(args.cache_dir, "train", repa=args.repa)
    dsv = CachedLatents(args.cache_dir, "val", repa=False)
    meta = ds.meta
    LOG(f"train items {len(ds)} ({ds.n_images()} images x {ds.K} variants) | val {len(dsv)}")
    if meta.get("repa") and not args.repa:
        LOG("cache HAS DINOv2 features but --repa was not passed: you are leaving "
            "the single largest convergence speed-up on the table.", "WARN")
    if args.repa and not meta.get("repa"):
        LOG("--repa requested but the cache has no DINOv2 features; re-run "
            "`cache --repa`. Continuing WITHOUT alignment.", "WARN")

    mcfg = ModelCfg.preset(args.preset)
    if args.dim: mcfg.dim = args.dim
    if args.depth: mcfg.depth = args.depth
    if args.heads: mcfg.heads = args.heads
    mcfg.drop_path = args.drop_path
    mcfg.repa_layer = args.repa_layer
    model = build_model(mcfg, meta, dev)
    model.grad_ckpt_every = args.grad_ckpt_every
    LOG(f"model {args.preset}: {count_params(model)/1e6:.1f}M params, "
        f"grid={model.grid}x{model.grid} tokens={model.grid**2}")

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.95), eps=1e-8)
    ema = PostHocEMA(model, sigma_rels=tuple(float(s) for s in args.ema_sigmas.split(",")))

    if args.compile:
        try:
            model_c = torch.compile(model, dynamic=False)
            LOG("torch.compile enabled")
        except Exception as e:
            LOG(f"torch.compile failed ({e}); continuing eager", "WARN")
            model_c = model
    else:
        model_c = model

    start_step = 0
    ckpt_path = run / "latest.pt"
    if args.resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        ema.load_state_dict(st["ema"], dev)
        start_step = st["step"]
        LOG(f"resumed from step {start_step}")

    pin = dev.type == "cuda"
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=pin,
                    persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers > 0 else None)
    dlv = DataLoader(dsv, batch_size=args.batch_size, shuffle=False, num_workers=0)

    unc_emb = ds.unc_emb.to(dev); unc_pool = ds.unc_pool.to(dev)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda")

    total_steps = args.max_steps
    t_start = time.time()
    step = start_step
    ema_loss = None
    csv = run / "metrics.csv"
    if not csv.exists():
        csv.write_text("step,loss,mse,repa_cos,lr,imgs_per_s,hours\n")

    LOG(f"effective batch {args.batch_size * args.grad_accum} | "
        f"budget {args.max_hours}h or {total_steps} steps")

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    it = infinite(dl)
    model.train()
    while step < total_steps and (time.time() - t_start) / 3600 < args.max_hours:
        lr = lr_at(step, args.lr, args.warmup, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum: Dict[str, float] = {}
        for _ in range(args.grad_accum):
            z, ctx, mask, pooled, aug, dino = [b.to(dev, non_blocking=True) for b in next(it)]
            ctx, mask, pooled = cfg_dropout(ctx, mask, pooled, unc_emb, unc_pool,
                                            ds.unc_len, args.cfg_dropout)
            with amp:
                loss, logs = flow_loss(
                    model_c, z, ctx, mask, pooled,
                    aug if args.aug_cond else None,
                    t_mode=args.t_mode, t_m=args.t_mean, t_s=args.t_std,
                    t_shift=args.train_shift, use_ot=bool(args.ot_coupling),
                    repa_feat=dino if (args.repa and ds.repa) else None,
                    repa_weight=args.repa_weight * repa_decay(step, args),
                    uncertainty_weight=bool(args.uncertainty_weight),
                )
            (loss / args.grad_accum).backward()
            for k, v in logs.items():
                accum[k] = accum.get(k, 0.0) + float(v) / args.grad_accum

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1
        ema.update(model, step)

        ema_loss = accum["loss"] if ema_loss is None else 0.98 * ema_loss + 0.02 * accum["loss"]
        if step % args.log_every == 0:
            el = time.time() - t_start
            ips = (step - start_step) * args.batch_size * args.grad_accum / max(el, 1e-6)
            eta = (total_steps - step) * el / max(step - start_step, 1) / 3600
            LOG(f"step {step:>7}/{total_steps} loss {ema_loss:.4f} mse {accum['mse']:.4f} "
                f"repa {accum.get('repa_cos', float('nan')):.3f} gn {float(gn):.2f} "
                f"lr {lr:.2e} {ips:.0f} img/s eta {eta:.1f}h {vram_report()}")
            with open(csv, "a") as f:
                f.write(f"{step},{ema_loss:.5f},{accum['mse']:.5f},"
                        f"{accum.get('repa_cos', 0):.5f},{lr:.3e},{ips:.1f},{el/3600:.3f}\n")

        if args.snapshot_every and step % args.snapshot_every == 0:
            ema.snapshot(run / "phema" / f"snap_{step:08d}.pt", step)
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(), "step": step,
                        "mcfg": asdict(model.cfg), "meta": meta,
                        "args": vars(args)}, ckpt_path)
        if args.preview_every and step % args.preview_every == 0:
            preview(model, ema, ds, meta, run, step, args, dev)
            model.train()
        if args.val_every and step % args.val_every == 0:
            LOG(f"  val mse {validate(model_c, dlv, dev, args, ds):.4f}")
            model.train()

    ema.snapshot(run / "phema" / f"snap_{step:08d}.pt", step)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "ema": ema.state_dict(), "step": step, "mcfg": asdict(model.cfg),
                "meta": meta, "args": vars(args)}, ckpt_path)
    torch.save({"model": {k: v for k, v in ema.emas[-1].items()}, "step": step,
                "mcfg": asdict(model.cfg), "meta": meta},
               run / "ema_last.pt")
    LOG(f"done at step {step} after {(time.time()-t_start)/3600:.2f}h. "
        f"Next: `sweep-ema` then `fid --sweep-cfg`.")


def repa_decay(step: int, args) -> float:
    if args.repa_decay_start <= 0:
        return 1.0
    s0 = int(args.repa_decay_start * args.max_steps)
    if step <= s0:
        return 1.0
    return max(0.0, 1.0 - (step - s0) / max(1, args.max_steps - s0))


@torch.no_grad()
def validate(model, dlv, dev, args, ds) -> float:
    model.eval()
    tot, n = 0.0, 0
    g = torch.Generator(device="cpu").manual_seed(1234)
    for batch in dlv:
        z, ctx, mask, pooled, aug, _ = [b.to(dev) for b in batch]
        B = z.shape[0]
        eps = torch.randn(z.shape, generator=g).to(dev)
        t = torch.rand(B, generator=g).to(dev).clamp(1e-3, 1 - 1e-3)
        tb = t.view(B, 1, 1, 1)
        zt = (1 - tb) * z + tb * eps
        with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
            v = model(zt, t, ctx, mask, pooled, aug if args.aug_cond else None)
        tot += float((v.float() - (eps - z).float()).pow(2).mean()) * B
        n += B
        if n >= 512:
            break
    return tot / max(n, 1)

def make_schedule(n_steps: int, shift: float = 1.0, device="cpu") -> torch.Tensor:
    u = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    return shift * u / (1.0 + (shift - 1.0) * u)


def _cfg_combine(v_cond, v_uncond, x, t, scale, rescale_phi, lo, hi):
    if scale == 1.0 or not (lo <= float(t) <= hi):
        return v_cond
    v = v_uncond + scale * (v_cond - v_uncond)
    if rescale_phi > 0:
        tt = max(float(t), 1e-3)
        x0_g = x - tt * v
        x0_c = x - tt * v_cond
        s_g = x0_g.flatten(1).std(1).view(-1, 1, 1, 1).clamp_min(1e-6)
        s_c = x0_c.flatten(1).std(1).view(-1, 1, 1, 1)
        x0_r = x0_g * (s_c / s_g)
        x0 = rescale_phi * x0_r + (1 - rescale_phi) * x0_g
        v = (x - x0) / tt
    return v


@torch.no_grad()
def sample_latents(model, n: int, ctx, mask, pooled, unc_ctx, unc_mask, unc_pool,
                   latent_size: int, channels: int, device, steps: int = 32,
                   cfg: float = 2.5, rescale: float = 0.7, shift: float = 1.0,
                   solver: str = "heun", cfg_lo: float = 0.0, cfg_hi: float = 1.0,
                   aug_dim: int = 4, generator=None, bad_model=None,
                   autoguidance: float = 0.0, progress: bool = False):
    model.eval()
    x = torch.randn(n, channels, latent_size, latent_size, device=device, generator=generator)
    ts = make_schedule(steps, shift, device)
    aug0 = torch.zeros(n, aug_dim, device=device)
    amp = torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda")

    def velocity(xx, tt):
        tt = max(float(tt), 1e-4)
        tv = torch.full((xx.shape[0],), tt, device=device)
        with amp:
            if autoguidance > 0.0 and bad_model is not None:
                v_good = model(xx, tv, ctx, mask, pooled, aug0).float()
                v_bad = bad_model(xx, tv, ctx, mask, pooled, aug0).float()
                return v_bad + autoguidance * (v_good - v_bad)
            if cfg == 1.0:
                return model(xx, tv, ctx, mask, pooled, aug0).float()
            xin = torch.cat([xx, xx], 0)
            tin = torch.cat([tv, tv], 0)
            cin = torch.cat([ctx, unc_ctx], 0)
            min_ = torch.cat([mask, unc_mask], 0)
            pin = torch.cat([pooled, unc_pool], 0)
            ain = torch.cat([aug0, aug0], 0)
            v = model(xin, tin, cin, min_, pin, ain).float()
            vc, vu = v.chunk(2, 0)
            return _cfg_combine(vc, vu, xx, tt, cfg, rescale, cfg_lo, cfg_hi)

    rng = range(steps)
    if progress:
        rng = tqdm(rng, desc="sample", leave=False)
    for i in rng:
        t_cur, t_next = ts[i], ts[i + 1]
        dt = (t_next - t_cur)
        v1 = velocity(x, t_cur)
        if solver == "euler":
            x = x + dt * v1
        elif solver == "heun":
            x_e = x + dt * v1
            v2 = velocity(x_e, t_next)
            x = x + dt * 0.5 * (v1 + v2)
        elif solver == "midpoint":
            x_m = x + 0.5 * dt * v1
            v2 = velocity(x_m, t_cur + 0.5 * dt)
            x = x + dt * v2
        else:
            raise ValueError(solver)
    return x


class Decoder:
    def __init__(self, ds_meta: dict, device, vae_name: Optional[str] = None):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(vae_name or ds_meta["vae"],
                                                 torch_dtype=torch.float16).to(device).eval()
        self.sf = ds_meta["scaling_factor"]
        self.ch_mean = torch.tensor(ds_meta["ch_mean"], device=device).view(1, -1, 1, 1)
        self.ch_std = torch.tensor(ds_meta["ch_std"], device=device).view(1, -1, 1, 1)
        self.device = device

    @torch.no_grad()
    def __call__(self, z: torch.Tensor, chunk: int = 8) -> torch.Tensor:
        z = (z.float() * self.ch_std + self.ch_mean) / self.sf
        outs = []
        for i in range(0, z.shape[0], chunk):
            im = self.vae.decode(z[i:i + chunk].half()).sample
            outs.append(((im.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).cpu())
        return torch.cat(outs, 0)


class TextEncoder:
    def __init__(self, meta: dict, device):
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tok = CLIPTokenizer.from_pretrained(meta["text_model"])
        self.mod = CLIPTextModel.from_pretrained(meta["text_model"],
                                                 torch_dtype=torch.float16).to(device).eval()
        self.L = meta["text_len"]
        self.device = device

    @torch.no_grad()
    def __call__(self, prompts: Sequence[str]):
        b = self.tok(list(prompts), padding="max_length", max_length=self.L,
                     truncation=True, return_tensors="pt").to(self.device)
        o = self.mod(**b)
        return (o.last_hidden_state.float(), b.attention_mask.float(), o.pooler_output.float())


def load_model_for_inference(ckpt_path: Path, device, meta_override=None):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = meta_override or st["meta"]
    mcfg = ModelCfg(**st["mcfg"])
    model = FlowDiT(mcfg, latent_size=meta["latent_size"]).to(device)
    sd = st["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        LOG(f"missing keys on load: {len(missing)} (ok if REPA/logvar heads)", "WARN")
    model.eval().requires_grad_(False)
    return model, meta, st.get("step", -1)


def save_grid(imgs: torch.Tensor, path: Path, nrow: int = 8):
    import torchvision.utils as vutils
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    g = vutils.make_grid(imgs.float() / 255.0, nrow=nrow, padding=2)
    arr = (g.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def preview(model, ema, ds, meta, run: Path, step: int, args, dev):
    shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({**shadow, **{k: v.to(shadow[k].dtype) for k, v in ema.emas[-1].items()}},
                          strict=False)
    try:
        caps = json.loads((Path(args.cache_dir) / "captions.json").read_text())
        rng = random.Random(0)
        prompts = [caps[rng.randrange(len(caps))] for _ in range(args.preview_n)]
        te = getattr(preview, "_te", None) or TextEncoder(meta, dev)
        preview._te = te
        ctx, mask, pooled = te(prompts)
        u_ctx, u_mask, u_pool = te([""] * len(prompts))
        z = sample_latents(model, len(prompts), ctx, mask, pooled, u_ctx, u_mask, u_pool,
                           meta["latent_size"], meta["latent_channels"], dev,
                           steps=args.sample_steps, cfg=args.cfg, rescale=args.cfg_rescale,
                           shift=args.sample_shift, solver="heun",
                           cfg_lo=args.cfg_lo, cfg_hi=args.cfg_hi)
        dec = getattr(preview, "_dec", None) or Decoder(meta, dev)
        preview._dec = dec
        imgs = dec(z)
        save_grid(imgs, run / "previews" / f"step_{step:08d}.png", nrow=int(math.sqrt(len(imgs))))
        (run / "previews" / f"step_{step:08d}.txt").write_text("\n".join(prompts))
    except Exception as e:
        LOG(f"preview failed: {e}", "WARN")
    finally:
        model.load_state_dict(shadow, strict=True)


class InceptionFeatures:
    def __init__(self, device):
        self.device = device
        self.backend = "torchvision"
        try:
            from pytorch_fid.inception import InceptionV3
            self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]],
                                   resize_input=True, normalize_input=True).to(device).eval()
            self.backend = "pytorch-fid(TF-Inception)"
            self._pfid = True
        except Exception:
            import torchvision
            w = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            net = torchvision.models.inception_v3(weights=w, aux_logits=True)
            net.fc = nn.Identity()
            self.net = net.to(device).eval()
            self._pfid = False
        LOG(f"FID feature backend: {self.backend}")

    @torch.no_grad()
    def __call__(self, imgs_uint8: torch.Tensor, batch: int = 64) -> np.ndarray:
        feats = []
        for i in range(0, imgs_uint8.shape[0], batch):
            x = imgs_uint8[i:i + batch].to(self.device).float() / 255.0
            if self._pfid:
                f = self.net(x)[0].squeeze(-1).squeeze(-1)
            else:
                x = F.interpolate(x, size=(299, 299), mode="bilinear",
                                  align_corners=False, antialias=True)
                mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
                f = self.net((x - mean) / std)
            feats.append(f.float().cpu().numpy())
        return np.concatenate(feats, 0)


def frechet_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    if _HAVE_SCIPY:
        try:
            covmean = _scipy_linalg.sqrtm(s1.dot(s2), disp=False)
        except TypeError:
            covmean = _scipy_linalg.sqrtm(s1.dot(s2))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
        if not np.isfinite(covmean).all():
            off = np.eye(s1.shape[0]) * 1e-6
            covmean = _scipy_linalg.sqrtm((s1 + off).dot(s2 + off))
            if isinstance(covmean, tuple):
                covmean = covmean[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr = np.trace(covmean)
    else:
        eig = np.linalg.eigvals(s1.dot(s2))
        tr = float(np.sum(np.sqrt(np.abs(eig))))
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr)


def kernel_distance(f1: np.ndarray, f2: np.ndarray, n_subsets: int = 100,
                    subset_size: int = 1000, seed: int = 0) -> Tuple[float, float]:
    rs = np.random.RandomState(seed)
    d = f1.shape[1]
    m = min(subset_size, f1.shape[0], f2.shape[0])
    vals = []
    for _ in range(n_subsets):
        x = f1[rs.choice(f1.shape[0], m, replace=False)]
        y = f2[rs.choice(f2.shape[0], m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def fid_noise_floor(real_feats: np.ndarray, seed: int = 0) -> float:
    rs = np.random.RandomState(seed)
    idx = rs.permutation(real_feats.shape[0])
    h = len(idx) // 2
    return frechet_distance(real_feats[idx[:h]], real_feats[idx[h:2 * h]])


@torch.no_grad()
def real_images_from_cache(ds: CachedLatents, decoder: Decoder, n: int,
                           device, batch: int = 32) -> torch.Tensor:
    idx = np.arange(ds.N)[:n] * ds.K
    outs = []
    for i in range(0, len(idx), batch):
        sl = idx[i:i + batch]
        mu = torch.from_numpy(np.asarray(ds.mu[sl], dtype=np.float32)).to(device)
        z = (mu * ds.sf - torch.tensor(ds.ch_mean, device=device)) / torch.tensor(ds.ch_std, device=device)
        outs.append(decoder(z))
    return torch.cat(outs, 0)


def real_images_from_disk(cache_dir: Path, args, n: int, size: int) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    recs, hf_ds = discover_records(args)
    out = []
    for r in tqdm(recs[:n], desc="real px"):
        from PIL import Image
        im = (Image.open(r.path) if r.path else hf_ds[r.hf_index][getattr(args, "_hf_image_col", "image")]).convert("RGB")
        crop, _ = make_variant(im, size, canonical=True, rng=random.Random(0))
        out.append((TF.to_tensor(crop) * 255).round().to(torch.uint8))
    return torch.stack(out)


@torch.no_grad()
def generate_batched(model, meta, ds, te, dec, n: int, device, args,
                     cfg: float, prompts: Optional[List[str]] = None,
                     bad_model=None, seed: int = 0) -> Tuple[torch.Tensor, List[str]]:
    caps = json.loads((Path(args.cache_dir) / "captions.json").read_text())
    rng = random.Random(seed)
    if prompts is None:
        prompts = [caps[rng.randrange(len(caps))] for _ in range(n)]
    imgs, used = [], []
    g = torch.Generator(device=device).manual_seed(seed)
    B = args.sample_batch
    for i in tqdm(range(0, n, B), desc=f"gen cfg={cfg}"):
        p = prompts[i:i + B]
        ctx, mask, pooled = te(p)
        u_ctx, u_mask, u_pool = te([""] * len(p))
        z = sample_latents(model, len(p), ctx, mask, pooled, u_ctx, u_mask, u_pool,
                           meta["latent_size"], meta["latent_channels"], device,
                           steps=args.sample_steps, cfg=cfg, rescale=args.cfg_rescale,
                           shift=args.sample_shift, solver=args.solver,
                           cfg_lo=args.cfg_lo, cfg_hi=args.cfg_hi, generator=g,
                           bad_model=bad_model, autoguidance=args.autoguidance)
        imgs.append(dec(z))
        used += p
    return torch.cat(imgs, 0), used


@torch.no_grad()
def clip_score(imgs_uint8: torch.Tensor, prompts: List[str], device,
               model_name: str = "openai/clip-vit-large-patch14", batch: int = 64) -> float:
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
    p = CLIPProcessor.from_pretrained(model_name)
    scores = []
    for i in range(0, len(prompts), batch):
        ims = [x.permute(1, 2, 0).numpy() for x in imgs_uint8[i:i + batch]]
        inp = p(text=prompts[i:i + batch], images=ims, return_tensors="pt",
                padding=True, truncation=True).to(device)
        ie = F.normalize(m.get_image_features(pixel_values=inp["pixel_values"].half()).float(), dim=-1)
        te_ = F.normalize(m.get_text_features(input_ids=inp["input_ids"],
                                              attention_mask=inp["attention_mask"]).float(), dim=-1)
        scores.append((ie * te_).sum(-1).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return float(np.concatenate(scores).mean() * 100.0)


def fid_main(args):
    dev = device_auto()
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    global LOG; LOG = Log(run / "eval.log")
    ds = CachedLatents(args.cache_dir, "all", repa=False)
    meta = ds.meta
    model, meta, step = load_model_for_inference(Path(args.ckpt), dev, meta)
    bad_model = None
    if args.autoguidance > 0 and args.bad_ckpt:
        bad_model, _, _ = load_model_for_inference(Path(args.bad_ckpt), dev, meta)

    dec = Decoder(meta, dev)
    te = TextEncoder(meta, dev)
    inc = InceptionFeatures(dev)

    n = min(args.n_fid, ds.N if args.fid_real == "vae" else args.n_fid)
    LOG(f"reference: {args.fid_real}, N_real={min(args.n_fid, ds.N)}, N_gen={args.n_fid}")
    if args.fid_real == "vae":
        real = real_images_from_cache(ds, dec, min(args.n_fid, ds.N), dev)
    else:
        real = real_images_from_disk(Path(args.cache_dir), args, min(args.n_fid, ds.N),
                                     meta["image_size"])
    f_real = inc(real)
    floor = fid_noise_floor(f_real)
    LOG(f"FID noise floor (real vs real, N/2={len(f_real)//2}): {floor:.2f}  <-- your practical lower bound")

    cfgs = [float(c) for c in args.sweep_cfg.split(",")] if args.sweep_cfg else [args.cfg]
    results = []
    for c in cfgs:
        imgs, prompts = generate_batched(model, meta, ds, te, dec, args.n_fid, dev, args,
                                         cfg=c, bad_model=bad_model, seed=args.seed)
        f_gen = inc(imgs)
        fid = frechet_distance(f_real, f_gen)
        kid, kid_sd = kernel_distance(f_real, f_gen)
        row = {"cfg": c, "fid": fid, "kid": kid, "kid_std": kid_sd,
               "fid_floor": floor, "steps": args.sample_steps, "solver": args.solver,
               "shift": args.sample_shift, "rescale": args.cfg_rescale,
               "cfg_lo": args.cfg_lo, "cfg_hi": args.cfg_hi, "n": args.n_fid,
               "backend": inc.backend, "ckpt": str(args.ckpt), "train_step": step}
        if args.clip_score:
            row["clip"] = clip_score(imgs[:2048], prompts[:2048], dev)
        LOG(f"cfg={c:<4} FID {fid:7.3f} (floor {floor:.2f})  KID {kid*1000:7.3f}e-3 "
            f"+-{kid_sd*1000:.3f}  " + (f"CLIP {row.get('clip', 0):.2f}" if args.clip_score else ""))
        results.append(row)
        if args.save_samples:
            save_grid(imgs[:64], run / "eval" / f"cfg{c}_samples.png", nrow=8)
    best = min(results, key=lambda r: r["fid"])
    LOG(f"BEST: cfg={best['cfg']} FID={best['fid']:.3f}")
    (run / "eval_results.json").write_text(json.dumps(results, indent=2))
    return results


def sweep_ema_main(args):
    dev = device_auto()
    run = Path(args.run_dir)
    global LOG; LOG = Log(run / "eval.log")
    snaps = sorted((run / "phema").glob("snap_*.pt"))
    if not snaps:
        raise SystemExit("no phema snapshots -- train with --snapshot-every > 0")
    LOG(f"{len(snaps)} snapshots, steps {torch.load(snaps[0], map_location='cpu', weights_only=False)['step']}"
        f"..{torch.load(snaps[-1], map_location='cpu', weights_only=False)['step']}")
    base = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    meta, mcfg = base["meta"], base["mcfg"]

    ds = CachedLatents(args.cache_dir, "all", repa=False)
    dec = Decoder(meta, dev); te = TextEncoder(meta, dev); inc = InceptionFeatures(dev)
    real = real_images_from_cache(ds, dec, min(args.n_fid, ds.N), dev)
    f_real = inc(real)
    floor = fid_noise_floor(f_real)
    LOG(f"noise floor {floor:.2f}")

    sigmas = [float(s) for s in args.ema_grid.split(",")]
    out = []
    best = (1e9, None, None)
    for sr in sigmas:
        sd = PostHocEMA.reconstruct(snaps, sr)
        model = FlowDiT(ModelCfg(**mcfg), latent_size=meta["latent_size"]).to(dev)
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
        model.eval().requires_grad_(False)
        imgs, prompts = generate_batched(model, meta, ds, te, dec, args.n_fid, dev, args,
                                         cfg=args.cfg, seed=args.seed)
        fid = frechet_distance(f_real, inc(imgs))
        LOG(f"sigma_rel={sr:.3f}  FID {fid:.3f}")
        out.append({"sigma_rel": sr, "fid": fid, "floor": floor})
        if fid < best[0]:
            best = (fid, sr, sd)
        del model
        torch.cuda.empty_cache()
    LOG(f"best sigma_rel={best[1]} FID={best[0]:.3f} -> {run/'ema_best.pt'}")
    torch.save({"model": {k: v.cpu() for k, v in best[2].items()}, "meta": meta,
                "mcfg": mcfg, "step": base["step"], "sigma_rel": best[1]}, run / "ema_best.pt")
    (run / "ema_sweep.json").write_text(json.dumps(out, indent=2))


@torch.no_grad()
def memcheck_main(args):
    from transformers import AutoModel
    import torchvision.transforms.functional as TF
    dev = device_auto()
    run = Path(args.run_dir)
    global LOG; LOG = Log(run / "eval.log")
    ds = CachedLatents(args.cache_dir, "all", repa=False)
    meta = ds.meta
    model, meta, _ = load_model_for_inference(Path(args.ckpt), dev, meta)
    dec = Decoder(meta, dev); te = TextEncoder(meta, dev)
    enc = AutoModel.from_pretrained(args.repa_model, torch_dtype=torch.float16).to(dev).eval()

    def embed(imgs):
        fs = []
        for i in range(0, imgs.shape[0], 32):
            x = imgs[i:i + 32].to(dev).float() / 255.0
            x = TF.normalize(TF.resize(x, [224, 224], antialias=True),
                             VariantDataset.IMAGENET_MEAN, VariantDataset.IMAGENET_STD)
            f = enc(pixel_values=x.half()).last_hidden_state[:, 0].float()
            fs.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(fs)

    real = real_images_from_cache(ds, dec, ds.N, dev)
    f_real = embed(real)
    gen, prompts = generate_batched(model, meta, ds, te, dec, args.n_mem, dev, args,
                                    cfg=args.cfg, seed=args.seed)
    f_gen = embed(gen)
    sim = f_gen @ f_real.T
    top, idx = sim.max(1)
    q = np.percentile(top.numpy(), [50, 90, 99, 100])
    LOG(f"NN cosine similarity to training set: median {q[0]:.3f} p90 {q[1]:.3f} "
        f"p99 {q[2]:.3f} max {q[3]:.3f}")
    LOG(f"fraction > {args.mem_threshold}: {(top > args.mem_threshold).float().mean()*100:.2f}%"
        "   (>0.95 usually means genuine near-duplicates; inspect the montage)")
    order = torch.argsort(top, descending=True)[:16]
    pairs = torch.stack([torch.stack([gen[i], real[idx[i]]]) for i in order]).flatten(0, 1)
    save_grid(pairs, run / "eval" / "memcheck_top16.png", nrow=8)
    (run / "eval" / "memcheck.json").write_text(json.dumps(
        {"median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "max": float(q[3])},
        indent=2))

def sample_main(args):
    dev = device_auto()
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    global LOG; LOG = Log(run / "eval.log")
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = st["meta"]
    model, meta, step = load_model_for_inference(Path(args.ckpt), dev, meta)
    dec = Decoder(meta, dev); te = TextEncoder(meta, dev)

    if args.prompt_file:
        prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    elif args.prompt:
        prompts = [args.prompt] * args.n
    else:
        caps = json.loads((Path(args.cache_dir) / "captions.json").read_text())
        rng = random.Random(args.seed)
        prompts = [caps[rng.randrange(len(caps))] for _ in range(args.n)]
    prompts = prompts[:args.n] if len(prompts) >= args.n else prompts * (args.n // len(prompts) + 1)
    prompts = prompts[:args.n]

    bad = None
    if args.autoguidance > 0 and args.bad_ckpt:
        bad, _, _ = load_model_for_inference(Path(args.bad_ckpt), dev, meta)
    imgs, used = generate_batched(model, meta, None, te, dec, args.n, dev, args,
                                  cfg=args.cfg, prompts=prompts, bad_model=bad, seed=args.seed)
    tag = f"cfg{args.cfg}_s{args.sample_steps}_{args.solver}"
    save_grid(imgs, run / "samples" / f"{tag}_step{step}.png",
              nrow=int(math.ceil(math.sqrt(args.n))))
    if args.save_individual:
        from PIL import Image
        d = run / "samples" / tag; d.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(imgs):
            Image.fromarray(im.permute(1, 2, 0).numpy()).save(d / f"{i:05d}.png")
    (run / "samples" / f"{tag}_step{step}.txt").write_text("\n".join(used))
    LOG(f"wrote {run/'samples'}/{tag}_step{step}.png")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Text-to-image latent rectified flow for Oxford Flowers-102 (12 GB / 50 h)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", choices=["cache", "train", "sample", "fid", "sweep-ema",
                                    "memcheck", "doctor", "selftest"])

    g = p.add_argument_group("paths / data")
    g.add_argument("--cache-dir", default="./cache256")
    g.add_argument("--run-dir", default="./runs/rf_base")
    g.add_argument("--images-dir", default=None, help="folder of jpgs (recursive)")
    g.add_argument("--captions", default=None, help="json/jsonl: file -> caption(s)")
    g.add_argument("--labels", default=None, help="json file->class, or imagelabels.mat")
    g.add_argument("--hf-dataset", default=None, help="e.g. nelorth/oxford-flowers")
    g.add_argument("--hf-split", default="train")
    g.add_argument("--hf-image-col", default="image")
    g.add_argument("--hf-caption-col", default="text")
    g.add_argument("--hf-label-col", default="label")
    g.add_argument("--captions-per-image", type=int, default=0,
                   help=">0 subsamples captions to cut text-cache size")

    g = p.add_argument_group("cache")
    g.add_argument("--image-size", type=int, default=256)
    g.add_argument("--aug-variants", type=int, default=8,
                   help="K cached crops per image (variant 0 is always canonical)")
    g.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    g.add_argument("--text-model", default="openai/clip-vit-large-patch14")
    g.add_argument("--text-len", type=int, default=77)
    g.add_argument("--repa", action="store_true", default=False,
                   help="cache DINOv2 features and use the REPA alignment loss")
    g.add_argument("--repa-model", default="facebook/dinov2-base")
    g.add_argument("--cache-batch", type=int, default=16)
    g.add_argument("--val-frac", type=float, default=0.03)

    g = p.add_argument_group("model")
    g.add_argument("--preset", default="base", choices=["small", "base", "large"])
    g.add_argument("--dim", type=int, default=0)
    g.add_argument("--depth", type=int, default=0)
    g.add_argument("--heads", type=int, default=0)
    g.add_argument("--patch-size", type=int, default=2,
                   help="cache-time only; train/sample always take it from meta.json")
    g.add_argument("--drop-path", type=float, default=0.05)
    g.add_argument("--aug-cond", type=int, default=1,
                   help="1 = condition on augmentation params and sample with aug=0")

    g = p.add_argument_group("objective")
    g.add_argument("--t-mode", default="logitnormal", choices=["logitnormal", "uniform", "cosmap"])
    g.add_argument("--t-mean", type=float, default=0.0)
    g.add_argument("--t-std", type=float, default=1.0)
    g.add_argument("--train-shift", type=float, default=1.0)
    g.add_argument("--ot-coupling", type=int, default=1)
    g.add_argument("--uncertainty-weight", type=int, default=1)
    g.add_argument("--repa-weight", type=float, default=0.5)
    g.add_argument("--repa-layer", type=int, default=6)
    g.add_argument("--repa-decay-start", type=float, default=0.7,
                   help="fraction of training after which the REPA term is annealed to 0")
    g.add_argument("--cfg-dropout", type=float, default=0.1)

    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--grad-accum", type=int, default=4)
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--weight-decay", type=float, default=0.02)
    g.add_argument("--warmup", type=int, default=1500)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--max-steps", type=int, default=45000)
    g.add_argument("--max-hours", type=float, default=50.0)
    g.add_argument("--grad-ckpt-every", type=int, default=2,
                   help="checkpoint every Nth block (0=off). 2 halves activation memory for ~17% time")
    g.add_argument("--ema-sigmas", default="0.05,0.10")
    g.add_argument("--snapshot-every", type=int, default=4000)
    g.add_argument("--ckpt-every", type=int, default=2000)
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--val-every", type=int, default=2000)
    g.add_argument("--preview-every", type=int, default=4000)
    g.add_argument("--preview-n", type=int, default=16)
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--compile", action="store_true")
    g.add_argument("--resume", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("sampling / eval")
    g.add_argument("--ckpt", default=None)
    g.add_argument("--bad-ckpt", default=None, help="weaker model for autoguidance")
    g.add_argument("--autoguidance", type=float, default=0.0)
    g.add_argument("--cfg", type=float, default=2.5)
    g.add_argument("--cfg-rescale", type=float, default=0.7)
    g.add_argument("--cfg-lo", type=float, default=0.0, help="apply CFG only for t >= lo")
    g.add_argument("--cfg-hi", type=float, default=1.0, help="apply CFG only for t <= hi")
    g.add_argument("--sample-steps", type=int, default=28)
    g.add_argument("--sample-shift", type=float, default=1.0)
    g.add_argument("--solver", default="heun", choices=["euler", "heun", "midpoint"])
    g.add_argument("--sample-batch", type=int, default=32)
    g.add_argument("--n", type=int, default=16)
    g.add_argument("--prompt", default=None)
    g.add_argument("--prompt-file", default=None)
    g.add_argument("--save-individual", action="store_true")
    g.add_argument("--n-fid", type=int, default=10000)
    g.add_argument("--fid-real", default="vae", choices=["vae", "pixel"],
                   help="'pixel' compares against the original files and needs --images-dir")
    g.add_argument("--sweep-cfg", default=None, help="e.g. 1.0,1.5,2.0,2.5,3.0")
    g.add_argument("--ema-grid", default="0.02,0.03,0.05,0.075,0.10,0.15,0.20")
    g.add_argument("--clip-score", action="store_true")
    g.add_argument("--save-samples", type=int, default=1)
    g.add_argument("--n-mem", type=int, default=2000)
    g.add_argument("--mem-threshold", type=float, default=0.90)
    g.add_argument("--tmp-dir", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "cache":
        if not args.images_dir and not args.hf_dataset:
            raise SystemExit("cache needs --images-dir or --hf-dataset")
        cache_main(args); return 0
    if args.mode == "train":
        train_main(args); return 0
    if args.ckpt is None:
        cands = [Path(args.run_dir) / n for n in ("ema_best.pt", "ema_last.pt", "latest.pt")]
        found = next((c for c in cands if c.exists()), None)
        if found is None:
            raise SystemExit(f"--ckpt required (nothing found in {args.run_dir})")
        args.ckpt = str(found)
        LOG(f"using checkpoint {args.ckpt}")
    if args.mode == "sample":
        sample_main(args)
    elif args.mode == "fid":
        fid_main(args)
    elif args.mode == "sweep-ema":
        sweep_ema_main(args)
    elif args.mode == "memcheck":
        memcheck_main(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
