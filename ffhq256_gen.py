import argparse
import glob
import json
import math
import os
import random
import time
from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, utils as vutils
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
SCRIPT_VERSION = "v11"

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_fast_matmul() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def resolve_precision(precision: str, device: torch.device) -> str:
    if device.type == "cuda":
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            print("[warn] bf16 unsupported on this GPU; falling back to fp16.")
            return "fp16"
        return precision
    if device.type == "mps":
        return precision
    if precision == "fp16":
        print("[warn] fp16 autocast is unreliable on CPU; using fp32.")
        return "fp32"
    return precision

def amp_dtype_of(precision: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

def human_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"

def load_vae(vae_name: str, device: torch.device, dtype: torch.dtype = torch.float32):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(vae_name).to(device=device, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    return vae

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    assert embed_dim % 2 == 0
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    return np.concatenate([emb_h, emb_w], axis=1)


def build_rope_2d(head_dim: int, grid_size: int, base: float = 10000.0
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial RoPE"
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, 2, dtype=torch.float64) / half))
    coords = torch.arange(grid_size, dtype=torch.float64)
    ang = torch.outer(coords, freqs)
    ang_h = ang[:, None, :].expand(grid_size, grid_size, -1)
    ang_w = ang[None, :, :].expand(grid_size, grid_size, -1)
    full = torch.cat([ang_h, ang_w], dim=-1).reshape(grid_size * grid_size, half)
    return full.cos().float(), full.sin().float()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    c = cos[None, None]
    s = sin[None, None]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2).to(orig_dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True,
                 use_rope: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope and rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden * 2, bias=True)
        self.w3 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256,
                 max_period: float = 10000.0, time_scale: float = 1000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.time_scale = time_scale
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * self.time_scale * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t))

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 style: str = "modern"):
        super().__init__()
        self.style = style
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, use_rope=(style == "modern"))
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if style == "legacy":
            mlp_hidden = int(hidden_size * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden, hidden_size),
            )
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            mlp_hidden = int(2 * hidden_size * mlp_ratio / 3)
            mlp_hidden = 64 * ((mlp_hidden + 63) // 64)
            self.mlp = SwiGLU(hidden_size, mlp_hidden)
            self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        if self.style == "legacy":
            mods = self.adaLN_modulation(c).chunk(6, dim=-1)
        else:
            mods = (c.reshape(c.shape[0], 6, -1) + self.scale_shift_table[None]).unbind(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))

class DiT(nn.Module):
    def __init__(self, input_size: int = 32, patch_size: int = 2, in_channels: int = 4,
                 hidden_size: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, use_grad_checkpoint: bool = False,
                 style: str = "modern"):
        super().__init__()
        assert input_size % patch_size == 0
        assert hidden_size % num_heads == 0
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_grad_checkpoint = use_grad_checkpoint
        self.style = style
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size,
                                    stride=patch_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        grid = input_size // patch_size
        num_patches = grid * grid
        if style == "legacy":
            self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size),
                                 persistent=False)
        else:
            assert self.head_dim % 4 == 0, (
                f"head_dim={self.head_dim} must be divisible by 4 for axial RoPE; "
                "adjust --hidden_size / --num_heads"
            )
            cos, sin = build_rope_2d(self.head_dim, grid)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, style=style) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic_init)

        if self.style == "legacy":
            grid = self.input_size // self.patch_size
            pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid)
            with torch.no_grad():
                self.pos_embed.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.x_embedder.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.style == "legacy":
            for blk in self.blocks:
                nn.init.zeros_(blk.adaLN_modulation[-1].weight)
                nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        else:
            nn.init.zeros_(self.t_block[-1].weight)
            nn.init.zeros_(self.t_block[-1].bias)
            for blk in self.blocks:
                nn.init.zeros_(blk.scale_shift_table)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c, p = self.out_channels, self.patch_size
        h = w = self.input_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        t_emb = self.t_embedder(t)

        if self.style == "legacy":
            x = x + self.pos_embed
            cond, rope = t_emb, None
        else:
            cond = self.t_block(t_emb)
            rope = (self.rope_cos, self.rope_sin)

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cond, rope, use_reentrant=False)
            else:
                x = blk(x, cond, rope)

        x = self.final_layer(x, t_emb)
        return self.unpatchify(x)

ARCH_PRESETS = {
    "legacy_s": dict(hidden_size=384, depth=12, num_heads=6,  style="legacy"),
    "s":        dict(hidden_size=448, depth=14, num_heads=7,  style="modern"),   # ~36M
    "m":        dict(hidden_size=640, depth=16, num_heads=10, style="modern"),   # ~83M
    "b":        dict(hidden_size=768, depth=20, num_heads=12, style="modern"),   # ~147M
}

def build_model(args: argparse.Namespace, grad_checkpoint: bool = False) -> DiT:
    preset = dict(ARCH_PRESETS[args.arch])
    if args.hidden_size is not None:
        preset["hidden_size"] = args.hidden_size
    if args.depth is not None:
        preset["depth"] = args.depth
    if args.num_heads is not None:
        preset["num_heads"] = args.num_heads
    return DiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=4,
        mlp_ratio=args.mlp_ratio,
        use_grad_checkpoint=grad_checkpoint,
        **preset,
    )

def arch_tag(args: argparse.Namespace, model: DiT) -> str:
    return (f"{SCRIPT_VERSION}|{args.arch}|style={model.style}|h={model.hidden_size}"
            f"|d={len(model.blocks)}|nh={model.num_heads}|p={model.patch_size}"
            f"|in={model.input_size}|mlp={args.mlp_ratio}")

class EMA:
    def __init__(self, model: nn.Module, decays: Sequence[float] = (0.9999,)):
        self.decays = [float(d) for d in decays]
        self.num_updates = 0
        sd = model.state_dict()
        self.keys = list(sd.keys())
        self.float_keys = [k for k in self.keys if sd[k].is_floating_point()]
        self.other_keys = [k for k in self.keys if not sd[k].is_floating_point()]
        self.shadows = [
            {k: (sd[k].detach().clone().float() if sd[k].is_floating_point()
                 else sd[k].detach().clone()) for k in self.keys}
            for _ in self.decays
        ]
        self._src_cache = None

    def _sources(self, model: nn.Module):
        if self._src_cache is None:
            sd = model.state_dict()
            self._src_cache = ([sd[k] for k in self.float_keys], sd)
        return self._src_cache

    def current_decays(self) -> List[float]:
        n = max(self.num_updates, 1)
        return [min(d, n / (n + 1.0)) for d in self.decays]

    @torch.no_grad()
    def update(self, model: nn.Module, steps: int = 1) -> None:
        self.num_updates += steps
        src_float, sd = self._sources(model)
        for decay, shadow in zip(self.current_decays(), self.shadows):
            d = decay ** steps
            tgt = [shadow[k] for k in self.float_keys]
            torch._foreach_mul_(tgt, d)
            torch._foreach_add_(tgt, src_float, alpha=1.0 - d)
            for k in self.other_keys:
                shadow[k].copy_(sd[k])

    @torch.no_grad()
    def copy_to(self, model: nn.Module, index: int = 0) -> None:
        msd = model.state_dict()
        for k, v in self.shadows[index].items():
            msd[k].copy_(v.to(msd[k].dtype))

    def state_dict(self) -> dict:
        return {"shadows": self.shadows, "decays": self.decays, "num_updates": self.num_updates}

    def load_state_dict(self, sd: dict) -> None:
        self.num_updates = int(sd["num_updates"])
        if "shadows" in sd:
            loaded_decays = [float(d) for d in sd["decays"]]
            loaded = sd["shadows"]
        else:
            loaded_decays, loaded = [float(sd.get("decay", 0.9999))], [sd["shadow"]]
        for i, d in enumerate(self.decays):
            j = loaded_decays.index(d) if d in loaded_decays else 0
            self.shadows[i] = {
                k: (v.detach().clone().float() if v.is_floating_point() else v.detach().clone())
                for k, v in loaded[j].items()
            }
        self._src_cache = None

def sample_timesteps(batch_size: int, device: torch.device, logit_normal: bool = True,
                     m: float = 0.0, s: float = 1.0) -> torch.Tensor:
    if logit_normal:
        t = torch.sigmoid(torch.randn(batch_size, device=device) * s + m)
    else:
        t = torch.rand(batch_size, device=device)
    return t.clamp(1e-5, 1.0 - 1e-5)

def flow_matching_loss(model: nn.Module, x0: torch.Tensor, logit_normal: bool = True,
                       logit_mean: float = 0.0, logit_std: float = 1.0) -> torch.Tensor:
    b = x0.shape[0]
    t = sample_timesteps(b, x0.device, logit_normal, logit_mean, logit_std)
    noise = torch.randn_like(x0)
    t_ = t.view(b, 1, 1, 1)
    xt = (1.0 - t_) * x0 + t_ * noise
    target = noise - x0
    v_pred = model(xt, t)
    return F.mse_loss(v_pred.float(), target.float())


def time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def make_timesteps(num_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    return time_shift(shift, t)


@torch.no_grad()
def ode_sample(model: nn.Module, shape: tuple, num_steps: int = 25, solver: str = "heun",
               t_shift: float = 1.0, device: Optional[torch.device] = None,
               generator: Optional[torch.Generator] = None,
               autocast_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    x = torch.randn(shape, device=device, generator=generator)
    ts = make_timesteps(num_steps, t_shift, device)

    use_amp = autocast_dtype is not None and device.type == "cuda"
    ctx = (torch.autocast(device_type=device.type, dtype=autocast_dtype)
           if use_amp else torch.autocast(device_type="cpu", enabled=False))

    with ctx:
        for i in range(num_steps):
            t0, t1 = ts[i], ts[i + 1]
            dt = t1 - t0
            v0 = model(x, t0.expand(shape[0])).float()
            if solver == "euler" or i == num_steps - 1:
                x = x + v0 * dt
            else:
                x_pred = x + v0 * dt
                v1 = model(x_pred, t1.expand(shape[0])).float()
                x = x + 0.5 * (v0 + v1) * dt
    if was_training:
        model.train()
    return x


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, scaling_factor: float,
                   batch_size: int = 16) -> torch.Tensor:
    vae_dtype = next(vae.parameters()).dtype
    out = []
    for i in range(0, latents.shape[0], batch_size):
        chunk = latents[i:i + batch_size] / scaling_factor
        img = vae.decode(chunk.to(vae_dtype)).sample
        out.append(((img.float().clamp(-1, 1) + 1) / 2))
    return torch.cat(out, dim=0)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

class ImageFolderFlat(Dataset):
    def __init__(self, root: str, resolution: int, hflip: bool = False):
        self.paths = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.paths.append(os.path.join(dirpath, fname))
        self.paths.sort()
        if not self.paths:
            raise ValueError(f"No images found under '{root}'")
        ops = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
        ]
        if hflip:
            ops.append(transforms.RandomHorizontalFlip(p=1.0))
        ops += [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), idx
        except Exception as e:
            print(f"[warn] failed to load '{self.paths[idx]}': {e}")
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class LatentStore:
    def __init__(self, cache_dir: str, device: torch.device, use_flip: bool = True,
                 pin: bool = True, to_gpu: bool = False):
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"'{meta_path}' not found. Run `cache` mode first "
                f"(or `cache --consolidate_legacy` if you have a v10 per-file cache)."
            )
        self.meta = json.load(open(meta_path))
        self.format = self.meta.get("format", "latents")
        self.scaling_factor = float(self.meta["scaling_factor"])
        self.latent_size = int(self.meta["latent_size"])
        self.device = device

        main = os.path.join(cache_dir, "latents.npy")
        if not os.path.exists(main):
            raise FileNotFoundError(f"'{main}' not found. Re-run `cache` mode.")
        self.data = [self._load(main, pin, to_gpu, device)]

        flip_path = os.path.join(cache_dir, "latents_flip.npy")
        self.use_flip = bool(use_flip and self.meta.get("flip", False) and os.path.exists(flip_path))
        if self.use_flip:
            self.data.append(self._load(flip_path, pin, to_gpu, device))
        elif use_flip and not self.meta.get("flip", False):
            print("[data] cache has no flipped copy; training without h-flip augmentation.")

        self.n = self.data[0].shape[0]
        nbytes = sum(t.numel() * t.element_size() for t in self.data)
        print(f"[data] {self.n} latents  format={self.format}  flip={self.use_flip}  "
              f"resident={human_bytes(nbytes)}  device={self.data[0].device}")

    @staticmethod
    def _load(path: str, pin: bool, to_gpu: bool, device: torch.device) -> torch.Tensor:
        arr = np.load(path, mmap_mode="r")
        t = torch.from_numpy(np.array(arr))
        if to_gpu:
            return t.to(device)
        if pin:
            try:
                t = t.pin_memory()
            except Exception as e:
                print(f"[warn] could not pin latent memory ({e}); continuing unpinned.")
        return t

    def batch(self, idx: torch.Tensor, flip_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_flip and flip_mask is not None:
            a = self.data[0][idx[~flip_mask]]
            b = self.data[1][idx[flip_mask]]
            raw = torch.empty((idx.shape[0],) + tuple(a.shape[1:]), dtype=a.dtype)
            raw[~flip_mask] = a
            raw[flip_mask] = b
        else:
            raw = self.data[0][idx]

        raw = raw.to(self.device, non_blocking=True).float()
        if self.format == "moments":
            mean, std = raw.chunk(2, dim=1)
            z = mean + std * torch.randn_like(mean)
        else:
            z = raw
        return z * self.scaling_factor

    def epoch_indices(self, batch_size: int, generator: torch.Generator):
        perm = torch.randperm(self.n, generator=generator)
        for i in range(0, self.n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            mask = (torch.rand(batch_size, generator=generator) < 0.5) if self.use_flip else None
            yield idx, mask

    def steps_per_epoch(self, batch_size: int) -> int:
        return self.n // batch_size

def _open_memmap(path: str, shape: tuple, dtype=np.float16):
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


@torch.no_grad()
def _encode_split(vae, dataset: Dataset, out_path: str, args: argparse.Namespace,
                  device: torch.device, channels: int, desc: str) -> None:
    n = len(dataset)
    lat = args.resolution // 8
    progress_path = out_path + ".progress"
    start = 0
    if os.path.exists(out_path) and os.path.exists(progress_path):
        try:
            start = int(open(progress_path).read().strip())
            mm = np.lib.format.open_memmap(out_path, mode="r+")
            if mm.shape != (n, channels, lat, lat):
                start, mm = 0, None
        except Exception:
            start, mm = 0, None
    else:
        mm = None
    if mm is None:
        mm = _open_memmap(out_path, (n, channels, lat, lat))
    if start >= n:
        print(f"[cache] {desc}: already complete ({n} items).")
        return

    loader = DataLoader(
        Subset(dataset, list(range(start, n))), batch_size=args.cache_batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate_skip_none,
    )
    cursor = start
    amp = args.cache_precision != "fp32"
    for batch in tqdm(loader, desc=desc):
        if batch is None:
            continue
        imgs, _ = batch
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype_of(args.cache_precision),
                            enabled=amp):
            dist = vae.encode(imgs).latent_dist
            out = (torch.cat([dist.mean, dist.std], dim=1) if channels == 8
                   else dist.sample())
        arr = out.float().cpu().numpy().astype(np.float16)
        mm[cursor:cursor + arr.shape[0]] = arr
        cursor += arr.shape[0]
        with open(progress_path, "w") as f:
            f.write(str(cursor))
    mm.flush()
    del mm
    os.remove(progress_path)


def cache_latents(args: argparse.Namespace) -> None:
    device = get_device()
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.consolidate_legacy:
        consolidate_legacy_cache(args)
        return

    print(f"[cache] loading VAE '{args.vae_name}' on {device} ...")
    vae = load_vae(args.vae_name, device, torch.float32)
    scaling_factor = float(vae.config.scaling_factor)
    channels = 8 if args.cache_moments else 4

    ds = ImageFolderFlat(args.data_dir, args.resolution, hflip=False)
    print(f"[cache] {len(ds)} images  ->  {channels}ch {args.resolution // 8}x{args.resolution // 8} "
          f"({'mean+std moments' if channels == 8 else 'sampled latents'})")
    _encode_split(vae, ds, os.path.join(args.cache_dir, "latents.npy"),
                  args, device, channels, "Encoding")

    if args.cache_flip:
        ds_f = ImageFolderFlat(args.data_dir, args.resolution, hflip=True)
        _encode_split(vae, ds_f, os.path.join(args.cache_dir, "latents_flip.npy"),
                      args, device, channels, "Encoding (h-flip)")

    meta = {
        "format": "moments" if channels == 8 else "latents",
        "n": len(ds), "channels": channels, "latent_size": args.resolution // 8,
        "resolution": args.resolution, "scaling_factor": scaling_factor,
        "vae_name": args.vae_name, "flip": bool(args.cache_flip),
        "script_version": SCRIPT_VERSION,
    }
    with open(os.path.join(args.cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.cache_dir, "index_map.json"), "w") as f:
        json.dump({str(i): p for i, p in enumerate(ds.paths)}, f)
    print(f"[cache] done -> '{args.cache_dir}'")


def consolidate_legacy_cache(args: argparse.Namespace) -> None:
    shards = sorted(glob.glob(os.path.join(args.cache_dir, "[0-9]" * 7 + ".npy")))
    if not shards:
        raise FileNotFoundError(f"No legacy shards (0000000.npy ...) in '{args.cache_dir}'")
    probe = np.load(shards[0])
    out = _open_memmap(os.path.join(args.cache_dir, "latents.npy"),
                       (len(shards),) + probe.shape)
    for i, p in enumerate(tqdm(shards, desc="Consolidating")):
        out[i] = np.load(p)
    out.flush()
    del out

    old_meta = {}
    old_path = os.path.join(args.cache_dir, "meta.json")
    if os.path.exists(old_path):
        try:
            old_meta = json.load(open(old_path))
        except Exception:
            pass
    meta = {
        "format": "latents", "n": len(shards), "channels": int(probe.shape[0]),
        "latent_size": int(probe.shape[-1]),
        "resolution": old_meta.get("resolution", args.resolution),
        "scaling_factor": float(old_meta.get("scaling_factor", 0.18215)),
        "vae_name": old_meta.get("vae_name", args.vae_name), "flip": False,
        "script_version": SCRIPT_VERSION, "consolidated_from_legacy": True,
    }
    with open(old_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] consolidated {len(shards)} shards -> latents.npy "
          f"({human_bytes(os.path.getsize(os.path.join(args.cache_dir, 'latents.npy')))})")
    print("[cache] the per-file shards are now redundant and can be deleted.")

class FIDEvaluator:
    def __init__(self, args: argparse.Namespace, device: torch.device):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.device = device
        self.args = args
        self.metric = FrechetInceptionDistance(
            feature=2048, normalize=True, reset_real_features=False
        ).to(device)
        self.cache_path = os.path.join(
            args.output_dir, f"fid_real_stats_{args.fid_real_samples}_{args.resolution}.pt"
        )
        self.ready = False

    def _restore_reals(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            blob = torch.load(self.cache_path, map_location=self.device, weights_only=False)
            self.metric.real_features_sum = blob["sum"].to(self.device)
            self.metric.real_features_cov_sum = blob["cov"].to(self.device)
            self.metric.real_features_num_samples = blob["n"].to(self.device)
            print(f"[fid] restored real statistics ({int(blob['n'])} images) from cache")
            return True
        except Exception as e:
            print(f"[fid] could not restore cached real stats ({e}); recomputing.")
            return False

    def prepare_reals(self) -> None:
        if self.ready:
            return
        if self._restore_reals():
            self.ready = True
            return
        ds = ImageFolderFlat(self.args.data_dir, self.args.resolution)
        n = min(self.args.fid_real_samples, len(ds))
        g = torch.Generator().manual_seed(self.args.fid_seed)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        loader = DataLoader(Subset(ds, idx), batch_size=self.args.fid_batch_size,
                            shuffle=False, num_workers=self.args.num_workers,
                            collate_fn=collate_skip_none)
        for batch in tqdm(loader, desc="FID reals", leave=False):
            if batch is None:
                continue
            imgs, _ = batch
            self.metric.update(((imgs.to(self.device) + 1) / 2).clamp(0, 1), real=True)
        try:
            torch.save({"sum": self.metric.real_features_sum.cpu(),
                        "cov": self.metric.real_features_cov_sum.cpu(),
                        "n": self.metric.real_features_num_samples.cpu()}, self.cache_path)
        except Exception as e:
            print(f"[fid] warning: could not cache real stats ({e})")
        self.ready = True

    @torch.no_grad()
    def score(self, model: nn.Module, vae, scaling_factor: float, num_samples: int,
              amp_dtype: Optional[torch.dtype] = None) -> float:
        self.prepare_reals()
        self.metric.reset()
        latent_res = self.args.resolution // 8
        gen = (torch.Generator(device=self.device).manual_seed(self.args.fid_seed)
               if self.args.fid_fixed_noise else None)
        done = 0
        with tqdm(total=num_samples, desc="FID fakes", leave=False) as pbar:
            while done < num_samples:
                bs = min(self.args.fid_batch_size, num_samples - done)
                z = ode_sample(model, (bs, 4, latent_res, latent_res),
                               num_steps=self.args.sample_steps, solver=self.args.solver,
                               t_shift=self.args.t_shift, device=self.device,
                               generator=gen, autocast_dtype=amp_dtype)
                imgs = decode_latents(vae, z, scaling_factor, self.args.decode_batch_size)
                self.metric.update(imgs.clamp(0, 1), real=False)
                done += bs
                pbar.update(bs)
        return float(self.metric.compute())

    @torch.no_grad()
    def vae_floor(self, vae, num_samples: int = 5000) -> float:
        self.prepare_reals()
        self.metric.reset()
        ds = ImageFolderFlat(self.args.data_dir, self.args.resolution)
        g = torch.Generator().manual_seed(self.args.fid_seed + 1)
        idx = torch.randperm(len(ds), generator=g)[:min(num_samples, len(ds))].tolist()
        loader = DataLoader(Subset(ds, idx), batch_size=self.args.decode_batch_size,
                            shuffle=False, num_workers=self.args.num_workers,
                            collate_fn=collate_skip_none)
        vae_dtype = next(vae.parameters()).dtype
        for batch in tqdm(loader, desc="VAE round-trip", leave=False):
            if batch is None:
                continue
            imgs, _ = batch
            imgs = imgs.to(self.device, dtype=vae_dtype)
            rec = vae.decode(vae.encode(imgs).latent_dist.sample()).sample
            self.metric.update(((rec.float().clamp(-1, 1) + 1) / 2), real=False)
        return float(self.metric.compute())

def save_checkpoint(path: str, model: nn.Module, ema: EMA, optimizer, scaler,
                    step: int, epoch: int, args: argparse.Namespace, tag: str,
                    best_fid: float) -> None:
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step, "epoch": epoch, "args": vars(args),
        "arch_tag": tag, "best_fid": best_fid, "script_version": SCRIPT_VERSION,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

def load_checkpoint(path: str, model: nn.Module, ema: EMA, optimizer, scaler,
                    device: torch.device, tag: str, strict_arch: bool = True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    old_tag = ckpt.get("arch_tag")
    if old_tag is not None and old_tag != tag:
        msg = (f"architecture mismatch:\n  checkpoint: {old_tag}\n  current:    {tag}")
        if strict_arch:
            raise RuntimeError(msg + "\nPass --no-strict_arch to force, or match the flags.")
        print(f"[warn] {msg}\n[warn] loading anyway (--no_strict_arch).")
    elif old_tag is None:
        print("[warn] checkpoint predates arch_tag versioning; assuming --arch legacy_s.")
    model.load_state_dict(ckpt["model"])
    try:
        ema.load_state_dict(ckpt["ema"])
    except Exception as e:
        print(f"[warn] could not restore EMA ({e}); re-seeding EMA from current weights.")
        ema.__init__(model, ema.decays)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt["epoch"], float(ckpt.get("best_fid", float("inf")))


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")), key=os.path.getmtime)
    return cands[-1] if cands else None


def get_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int,
           min_lr_ratio: float = 0.1, schedule: str = "cosine") -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return base_lr
    total_steps = max(total_steps, warmup_steps + 1)
    p = min(max((step - warmup_steps) / max(1, total_steps - warmup_steps), 0.0), 1.0)
    if schedule == "linear":
        factor = 1.0 - p
    else:
        factor = 0.5 * (1 + math.cos(math.pi * p))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * factor)


@torch.no_grad()
def generate_sample_grid(model: nn.Module, vae, args: argparse.Namespace,
                         scaling_factor: float, device: torch.device, step: int,
                         out_dir: str, amp_dtype: Optional[torch.dtype], suffix: str = "") -> str:
    n = args.num_sample_images
    latent_res = args.resolution // 8
    gen = torch.Generator(device=device).manual_seed(args.seed)
    z = ode_sample(model, (n, 4, latent_res, latent_res), num_steps=args.sample_steps,
                   solver=args.solver, t_shift=args.t_shift, device=device,
                   generator=gen, autocast_dtype=amp_dtype)
    imgs = decode_latents(vae, z, scaling_factor, args.decode_batch_size)
    grid = vutils.make_grid(imgs.cpu(), nrow=max(1, int(math.sqrt(n))))
    path = os.path.join(out_dir, f"samples_step{step:08d}{suffix}.png")
    vutils.save_image(grid, path)
    return path

def train(args: argparse.Namespace) -> None:
    device = get_device()
    set_seed(args.seed)
    enable_fast_matmul()
    samples_dir = os.path.join(args.output_dir, "samples")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "metrics.jsonl")
    precision = resolve_precision(args.precision, device)
    amp_dtype = amp_dtype_of(precision)
    use_amp = precision != "fp32"
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    sample_amp = None
    if use_amp and args.amp_sampling:
        sample_amp = (torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported())
                      else (amp_dtype if amp_dtype is torch.bfloat16 else None))

    store = LatentStore(args.cache_dir, device, use_flip=args.hflip,
                        pin=args.pin_latents, to_gpu=args.latents_on_gpu)
    steps_per_epoch = store.steps_per_epoch(args.batch_size) // args.grad_accum_steps
    if steps_per_epoch < 1:
        raise ValueError("batch_size * grad_accum_steps exceeds the dataset size")

    model = build_model(args, grad_checkpoint=args.grad_checkpoint).to(device)
    tag = arch_tag(args, model)
    n_params = model.num_parameters()
    print(f"[train] device={device}  precision={precision}  arch={args.arch}  "
          f"params={n_params / 1e6:.1f}M")
    print(f"[train] arch_tag = {tag}")

    ema = EMA(model, decays=args.ema_decays)
    ema_model = build_model(args, grad_checkpoint=False).to(device)
    ema_model.eval().requires_grad_(False)

    fused_ok = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=tuple(args.betas), eps=1e-8,
        weight_decay=args.weight_decay, **({"fused": True} if fused_ok else {}),
    )

    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    if total_steps <= 0:
        raise ValueError("Nothing to do: set --max_steps > 0 (or --epochs > 0).")
    lr_total = args.lr_total_steps if args.lr_total_steps > 0 else total_steps
    if lr_total != total_steps:
        print(f"[train] LR cosine annealed over {lr_total} steps (budget {total_steps}).")

    global_step, start_epoch, best_fid = 0, 0, float("inf")
    if args.resume:
        rp = find_latest_checkpoint(ckpt_dir) if args.resume == "auto" else args.resume
        if rp and os.path.isfile(rp):
            global_step, start_epoch, best_fid = load_checkpoint(
                rp, model, ema, optimizer, scaler, device, tag, strict_arch=args.strict_arch)
            print(f"[resume] '{rp}': step={global_step} epoch={start_epoch} best_fid={best_fid:.3f}")
            if args.reset_ema:
                ema = EMA(model, decays=args.ema_decays)
                print("[resume] EMA re-seeded from the current weights (--reset_ema).")
        else:
            print("[resume] no checkpoint found; starting fresh.")

    net = torch.compile(model) if args.compile else model
    if args.compile:
        print("[train] torch.compile enabled (first ~100 steps will be slow).")

    vae = load_vae(args.vae_name, device, torch.bfloat16 if use_amp else torch.float32)
    fid_eval = FIDEvaluator(args, device) if args.fid_every_steps > 0 else None

    print(f"[train] {store.n} latents | {steps_per_epoch} opt-steps/epoch | "
          f"budget {total_steps} steps ({total_steps * args.batch_size * args.grad_accum_steps / 1e6:.1f}M images)")
    print(f"[train] EMA horizons {args.ema_decays} -> windows "
          f"{[int(1 / (1 - d)) for d in args.ema_decays]} steps")

    model.train()
    data_gen = torch.Generator().manual_seed(args.seed + start_epoch)
    loss_acc = torch.zeros((), device=device)
    n_acc, grad_norm = 0, torch.zeros((), device=device)
    t_last = time.time()
    micro = 0
    epoch = start_epoch
    stop = False

    for epoch in range(start_epoch, 10 ** 9):
        if stop:
            break
        optimizer.zero_grad(set_to_none=True)
        for idx, flip_mask in store.epoch_indices(args.batch_size, data_gen):
            x0 = store.batch(idx, flip_mask)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = flow_matching_loss(net, x0, args.logit_normal_t,
                                          args.logit_mean, args.logit_std)
            (scaler.scale(loss / args.grad_accum_steps) if use_scaler
             else loss / args.grad_accum_steps).backward()

            loss_acc += loss.detach()
            micro += 1
            if micro % args.grad_accum_steps != 0:
                continue

            lr = get_lr(global_step, lr_total, args.lr, args.warmup_steps,
                        args.min_lr_ratio, args.lr_schedule)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            n_acc += 1

            if global_step % args.ema_every == 0:
                ema.update(model, steps=args.ema_every)

            if global_step % args.log_every == 0:
                avg_loss = (loss_acc / max(n_acc * args.grad_accum_steps, 1)).item()
                dt = max(time.time() - t_last, 1e-6)
                ips = n_acc * args.batch_size * args.grad_accum_steps / dt
                d_eff = ema.current_decays()[0]
                print(f"step {global_step}/{total_steps}  ep {epoch}  loss {avg_loss:.4f}  "
                      f"lr {lr:.2e}  gnorm {grad_norm.item():.3f}  {ips:.1f} img/s  "
                      f"ema_win {1 / (1 - d_eff):.0f}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": global_step, "epoch": epoch, "loss": avg_loss,
                                        "lr": lr, "grad_norm": grad_norm.item(),
                                        "imgs_per_sec": ips, "ema_decay": d_eff}) + "\n")
                loss_acc = torch.zeros((), device=device)
                n_acc = 0
                t_last = time.time()

            if args.sample_every > 0 and global_step % args.sample_every == 0:
                ema.copy_to(ema_model, 0)
                p = generate_sample_grid(ema_model, vae, args, store.scaling_factor,
                                         device, global_step, samples_dir, sample_amp)
                print(f"[sample] {p}")
                if args.sample_raw_too:
                    generate_sample_grid(model, vae, args, store.scaling_factor, device,
                                         global_step, samples_dir, sample_amp, suffix="_raw")
                model.train()
                loss_acc, n_acc, t_last = torch.zeros((), device=device), 0, time.time()

            if args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), model, ema, optimizer,
                                scaler, global_step, epoch, args, tag, best_fid)
            if args.snapshot_every > 0 and global_step % args.snapshot_every == 0:
                save_checkpoint(os.path.join(ckpt_dir, f"step_{global_step:08d}.pt"), model,
                                ema, optimizer, scaler, global_step, epoch, args, tag, best_fid)

            if fid_eval is not None and global_step % args.fid_every_steps == 0:
                scores = []
                for i, d in enumerate(args.ema_decays):
                    ema.copy_to(ema_model, i)
                    s = fid_eval.score(ema_model, vae, store.scaling_factor,
                                       args.fid_samples, sample_amp)
                    scores.append(s)
                    print(f"[fid] step {global_step}  EMA {d}  FID({args.fid_samples}) = {s:.3f}")
                    if not args.fid_all_emas:
                        break
                best_i = int(np.argmin(scores))
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": global_step, "epoch": epoch,
                                        "fid": scores, "ema_decays": args.ema_decays[:len(scores)],
                                        "fid_samples": args.fid_samples}) + "\n")
                if scores[best_i] < best_fid:
                    best_fid = scores[best_i]
                    save_checkpoint(os.path.join(ckpt_dir, "best_fid.pt"), model, ema, optimizer,
                                    scaler, global_step, epoch, args, tag, best_fid)
                    print(f"[fid] new best ({best_fid:.3f}) -> best_fid.pt")
                model.train()
                loss_acc, n_acc, t_last = torch.zeros((), device=device), 0, time.time()

            if global_step >= total_steps:
                stop = True
                break

    save_checkpoint(os.path.join(ckpt_dir, "final.pt"), model, ema, optimizer, scaler,
                    global_step, epoch, args, tag, best_fid)
    print(f"[train] finished at step {global_step}. best FID so far: {best_fid:.3f}")

    if fid_eval is not None and args.final_fid_samples > 0:
        for i, d in enumerate(args.ema_decays):
            ema.copy_to(ema_model, i)
            s = fid_eval.score(ema_model, vae, store.scaling_factor,
                               args.final_fid_samples, sample_amp)
            print(f"[fid] FINAL EMA {d}  FID({args.final_fid_samples}) = {s:.3f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": global_step, "final_fid": s, "ema_decay": d,
                                    "fid_samples": args.final_fid_samples}) + "\n")
            if not args.fid_all_emas:
                break

def sample_mode(args: argparse.Namespace) -> None:
    device = get_device()
    set_seed(args.seed)
    enable_fast_matmul()
    out_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    targs = argparse.Namespace(**ckpt["args"])
    if not hasattr(targs, "arch"):                       # v10 checkpoint
        targs.arch = "legacy_s"
        targs.mlp_ratio = getattr(targs, "mlp_ratio", 4.0)
        print("[sample] checkpoint predates arch presets; assuming --arch legacy_s.")
    for k in ("sample_steps", "solver", "t_shift", "decode_batch_size", "resolution"):
        setattr(targs, k, getattr(args, k, getattr(targs, k, None)))

    model = build_model(targs, grad_checkpoint=False).to(device)
    if args.use_raw_weights:
        model.load_state_dict(ckpt["model"])
        which = "raw weights"
    else:
        ema = EMA(model, decays=ckpt["ema"].get("decays", [0.9999]))
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model, min(args.ema_index, len(ema.decays) - 1))
        which = f"EMA[{args.ema_index}] decay={ema.decays[min(args.ema_index, len(ema.decays) - 1)]}"
    model.eval()
    print(f"[sample] step={ckpt.get('step')}  using {which}  solver={args.solver} "
          f"steps={args.sample_steps} shift={args.t_shift}")

    precision = resolve_precision(args.precision, device)
    amp = amp_dtype_of(precision) if (precision != "fp32" and args.amp_sampling) else None
    vae = load_vae(targs.vae_name, device, torch.bfloat16 if amp is not None else torch.float32)
    scaling_factor = float(vae.config.scaling_factor)

    latent_res = targs.resolution // 8
    saved, done = [], 0
    while done < args.num_samples:
        bs = min(args.gen_batch_size, args.num_samples - done)
        z = ode_sample(model, (bs, 4, latent_res, latent_res), num_steps=args.sample_steps,
                       solver=args.solver, t_shift=args.t_shift, device=device,
                       autocast_dtype=amp)
        imgs = decode_latents(vae, z, scaling_factor, args.decode_batch_size)
        for img in imgs:
            p = os.path.join(out_dir, f"sample_{len(saved):05d}.png")
            vutils.save_image(img.cpu(), p)
            saved.append(p)
        done += bs

    k = min(64, len(saved))
    if k:
        tt = transforms.ToTensor()
        grid = vutils.make_grid(torch.stack([tt(Image.open(p)) for p in saved[:k]]),
                                nrow=max(1, int(math.sqrt(k))))
        vutils.save_image(grid, os.path.join(out_dir, "grid.png"))
    print(f"[sample] {len(saved)} images -> '{out_dir}'")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"FFHQ-256 Latent Rectified-Flow (DiT + Flow Matching) {SCRIPT_VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", choices=["cache", "train", "sample", "doctor"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--vae_name", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./runs/ffhq_v11")
    p.add_argument("--cache_dir", type=str, default="./latent_cache")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_batch_size", type=int, default=32)
    p.add_argument("--cache_precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--cache_moments", action=argparse.BooleanOptionalAction, default=True,
                   help="Store VAE mean+std (8ch) so a fresh posterior sample is drawn each epoch.")
    p.add_argument("--cache_flip", action=argparse.BooleanOptionalAction, default=True,
                   help="Also encode the horizontally flipped image (exact flip augmentation).")
    p.add_argument("--consolidate_legacy", action="store_true",
                   help="Merge a v10 per-file latent cache into latents.npy and exit.")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=0, help="Ignored when --max_steps > 0.")
    p.add_argument("--max_steps", type=int, default=250000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--lr_schedule", choices=["cosine", "linear", "constant"], default="cosine")
    p.add_argument("--lr_total_steps", type=int, default=0,
                   help="Anneal the LR over this many steps instead of the full budget.")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pin_latents", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--latents_on_gpu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--amp_sampling", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--arch", choices=list(ARCH_PRESETS.keys()), default="s")
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--logit_normal_t", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--ema_decays", type=float, nargs="+", default=[0.9995, 0.9999])
    p.add_argument("--ema_every", type=int, default=1)
    p.add_argument("--reset_ema", action="store_true",
                   help="On resume, restart the EMA from the current weights "
                        "(use this once when migrating off a v10 checkpoint).")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--sample_raw_too", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--snapshot_every", type=int, default=25000)
    p.add_argument("--num_sample_images", type=int, default=16)
    p.add_argument("--resume", type=str, default=None, help="'auto' or a path.")
    p.add_argument("--strict_arch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fid_every_steps", type=int, default=20000)
    p.add_argument("--fid_samples", type=int, default=10000)
    p.add_argument("--final_fid_samples", type=int, default=50000)
    p.add_argument("--fid_real_samples", type=int, default=50000)
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_seed", type=int, default=1234)
    p.add_argument("--fid_fixed_noise", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fid_all_emas", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--decode_batch_size", type=int, default=16)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--solver", choices=["euler", "heun"], default="heun")
    p.add_argument("--t_shift", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--use_raw_weights", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ema_index", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.resolution % 8 != 0:
        raise ValueError("--resolution must be divisible by 8")
    args.ema_decays = [float(d) for d in args.ema_decays]

    if args.mode == "cache":
        if not args.data_dir and not args.consolidate_legacy:
            raise ValueError("--data_dir is required for `cache`")
        cache_latents(args)
    elif args.mode == "train":
        if args.fid_every_steps > 0 and not args.data_dir:
            raise ValueError("--data_dir is required for FID; pass --fid_every_steps 0 to skip.")
        train(args)
    elif args.mode == "sample":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for `sample`")
        sample_mode(args)

if __name__ == "__main__":
    main()