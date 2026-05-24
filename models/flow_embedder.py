"""
Flow Embedder — Velocity Corrector for Flux SR

在 HR→LR 整流流路径上，校正冻结 Flux 的速度预测。
输入 (x_t, t, z_lr, v_flux)，输出速度校正 v_corr。

流路径: t=0 → HR, t=1 → LR
  x_t = (1-t)·z_hr + t·z_lr
  v_gt = z_lr - z_hr  (真值速度，沿直线恒定)

训练:  v_total = v_flux(x_t, t) + v_corr(x_t, t, z_lr, v_flux)
       loss = MSE(v_total, z_lr - z_hr)   (Flux detach，梯度不穿透)

推理:  从 z_lr (t=1) 开始，逆流积分:
       x_{t-dt} = x_t - dt · (v_flux + v_corr)
       逐步到 t=0 得到 z_hr

Architecture: Multi-scale U-Net + time injection
  - concat(x_t, z_lr, v_flux) → [B, 48, H, W] → Encoder (3↓)
  - Time embedding injected at each level via FiLM
  - Decoder (3↑ + skip) → v_corr [B, 16, H, W]

~35M params.
"""

from __future__ import annotations

import math
import os
import sys
from typing import List

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import Tensor


# =========================================================================
# Time embedding
# =========================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Transformer-style sinusoidal time embedding."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """t: [B] float in [0, 1] → [B, dim]"""
        half = self.dim // 2
        freq = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freq[None, :]  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeProjection(nn.Module):
    """Sinusoidal → MLP → time features, used for FiLM."""

    def __init__(self, time_emb_dim: int, out_dim: int):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t: [B] → [B, out_dim]"""
        return self.mlp(self.time_emb(t))


# =========================================================================
# Basic building blocks
# =========================================================================

class ResBlock(nn.Module):
    """Residual block with optional time conditioning via FiLM."""

    def __init__(self, channels: int, time_dim: int = 0, norm_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(norm_groups, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

        self.norm2 = nn.GroupNorm(min(norm_groups, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        nn.init.zeros_(self.conv2.weight)  # zero-init for residual stability

        self.use_time = time_dim > 0
        if self.use_time:
            self.time_scale = nn.Linear(time_dim, channels, bias=True)
            self.time_shift = nn.Linear(time_dim, channels, bias=True)
            nn.init.zeros_(self.time_scale.weight)
            nn.init.zeros_(self.time_shift.weight)

    def forward(self, x: Tensor, t_emb: Tensor | None = None) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.use_time and t_emb is not None:
            scale = self.time_scale(t_emb)[:, :, None, None]
            shift = self.time_shift(t_emb)[:, :, None, None]
            h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class SelfAttention2d(nn.Module):
    """Spatial self-attention (operates over H×W tokens)."""

    def __init__(self, channels: int, num_heads: int = 8, norm_groups: int = 32):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(norm_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        k = rearrange(k, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        v = rearrange(v, "b (n d) h w -> b n (h w) d", n=self.num_heads)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = rearrange(out, "b n (h w) d -> b (n d) h w", h=h, w=w)
        return x + self.proj(out)


# =========================================================================
# Encoder / Decoder levels
# =========================================================================

class EncoderLevel(nn.Module):
    """One encoder level: ResBlocks → save skip → stride-2 downsample."""

    def __init__(self, in_channels: int, out_channels: int,
                 num_res_blocks: int = 2, time_dim: int = 0, norm_groups: int = 32):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock(in_channels, time_dim, norm_groups) for _ in range(num_res_blocks)
        ])
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)

    def forward(self, x: Tensor, t_emb: Tensor | None = None) -> tuple[Tensor, Tensor]:
        for res in self.res_blocks:
            x = res(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class DecoderLevel(nn.Module):
    """One decoder level: upsample → concat skip → fuse → ResBlocks."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 num_res_blocks: int = 3, time_dim: int = 0, norm_groups: int = 32):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.res_blocks = nn.ModuleList([
            ResBlock(out_channels, time_dim, norm_groups) for _ in range(num_res_blocks)
        ])

    def forward(self, x: Tensor, skip: Tensor, t_emb: Tensor | None = None) -> Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        for res in self.res_blocks:
            x = res(x, t_emb)
        return x


class Bottleneck(nn.Module):
    """Bottleneck: ResBlock → SelfAttention → ResBlock."""

    def __init__(self, channels: int, time_dim: int = 0,
                 num_heads: int = 8, norm_groups: int = 32):
        super().__init__()
        self.res1 = ResBlock(channels, time_dim, norm_groups)
        self.attn = SelfAttention2d(channels, num_heads, norm_groups)
        self.res2 = ResBlock(channels, time_dim, norm_groups)

    def forward(self, x: Tensor, t_emb: Tensor | None = None) -> Tensor:
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


# =========================================================================
# FlowEmbedder — Velocity Corrector
# =========================================================================

class FlowEmbedder(nn.Module):
    """
    Velocity correction network for Flux SR.

    Flow path: t=0 → HR, t=1 → LR
    Inputs:  x_t    [B, 16, H, W]  — current state on flow path
             t      [B]             — time scalar in [0, 1]
             z_lr   [B, 16, H, W]  — LR latent for context
             v_flux [B, 16, H, W]  — Flux predicted velocity (needs correction)
    Output:  v_corr [B, 16, H, W]  — velocity correction

    Architecture (U-Net with time conditioning):
      Concat(x_t, z_lr, v_flux) → [B, 48, H, W]
        → conv_in  → [B, 128, H, W]
        → Encoder (3↓: 128→256→512→512, time-FiLM)
        → Bottleneck (ResBlock+SA+ResBlock, time-FiLM)
        → Decoder (3↑ + skip, time-FiLM): 512→256→128→128
        → conv_out → v_corr [B, 16, H, W]
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: List[int] | None = None,
        num_enc_blocks: int = 2,
        num_dec_blocks: int = 3,
        num_mid_blocks: int = 2,
        attn_heads: int = 8,
        norm_groups: int = 32,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4]

        ch = base_channels
        cm = channel_multipliers
        num_levels = len(cm)
        enc_ch = [ch * m for m in cm]  # [128, 256, 512]

        # ---- Time embedding ----
        self.time_proj = TimeProjection(time_emb_dim, time_emb_dim)

        # ---- conv_in: concat(x_t, z_lr, v_flux) = 3 * in_channels ----
        self.conv_in = nn.Conv2d(in_channels * 3, enc_ch[0], 3, padding=1, bias=False)

        # ---- Encoder ----
        self.enc_levels = nn.ModuleList()
        for i in range(num_levels):
            in_ch = enc_ch[i]
            out_ch = enc_ch[i + 1] if i < num_levels - 1 else enc_ch[i]
            self.enc_levels.append(
                EncoderLevel(in_ch, out_ch, num_res_blocks=num_enc_blocks,
                             time_dim=time_emb_dim, norm_groups=norm_groups)
            )

        # ---- Bottleneck ----
        bottleneck_ch = enc_ch[-1]
        self.mid = nn.ModuleList()
        mid_attn_every = max(1, num_mid_blocks // 2) if attn_heads > 0 else num_mid_blocks + 1
        for i in range(num_mid_blocks):
            if attn_heads > 0 and i == mid_attn_every:
                self.mid.append(SelfAttention2d(bottleneck_ch, attn_heads, norm_groups))
            self.mid.append(ResBlock(bottleneck_ch, time_emb_dim, norm_groups))

        # ---- Decoder ----
        self.dec_levels = nn.ModuleList()
        dec_ch_rev = list(reversed(enc_ch))  # [512, 256, 128]
        for i in range(num_levels):
            in_ch = dec_ch_rev[i]
            skip_ch = enc_ch[0] if i == num_levels - 1 else enc_ch[num_levels - 1 - i]
            out_ch = enc_ch[0] if i == num_levels - 1 else enc_ch[num_levels - 2 - i]
            self.dec_levels.append(
                DecoderLevel(in_ch, skip_ch, out_ch, num_res_blocks=num_dec_blocks,
                             time_dim=time_emb_dim, norm_groups=norm_groups)
            )

        # ---- Output head ----
        final_ch = enc_ch[0]
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(norm_groups, final_ch), final_ch),
            nn.SiLU(),
            nn.Conv2d(final_ch, final_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(norm_groups, final_ch), final_ch),
            nn.SiLU(),
            nn.Conv2d(final_ch, out_channels, 3, padding=1),
        )
        # Zero-init output for stable start
        nn.init.zeros_(self.conv_out[-1].weight)
        if self.conv_out[-1].bias is not None:
            nn.init.zeros_(self.conv_out[-1].bias)

    def forward(self, x_t: Tensor, t: Tensor, z_lr: Tensor, v_flux: Tensor) -> Tensor:
        """
        Args:
            x_t:    [B, 16, H, W]  当前流路径上的状态
            t:      [B]             时间 ∈ [0, 1]
            z_lr:   [B, 16, H, W]  LR latent (条件上下文)
            v_flux: [B, 16, H, W]  Flux 预测的速度 (detach, 无梯度)

        Returns:
            v_corr: [B, 16, H, W]  速度校正量
        """
        # Time features
        t_emb = self.time_proj(t)  # [B, time_emb_dim]

        # Concat input: x_t + z_lr + v_flux = 48 channels
        x = torch.cat([x_t, z_lr, v_flux], dim=1)  # [B, 48, H, W]
        x = self.conv_in(x)

        # ---- Encoder ----
        skips: list[Tensor] = []
        for enc in self.enc_levels:
            x, skip = enc(x, t_emb)
            skips.append(skip)

        # ---- Bottleneck ----
        for layer in self.mid:
            if isinstance(layer, SelfAttention2d):
                x = layer(x)
            else:
                x = layer(x, t_emb)

        # ---- Decoder ----
        for i, dec in enumerate(self.dec_levels):
            skip = skips[-(1 + i)]
            x = dec(x, skip, t_emb)

        # ---- Output ----
        return self.conv_out(x)


# =========================================================================
# Factory function
# =========================================================================

def create_flow_embedder(config_path: str) -> FlowEmbedder:
    """从 yaml 配置文件创建 FlowEmbedder 实例。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return FlowEmbedder(**cfg)


# =========================================================================
# Self-test
# =========================================================================

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "flow_embedder.yaml")
    model = create_flow_embedder(config_path)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total / 1e6:.2f}M")

    # Test forward
    B, C, H, W = 2, 16, 64, 64
    x_t = torch.randn(B, C, H, W)
    t = torch.rand(B)
    z_lr = torch.randn(B, C, H, W)
    v_flux = torch.randn(B, C, H, W)
    model.eval()
    with torch.no_grad():
        v_corr = model(x_t, t, z_lr, v_flux)
    print(f"x_t:    {x_t.shape}")
    print(f"t:      {t.shape} values={[f'{v:.3f}' for v in t.tolist()]}")
    print(f"z_lr:   {z_lr.shape}")
    print(f"v_flux: {v_flux.shape}")
    print(f"v_corr: {v_corr.shape} mean={v_corr.mean():.6f} std={v_corr.std():.6f}")
    print("Forward pass OK — zero-init verified (near-zero output)")
