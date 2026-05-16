"""
Flow Embedder Δ_φ — 基于 Flux SingleStreamBlock 的轻量 DiT

将 LR latent 嵌入到预训练 T2I 的噪声→HR 整流流中。
输入 z_LR (latent) + timestep t，输出残差 Δ，使得 z_LR + Δ ≈ z_HR。
复用 Flux 的 RoPE、timestep embedding、SingleStreamBlock、LastLayer。
"""

from __future__ import annotations

import os
import sys

# 确保项目根目录在 sys.path 中，使 `flux` 包可被导入
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import yaml
from torch import Tensor, nn

from flux.modules.layers import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


class FlowEmbedder(nn.Module):
    """
    轻量 DiT，将 LR latent 嵌入到预训练 T2I 的整流流中。

    前向流程:
        z_LR [B,16,H,W] → pack [B,seq,64]
          → img_in → SingleStreamBlock × depth → LastLayer
        → unpack [B,16,H,W] = Δ
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        hidden_size: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        axes_dim: list[int] | None = None,
        theta: int = 10_000,
        qkv_bias: bool = True,
    ):
        super().__init__()
        if axes_dim is None:
            axes_dim = [4, 30, 30]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth

        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(
                f"axes_dim {axes_dim} 之和应等于 pe_dim={pe_dim} "
                f"(hidden_size/num_heads = {hidden_size}/{num_heads})"
            )

        # ---- 输入投影 ----
        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)

        # ---- 时间步条件 ----
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)

        # ---- RoPE 位置编码 ----
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=theta, axes_dim=axes_dim,
        )

        # ---- Transformer 主体 ----
        self.blocks = nn.ModuleList([
            SingleStreamBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        # ---- 输出投影 ----
        self.final_layer = LastLayer(
            hidden_size, 1, out_channels,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # 权重初始化：残差输出层零初始化，确保训练初期 Δ ≈ 0
    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        adaLN_linear = self.final_layer.adaLN_modulation[-1]
        assert isinstance(adaLN_linear, nn.Linear)
        nn.init.zeros_(adaLN_linear.weight)
        nn.init.zeros_(adaLN_linear.bias)

    # ------------------------------------------------------------------
    # Pack / Unpack — 与 Flux 的 sampling.py 保持一致
    # ------------------------------------------------------------------
    @staticmethod
    def pack(x: Tensor) -> Tensor:
        """[B, C, H, W] → [B, (H/2)*(W/2), C*4]   (patch_size=2)"""
        from einops import rearrange
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def unpack(x: Tensor, h: int, w: int) -> Tensor:
        """[B, seq, C*4] → [B, C, H, W]"""
        from einops import rearrange
        return rearrange(
            x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h, w=w, ph=2, pw=2,
        )

    @staticmethod
    def make_img_ids(h_lat: int, w_lat: int, device: torch.device) -> Tensor:
        """生成 RoPE 用的 img_ids [1, seq_len, 3]，与 Flux infer.py 一致。"""
        img_ids = torch.zeros(h_lat, w_lat, 3, device=device)
        img_ids[..., 1] = torch.arange(h_lat, device=device)[:, None]
        img_ids[..., 2] = torch.arange(w_lat, device=device)[None, :]
        return img_ids.reshape(1, h_lat * w_lat, 3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, z_lr: Tensor, timesteps: Tensor) -> Tensor:
        """
        Args:
            z_lr:      [B, 16, H, W]  VAE latent of upsampled LR image
            timesteps: [B]            float in [0, 1]

        Returns:
            delta:     [B, 16, H, W]  嵌入残差
        """
        b, c, h, w = z_lr.shape
        h_half, w_half = h // 2, w // 2   # pack 后的 grid 尺寸

        # 1. pack
        img = self.pack(z_lr)                          # [B, seq, 64]

        # 2. 投影到 hidden_size
        img = self.img_in(img)                         # [B, seq, hidden]

        # 3. 时间步条件
        vec = self.time_in(                            # [B, hidden]
            timestep_embedding(timesteps, 256)
        )

        # 4. RoPE 位置编码
        img_ids = self.make_img_ids(h_half, w_half, z_lr.device)
        img_ids = img_ids.expand(b, -1, -1)
        pe = self.pe_embedder(img_ids)                 # [B, 1, seq, pe_dim]

        # 5. Transformer
        for block in self.blocks:
            img = block(img, vec=vec, pe=pe)

        # 6. 输出投影
        img = self.final_layer(img, vec)               # [B, seq, 64]

        # 7. unpack → 残差
        delta = self.unpack(img, h_half, w_half)       # [B, 16, H, W]

        return delta


def create_flow_embedder(config_path: str) -> FlowEmbedder:
    """从 yaml 配置文件创建 FlowEmbedder 实例。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return FlowEmbedder(**cfg)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "flow_embedder.yaml")
    model = create_flow_embedder(config_path)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Config:    {config_path}")
    print(f"Total:     {total / 1e6:.2f}M")
    print(f"Trainable: {trainable / 1e6:.2f}M")
