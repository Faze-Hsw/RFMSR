"""
DiT Flow Embedder — LightningDiT 替代 U-Net 做速度预测

输入:
  z_lr [B, 16, H, W]   LR latent (VAE encode 上采样 LR)
  x_t  [B, 16, H, W]   当前流状态
  t    [B]              时间 ∈ [0,1]

输出:
  v    [B, 16, H, W]    速度预测

架构:
  cat(z_lr, x_t) → [B, 32, H, W]
    → PatchEmbed(patch_size=2) → tokens [B, N, 1024]
    → LightningDiT × 28 blocks
    → unpatchify → [B, 16, H, W]
"""

import torch
import torch.nn as nn
import yaml

from .lightningdit import LightningDiT


class DiTFlowEmbedder(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 32,
        out_channels: int = 16,
        hidden_size: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.dit = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            use_checkpoint=use_checkpoint,
            z_dims=None,                # 关闭 DinoV2 cross-attention
            auxiliary_time_cond=False,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:  [B, 16, H, W]  当前流状态
            t:    [B]             时间
            z_lr: [B, 16, H, W]  LR latent（支持任意分辨率，动态 RoPE）

        Returns:
            v:    [B, 16, H, W]  速度预测
        """
        inp = torch.cat([z_lr, x_t], dim=1)  # [B, 32, H, W]
        return self.dit.forward_flexible(inp, t)


def create_dit_flow_embedder(cfg_path: str) -> DiTFlowEmbedder:
    """从 YAML 配置文件创建 DiTFlowEmbedder。"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    arch = cfg.get("dit_arch", {})
    return DiTFlowEmbedder(
        input_size=arch.get("input_size", 64),
        patch_size=arch.get("patch_size", 2),
        in_channels=arch.get("in_channels", 32),
        out_channels=arch.get("out_channels", 16),
        hidden_size=arch.get("hidden_size", 1024),
        depth=arch.get("depth", 28),
        num_heads=arch.get("num_heads", 16),
        mlp_ratio=arch.get("mlp_ratio", 4.0),
        use_qknorm=arch.get("use_qknorm", True),
        use_swiglu=arch.get("use_swiglu", True),
        use_rope=arch.get("use_rope", True),
        use_rmsnorm=arch.get("use_rmsnorm", True),
        wo_shift=arch.get("wo_shift", False),
        use_checkpoint=arch.get("use_checkpoint", False),
    )
