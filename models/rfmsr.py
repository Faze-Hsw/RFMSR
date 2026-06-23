"""
RFMSR — Residual Flow Matching DiT (SD2.1 VAE 潜空间)

输入:
  z_lr [B, 4, H, W]    LR latent (VAE encode 上采样 LR)
  x_t  [B, 4, H, W]    当前流状态
  t    [B]              时间 ∈ [0,1]

输出:
  v    [B, 4, H, W]     速度预测

架构:
  cat(z_lr, x_t) → [B, 8, H, W]
    → PatchEmbed(patch_size=2) → tokens [B, N, 1024]
    → LightningDiT × 28 blocks
    → unpatchify → [B, 4, H, W]
"""

import math
import torch
import torch.nn as nn
import yaml
from safetensors.torch import load_file as safetensors_load

from .lightningdit import LightningDiT


class RFMSR(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 8,
        out_channels: int = 4,
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
        z_dims: int | None = None,
        num_fused_layers: int = 1,
        encdim_ratio: int = 2,
    ):
        super().__init__()

        self.z_dims = z_dims

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
            z_dims=z_dims,
            num_fused_layers=num_fused_layers,
            encdim_ratio=encdim_ratio,
            auxiliary_time_cond=False,
        )

    def load_pretrained(self, ckpt_path: str, verbose: bool = True):
        """从 VOSR 预训练权重初始化 RFMSR（兼容 key 前缀和 pos_embed 尺寸差异）。

        VOSR checkpoint 的 key 为裸 LightningDiT (如 blocks.0.attn.qkv.weight)，
        RFMSR 内部用 self.dit 包裹，key 多了 dit. 前缀，此处自动匹配。
        """
        if ckpt_path.endswith(".safetensors"):
            state_dict = safetensors_load(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        target_state = self.state_dict()
        new_state_dict = {}
        skipped = 0
        loaded = 0

        # 自动检测是否需要 dit. 前缀
        need_prefix = "dit." if any(k.startswith("dit.") for k in target_state) else ""

        for k, v in state_dict.items():
            target_k = k
            if k not in target_state and need_prefix:
                target_k = need_prefix + k

            if target_k not in target_state:
                skipped += 1
                if verbose and skipped <= 3:
                    print(f"[RFMSR] Skipping {k} (not in model)")
                continue
            # 跳过 RoPE/freqs（模型会根据当前 input_size 自动生成）
            if "rope" in k or "freqs_cos" in k or "freqs_sin" in k:
                continue
            # pos_embed 尺寸不匹配时 bicubic 插值
            if "pos_embed" in target_k and v.shape != target_state[target_k].shape:
                if verbose:
                    print(f"[RFMSR] Interpolating pos_embed: {v.shape} → {target_state[target_k].shape}")
                v_len = v.shape[1]
                target_len = target_state[target_k].shape[1]
                dim = v.shape[-1]
                src_size = int(math.sqrt(v_len))
                tgt_size = int(math.sqrt(target_len))
                v_img = v.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
                v_img = nn.functional.interpolate(
                    v_img, size=(tgt_size, tgt_size), mode="bicubic", align_corners=False
                )
                v = v_img.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, dim)
            new_state_dict[target_k] = v
            loaded += 1

        msg = self.load_state_dict(new_state_dict, strict=False)
        if verbose:
            if skipped > 0:
                print(f"[RFMSR] Skipped {skipped} incompatible keys")
            if msg.missing_keys:
                print(f"[RFMSR] Missing keys: {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"[RFMSR] Unexpected keys: {len(msg.unexpected_keys)}")
            print(f"[RFMSR] Loaded {loaded} params from {ckpt_path}")

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z_lr: torch.Tensor,
                venc_fea=None) -> torch.Tensor:
        """
        Args:
            x_t:  [B, 4, H, W]   当前流状态
            t:    [B]             时间
            z_lr: [B, 4, H, W]   LR latent（channel-concat 条件）
            venc_fea: DINOv2 特征列表 [tensor[B,N,C]] 或 None（Cross-Attn 条件）

        Returns:
            v:    [B, 4, H, W]   速度预测
        """
        inp = torch.cat([z_lr, x_t], dim=1)  # [B, 8, H, W]
        return self.dit.forward_flexible(inp, t, z=venc_fea)


def create_rfmsr(cfg_path: str) -> RFMSR:
    """从 YAML 配置文件创建 RFMSR。"""
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    arch = cfg.get("dit_arch", {})
    dv2 = cfg.get("dinov2", {}) or {}

    z_dims = dv2.get("enc_dim", None)
    num_fused_layers = len(dv2.get("layer_dinov2b_list", [1]))
    encdim_ratio = dv2.get("encdim_ratio", 2)

    return RFMSR(
        input_size=arch.get("input_size", 64),
        patch_size=arch.get("patch_size", 2),
        in_channels=arch.get("in_channels", 8),
        out_channels=arch.get("out_channels", 4),
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
        z_dims=z_dims,
        num_fused_layers=num_fused_layers,
        encdim_ratio=encdim_ratio,
    )
