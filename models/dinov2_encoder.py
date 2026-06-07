"""
DINOv2 编码器 — 冻结，从 LR 图片提取语义特征注入 DiT Cross-Attention。

使用 torch.hub 下载权重（与 VOSR 一致），缓存至 ckpts/torch_cache。

用法:
    encoder = create_dinov2_encoder("configs/resflow.yaml", device="cuda")
    features = encoder(lr_tensor)  # lr: [B,3,H,W] float [0,1] → list[[B,N,enc_dim]]
"""

import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DINOV2_HUB_NAMES = {
    "dinov2b": "dinov2_vitb14",
    "dinov2l": "dinov2_vitl14",
    "dinov2g": "dinov2_vitg14",
}


class Dinov2Encoder(nn.Module):
    """冻结的 DINOv2 特征提取器，输出指定中间层特征。"""

    def __init__(
        self,
        enc_type: str = "dinov2b",
        dinov2_size: int = 448,
        layer_indices: list[int] | None = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.dinov2_size = dinov2_size
        self.layer_indices = layer_indices or [8]

        hub_name = DINOV2_HUB_NAMES.get(enc_type)
        if hub_name is None:
            raise ValueError(
                f"Unknown DINOv2 type: {enc_type}, "
                f"expected one of {list(DINOV2_HUB_NAMES)}"
            )

        # 设置 torch.hub 缓存目录（与 VOSR 一致）
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "ckpts", "torch_cache")
        os.makedirs(cache_dir, exist_ok=True)
        torch.hub.set_dir(cache_dir)

        print(f"Loading DINOv2 from torch.hub: facebookresearch/dinov2 → {hub_name} ...")
        encoder = torch.hub.load('facebookresearch/dinov2', hub_name)

        # 去掉分类头，替换为 Identity
        del encoder.head
        encoder.head = torch.nn.Identity()

        # 注入 forward_with_features 方法（与 VOSR 完全一致）
        def forward_with_features(self, x, masks=None):
            features = {}
            layer_indices = list(range(len(self.blocks)))
            if isinstance(x, list):
                return self.forward_features_list(x, masks)
            x = self.prepare_tokens_with_masks(x, masks)
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in layer_indices:
                    features[f'layer_{i}'] = x[:, 1:]  # 去掉 CLS token
            x_norm = self.norm(x)
            return features, x_norm[:, 1:]

        encoder.forward_with_features = types.MethodType(forward_with_features, encoder)

        self.encoder = encoder.to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        print(f"✅ DINOv2 encoder loaded, layers={self.layer_indices}")

    def preprocess(self, lr: torch.Tensor) -> torch.Tensor:
        """
        lr: [B, 3, H, W] float [0, 1]
        → resize → clamp → ImageNet 标准化
        """
        x = F.interpolate(lr, size=self.dinov2_size, mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        return x

    @torch.no_grad()
    def forward(self, lr: torch.Tensor) -> list[torch.Tensor]:
        """
        lr: [B, 3, H, W] float [0, 1]
        → list of [B, N_patches, enc_dim]  每个 tensor 对应 layer_indices 中的一个指定层
        """
        x = self.preprocess(lr)

        features, x_norm = self.encoder.forward_with_features(x)
        z = [v for k, v in features.items() if k.startswith('layer_')]
        z[-1] = x_norm
        z = [z[i] for i in self.layer_indices]

        return z


def create_dinov2_encoder(config_path: str, device: str = "cuda") -> Dinov2Encoder | None:
    """从 YAML 配置创建 DINOv2 编码器，如未配置则返回 None。"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dv2 = cfg.get("dinov2", {}) or {}
    if not dv2:
        return None

    return Dinov2Encoder(
        enc_type=dv2.get("enc_type", "dinov2b"),
        dinov2_size=dv2.get("dinov2_size", 448),
        layer_indices=dv2.get("layer_dinov2b_list", [8]),
        device=device,
    )
