"""
DINOv2 编码器 — 冻结，从 LR 图片提取语义特征注入 DiT Cross-Attention。

用法:
    encoder = load_dinov2_encoder("dinov2b", device="cuda")
    features = encoder(lr_tensor)  # lr: [B,3,H,W] float [0,1] → list[[B,N,enc_dim]]
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


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

        # 加载 DINOv2
        print(f"Loading DINOv2 encoder ({enc_type}) ...")
        encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{enc_type[-1]}14")

        # 去掉分类头，添加 forward_with_features
        del encoder.head
        encoder.head = nn.Identity()

        self._patch_forward_with_features(encoder, self.layer_indices)
        self.encoder = encoder.to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        print(f"✅ DINOv2 encoder loaded, layers={self.layer_indices}")

    @staticmethod
    def _patch_forward_with_features(encoder, layer_indices):
        """给 DINOv2 ViT 打 monkey-patch，使其 forward 返回中间层特征。"""
        def forward_with_features(self, x, masks=None):
            features = {}
            if isinstance(x, list):
                return self.forward_features_list(x, masks)
            x = self.prepare_tokens_with_masks(x, masks)
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in layer_indices:
                    features[f"layer_{i}"] = x[:, 1:]  # 去 CLS token
            x_norm = self.norm(x)
            return features, x_norm[:, 1:]

        encoder.forward_with_features = types.MethodType(forward_with_features, encoder)

    def preprocess(self, lr: torch.Tensor) -> torch.Tensor:
        """
        lr: [B, 3, H, W] float [0, 1]
        → resize 448, ImageNet 标准化 → [B, 3, 448, 448]
        """
        x = F.interpolate(lr, size=self.dinov2_size, mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        return x

    @torch.no_grad()
    def forward(self, lr: torch.Tensor) -> list[torch.Tensor]:
        """
        lr: [B, 3, H, W] float [0, 1]
        → list of [B, N_patches, enc_dim]  每个 tensor 对应一个指定层
        """
        x = self.preprocess(lr)
        features, x_norm = self.encoder.forward_with_features(x)

        z_list = [features[f"layer_{i}"] for i in self.layer_indices]
        # 最后一层用 x_norm 替换
        z_list[-1] = x_norm

        return z_list


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
