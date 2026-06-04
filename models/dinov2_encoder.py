"""
DINOv2 编码器 — 冻结，从 LR 图片提取语义特征注入 DiT Cross-Attention。

使用 HuggingFace transformers 自动下载权重：
  export HF_ENDPOINT=https://hf-mirror.com   # AutoDL 等国内服务器

用法:
    encoder = create_dinov2_encoder("configs/resflow.yaml", device="cuda")
    features = encoder(lr_tensor)  # lr: [B,3,H,W] float [0,1] → list[[B,N,enc_dim]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DINOV2_HF_MODELS = {
    "dinov2b": "facebook/dinov2-base",
    "dinov2l": "facebook/dinov2-large",
    "dinov2g": "facebook/dinov2-giant",
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

        model_name = DINOV2_HF_MODELS.get(enc_type)
        if model_name is None:
            raise ValueError(
                f"Unknown DINOv2 type: {enc_type}, "
                f"expected one of {list(DINOV2_HF_MODELS)}"
            )

        print(f"Loading DINOv2 from HuggingFace: {model_name} ...")
        try:
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DINOv2 from HuggingFace: {e}\n"
                f"设置镜像重试: export HF_ENDPOINT=https://hf-mirror.com"
            )

        self.encoder = self.encoder.to(device).eval()
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

        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        )

        # hidden_states: (embedding, block_0, block_1, ..., block_L)
        # hs[0] = patch_embed + pos_embed, hs[i] = block i-1 的输出
        hidden_states = outputs.hidden_states

        z_list = []
        for idx in self.layer_indices:
            z_list.append(hidden_states[idx + 1][:, 1:, :])  # +1 跳过 embedding 层，去 CLS token

        # 最后一层用最终 hidden state
        z_list[-1] = hidden_states[-1][:, 1:, :]

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
