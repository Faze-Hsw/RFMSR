"""
Latent Discriminator — 无条件多尺度 UNet 判别器
在 Flux 潜空间（16 通道）上做真伪判别，仅基于图像本身，无文本/时间步条件。

包含 GAN 损失函数：
  - hinge_d_loss: 判别器 Hinge 损失
  - gen_loss: 生成器非饱和损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# 基础组件（无条件版本）
# ═══════════════════════════════════════════════

class ResnetBlock2D(nn.Module):
    """Conv-Norm-SiLU residual block，无条件。"""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-5)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ═══════════════════════════════════════════════
# 编解码 Block
# ═══════════════════════════════════════════════

class DownBlock2D(nn.Module):
    """ResNet blocks + stride-2 下采样。返回每层输出用于 U-Net skip。"""
    def __init__(self, in_ch, out_ch, num_layers, add_down=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            _in = in_ch if i == 0 else out_ch
            self.blocks.append(ResnetBlock2D(_in, out_ch))
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if add_down else None

    def forward(self, x):
        layer_outputs = []
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)
        if self.down is not None:
            x = self.down(x)
        return x, layer_outputs


class UpBlock2D(nn.Module):
    """ResNet blocks + upsampling（支持 U-Net skip connection）。"""
    def __init__(self, in_ch, out_ch, num_layers, skip_ch=None, add_up=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            _in = (in_ch + skip_ch) if (i == 0 and skip_ch is not None) else out_ch
            self.blocks.append(ResnetBlock2D(_in, out_ch))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if add_up else None

    def forward(self, x, res_hidden_states=None):
        for i, block in enumerate(self.blocks):
            if i == 0 and res_hidden_states is not None:
                x = torch.cat([x, res_hidden_states], dim=1)
            x = block(x)
        if self.up is not None:
            x = self.up(x)
        return x


# ═══════════════════════════════════════════════
# 主模型
# ═══════════════════════════════════════════════

class LatentDiscriminator(nn.Module):
    """
    无条件多尺度 UNet 判别器，仅接受 Flux 潜空间图像输入。

    Args:
        in_channels:       输入通道数（Flux VAE = 16）
        block_out_channels: 各级通道数 [128, 256, 512]
        layers_per_block:   各级 ResNet 层数 [1, 2, 2]
        norm_num_groups:    输入特征投影通道数（拼接用）
    """
    def __init__(
        self,
        in_channels=16,
        block_out_channels=(128, 256, 512),
        layers_per_block=(1, 2, 2),
        norm_num_groups=32,
        **__,
    ):
        super().__init__()
        n_levels = len(block_out_channels)
        assert n_levels == len(layers_per_block)

        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.feature_in = nn.Conv2d(in_channels, norm_num_groups, 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        for i in range(n_levels):
            in_ch = block_out_channels[i - 1] if i > 0 else block_out_channels[0]
            in_ch = in_ch + norm_num_groups  # 拼接原始特征
            out_ch = block_out_channels[i]
            n_layers = layers_per_block[i]
            is_last = (i == n_levels - 1)
            block = DownBlock2D(in_ch, out_ch, n_layers, add_down=not is_last)
            self.down_blocks.append(block)

        # 中间块（简单 ResNet，无 attention）
        mid_ch = block_out_channels[-1]
        self.mid_block = nn.Sequential(ResnetBlock2D(mid_ch, mid_ch))

        # 输出头
        self.out_blocks = nn.ModuleList()
        self.out_blocks.append(self._make_out_head(mid_ch))

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(block_out_channels))
        rev_layers = list(reversed(layers_per_block))
        for i in range(n_levels - 1):
            in_ch = rev_channels[i]
            out_ch = rev_channels[i + 1]
            n_layers = rev_layers[i + 1]
            is_last = (i == n_levels - 2)
            block = UpBlock2D(
                in_ch, out_ch, n_layers,
                skip_ch=in_ch, add_up=not is_last,
            )
            self.up_blocks.append(block)
            self.out_blocks.append(self._make_out_head(out_ch))

    def _make_out_head(self, ch):
        return nn.Sequential(
            nn.GroupNorm(32, ch, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, sample):
        """
        Args:
            sample: [B, 16, H, W] Flux latent
        Returns:
            list of [B, 1, h, w] logit maps at different scales
        """
        sample = sample.float()

        # 预处理
        inputs = sample
        sample = self.conv_in(sample)
        inputs_feature = self.feature_in(inputs)

        # 下采样 + 记录跳跃连接
        skips = []
        for block in self.down_blocks:
            inputs_down = F.interpolate(inputs_feature, size=sample.shape[-2:],
                                        mode="bilinear", align_corners=False)
            x_in = torch.cat([sample, inputs_down], dim=1)
            sample, res = block(x_in)
            skips.append(res[-1])

        # 中间块
        sample = self.mid_block(sample)

        # 输出头 0（瓶颈）
        out = [self.out_blocks[0](sample)]

        # 上采样 + 多尺度输出
        for i, block in enumerate(self.up_blocks):
            res_hidden = skips.pop()
            sample = block(sample, res_hidden_states=res_hidden)
            out.append(self.out_blocks[i + 1](sample))

        return out


# ═══════════════════════════════════════════════
# GAN 损失函数
# ═══════════════════════════════════════════════

def hinge_d_loss(logits_real, logits_fake):
    """判别器 Hinge 损失。支持多尺度 logits 列表。"""
    if not isinstance(logits_real, list):
        logits_real, logits_fake = [logits_real], [logits_fake]

    loss = 0.0
    for lr, lf in zip(logits_real, logits_fake):
        loss_real = F.relu(1.0 - lr)
        loss_fake = F.relu(1.0 + lf)
        loss = loss + (loss_real + loss_fake).mean(dim=list(range(1, lr.ndim))).mean()

    return loss / len(logits_real)


def gen_loss(logits_fake):
    """生成器非饱和 GAN 损失。支持多尺度 logits 列表。"""
    if not isinstance(logits_fake, list):
        logits_fake = [logits_fake]

    loss = 0.0
    for lf in logits_fake:
        loss = loss - lf.mean(dim=list(range(1, lf.ndim))).mean()

    return loss / len(logits_fake)
