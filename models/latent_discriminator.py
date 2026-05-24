"""
UNet Discriminator — 基于 LPNSR 的标准 UNet 判别器

在 Flux 潜空间（16 通道）上做 PatchGAN 真伪判别。
使用 spectral_norm 稳定训练，skip connection (ADD) 结合全局与局部判断。

参考: Real-ESRGAN, LPNSR

包含 GAN 损失函数：
  - hinge_d_loss: 判别器 Hinge 损失
  - gen_loss: 生成器非饱和损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetDiscriminator(nn.Module):
    """
    UNet 风格判别器，结合全局与局部判别能力。

    Args:
        input_nc: 输入通道数（Flux VAE latent = 16）
        ndf: 基础通道数
        skip_connection: 是否使用 skip connection (ADD 方式)
    """

    def __init__(self, input_nc=16, ndf=64, skip_connection=True):
        super().__init__()

        self.skip_connection = skip_connection

        # ---- Encoder ----
        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)

        self.conv1 = spectral_norm(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        )
        self.conv3 = spectral_norm(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        )

        # ---- Decoder ----
        self.conv4 = spectral_norm(
            nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv5 = spectral_norm(
            nn.Conv2d(ndf * 4, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv6 = spectral_norm(
            nn.Conv2d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # ---- 最终输出头 ----
        self.conv7 = spectral_norm(
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv8 = spectral_norm(
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv9 = nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] Flux latent

        Returns:
            [B, 1, H, W] logit map（与输入同分辨率）
        """
        # ---- Encoder ----
        feat0 = self.lrelu(self.conv0(x))                           # H   * ndf
        feat1 = self.lrelu(self.conv1(feat0))                       # H/2 * ndf*2
        feat2 = self.lrelu(self.conv2(feat1))                       # H/4 * ndf*4
        feat3 = self.lrelu(self.conv3(feat2))                       # H/8 * ndf*8

        # ---- Decoder + Upsampling ----
        feat3 = F.interpolate(feat3, scale_factor=2, mode="bilinear", align_corners=False)
        feat4 = self.lrelu(self.conv4(feat3))                       # H/4 * ndf*4
        if self.skip_connection:
            feat4 = feat4 + feat2

        feat4 = F.interpolate(feat4, scale_factor=2, mode="bilinear", align_corners=False)
        feat5 = self.lrelu(self.conv5(feat4))                       # H/2 * ndf*2
        if self.skip_connection:
            feat5 = feat5 + feat1

        feat5 = F.interpolate(feat5, scale_factor=2, mode="bilinear", align_corners=False)
        feat6 = self.lrelu(self.conv6(feat5))                       # H * ndf
        if self.skip_connection:
            feat6 = feat6 + feat0

        # ---- 最终输出 ----
        out = self.lrelu(self.conv7(feat6))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)                                       # [B, 1, H, W]

        return out


# ═══════════════════════════════════════════════
# GAN 损失函数
# ═══════════════════════════════════════════════

def hinge_d_loss(logits_real, logits_fake):
    """判别器 Hinge 损失。支持单尺度或多尺度 logit list。"""
    if not isinstance(logits_real, list):
        logits_real, logits_fake = [logits_real], [logits_fake]

    loss = 0.0
    for lr, lf in zip(logits_real, logits_fake):
        loss_real = F.relu(1.0 - lr)
        loss_fake = F.relu(1.0 + lf)
        loss = loss + (loss_real + loss_fake).mean(dim=list(range(1, lr.ndim))).mean()

    return loss / len(logits_real)


def gen_loss(logits_fake):
    """生成器非饱和 GAN 损失。支持单尺度或多尺度 logit list。"""
    if not isinstance(logits_fake, list):
        logits_fake = [logits_fake]

    loss = 0.0
    for lf in logits_fake:
        loss = loss - lf.mean(dim=list(range(1, lf.ndim))).mean()

    return loss / len(logits_fake)
