#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAN Loss Functions
移植自 LPNSR/losses/gan_loss.py

Includes:
1. PatchGAN discriminator (for adversarial training in image space)
2. GAN loss calculation (supports multiple GAN types)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN Discriminator

    Divides image into multiple patches, each patch independently determines real/fake
    Output is a feature map, each position represents real/fake judgment for corresponding patch

    Reference: pix2pix, pix2pixHD
    """

    def __init__(
        self, input_nc=3, ndf=64, n_layers=3, norm_type="spectral", use_sigmoid=False
    ):
        """
        Args:
            input_nc: Number of input channels
            ndf: Number of channels in first convolution
            n_layers: Number of discriminator layers
            norm_type: Normalization type ('batch', 'instance', 'spectral', 'none')
            use_sigmoid: Whether to use sigmoid at output (needed for vanilla GAN, not for LSGAN/WGAN)
        """
        super().__init__()

        self.use_sigmoid = use_sigmoid

        # Choose normalization layer
        if norm_type == "batch":
            norm_layer = nn.BatchNorm2d
            use_spectral = False
        elif norm_type == "instance":
            norm_layer = nn.InstanceNorm2d
            use_spectral = False
        elif norm_type == "spectral":
            norm_layer = None
            use_spectral = True
        else:
            norm_layer = None
            use_spectral = False

        kw = 4  # Kernel size
        padw = 1  # Padding size

        # First layer: no normalization
        sequence = []
        if use_spectral:
            sequence.append(
                spectral_norm(
                    nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
                )
            )
        else:
            sequence.append(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            )
        sequence.append(nn.LeakyReLU(0.2, True))

        # Middle layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)

            if use_spectral:
                sequence.append(
                    spectral_norm(
                        nn.Conv2d(
                            ndf * nf_mult_prev,
                            ndf * nf_mult,
                            kernel_size=kw,
                            stride=2,
                            padding=padw,
                        )
                    )
                )
            else:
                sequence.append(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                    )
                )

            if norm_layer is not None:
                sequence.append(norm_layer(ndf * nf_mult))

            sequence.append(nn.LeakyReLU(0.2, True))

        # Second to last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        if use_spectral:
            sequence.append(
                spectral_norm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=1,
                        padding=padw,
                    )
                )
            )
        else:
            sequence.append(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                )
            )

        if norm_layer is not None:
            sequence.append(norm_layer(ndf * nf_mult))

        sequence.append(nn.LeakyReLU(0.2, True))

        # Last layer: output 1 channel
        if use_spectral:
            sequence.append(
                spectral_norm(
                    nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
                )
            )
        else:
            sequence.append(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            )

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        """
        Args:
            x: Input image [B, C, H, W]

        Returns:
            Discrimination result [B, 1, H', W']
        """
        out = self.model(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


class UNetDiscriminator(nn.Module):
    """
    UNet-style Discriminator

    Combines global and local judgment capabilities, suitable for high-resolution images
    Reference: Real-ESRGAN
    """

    def __init__(self, input_nc=3, ndf=64, skip_connection=True):
        """
        Args:
            input_nc: Number of input channels
            ndf: Base number of channels
            skip_connection: Whether to use skip connections
        """
        super().__init__()

        self.skip_connection = skip_connection

        # Encoder
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

        # Decoder
        self.conv4 = spectral_norm(
            nn.Conv2d(ndf * 8, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv5 = spectral_norm(
            nn.Conv2d(ndf * 4, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv6 = spectral_norm(
            nn.Conv2d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # Final output
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
            x: Input image [B, C, H, W]

        Returns:
            Discrimination result [B, 1, H, W]
        """
        # Encoding
        feat0 = self.lrelu(self.conv0(x))
        feat1 = self.lrelu(self.conv1(feat0))
        feat2 = self.lrelu(self.conv2(feat1))
        feat3 = self.lrelu(self.conv3(feat2))

        # Decoding + upsampling
        feat3 = F.interpolate(
            feat3, scale_factor=2, mode="bilinear", align_corners=False
        )
        feat4 = self.lrelu(self.conv4(feat3))
        if self.skip_connection:
            feat4 = feat4 + feat2

        feat4 = F.interpolate(
            feat4, scale_factor=2, mode="bilinear", align_corners=False
        )
        feat5 = self.lrelu(self.conv5(feat4))
        if self.skip_connection:
            feat5 = feat5 + feat1

        feat5 = F.interpolate(
            feat5, scale_factor=2, mode="bilinear", align_corners=False
        )
        feat6 = self.lrelu(self.conv6(feat5))
        if self.skip_connection:
            feat6 = feat6 + feat0

        # Final output
        out = self.lrelu(self.conv7(feat6))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)

        return out


class GANLoss(nn.Module):
    """
    GAN Loss Calculation

    Supports multiple GAN types:
    - vanilla: Original GAN (BCE loss)
    - lsgan: Least Squares GAN (MSE loss)
    - wgan: Wasserstein GAN
    - wgan-gp: WGAN with gradient penalty
    - hinge: Hinge loss (commonly used in BigGAN, StyleGAN)
    """

    def __init__(
        self, gan_type="lsgan", real_label=1.0, fake_label=0.0, loss_weight=1.0
    ):
        """
        Args:
            gan_type: GAN type ('vanilla', 'lsgan', 'wgan', 'wgan-gp', 'hinge')
            real_label: Real label value
            fake_label: Fake label value
            loss_weight: Loss weight
        """
        super().__init__()

        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss_weight = loss_weight

        if gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_type in ["wgan", "wgan-gp", "hinge"]:
            self.loss = None
        else:
            raise ValueError(f"Unsupported GAN type: {gan_type}")

    def _get_target_label(self, pred, target_is_real):
        """Get target label"""
        if target_is_real:
            target_label = self.real_label
        else:
            target_label = self.fake_label
        return torch.full_like(pred, target_label)

    def forward(self, pred, target_is_real, is_disc=False):
        """
        Calculate GAN loss

        Args:
            pred: Discriminator output
            target_is_real: Whether target is real image
            is_disc: Whether it's discriminator loss (for hinge loss)

        Returns:
            GAN loss value
        """
        if self.gan_type == "vanilla" or self.gan_type == "lsgan":
            target_label = self._get_target_label(pred, target_is_real)
            loss = self.loss(pred, target_label)

        elif self.gan_type == "wgan":
            if target_is_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()

        elif self.gan_type == "wgan-gp":
            if target_is_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()

        elif self.gan_type == "hinge":
            if is_disc:
                if target_is_real:
                    loss = F.relu(1.0 - pred).mean()
                else:
                    loss = F.relu(1.0 + pred).mean()
            else:
                # Generator loss
                loss = -pred.mean()
        else:
            raise ValueError(f"Unsupported GAN type: {self.gan_type}")

        return self.loss_weight * loss

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples):
        """
        Calculate gradient penalty for WGAN-GP

        Args:
            discriminator: Discriminator
            real_samples: Real samples
            fake_samples: Generated samples

        Returns:
            Gradient penalty value
        """
        batch_size = real_samples.size(0)
        device = real_samples.device

        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)

        # Interpolated samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Discriminator output
        d_interpolates = discriminator(interpolates)

        # Calculate gradient
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


def create_discriminator(
    disc_type="patch", input_nc=3, ndf=64, n_layers=3, norm_type="spectral"
):
    """
    Create discriminator

    Args:
        disc_type: Discriminator type ('patch', 'unet')
        input_nc: Number of input channels
        ndf: Base number of channels
        n_layers: Number of layers (PatchGAN only)
        norm_type: Normalization type

    Returns:
        Discriminator instance
    """
    if disc_type == "patch":
        return NLayerDiscriminator(
            input_nc=input_nc,
            ndf=ndf,
            n_layers=n_layers,
            norm_type=norm_type,
            use_sigmoid=False,
        )
    elif disc_type == "unet":
        return UNetDiscriminator(input_nc=input_nc, ndf=ndf, skip_connection=True)
    else:
        raise ValueError(f"Unsupported discriminator type: {disc_type}")


if __name__ == "__main__":
    # Test code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test PatchGAN discriminator
    print("Testing PatchGAN discriminator...")
    disc_patch = create_discriminator("patch", input_nc=3, ndf=64, n_layers=3).to(
        device
    )
    x = torch.randn(2, 3, 256, 256).to(device)
    out = disc_patch(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in disc_patch.parameters()) / 1e6:.2f}M")

    # Test UNet discriminator
    print("\nTesting UNet discriminator...")
    disc_unet = create_discriminator("unet", input_nc=3, ndf=64).to(device)
    out = disc_unet(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in disc_unet.parameters()) / 1e6:.2f}M")

    # Test GAN loss
    print("\nTesting GAN loss...")
    gan_loss = GANLoss(gan_type="lsgan", loss_weight=1.0)

    fake_pred = torch.randn(2, 1, 30, 30).to(device)
    real_pred = torch.randn(2, 1, 30, 30).to(device)

    # Discriminator loss
    d_loss_real = gan_loss(real_pred, target_is_real=True, is_disc=True)
    d_loss_fake = gan_loss(fake_pred, target_is_real=False, is_disc=True)
    d_loss = d_loss_real + d_loss_fake
    print(f"  Discriminator loss: {d_loss.item():.4f}")

    # Generator loss
    g_loss = gan_loss(fake_pred, target_is_real=True, is_disc=False)
    print(f"  Generator loss: {g_loss.item():.4f}")

    print("\nTest completed!")
