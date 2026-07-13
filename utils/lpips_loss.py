"""
LPIPS Perceptual Loss (Learned Perceptual Image Patch Similarity)
Ported from LPNSR/losses/lpips_loss.py

Design Philosophy:
LPIPS is a deep learning-based perceptual similarity metric that measures perceptual differences
between images using features extracted from pre-trained networks.
Compared to traditional pixel-level losses (such as L1, L2), LPIPS can better
capture the human visual system's perception of image quality.

Core Advantages:
1. Perceptual consistency: Highly correlated with human perception
2. Feature-level comparison: Captures high-level semantic information
3. Pre-trained advantage: Leverages representations learned from large-scale datasets
4. Multi-scale features: Combines feature information at different hierarchical levels

Reference Paper:
The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (CVPR 2018)
https://arxiv.org/abs/1801.03924
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    import lpips

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


class LPIPSLoss(nn.Module):
    """
    LPIPS Perceptual Loss

    This loss function uses pre-trained deep networks to extract features and
    calculate perceptual similarity between images.
    LPIPS has been proven to be highly correlated with human perception
    and is widely used in image generation, super-resolution, and other tasks.

    Args:
        loss_weight: Loss weight for multi-loss weighting
        net_type: Pre-trained network type, optional 'alex'(AlexNet), 'vgg'(VGG), 'squeeze'(SqueezeNet)
                  Default is 'alex', which has the highest computational efficiency and good performance
        use_gpu: Whether to use GPU acceleration, default True
        spatial: Whether to return spatial dimension loss map, default False
        normalize: Whether to normalize input to [-1, 1] range, default True

    Examples:
        >>> criterion = LPIPSLoss(loss_weight=0.5, net_type='alex')
        >>> pred = torch.randn(4, 3, 256, 256)  # Needs 3-channel RGB image
        >>> target = torch.randn(4, 3, 256, 256)
        >>> loss = criterion(pred, target)

    Note:
        - Input images should be in RGB format, 3 channels
        - Input range should be [-1, 1] or [0, 1] (automatically handled if normalize=True)
        - LPIPS model parameters are frozen and do not participate in training
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        net_type: str = "alex",
        use_gpu: bool = True,
        spatial: bool = False,
        normalize: bool = True,
    ):
        super().__init__()

        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips library is not installed. Please run 'pip install lpips' to install."
            )

        self.loss_weight = loss_weight
        self.net_type = net_type
        self.spatial = spatial
        self.normalize = normalize

        # Create LPIPS model
        # net_type options: 'alex', 'vgg', 'squeeze'
        # alex: AlexNet, fastest computation, good performance
        # vgg: VGG, slower computation, but better performance in some scenarios
        # squeeze: SqueezeNet, in between the two
        self.lpips_model = lpips.LPIPS(net=net_type, spatial=spatial, verbose=False)

        # Freeze LPIPS model parameters
        for param in self.lpips_model.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.lpips_model.eval()

    def _convert_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input to 3-channel RGB format

        If input is 4-channel (such as latent space), take first 3 channels
        If input is 1-channel (grayscale), copy to 3 channels

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            rgb: RGB image [B, 3, H, W]
        """
        c = x.shape[1]

        if c == 3:
            return x
        elif c == 4:
            # Take first 3 channels
            return x[:, :3, :, :]
        elif c == 1:
            # Copy to 3 channels
            return x.repeat(1, 3, 1, 1)
        else:
            # Other cases, take first 3 channels or copy
            if c > 3:
                return x[:, :3, :, :]
            else:
                # Pad to 3 channels
                pad = torch.zeros(
                    x.shape[0],
                    3 - c,
                    x.shape[2],
                    x.shape[3],
                    device=x.device,
                    dtype=x.dtype,
                )
                return torch.cat([x, pad], dim=1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate LPIPS perceptual loss

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            weight: Optional pixel weight (only effective when spatial=True)

        Returns:
            loss: LPIPS loss value
        """
        assert pred.shape == target.shape, (
            f"pred and target shapes must be the same, got {pred.shape} and {target.shape}"
        )

        # Convert to RGB format
        pred_rgb = self._convert_to_rgb(pred)
        target_rgb = self._convert_to_rgb(target)

        # Ensure LPIPS model is on correct device
        if pred.device != next(self.lpips_model.parameters()).device:
            self.lpips_model = self.lpips_model.to(pred.device)

        # Calculate LPIPS loss
        # LPIPS model expects input range [-1, 1]
        # When normalize=True, [0,1] range input is automatically handled
        loss = self.lpips_model(pred_rgb, target_rgb, normalize=self.normalize)

        # If returning spatial dimension loss map, apply weight
        if self.spatial and weight is not None:
            loss = loss * weight

        # Take average
        loss = loss.mean()

        return self.loss_weight * loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"loss_weight={self.loss_weight}, "
            f"net_type='{self.net_type}', "
            f"spatial={self.spatial}, "
            f"normalize={self.normalize})"
        )
