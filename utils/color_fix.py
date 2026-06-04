"""
Color correction utilities for super-resolution outputs.

Three methods are provided to align the color statistics of SR results
back to the input LR image:

- AdaIN: Global adaptive instance normalization (mean/std matching)
- Wavelet: Multi-level wavelet decomposition, replacing low-frequency color
- YCbCr: Direct color space replacement (Y from SR, CbCr from LR)

Original wavelet/AdaIN reference:
    https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py
"""

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import ToTensor, ToPILImage


# ---------------------------------------------------------------------------
# YCbCr color-space conversions (PyTorch, range [0,1])
# ---------------------------------------------------------------------------

def rgb2ycbcr_torch(im: Tensor, only_y: bool = True) -> Tensor:
    """MATLAB-compatible rgb2ycbcr.

    Args:
        im:  float tensor, N×3×H×W, values in [0, 1].
        only_y:  if True, return only the Y channel (N×1×H×W).

    Returns:
        YCbCr tensor in [0, 1].
    """
    im_temp = im.permute(0, 2, 3, 1) * 255.0  # N×H×W×3

    if only_y:
        coeff = torch.tensor([65.481, 128.553, 24.966],
                             device=im.device, dtype=im.dtype).view(3, 1) / 255.0
        result = torch.matmul(im_temp, coeff) + 16.0
    else:
        scale = torch.tensor(
            [[65.481, -37.797, 112.0],
             [128.553, -74.203, -93.786],
             [24.966, 112.0, -18.214]],
            device=im.device, dtype=im.dtype
        ) / 255.0
        bias = torch.tensor([16, 128, 128],
                            device=im.device, dtype=im.dtype).view(1, 1, 1, 3)
        result = torch.matmul(im_temp, scale) + bias

    result = result / 255.0
    result = result.clamp_(0.0, 1.0)
    return result.permute(0, 3, 1, 2)


def ycbcr2rgb_torch(im: Tensor) -> Tensor:
    """MATLAB-compatible ycbcr2rgb.

    Args:
        im:  float tensor, N×3×H×W, values in [0, 1].

    Returns:
        RGB tensor in [0, 1].
    """
    im_temp = im.permute(0, 2, 3, 1) * 255.0  # N×H×W×3

    scale = torch.tensor(
        [[0.00456621, 0.00456621, 0.00456621],
         [0, -0.00153632, 0.00791071],
         [0.00625893, -0.00318811, 0]],
        device=im.device, dtype=im.dtype
    ) * 255.0
    bias = torch.tensor(
        [-222.921, 135.576, -276.836],
        device=im.device, dtype=im.dtype
    ).view(1, 1, 1, 3)

    result = torch.matmul(im_temp, scale) + bias
    result = result / 255.0
    result = result.clamp_(0.0, 1.0)
    return result.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# AdaIN color fix
# ---------------------------------------------------------------------------

def _calc_mean_std(feat: Tensor, eps: float = 1e-5):
    """Compute per-channel mean and std for a 4D tensor."""
    b, c = feat.size()[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std


def _adain(content: Tensor, style: Tensor) -> Tensor:
    """Adaptive instance normalization: match content stats to style stats."""
    style_mean, style_std = _calc_mean_std(style)
    content_mean, content_std = _calc_mean_std(content)
    size = content.size()
    normalized = (content - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def adain_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """Globally match the mean & std of *target* to *source* via AdaIN.

    Args:
        target:  SR output (PIL Image).
        source:  LR input (PIL Image).

    Returns:
        Color-corrected PIL Image.
    """
    target_t = ToTensor()(target).unsqueeze(0)
    source_t = ToTensor()(source).unsqueeze(0)
    result_t = _adain(target_t, source_t)
    result_t = result_t.squeeze(0).clamp_(0.0, 1.0)
    return ToPILImage()(result_t)


# ---------------------------------------------------------------------------
# Wavelet color fix
# ---------------------------------------------------------------------------

_WAVELET_KERNEL = torch.tensor(
    [[0.0625, 0.125, 0.0625],
     [0.125, 0.25, 0.125],
     [0.0625, 0.125, 0.0625]]
)


def _wavelet_blur(image: Tensor, radius: int) -> Tensor:
    """Low-pass filter via dilated Gaussian-like conv.

    Args:
        image:  (1, 3, H, W) tensor.
        radius: dilation radius.

    Returns:
        Blurred tensor of same shape.
    """
    kernel = _WAVELET_KERNEL.to(dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)  # (3,1,3,3)
    padded = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    return F.conv2d(padded, kernel, groups=3, dilation=radius)


def _wavelet_decomp(image: Tensor, levels: int = 5):
    """Multi-level wavelet decomposition.

    Returns:
        (high_freq, low_freq): both (1, 3, H, W).
    """
    high_freq = torch.zeros_like(image)
    cur = image
    for i in range(levels):
        radius = 2 ** i
        low = _wavelet_blur(cur, radius)
        high_freq = high_freq + (cur - low)
        cur = low
    return high_freq, cur  # high, low


def _wavelet_reconstruct(content: Tensor, style: Tensor) -> Tensor:
    """Replace low-frequency of *content* with that of *style*."""
    content_high, _ = _wavelet_decomp(content)
    _, style_low = _wavelet_decomp(style)
    return content_high + style_low


def wavelet_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """Multi-level wavelet color correction.

    Replaces the low-frequency component of *target* (SR) with that of
    *source* (LR), preserving SR high-frequency details.

    Args:
        target:  SR output (PIL Image).
        source:  LR input (PIL Image).

    Returns:
        Color-corrected PIL Image.
    """
    target_t = ToTensor()(target).unsqueeze(0)
    source_t = ToTensor()(source).unsqueeze(0)
    result_t = _wavelet_reconstruct(target_t, source_t)
    result_t = result_t.squeeze(0).clamp_(0.0, 1.0)
    return ToPILImage()(result_t)


# ---------------------------------------------------------------------------
# YCbCr color fix
# ---------------------------------------------------------------------------

def ycbcr_color_fix(target: Image.Image, source: Image.Image) -> Image.Image:
    """Replace Cb/Cr channels of *target* with those of *source*.

    Keeps SR luminance (Y) and borrows LR chrominance (CbCr), which is
    the most direct way to restore color fidelity.

    Args:
        target:  SR output (PIL Image).
        source:  LR input (PIL Image).

    Returns:
        Color-corrected PIL Image.
    """
    target_t = ToTensor()(target).unsqueeze(0)
    source_t = ToTensor()(source).unsqueeze(0)

    content_y = rgb2ycbcr_torch(target_t, only_y=True)          # (1,1,H,W)
    style_ycbcr = rgb2ycbcr_torch(source_t, only_y=False)      # (1,3,H,W)

    combined = torch.cat([content_y, style_ycbcr[:, 1:]], dim=1)  # Y_sr + CbCr_lr
    result_t = ycbcr2rgb_torch(combined)

    result_t = result_t.squeeze(0).clamp_(0.0, 1.0)
    return ToPILImage()(result_t)


# ---------------------------------------------------------------------------
# Unified entry
# ---------------------------------------------------------------------------

COLOR_FIX_METHODS = {
    'adain': adain_color_fix,
    'wavelet': wavelet_color_fix,
    'ycbcr': ycbcr_color_fix,
    'none': lambda target, source: target,
}


def apply_color_fix(target: Image.Image, source: Image.Image,
                    method: str = 'ycbcr') -> Image.Image:
    """Apply the specified color correction method.

    Args:
        target:  SR output image.
        source:  LR input image.
        method:  one of 'adain', 'wavelet', 'ycbcr', 'none'.

    Returns:
        Color-corrected image.
    """
    if method not in COLOR_FIX_METHODS:
        raise ValueError(
            f"Unknown color fix method: '{method}'. "
            f"Available: {list(COLOR_FIX_METHODS.keys())}"
        )
    return COLOR_FIX_METHODS[method](target, source)
