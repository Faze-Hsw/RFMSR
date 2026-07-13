"""
RealESRGAN Image Degradation Pipeline
Based on the official basicsr implementation. Install: pip install basicsr
"""

import math
import random
import sys
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
import yaml
from typing import Dict, Optional, Tuple

# ---- compatibility: basicsr depends on torchvision.transforms.functional_tensor ----
# torchvision >= 0.15 removed this module, need to create a compatibility shim
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    _ft = ModuleType('torchvision.transforms.functional_tensor')
    _ft.rgb_to_grayscale = tv_F.rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _ft

from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    random_mixed_kernels,
    circular_lowpass_kernel,
)


class RealESRGANDegradation:
    """RealESRGAN two-stage degradation pipeline"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.opts = self.config['opts']
        self.degradation = self.config['degradation']

        self.kernel_range1 = [x for x in range(3, self.opts['blur_kernel_size'], 2)]
        self.kernel_range2 = [x for x in range(3, self.opts['blur_kernel_size2'], 2)]

        self.pulse_tensor = torch.zeros(
            self.opts['blur_kernel_size2'],
            self.opts['blur_kernel_size2'],
        ).float()
        self.pulse_tensor[
            self.opts['blur_kernel_size2'] // 2,
            self.opts['blur_kernel_size2'] // 2,
        ] = 1

        self.jpeger = DiffJPEG(differentiable=False).cpu()

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Kernel generation
    # ------------------------------------------------------------------
    def generate_kernels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate first-order blur kernel, second-order blur kernel, final sinc kernel"""

        # --- first blur kernel ---
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opts['sinc_prob']:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 \
                else np.random.uniform(np.pi / 5, np.pi)
            kernel1 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel1 = random_mixed_kernels(
                self.opts['kernel_list'], self.opts['kernel_prob'],
                kernel_size,
                self.opts['blur_sigma'], self.opts['blur_sigma'],
                [-math.pi, math.pi],
                self.opts['betag_range'], self.opts['betap_range'],
            )
        pad = (self.opts['blur_kernel_size'] - kernel_size) // 2
        kernel1 = np.pad(kernel1, ((pad, pad), (pad, pad)))

        # --- second blur kernel ---
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opts['sinc_prob2']:
            omega_c = np.random.uniform(np.pi / 3, np.pi) if kernel_size < 13 \
                else np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.opts['kernel_list2'], self.opts['kernel_prob2'],
                kernel_size,
                self.opts['blur_sigma2'], self.opts['blur_sigma2'],
                [-math.pi, math.pi],
                self.opts['betag_range2'], self.opts['betap_range2'],
            )
        pad = (self.opts['blur_kernel_size2'] - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad, pad), (pad, pad)))

        # --- final sinc kernel ---
        if np.random.uniform() < self.opts['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=self.opts['blur_kernel_size2'],
            )
        else:
            sinc_kernel = self.pulse_tensor.numpy()

        return kernel1, kernel2, sinc_kernel

    # ------------------------------------------------------------------
    # Degradation pipeline
    # ------------------------------------------------------------------
    def degrade(
        self,
        img_gt: torch.Tensor,
        kernel1: Optional[torch.Tensor] = None,
        kernel2: Optional[torch.Tensor] = None,
        sinc_kernel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply RealESRGAN two-stage degradation to HR image.

        Args:
            img_gt: [B, C, H, W], float32, [0, 1]

        Returns:
            {'lq': [B, C, H/sf, W/sf], 'gt': img_gt}
        """
        if kernel1 is None or kernel2 is None or sinc_kernel is None:
            k1, k2, sk = self.generate_kernels()
            kernel1 = torch.FloatTensor(k1) if kernel1 is None else kernel1
            kernel2 = torch.FloatTensor(k2) if kernel2 is None else kernel2
            sinc_kernel = torch.FloatTensor(sk) if sinc_kernel is None else sinc_kernel

        device = img_gt.device
        kernel1 = kernel1.to(device)
        kernel2 = kernel2.to(device)
        sinc_kernel = sinc_kernel.to(device)
        self.jpeger = self.jpeger.to(device)

        ori_h, ori_w = img_gt.size()[2:4]
        sf = self.degradation['sf']

        # ==================== First degradation ====================
        out = filter2D(img_gt, kernel1)

        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'], self.degradation['resize_prob'])[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.degradation['resize_range'][0], 1)
        else:
            scale = 1
        out = F.interpolate(out, scale_factor=scale,
                            mode=random.choice(['area', 'bilinear', 'bicubic']))

        # noise
        if random.random() < self.degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.degradation['noise_range'],
                clip=True, rounds=False,
                gray_prob=self.degradation['gray_noise_prob'])
        else:
            out = random_add_poisson_noise_pt(
                out, scale_range=self.degradation['poisson_scale_range'],
                gray_prob=self.degradation['gray_noise_prob'],
                clip=True, rounds=False)

        # JPEG
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ==================== Second degradation ====================
        if random.random() < self.degradation['second_order_prob']:
            if random.random() < self.degradation['second_blur_prob']:
                out = filter2D(out, kernel2)

            updown_type = random.choices(
                ['up', 'down', 'keep'], self.degradation['resize_prob2'])[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.degradation['resize_range2'][0], 1)
            else:
                scale = 1
            out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                mode=random.choice(['area', 'bilinear', 'bicubic']))

            if random.random() < self.degradation['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.degradation['noise_range2'],
                    clip=True, rounds=False,
                    gray_prob=self.degradation['gray_noise_prob2'])
            else:
                out = random_add_poisson_noise_pt(
                    out, scale_range=self.degradation['poisson_scale_range2'],
                    gray_prob=self.degradation['gray_noise_prob2'],
                    clip=True, rounds=False)

        # ==================== JPEG + final sinc ====================
        if random.random() < 0.5:
            out = F.interpolate(
                out, size=(ori_h // sf, ori_w // sf),
                mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            out = F.interpolate(
                out, size=(ori_h // sf, ori_w // sf),
                mode=random.choice(['area', 'bilinear', 'bicubic']))
            out = filter2D(out, sinc_kernel)

        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return {'lq': im_lq.contiguous(), 'gt': img_gt}
