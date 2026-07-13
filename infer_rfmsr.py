"""
RFMSR Inference Script — Residual Flow Matching Reverse Integration (SD2.1 VAE)

Uses RFMSR (LightningDiT) for velocity prediction, SD2.1 VAE for encoding/decoding.
Flow path: x_t = z_hr + t*(z_lr - z_hr) + t*sigma*epsilon
Reverse integration: t=1 (LR+noise) → t=0 (HR)

Usage:
  python infer_rfmsr.py --input input.png
"""

import os
import math
import glob
import fire
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm

from safetensors.torch import load_file as safe_load
from models.rfmsr import create_rfmsr
from models.dinov2_encoder import create_dinov2_encoder
from utils.color_fix import apply_color_fix


# ================================================================================
# Config loading
# ================================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "infer_rfmsr.yaml")


def load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        print(f"Config file {_CONFIG_PATH} not found, using built-in defaults")
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CFG = load_config()

VAE_PATH = _CFG.get("vae_path", "ckpts/stable-diffusion-2-1-base")
RFMSR_PATH = _CFG.get("rfmsr_path", "ckpts/rfmsr.safetensors")
MODEL_CONFIG = _CFG.get("model_config", "configs/rfmsr.yaml")
FLOW_SIGMA = _CFG.get("flow_sigma", 1.0)
INFER_STEPS = _CFG.get("steps", 28)
SCALE = _CFG.get("scale", 4.0)
SEED = _CFG.get("seed", 42)
OUTPUT = _CFG.get("output", "outputs")
DENOISE_DEVICE = _CFG.get("denoise_device", "cuda")

_CHOPPING_CFG = _CFG.get("chopping", {})
CHOPPING_ENABLED = _CHOPPING_CFG.get("enabled", False)
TILE_SIZE = _CHOPPING_CFG.get("tile_size", 512)
TILE_STRIDE = _CHOPPING_CFG.get("tile_stride", 256)      # pixel-space stride

COLOR_CORRECTION = _CFG.get("color_correction", "none")


# ================================================================================
# RFMSR Inferencer
# ================================================================================

class RFMSRInferencer:

    def __init__(self):
        pass

    # ---- Model loading ----

    def load(self, vae_path=None, rfmsr_path=None, model_config=None, denoise_device=None):
        """Load SD2.1 VAE + RFMSR + DINOv2. CLI args override YAML defaults."""
        from diffusers import AutoencoderKL

        _vae = vae_path or VAE_PATH
        _rfmsr = rfmsr_path or RFMSR_PATH
        _model = model_config or MODEL_CONFIG
        _device = denoise_device or DENOISE_DEVICE
        self._device = _device

        print(f"Loading SD2.1 VAE from {_vae} -> {_device}...")
        self.ae = AutoencoderKL.from_pretrained(_vae, subfolder="vae")
        self.ae = self.ae.to(_device).eval()
        self.ae.requires_grad_(False)
        print(f"  VAE scaling_factor: {self.ae.config.scaling_factor}")

        print(f"Loading RFMSR from {_rfmsr} ...")
        self.rfmsr = create_rfmsr(_model)
        sd = safe_load(_rfmsr)
        # VOSR checkpoint: raw DiT params (no prefix) → RFMSR expects "dit." prefix
        sd.pop("ema_scale", None)
        sd = {"dit." + k if not k.startswith("dit.") else k: v for k, v in sd.items()}
        missing, unexpected = self.rfmsr.load_state_dict(sd, strict=False)
        self.rfmsr = self.rfmsr.to(_device, dtype=torch.float32)
        self.rfmsr.eval()
        self.rfmsr.dit.use_checkpoint = False
        n = sum(p.numel() for p in self.rfmsr.parameters()) / 1e6
        print(f"  Params: {n:.2f}M")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

        print(f"Loading DINOv2 encoder ...")
        self.venc = create_dinov2_encoder(_model, device=_device)
        if self.venc is not None:
            print(f"  DINOv2: loaded")

        print("All models loaded.")

    # ---- VAE encode/decode ----

    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """img [-1,1] → SD2.1 latent [B,4,H,W] (scaled)."""
        return self.ae.encode(img.float()).latent_dist.sample() * self.ae.config.scaling_factor

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """SD2.1 latent [B,4,H,W] (scaled) → pixel [0,1]."""
        latent = latent / self.ae.config.scaling_factor
        img = self.ae.decode(latent).sample
        return torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)

    # ---- Tile helpers ----

    @staticmethod
    def _make_tile_grid(length: int, tile: int, stride: int) -> list[tuple[int, int]]:
        """Return (start, end) tile positions covering the entire dimension."""
        if length <= tile:
            return [(0, length)]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] + tile < length:
            positions.append(length - tile)
        return [(p, p + tile) for p in sorted(set(positions))]

    @staticmethod
    def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device: torch.device) -> torch.Tensor:
        """2D Gaussian blend weights, OpenCV adaptive sigma."""
        def _kernel_1d(ksize):
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
            if ksize % 2 == 0:
                kernel = cv2.getGaussianKernel(ksize=ksize + 1, sigma=sigma, ktype=cv2.CV_64F)
                kernel = kernel[1:, ]
            else:
                kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_64F)
            return kernel

        kernel_h = _kernel_1d(tile_h)       # (H, 1)
        kernel_w = _kernel_1d(tile_w)       # (W, 1)
        w = np.matmul(kernel_h, kernel_w.T) # (H, W)
        w = torch.from_numpy(w).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return w.to(device).expand(1, channels, -1, -1)

    # ---- Reverse integration (full image) ----

    @torch.no_grad()
    def reverse_flow(self, z_lr: torch.Tensor, steps: int = 28,
                     flow_sigma: float = None, seed: int = 42,
                     lr_pixel=None) -> torch.Tensor:
        """RFMSR reverse flow integration: t=1 → t=0."""
        device = z_lr.device
        B, C, H, W = z_lr.shape
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA

        # DINOv2: full image once
        venc_fea = None
        if self.venc is not None and lr_pixel is not None:
            venc_fea = self.venc(lr_pixel.float())

        # Timesteps: 1.0 → 0.0
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # Initial state: LR latent + noise
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + _sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t_curr, t_prev in tqdm(step_pairs, desc="RFMSR",
                                        total=len(step_pairs), leave=False):
                t_batch = torch.full((B,), t_curr, device=device)
                dt = t_prev - t_curr

                v = self.rfmsr(x, t_batch, z_lr, venc_fea=venc_fea).float()
                x = x + dt * v

        return x

    # ---- Reverse integration (tiled, VOSR-style per-step blending) ----

    @torch.no_grad()
    def reverse_flow_tiled(self, z_lr: torch.Tensor, steps: int = 28,
                           flow_sigma: float = None, seed: int = 42,
                           lt_size: int = 64, lt_stride: int = 32,
                           lr_pixel=None) -> torch.Tensor:
        """Per-step tiled velocity prediction with Gaussian-weighted blending back to full latent."""
        device = z_lr.device
        B, C, H, W = z_lr.shape
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
        AE_FACTOR = 8

        # Tile grid
        h_tiles = self._make_tile_grid(H, lt_size, lt_stride)
        w_tiles = self._make_tile_grid(W, lt_size, lt_stride)

        # Per-tile DINOv2: pre-compute
        tile_venc = {}
        use_venc = self.venc is not None and lr_pixel is not None
        if use_venc:
            with torch.no_grad():
                for hs, he in h_tiles:
                    for ws, we in w_tiles:
                        ph_s, pw_s = hs * AE_FACTOR, ws * AE_FACTOR
                        ph_e = min(he * AE_FACTOR, lr_pixel.shape[2])
                        pw_e = min(we * AE_FACTOR, lr_pixel.shape[3])
                        lq_crop = lr_pixel[:, :, ph_s:ph_e, pw_s:pw_e]
                        tile_venc[(hs, ws)] = self.venc(lq_crop)

        # Timesteps
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # Initial state: full-image noise (shared across tiles)
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + _sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        # Gaussian blend weights
        g_weight = self._gaussian_weights(lt_size, lt_size, C, device)

        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        for t_curr, t_prev in tqdm(step_pairs, desc="Tiled RFMSR",
                                    total=len(step_pairs), leave=False):
            t_batch = torch.full((B,), t_curr, device=device)
            dt = t_prev - t_curr

            v_acc = torch.zeros(B, C, H, W, device=device)
            w_acc = torch.zeros(B, C, H, W, device=device)

            for hs, he in h_tiles:
                for ws, we in w_tiles:
                    x_tile = x[:, :, hs:he, ws:we]
                    z_lr_tile = z_lr[:, :, hs:he, ws:we]
                    tile_fea = tile_venc.get((hs, ws), None) if use_venc else None

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        v_tile = self.rfmsr(
                            x_tile, t_batch, z_lr_tile, venc_fea=tile_fea
                        ).float()

                    v_acc[:, :, hs:he, ws:we] += v_tile * g_weight
                    w_acc[:, :, hs:he, ws:we] += g_weight

            v_total = v_acc / w_acc.clamp(min=1e-8)
            x = x + dt * v_total

        return x

    # ---- Main entry ----

    def infer(self, init_image: str, scale: float = SCALE, steps: int = None,
              flow_sigma: float = None, seed: int = SEED,
              chopping: bool = None, tile_size: int = None,
              tile_stride: int = None, color_correction: str = 'none') -> Image.Image:
        """Input image → RFMSR super-resolution → output image.

        Args:
            color_correction: Color correction method ('adain', 'wavelet', 'ycbcr', 'none').
        """
        _steps = steps or INFER_STEPS
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
        _chopping = chopping if chopping is not None else CHOPPING_ENABLED
        _tile_size = tile_size or TILE_SIZE
        _tile_stride = tile_stride or TILE_STRIDE

        AE_FACTOR = 8
        PATCH_SIZE = 2
        MOD_PIXEL = 16

        # ---- Load & resize LR ----
        src = Image.open(init_image).convert("RGB")
        exact_w = int(src.size[0] * scale)
        exact_h = int(src.size[1] * scale)
        target = src.resize((exact_w, exact_h), Image.BICUBIC)

        im_np = np.array(target).astype(np.float32) / 255.0
        im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0)
        im_cond = im_cond.to(dtype=torch.bfloat16, device=self._device)
        ori_h, ori_w = im_cond.shape[-2:]

        # ---- Align to multiple of 16 ----
        h, w = im_cond.shape[-2:]
        pad_h = (math.ceil(h / MOD_PIXEL) * MOD_PIXEL) - h
        pad_w = (math.ceil(w / MOD_PIXEL) * MOD_PIXEL) - w
        if pad_h > 0 or pad_w > 0:
            im_cond = F.pad(im_cond, (0, pad_w, 0, pad_h), mode="reflect")

        # ---- VAE global encode ----
        image_tensor = im_cond * 2.0 - 1.0
        z_lr = self.vae_encode(image_tensor)
        lh, lw = z_lr.shape[2], z_lr.shape[3]

        # ---- LR pixel space (for DINOv2) ----
        lr_pixel = None
        if self.venc is not None:
            lr_pixel = im_cond.float()  

        # ---- Tile params ----
        lt_size = max((_tile_size // AE_FACTOR // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
        lt_stride = max((_tile_stride // AE_FACTOR // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
        lt_size = min(lt_size, min(lh, lw))
        lt_stride = min(lt_stride, lt_size)

        use_tiling = _chopping and (lh > lt_size or lw > lt_size)

        if not use_tiling:
            z_hr = self.reverse_flow(z_lr, steps=_steps, flow_sigma=_sigma,
                                     seed=seed, lr_pixel=lr_pixel)
        else:
            z_hr = self.reverse_flow_tiled(
                z_lr, steps=_steps, flow_sigma=_sigma, seed=seed,
                lt_size=lt_size, lt_stride=lt_stride,
                lr_pixel=lr_pixel,
            )

        # ---- VAE global decode ----
        res_sr = self.vae_decode(z_hr)
        res_sr = res_sr[:, :, 0:ori_h, 0:ori_w]

        img = torch.clamp(res_sr, 0.0, 1.0)[0]
        decoded = 255.0 * np.moveaxis(img.cpu().float().numpy(), 0, 2)
        decoded = decoded.astype(np.uint8)
        sr_image = Image.fromarray(decoded)

        # ---- Color correction ----
        if color_correction != 'none':
            sr_image = apply_color_fix(sr_image, target, method=color_correction)

        return sr_image


# ================================================================================
# CLI entry
# ================================================================================

def main(
    input: str = None,
    output: str = OUTPUT,
    scale: float = SCALE,
    steps: int = None,
    flow_sigma: float = None,
    seed: int = SEED,
    chopping: bool = None,
    tile_size: int = None,
    tile_stride: int = None,
    color_correction: str = COLOR_CORRECTION,
    vae_path: str = None,
    rfmsr_path: str = None,
    model_config: str = None,
    denoise_device: str = None,
):
    """RFMSR Residual Flow Matching super-resolution inference.

    Args:
        input:           Input image or folder path (required)
        output:          Output directory
        scale:           Upscale factor
        steps:           Reverse integration steps (default from config)
        flow_sigma:      Noise std (default from config)
        seed:            Random seed
        chopping:        Enable tiled inference (default from config)
        tile_size:       Pixel-space tile size
        tile_stride:     Pixel-space stride
        color_correction: Color correction method ('adain', 'wavelet', 'ycbcr', 'none').
        vae_path:        SD2.1 VAE path (default from config)
        rfmsr_path:      RFMSR weight path (default from config)
        model_config:    DiT architecture config path (default from config)
        denoise_device:  Inference device (default from config)
    """
    init_image = input
    if init_image is None:
        raise ValueError("Must provide --input argument.")

    _steps = steps or INFER_STEPS
    _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
    src_path = Path(init_image)
    if src_path.is_dir():
        # Batch folder inference
        IMG_EXTS = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff"}
        img_files = []
        for ext in IMG_EXTS:
            img_files.extend(glob.glob(str(src_path / f"**/{ext}"), recursive=True))
        img_files = sorted(set(img_files))
        if not img_files:
            raise ValueError(f"No image files found in folder {init_image}")
        img_files_base = [p for p in img_files]  # for relative path computation
    else:
        img_files_base = [init_image]
        img_files = [init_image]

    print(f"\n{'=' * 50}")
    print(f"RFMSR Residual FM Inference")
    print(f"  Input:    {init_image}")
    if src_path.is_dir():
        print(f"  Images:   {len(img_files)}")
    print(f"  Scale:    {scale}×")
    print(f"  Steps:    {_steps}")
    print(f"  Sigma:    {_sigma}")
    print(f"  Seed:     {seed}")
    print(f"  Tiling:   {chopping if chopping is not None else CHOPPING_ENABLED}")
    print(f"  Color Fix:{color_correction}")
    print(f"  Output:   {output}")
    print(f"{'=' * 50}\n")

    inferencer = RFMSRInferencer()
    inferencer.load(vae_path=vae_path, rfmsr_path=rfmsr_path,
                    model_config=model_config, denoise_device=denoise_device)

    out_root = Path(output)
    pbar = tqdm(img_files, desc="Inference", unit="img")
    for img_path in pbar:
        rel_path = os.path.relpath(img_path, init_image) if src_path.is_dir() else os.path.basename(img_path)
        base_name = os.path.splitext(rel_path)[0]
        save_path = out_root / f"{base_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pbar.set_postfix_str(rel_path[:40])
        try:
            result = inferencer.infer(
                init_image=img_path, scale=scale, steps=_steps,
                flow_sigma=_sigma, seed=seed,
                chopping=chopping, tile_size=tile_size, tile_stride=tile_stride,
                color_correction=color_correction,
            )
            result.save(str(save_path))
        except Exception as e:
            tqdm.write(f"  ❌ {rel_path}: {e}")

    print(f"\nDone. Output directory: {out_root.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
