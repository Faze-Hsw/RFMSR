"""
ResFlow 推理脚本 — Residual Flow Matching 逆流积分

使用 ResFlow (LightningDiT) 做速度预测，无 Flux 第二阶段。
流路径: x_t = z_hr + t·(z_lr - z_hr) + t·σ·ε
逆流积分: t=1(LR+noise) → t=0(HR)

用法:
  python infer_resflow.py --input input.png
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
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
from flux.util import load_ae
from models.resflow import create_resflow
from models.dinov2_encoder import create_dinov2_encoder
from utils.color_fix import apply_color_fix


# ================================================================================
# 配置加载
# ================================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "infer_resflow.yaml")


def load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        print(f"配置文件 {_CONFIG_PATH} 不存在，使用内置默认值")
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CFG = load_config()

MODEL_NAME = _CFG.get("model_name", "flux-dev")
RESFLOW_PATH = _CFG.get("resflow_path", "checkpoints/resflow.safetensors")
MODEL_CONFIG = _CFG.get("model_config", "configs/resflow.yaml")
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
# ResFlow 推理器
# ================================================================================

class ResFlowInferencer:

    def __init__(self):
        pass

    # ---- 模型加载 ----

    def load(self):
        """加载 VAE + ResFlow + DINOv2。"""
        print(f"Loading VAE '{MODEL_NAME}' -> {DENOISE_DEVICE}...")
        self.ae = load_ae(MODEL_NAME, device=DENOISE_DEVICE)
        self.ae.eval()
        self.ae.requires_grad_(False)

        print(f"Loading ResFlow from {RESFLOW_PATH} ...")
        self.resflow = create_resflow(MODEL_CONFIG)
        sd = safe_load(RESFLOW_PATH)
        self.resflow.load_state_dict(sd, strict=True)
        self.resflow = self.resflow.to(DENOISE_DEVICE, dtype=torch.float32)
        self.resflow.eval()
        self.resflow.dit.use_checkpoint = False  # 推理不需要梯度检查点
        n = sum(p.numel() for p in self.resflow.parameters()) / 1e6
        print(f"  Params: {n:.2f}M")

        print(f"Loading DINOv2 encoder ...")
        self.venc = create_dinov2_encoder(MODEL_CONFIG, device=DENOISE_DEVICE)
        if self.venc is not None:
            print(f"  DINOv2: loaded")

        print("All models loaded.")

    # ---- VAE 编解码 ----

    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """[-1,1] bf16 → latent [B,16,H,W]."""
        return self.ae.encode(img.to(DENOISE_DEVICE, dtype=torch.bfloat16))

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [B,16,H,W] → pixel [0,1]."""
        latent = latent.to(DENOISE_DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            img = self.ae.decode(latent)
        img = img.float()
        return torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)

    # ---- Tile 辅助 ----

    @staticmethod
    def _make_tile_grid(length: int, tile: int, stride: int) -> list[tuple[int, int]]:
        """返回覆盖整个维度的 (start, end) tile 位置列表。"""
        if length <= tile:
            return [(0, length)]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] + tile < length:
            positions.append(length - tile)
        return [(p, p + tile) for p in sorted(set(positions))]

    @staticmethod
    def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device: torch.device) -> torch.Tensor:
        """2D 高斯融合权重，OpenCV 自适应 sigma。"""
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

    # ---- 逆流积分 (整张) ----

    @torch.no_grad()
    def reverse_flow(self, z_lr: torch.Tensor, steps: int = 28,
                     flow_sigma: float = None, seed: int = 42,
                     lr_pixel=None) -> torch.Tensor:
        """ResFlow 逆流积分: t=1 → t=0."""
        device = z_lr.device
        B, C, H, W = z_lr.shape
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA

        # DINOv2 全图一次
        venc_fea = None
        if self.venc is not None and lr_pixel is not None:
            venc_fea = self.venc(lr_pixel.float())

        # 时间步: 1.0 → 0.0
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # 初始状态: LR latent + 噪声
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + _sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t_curr, t_prev in tqdm(step_pairs, desc="ResFlow",
                                        total=len(step_pairs), leave=False):
                t_batch = torch.full((B,), t_curr, device=device)
                dt = t_prev - t_curr

                v = self.resflow(x, t_batch, z_lr, venc_fea=venc_fea).float()
                x = x + dt * v

        return x

    # ---- 逆流积分 (分 tile, VOSR 风格 per-step 融合) ----

    @torch.no_grad()
    def reverse_flow_tiled(self, z_lr: torch.Tensor, steps: int = 28,
                           flow_sigma: float = None, seed: int = 42,
                           lt_size: int = 64, lt_stride: int = 32,
                           lr_pixel=None) -> torch.Tensor:
        """每步逐 tile 预测速度场并高斯加权融合回全图潜变量。"""
        device = z_lr.device
        B, C, H, W = z_lr.shape
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
        AE_FACTOR = 8

        # tile 网格
        h_tiles = self._make_tile_grid(H, lt_size, lt_stride)
        w_tiles = self._make_tile_grid(W, lt_size, lt_stride)

        # Per-tile DINOv2 预计算
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

        # 时间步
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        # 初始状态: 全图噪声（所有 tile 共享）
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + _sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        # 高斯融合权重
        g_weight = self._gaussian_weights(lt_size, lt_size, C, device)

        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        for t_curr, t_prev in tqdm(step_pairs, desc="Tiled ResFlow",
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
                        v_tile = self.resflow(
                            x_tile, t_batch, z_lr_tile, venc_fea=tile_fea
                        ).float()

                    v_acc[:, :, hs:he, ws:we] += v_tile * g_weight
                    w_acc[:, :, hs:he, ws:we] += g_weight

            v_total = v_acc / w_acc.clamp(min=1e-8)
            x = x + dt * v_total

        return x

    # ---- 主入口 ----

    def infer(self, init_image: str, scale: float = SCALE, steps: int = None,
              flow_sigma: float = None, seed: int = SEED,
              chopping: bool = None, tile_size: int = None,
              tile_stride: int = None, color_correction: str = 'none') -> Image.Image:
        """输入图片 → ResFlow 超分 → 输出图片。

        Args:
            color_correction: 颜色校正方法 ('adain', 'wavelet', 'ycbcr', 'none').
        """
        _steps = steps or INFER_STEPS
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
        _chopping = chopping if chopping is not None else CHOPPING_ENABLED
        _tile_size = tile_size or TILE_SIZE
        _tile_stride = tile_stride or TILE_STRIDE

        AE_FACTOR = 8
        PATCH_SIZE = 2
        MOD_PIXEL = 16

        # ---- 加载 & resize LR ----
        src = Image.open(init_image).convert("RGB")
        exact_w = int(src.size[0] * scale)
        exact_h = int(src.size[1] * scale)
        target = src.resize((exact_w, exact_h), Image.BICUBIC)

        im_np = np.array(target).astype(np.float32) / 255.0
        im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0)
        im_cond = im_cond.to(dtype=torch.bfloat16, device=DENOISE_DEVICE)
        ori_h, ori_w = im_cond.shape[-2:]

        # ---- 对齐到 16 倍数 ----
        h, w = im_cond.shape[-2:]
        pad_h = (math.ceil(h / MOD_PIXEL) * MOD_PIXEL) - h
        pad_w = (math.ceil(w / MOD_PIXEL) * MOD_PIXEL) - w
        if pad_h > 0 or pad_w > 0:
            im_cond = F.pad(im_cond, (0, pad_w, 0, pad_h), mode="reflect")

        # ---- VAE 全局编码 ----
        image_tensor = im_cond * 2.0 - 1.0
        z_lr = self.vae_encode(image_tensor)
        lh, lw = z_lr.shape[2], z_lr.shape[3]

        # ---- LR 像素空间 (供 DINOv2 使用) ----
        lr_pixel = None
        if self.venc is not None:
            lr_pixel = (im_cond + 1.0) / 2.0   # [-1,1] bf16 → [0,1]

        # ---- tile 参数 ----
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

        # ---- VAE 全局解码 ----
        res_sr = self.vae_decode(z_hr)
        res_sr = res_sr[:, :, 0:ori_h, 0:ori_w]

        img = torch.clamp(res_sr, 0.0, 1.0)[0]
        decoded = 255.0 * np.moveaxis(img.cpu().float().numpy(), 0, 2)
        decoded = decoded.astype(np.uint8)
        sr_image = Image.fromarray(decoded)

        # ---- 颜色校正 ----
        if color_correction != 'none':
            sr_image = apply_color_fix(sr_image, target, method=color_correction)

        return sr_image


# ================================================================================
# CLI 入口
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
):
    """ResFlow Residual Flow Matching 超分推理。

    Args:
        input:        输入图片或文件夹路径 (必填)
        output:       输出目录
        scale:        放大倍数
        steps:        逆流积分步数 (默认从配置读取)
        flow_sigma:   噪声标准差 (默认从配置读取)
        seed:         随机种子
        chopping:     是否启用分块推理 (默认从配置读取)
        tile_size:    像素空间 tile 大小
        tile_stride:  像素空间 stride
        color_correction: 颜色校正方法 ('adain', 'wavelet', 'ycbcr', 'none').
    """
    init_image = input
    if init_image is None:
        raise ValueError("必须提供 --input 参数。")

    _steps = steps or INFER_STEPS
    _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
    src_path = Path(init_image)
    if src_path.is_dir():
        # 批量文件夹推理
        IMG_EXTS = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff"}
        img_files = []
        for ext in IMG_EXTS:
            img_files.extend(glob.glob(str(src_path / f"**/{ext}"), recursive=True))
        img_files = sorted(set(img_files))
        if not img_files:
            raise ValueError(f"文件夹 {init_image} 中未找到图片文件")
        img_files_base = [p for p in img_files]  # for relative path computation
    else:
        img_files_base = [init_image]
        img_files = [init_image]

    print(f"\n{'=' * 50}")
    print(f"ResFlow Residual FM Inference")
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

    inferencer = ResFlowInferencer()
    inferencer.load()

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

    print(f"\nDone. 输出目录: {out_root.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
