"""
Flux 超分推理脚本 — ResFlow 两阶段 Residual Flow 版本

两阶段逆流积分:
  Phase 1 (t=1 → switch_t): ResFlow — Residual FM, z_lr+noise 起点还原细节
  Phase 2 (switch_t → t=0): Flux    — 质量精修，去伪影、加细节

用法:
  python infer_flux.py --input input.png --scale 2.0 --switch_t 0.5
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
from pathlib import Path
import torch.nn.functional as F
import yaml
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm

from safetensors.torch import load_file as safe_load
from flux.util import load_flow_model, load_t5, load_clip, load_ae
from models.resflow import create_resflow
from models.dinov2_encoder import create_dinov2_encoder
from utils.color_fix import apply_color_fix


#################################################################################################
### 配置文件加载 & 默认参数
#################################################################################################

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "infer_flux.yaml")


def load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        print(f"⚠️  配置文件 {_CONFIG_PATH} 不存在，使用内置默认值")
        return {}
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


_CFG = load_config()

MODEL_NAME = _CFG.get("model_name", "flux-dev")
PROMPT = _CFG.get("prompt", "")
CFG_SCALE = _CFG.get("cfg", 3.5)
SEED = _CFG.get("seed", 42)

_WEIGHTS_CFG = _CFG.get("weights", {})
T5XXL_PATH = _WEIGHTS_CFG.get("t5xxl", None)
CLIP_PATH = _WEIGHTS_CFG.get("clip", None)
OUTDIR = _CFG.get("output", "outputs")
VERBOSE = _CFG.get("verbose", False)
SCALE = _CFG.get("scale", 1.0)
SHIFT = _CFG.get("shift", True)
TEXT_ENCODER_DEVICE = _CFG.get("text_encoder_device", "cuda")
DENOISE_DEVICE = _CFG.get("denoise_device", "cuda")

_CHOPPING_CFG = _CFG.get("chopping", {})
CHOPPING_ENABLED = _CHOPPING_CFG.get("enabled", False)
TILE_SIZE = _CHOPPING_CFG.get("tile_size", 512)          # pixel-space tile size
TILE_STRIDE = _CHOPPING_CFG.get("tile_stride", 256)       # pixel-space stride

RESFLOW_PATH = _CFG.get("resflow_path", None)
NO_RESFLOW = _CFG.get("no_resflow", False)
FLOW_SIGMA = _CFG.get("flow_sigma", 1.0)
T5_MAX_LENGTH = _CFG.get("t5_max_length", 512)

# 推理步数（逆流积分步数）
INFER_STEPS = _CFG.get("steps", 28)

# 两阶段切换时间点 switch_t ∈ [0,1]
# t ∈ (switch_t, 1.0] → DiT Embedder (Residual FM)
# t ∈ [0, switch_t]   → Flux 质量精修
SWITCH_T = _CFG.get("switch_t", 0.5)

COLOR_CORRECTION = _CFG.get("color_correction", "none")


#################################################################################################
### 主推理逻辑
#################################################################################################


class FluxInferencer:

    def __init__(self):
        self.verbose = False

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(self, model_name=MODEL_NAME, verbose=False, denoise_device="cuda"):
        """加载 Flux 主模型 + VAE。"""
        self.verbose = verbose
        self.model_name = model_name
        self.denoise_device = denoise_device

        print(f"Loading Flux model '{model_name}' ({torch.bfloat16}) -> {denoise_device}...")
        self.model = load_flow_model(model_name, device=denoise_device, verbose=verbose)
        self.model.eval()

        print(f"Loading VAE -> {denoise_device}...")
        self.ae = load_ae(model_name, device=denoise_device)
        self.ae.eval()

        print("✅ Models loaded.")

    def load_resflow(self, ckpt_path: str, config_path: str = None):
        """加载训练好的 ResFlow + DINOv2 编码器。"""
        if config_path is None:
            config_path = os.path.join(_SCRIPT_DIR, "configs", "resflow.yaml")
        print(f"Loading ResFlow from {ckpt_path} ...")

        self.resflow = create_resflow(config_path)
        sd = safe_load(ckpt_path)
        self.resflow.load_state_dict(sd, strict=True)
        self.resflow = self.resflow.to(self.denoise_device, dtype=torch.float32)
        self.resflow.eval()
        self.resflow.dit.use_checkpoint = False  
        n = sum(p.numel() for p in self.resflow.parameters()) / 1e6
        print(f"  Params: {n:.2f}M")

        # DINOv2 编码器
        self.venc = create_dinov2_encoder(config_path, device=self.denoise_device)
        if self.venc is not None:
            print(f"  DINOv2: loaded ({self.venc.encoder.__class__.__name__})")

        print("✅ ResFlow loaded.")

    def load_text_encoders(self, device="cuda", t5_max_length=T5_MAX_LENGTH,
                           t5xxl_path=None, clip_path=None):
        """加载 T5 + CLIP 文本编码器。"""
        print(f"Loading T5 ({torch.bfloat16}) -> {device}...")
        self.t5 = load_t5(device, max_length=t5_max_length, ckpt_path=t5xxl_path)
        print(f"Loading CLIP ({torch.bfloat16}) -> {device}...")
        self.clip = load_clip(device, ckpt_path=clip_path)
        print("✅ Text encoders loaded.")

    def free_text_encoders(self):
        print("Freeing text encoders...")
        del self.t5, self.clip
        self.t5 = None
        self.clip = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Text encoders freed.")

    def prepare_conditions(self, prompt):
        """编码 prompt，保存 txt/vec。"""
        print("Encoding prompt with T5 and CLIP...")
        with torch.no_grad():
            self._prompt_bs = 1
            self._cached_txt = self.t5(prompt)
            self._cached_vec = self.clip(prompt)
        self._cached_txt_ids = torch.zeros(1, self._cached_txt.shape[1], 3, dtype=torch.float32)
        print("✅ Prompt encoded.")

    # ------------------------------------------------------------------
    # 逆流积分采样
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reverse_flow_sampling(self, z_lr: torch.Tensor,
                               switch_t: float = 0.5,
                               steps: int = 28, guidance: float = 3.5,
                               shift: bool = True, seed: int = 42,
                               lr_pixel=None, flow_sigma=None) -> torch.Tensor:
        """
        两阶段逆流积分: t=1(LR+noise) → t=0(HR)

        Phase 1 (t ∈ [switch_t, 1.0]): ResFlow — Residual FM，从 LR 还原细节
        Phase 2 (t ∈ [0, switch_t]):   Flux    — 质量精修
        """
        device = z_lr.device
        B, C, H, W = z_lr.shape
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
        h_pack, w_pack = H // 2, W // 2
        seq_len = h_pack * w_pack

        # ---- DINOv2 语义特征（全图一次提取） ----
        venc_fea = None
        if hasattr(self, 'venc') and self.venc is not None and lr_pixel is not None:
            venc_fea = self.venc(lr_pixel.float())

        # ---- 准备 Flux 条件 ----
        flux_dtype = torch.bfloat16
        txt = self._cached_txt.expand(B, -1, -1).to(device, dtype=flux_dtype)
        txt_ids = self._cached_txt_ids.expand(B, -1, -1).to(device)
        vec = self._cached_vec.expand(B, -1).to(device, dtype=flux_dtype)
        guidance_vec = torch.full((B,), guidance, device=device, dtype=flux_dtype)
        img_ids = self._make_img_ids(B, h_pack, w_pack, device)

        # ---- 时间步调度 ----
        timesteps = self._get_schedule(steps, seq_len, shift=shift)
        self.print(f"   2-stage flow: {steps} steps, t: 1.0 → 0.0"
                   f"{', shift' if shift else ''}, switch_t={switch_t:.3f}, σ={_sigma}")

        # ---- 初始状态: LR latent + 扰动 (Residual FM 起点) ----
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + _sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        # ---- 逐步逆流积分 ----
        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        n_embed, n_flux = 0, 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t_curr, t_prev in tqdm(step_pairs, desc="2-stage flow", total=len(step_pairs), leave=False):
                t_batch = torch.full((B,), t_curr, device=device)
                dt = t_prev - t_curr

                if t_curr > switch_t:
                    # Phase 1: DiT Embedder — LR latent + DINOv2 语义条件
                    if hasattr(self, 'resflow'):
                        v_total = self.resflow(x, t_batch, z_lr, venc_fea=venc_fea).float()
                    else:
                        v_total = torch.zeros(B, C, H, W, device=device)
                    n_embed += 1
                else:
                    # Phase 2: Flux — 质量精修
                    x_packed = self._pack(x.to(flux_dtype))
                    v_flux_packed = self.model(
                        img=x_packed, img_ids=img_ids,
                        txt=txt, txt_ids=txt_ids,
                        timesteps=t_batch.to(flux_dtype), y=vec, guidance=guidance_vec,
                    )
                    v_total = self._unpack(v_flux_packed.float(), h_pack, w_pack)

                    # NaN/极端值净化
                    v_total = torch.nan_to_num(v_total, nan=0.0, posinf=10.0, neginf=-10.0)
                    v_total = v_total.clamp(-20.0, 20.0)
                    n_flux += 1

                # Euler 步
                x = x + dt * v_total

        self.print(f"   Steps: {n_embed} DiT + {n_flux} Flux")
        return x  # t=0 → z_hr

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _pack(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def _unpack(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                         h=h, w=w, ph=2, pw=2)

    @staticmethod
    def _make_img_ids(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        ids = torch.zeros(h, w, 3, device=device)
        ids[..., 1] = torch.arange(h, device=device)[:, None]
        ids[..., 2] = torch.arange(w, device=device)[None, :]
        return repeat(ids, "h w c -> b (h w) c", b=b)

    @staticmethod
    def _get_schedule(steps: int, seq_len: int, shift: bool = True) -> list[float]:
        """从 1.0 到 0.0 的时间步调度（标准 Flux time shift）。"""
        from flux.sampling import time_shift, get_lin_function
        timesteps = torch.linspace(1.0, 0.0, steps + 1)
        if shift:
            mu = get_lin_function(y1=0.5, y2=1.15)(seq_len)
            timesteps = time_shift(mu, 1.0, timesteps)
        return timesteps.tolist()

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    def vae_encode_tensor(self, image_tensor):
        """像素 tensor [-1,1] → spatial latent [B,16,H,W]."""
        self.ae = self.ae.to(self.denoise_device)
        return self.ae.encode(image_tensor)

    def vae_decode_tensor(self, latent):
        """Spatial latent [B,16,H,W] → 像素 tensor [0,1]."""
        self.ae = self.ae.to(self.denoise_device)
        latent = latent.to(self.denoise_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image = self.ae.decode(latent)
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        return image

    # ------------------------------------------------------------------
    # Latent tiling helpers (VOSR-style)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tile_grid(length: int, tile: int, stride: int):
        """返回覆盖整个维度的 (start, end) tile 位置列表。"""
        if length <= tile:
            return [(0, length)]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] + tile < length:
            positions.append(length - tile)
        return [(p, p + tile) for p in sorted(set(positions))]

    @staticmethod
    def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device: torch.device):
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

    # ------------------------------------------------------------------
    # Tiled reverse flow sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reverse_flow_sampling_tiled(
        self, z_lr: torch.Tensor,
        switch_t: float = 0.5, steps: int = 28, guidance: float = 3.5,
        shift: bool = True, seed: int = 42,
        lt_size: int = 64, lt_stride: int = 32,
        lr_pixel=None, flow_sigma=None,
    ) -> torch.Tensor:
        """Latent 空间分 tile 的两阶段逆流积分（VOSR 风格：每 tile 独立 DINOv2）。"""
        device = z_lr.device
        B, C, H, W = z_lr.shape
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA
        h_pack_full, w_pack_full = H // 2, W // 2
        seq_len = h_pack_full * w_pack_full

        # ---- Flux 条件（bfloat16，复用缓存） ----
        flux_dtype = torch.bfloat16
        txt = self._cached_txt.expand(B, -1, -1).to(device, dtype=flux_dtype)
        txt_ids = self._cached_txt_ids.expand(B, -1, -1).to(device)
        vec = self._cached_vec.expand(B, -1).to(device, dtype=flux_dtype)
        guidance_vec = torch.full((B,), guidance, device=device, dtype=flux_dtype)

        # ---- 时间步调度 ----
        timesteps = self._get_schedule(steps, seq_len, shift=shift)
        self.print(f"   Tiled 2-stage flow: {steps} steps, t: 1.0 → 0.0"
                   f"{', shift' if shift else ''}, switch_t={switch_t:.3f}, σ={_sigma}")

        # ---- tile 网格 ----
        h_tiles = self._make_tile_grid(H, lt_size, lt_stride)
        w_tiles = self._make_tile_grid(W, lt_size, lt_stride)
        self.print(f"   Tile grid: {len(h_tiles)}×{len(w_tiles)} "
                   f"({lt_size}×{lt_size} latent, stride={lt_stride})")

        # ---- Per-tile DINOv2 features（预计算一次，VOSR 风格） ----
        AE_FACTOR = 8
        tile_venc = {}
        use_venc = hasattr(self, 'venc') and self.venc is not None and lr_pixel is not None
        if use_venc:
            with torch.no_grad():
                for hs, he in h_tiles:
                    for ws, we in w_tiles:
                        ph_s, pw_s = hs * AE_FACTOR, ws * AE_FACTOR
                        ph_e = min(he * AE_FACTOR, lr_pixel.shape[2])
                        pw_e = min(we * AE_FACTOR, lr_pixel.shape[3])
                        lq_crop = lr_pixel[:, :, ph_s:ph_e, pw_s:pw_e]
                        tile_venc[(hs, ws)] = self.venc(lq_crop)
            self.print(f"   DINOv2: {len(tile_venc)} per-tile features extracted")

        # ---- 初始状态: LR latent + 扰动 (Residual FM 起点) ----
        generator = torch.Generator(device=device).manual_seed(seed)
        x = z_lr + _sigma * torch.randn(B, C, H, W, generator=generator, device=device)

        # ---- 高斯融合权重 ----
        g_weight = self._gaussian_weights(lt_size, lt_size, C, device)

        # ---- 逐步逆流积分 ----
        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        n_embed, n_flux = 0, 0

        for t_curr, t_prev in tqdm(step_pairs, desc="Tiled flow", total=len(step_pairs), leave=False):
            t_batch = torch.full((B,), t_curr, device=device)
            dt = t_prev - t_curr

            is_embedder_step = t_curr > switch_t
            if is_embedder_step:
                n_embed += 1
            else:
                n_flux += 1

            # 累积器
            v_acc = torch.zeros(B, C, H, W, device=device)
            w_acc = torch.zeros(B, C, H, W, device=device)

            for hs, he in h_tiles:
                for ws, we in w_tiles:
                    x_tile = x[:, :, hs:he, ws:we]       # [B, C, lt_size, lt_size]
                    z_lr_tile = z_lr[:, :, hs:he, ws:we]

                    # 该 tile 对应的 DINOv2 特征
                    tile_fea = tile_venc.get((hs, ws), None) if use_venc else None

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if is_embedder_step:
                            # Phase 1: DiT Embedder + 该 tile 独立的 DINOv2 语义条件
                            if hasattr(self, 'resflow'):
                                v_tile = self.resflow(x_tile, t_batch, z_lr_tile, venc_fea=tile_fea).float()
                            else:
                                v_tile = torch.zeros_like(x_tile)
                        else:
                            # Phase 2: Flux
                            tile_hp, tile_wp = x_tile.shape[2] // 2, x_tile.shape[3] // 2
                            tile_img_ids = self._make_img_ids(B, tile_hp, tile_wp, device)
                            x_packed = self._pack(x_tile.to(flux_dtype))
                            v_flux_packed = self.model(
                                img=x_packed, img_ids=tile_img_ids,
                                txt=txt, txt_ids=txt_ids,
                                timesteps=t_batch.to(flux_dtype), y=vec, guidance=guidance_vec,
                            )
                            v_tile = self._unpack(v_flux_packed.float(), tile_hp, tile_wp)
                            v_tile = torch.nan_to_num(v_tile, nan=0.0, posinf=10.0, neginf=-10.0)
                            v_tile = v_tile.clamp(-20.0, 20.0)

                    # 高斯加权累加
                    v_acc[:, :, hs:he, ws:we] += v_tile * g_weight
                    w_acc[:, :, hs:he, ws:we] += g_weight

            # 归一化 + Euler 步
            v_total = v_acc / w_acc.clamp(min=1e-8)
            x = x + dt * v_total

        self.print(f"   Steps: {n_embed} DiT + {n_flux} Flux")
        return x

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def gen_image(self, prompt="", neg_prompt="", steps=None,
                  cfg_scale=CFG_SCALE, seed=SEED, out_dir=OUTDIR,
                  init_image=None, scale=1.0, switch_t=0.5, shift=True,
                  chopping_enabled=False, tile_size=512, tile_stride=256,
                  color_correction='none', flow_sigma=None):
        """img2img 超分/增强。"""
        if init_image is None:
            raise ValueError("必须提供 init_image 参数。")

        _steps = steps if steps is not None else INFER_STEPS
        _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA

        image = self._gen_img2img(
            init_image, scale, switch_t, seed, _steps, cfg_scale, shift,
            chopping_enabled, tile_size, tile_stride, color_correction,
            _sigma,
        )

        base_name = os.path.splitext(os.path.basename(init_image))[0]
        save_path = os.path.join(out_dir, f"{base_name}.png")
        self.print(f"Saving to {save_path}")
        image.save(save_path)
        self.print("Done")

    def _gen_img2img(self, init_image, scale, switch_t, seed,
                     steps, cfg_scale, shift,
                     chopping_enabled, tile_size, tile_stride,
                     color_correction='none', flow_sigma=None) -> Image.Image:
        """img2img 核心逻辑：VAE 全局编码 → latent 分 tile 推理 → 高斯融合 → VAE 全局解码。"""
        AE_FACTOR = 8
        PATCH_SIZE = 2  # LightningDiT patch size

        src_image = Image.open(init_image).convert("RGB")
        exact_w = int(src_image.size[0] * scale)
        exact_h = int(src_image.size[1] * scale)

        target_image = src_image.resize((exact_w, exact_h), Image.BICUBIC)
        im_np = np.array(target_image).astype(np.float32) / 255.0
        im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0)
        im_cond = im_cond.to(dtype=torch.bfloat16, device=self.denoise_device)

        ori_h, ori_w = im_cond.shape[-2:]

        # ---- 对齐到 16 的倍数 ----
        mod_pixel = 16
        h, w = im_cond.shape[-2:]
        pad_h = (math.ceil(h / mod_pixel) * mod_pixel) - h
        pad_w = (math.ceil(w / mod_pixel) * mod_pixel) - w
        if pad_h > 0 or pad_w > 0:
            im_cond = F.pad(im_cond, (0, pad_w, 0, pad_h), mode='reflect')
            self.print(f"Align pad: +({pad_w},{pad_h}) → {im_cond.shape[-1]}x{im_cond.shape[-2]}")

        # ---- VAE 全局编码一次 ----
        image_tensor = im_cond * 2.0 - 1.0
        z_lr = self.vae_encode_tensor(image_tensor)  # [1, 16, H/8, W/8]
        lh, lw = z_lr.shape[2], z_lr.shape[3]

        # ---- LR 像素空间（VOSR 风格：由各推理函数自行决定 DINOv2 提取策略） ----
        lr_pixel = None
        if hasattr(self, 'venc') and self.venc is not None:
            lr_pixel = (im_cond + 1.0) / 2.0   # [-1,1] bf16 → [0,1] float

        # ---- latent tile 参数 ----
        lt_size = max((tile_size // AE_FACTOR // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
        lt_stride = max((tile_stride // AE_FACTOR // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
        lt_size = min(lt_size, min(lh, lw))
        lt_stride = min(lt_stride, lt_size)

        use_tiling = chopping_enabled and (lh > lt_size or lw > lt_size)

        if not use_tiling:
            print(f"📐 整张 latent 推理: {ori_w}x{ori_h} (latent {lw}×{lh}), "
                  f"switch_t={switch_t:.2f}, {steps} steps")
            z_hr = self.reverse_flow_sampling(
                z_lr, switch_t=switch_t,
                steps=steps, guidance=cfg_scale, shift=shift, seed=seed,
                lr_pixel=lr_pixel, flow_sigma=flow_sigma,
            )
        else:
            print(f"📐 Latent tiling 推理: {ori_w}x{ori_h} (latent {lw}×{lh}), "
                  f"tile={lt_size}×{lt_size} latent (~{tile_size}px), stride={lt_stride}")
            z_hr = self.reverse_flow_sampling_tiled(
                z_lr, switch_t=switch_t,
                steps=steps, guidance=cfg_scale, shift=shift, seed=seed,
                lt_size=lt_size, lt_stride=lt_stride,
                lr_pixel=lr_pixel, flow_sigma=flow_sigma,
            )

        # ---- VAE 全局解码一次 ----
        res_sr = self.vae_decode_tensor(z_hr)

        # ---- crop 回精确目标尺寸 ----
        res_sr = res_sr[:, :, 0:ori_h, 0:ori_w]
        self.print(f"Cropped to exact size: {res_sr.shape[-1]}x{res_sr.shape[-2]}")

        image = torch.clamp(res_sr, 0.0, 1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().float().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        sr_image = Image.fromarray(decoded_np)

        # ---- 颜色校正 ----
        if color_correction != 'none':
            sr_image = apply_color_fix(sr_image, target_image, method=color_correction)

        return sr_image


#################################################################################################
### CLI 入口
#################################################################################################


@torch.no_grad()
def main(
    model_name=MODEL_NAME,
    prompt=PROMPT,
    output=OUTDIR,
    seed=SEED,
    steps=None,
    cfg=None,
    verbose=VERBOSE,
    text_encoder_device=TEXT_ENCODER_DEVICE,
    denoise_device=DENOISE_DEVICE,
    input=None,
    scale=SCALE,
    switch_t=SWITCH_T,
    shift=SHIFT,
    t5_max_length=T5_MAX_LENGTH,
    chopping_enabled=CHOPPING_ENABLED,
    tile_size=TILE_SIZE,
    tile_stride=TILE_STRIDE,
    no_resflow=None,
    color_correction=COLOR_CORRECTION,
    flow_sigma=None,
):
    """Flux 两阶段 Residual Flow 超分推理 (ResFlow + Flux)。

    Phase 1 (t ∈ [switch_t, 1.0]): ResFlow — z_lr+noise 起点，Residual FM
    Phase 2 (t ∈ [0, switch_t]):   Flux    — 质量精修
    """
    init_image = input
    _steps = steps if steps is not None else INFER_STEPS
    _cfg = cfg if cfg is not None else CFG_SCALE
    _sigma = flow_sigma if flow_sigma is not None else FLOW_SIGMA

    src_path = Path(init_image) if init_image else None
    if src_path is None:
        raise ValueError("必须提供 --input 参数。")
    if src_path.is_dir():
        IMG_EXTS = {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff"}
        img_files = []
        for ext in IMG_EXTS:
            img_files.extend(glob.glob(str(src_path / f"**/{ext}"), recursive=True))
        img_files = sorted(set(img_files))
        if not img_files:
            raise ValueError(f"文件夹 {init_image} 中未找到图片文件")
    else:
        img_files = [init_image]

    print(f"\n{'=' * 50}")
    print(f"Flux + ResFlow 两阶段推理")
    if src_path.is_dir():
        print(f"  Images:   {len(img_files)}")

    inferencer = FluxInferencer()

    # Phase 1: Prompt 编码
    inferencer.load_text_encoders(
        device=text_encoder_device, t5_max_length=t5_max_length,
        t5xxl_path=T5XXL_PATH, clip_path=CLIP_PATH,
    )
    inferencer.prepare_conditions(prompt)
    inferencer.free_text_encoders()

    # Phase 2: Flux + VAE (batch 时禁用 verbose 避免破坏进度条)
    is_batch = src_path.is_dir()
    inferencer.load(model_name, verbose and not is_batch, denoise_device)

    # Phase 3: ResFlow
    resflow_path = RESFLOW_PATH
    _no_resflow = no_resflow if no_resflow is not None else NO_RESFLOW
    if _no_resflow:
        print("⏭️  ResFlow disabled (pure Flux reverse-flow).")
    elif resflow_path:
        inferencer.load_resflow(resflow_path)
    else:
        print("⚠️  ResFlow not configured.")

    # Phase 4: 推理
    out_root = Path(output)
    pbar = tqdm(img_files, desc="Inference", unit="img")
    for img_path in pbar:
        rel_path = os.path.relpath(img_path, init_image) if src_path.is_dir() else os.path.basename(img_path)
        base_name = os.path.splitext(rel_path)[0]
        save_path = out_root / f"{base_name}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pbar.set_postfix_str(rel_path[:40])
        try:
            inferencer.gen_image(
                prompt, "", _steps, _cfg, seed, str(save_path.parent),
                init_image=img_path, scale=scale, switch_t=switch_t, shift=shift,
                chopping_enabled=chopping_enabled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                color_correction=color_correction,
                flow_sigma=_sigma,
            )
        except Exception as e:
            tqdm.write(f"  ❌ {rel_path}: {e}")

    print(f"\nDone. 输出目录: {out_root.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
