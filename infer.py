"""
Flux 超分推理脚本 — DiT Embedder 两阶段流版本

两阶段逆流积分:
  Phase 1 (t=1 → switch_t): DiT Embedder — 纯噪声起点，LR latent 条件，锚定结构
  Phase 2 (switch_t → t=0): Flux         — 质量精修，去伪影、加细节

用法:
  python infer.py --init_image input.png --scale 2.0 --switch_t 0.5
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import math
import fire
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm

from safetensors.torch import load_file as safe_load
from flux.util import load_flow_model, load_t5, load_clip, load_ae
from models.dit_flow_embedder import create_dit_flow_embedder


#################################################################################################
### 配置文件加载 & 默认参数
#################################################################################################

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "infer.yaml")


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
OUTDIR = _CFG.get("out_dir", "outputs")
VERBOSE = _CFG.get("verbose", False)
SCALE = _CFG.get("scale", 1.0)
SHIFT = _CFG.get("shift", True)
TEXT_ENCODER_DEVICE = _CFG.get("text_encoder_device", "cuda")
DENOISE_DEVICE = _CFG.get("denoise_device", "cuda")

_CHOPPING_CFG = _CFG.get("chopping", {})
CHOPPING_ENABLED = _CHOPPING_CFG.get("enabled", False)
TILE_SIZE = _CHOPPING_CFG.get("tile_size", 512)          # pixel-space tile size
TILE_OVERLAP = _CHOPPING_CFG.get("tile_overlap", 4)      # pixel-space overlap

FLOW_EMBEDDER_PATH = _CFG.get("flow_embedder_path", None)
NO_FLOW_EMBEDDER = _CFG.get("no_flow_embedder", False)
T5_MAX_LENGTH = _CFG.get("t5_max_length", 512)

# 推理步数（逆流积分步数）
INFER_STEPS = _CFG.get("infer_steps", 28)

# 两阶段切换时间点 switch_t ∈ [0,1]
# t ∈ (switch_t, 1.0] → DiT Embedder 锚定结构
# t ∈ [0, switch_t]   → Flux 提升质量
SWITCH_T = _CFG.get("switch_t", 0.5)


#################################################################################################
### 主推理逻辑
#################################################################################################


class FluxInferencer:

    def __init__(self):
        self.verbose = False
        self.model_name = "flux-dev"

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

    def load_flow_embedder(self, ckpt_path: str, config_path: str = None):
        """加载训练好的 DiT Flow Embedder。"""
        if config_path is None:
            config_path = os.path.join(_SCRIPT_DIR, "configs", "flow_embedder.yaml")
        print(f"Loading DiT FlowEmbedder from {ckpt_path} ...")

        self.flow_embedder = create_dit_flow_embedder(config_path)
        sd = safe_load(ckpt_path)
        self.flow_embedder.load_state_dict(sd, strict=True)
        self.flow_embedder = self.flow_embedder.to(self.denoise_device, dtype=torch.float32)
        self.flow_embedder.eval()
        n = sum(p.numel() for p in self.flow_embedder.parameters()) / 1e6
        print(f"  Params: {n:.2f}M")
        print("✅ DiT FlowEmbedder loaded.")

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
                               shift: bool = True, seed: int = 42) -> torch.Tensor:
        """
        两阶段逆流积分: t=1(纯噪声) → t=0(HR)

        Phase 1 (t ∈ [switch_t, 1.0]): DiT Embedder — 纯噪声→锚定结构 (LR latent 条件)
        Phase 2 (t ∈ [0, switch_t]):   Flux         — 质量精修
        """
        device = z_lr.device
        B, C, H, W = z_lr.shape
        h_pack, w_pack = H // 2, W // 2
        seq_len = h_pack * w_pack

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
                   f"{', shift' if shift else ''}, switch_t={switch_t:.3f}")

        # ---- 初始状态: 纯噪声 ----
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(B, C, H, W, generator=generator, device=device)

        # ---- 逐步逆流积分 ----
        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        n_embed, n_flux = 0, 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t_curr, t_prev in tqdm(step_pairs, desc="2-stage flow", total=len(step_pairs), leave=False):
                t_batch = torch.full((B,), t_curr, device=device)
                dt = t_prev - t_curr

                if t_curr > switch_t:
                    # Phase 1: DiT Embedder — LR latent 条件
                    if hasattr(self, 'flow_embedder'):
                        v_total = self.flow_embedder(x, t_batch, z_lr).float()
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
    def _make_tile_grid(length: int, tile: int, overlap: int):
        """返回覆盖整个维度的 (start, end) tile 位置列表。每个 tile 大小 = tile。"""
        stride = max(tile - overlap, 1)
        if length <= tile:
            return [(0, length)]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] + tile < length:
            positions.append(length - tile)
        return [(p, p + tile) for p in sorted(set(positions))]

    @staticmethod
    def _gaussian_weights(tile_h: int, tile_w: int, channels: int, device: torch.device):
        """2D 高斯融合权重，中心最高、边缘衰减。"""
        var = 0.01
        mid_h, mid_w = (tile_h - 1) / 2, (tile_w - 1) / 2
        y = torch.arange(tile_h, dtype=torch.float32, device=device)
        x = torch.arange(tile_w, dtype=torch.float32, device=device)
        wy = torch.exp(-((y - mid_h) / tile_h) ** 2 / (2 * var))
        wx = torch.exp(-((x - mid_w) / tile_w) ** 2 / (2 * var))
        w = wy[:, None] * wx[None, :]
        return w.unsqueeze(0).unsqueeze(0).expand(1, channels, -1, -1)

    # ------------------------------------------------------------------
    # Tiled reverse flow sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reverse_flow_sampling_tiled(
        self, z_lr: torch.Tensor,
        switch_t: float = 0.5, steps: int = 28, guidance: float = 3.5,
        shift: bool = True, seed: int = 42,
        lt_size: int = 64, lt_overlap: int = 8,
    ) -> torch.Tensor:
        """Latent 空间分 tile 的两阶段逆流积分（VOSR 风格）。"""
        device = z_lr.device
        B, C, H, W = z_lr.shape
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
                   f"{', shift' if shift else ''}, switch_t={switch_t:.3f}")
        self.print(f"   Tile grid: {len(self._make_tile_grid(H, lt_size, lt_overlap))}×"
                   f"{len(self._make_tile_grid(W, lt_size, lt_overlap))} "
                   f"({lt_size}×{lt_size} latent, overlap={lt_overlap})")

        # ---- 初始噪声 ----
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(B, C, H, W, generator=generator, device=device)

        # ---- 高斯融合权重 ----
        g_weight = self._gaussian_weights(lt_size, lt_size, C, device)

        # ---- tile 网格 ----
        h_tiles = self._make_tile_grid(H, lt_size, lt_overlap)
        w_tiles = self._make_tile_grid(W, lt_size, lt_overlap)

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

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if is_embedder_step:
                            # Phase 1: DiT Embedder
                            if hasattr(self, 'flow_embedder'):
                                v_tile = self.flow_embedder(x_tile, t_batch, z_lr_tile).float()
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
                  chopping_enabled=False, tile_size=512, tile_overlap=32):
        """img2img 超分/增强。"""
        if init_image is None:
            raise ValueError("必须提供 init_image 参数。")

        _steps = steps if steps is not None else INFER_STEPS

        image = self._gen_img2img(
            init_image, scale, switch_t, seed, _steps, cfg_scale, shift,
            chopping_enabled, tile_size, tile_overlap,
        )

        base_name = os.path.splitext(os.path.basename(init_image))[0]
        save_path = os.path.join(out_dir, f"{base_name}_sr.png")
        self.print(f"Saving to {save_path}")
        image.save(save_path)
        self.print("Done")

    def _gen_img2img(self, init_image, scale, switch_t, seed,
                     steps, cfg_scale, shift,
                     chopping_enabled, tile_size, tile_overlap) -> Image.Image:
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

        # ---- latent tile 参数 ----
        lt_size = max((tile_size // AE_FACTOR // PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)
        lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
        lt_size = min(lt_size, min(lh, lw))
        lt_overlap = min(lt_overlap, lt_size - 1)

        use_tiling = chopping_enabled and (lh > lt_size or lw > lt_size)

        if not use_tiling:
            print(f"📐 整张 latent 推理: {ori_w}x{ori_h} (latent {lw}×{lh}), "
                  f"switch_t={switch_t:.2f}, {steps} steps")
            z_hr = self.reverse_flow_sampling(
                z_lr, switch_t=switch_t,
                steps=steps, guidance=cfg_scale, shift=shift, seed=seed,
            )
        else:
            print(f"📐 Latent tiling 推理: {ori_w}x{ori_h} (latent {lw}×{lh}), "
                  f"tile={lt_size}×{lt_size} latent (~{tile_size}px), overlap={lt_overlap}")
            z_hr = self.reverse_flow_sampling_tiled(
                z_lr, switch_t=switch_t,
                steps=steps, guidance=cfg_scale, shift=shift, seed=seed,
                lt_size=lt_size, lt_overlap=lt_overlap,
            )

        # ---- VAE 全局解码一次 ----
        res_sr = self.vae_decode_tensor(z_hr)

        # ---- crop 回精确目标尺寸 ----
        res_sr = res_sr[:, :, 0:ori_h, 0:ori_w]
        self.print(f"Cropped to exact size: {res_sr.shape[-1]}x{res_sr.shape[-2]}")

        image = torch.clamp(res_sr, 0.0, 1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().float().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        return Image.fromarray(decoded_np)


#################################################################################################
### CLI 入口
#################################################################################################


@torch.no_grad()
def main(
    model_name=MODEL_NAME,
    prompt=PROMPT,
    out_dir=OUTDIR,
    seed=SEED,
    steps=None,
    cfg=None,
    verbose=VERBOSE,
    text_encoder_device=TEXT_ENCODER_DEVICE,
    denoise_device=DENOISE_DEVICE,
    init_image=None,
    scale=SCALE,
    switch_t=SWITCH_T,
    shift=SHIFT,
    t5_max_length=T5_MAX_LENGTH,
    chopping_enabled=CHOPPING_ENABLED,
    tile_size=TILE_SIZE,
    tile_overlap=TILE_OVERLAP,
    no_flow_embedder=False,
):
    """Flux 两阶段流超分推理 (DiT Embedder + Flux)。

    Phase 1 (t ∈ [switch_t, 1.0]): DiT Embedder — 纯噪声起点，LR latent 锚定结构
    Phase 2 (t ∈ [0, switch_t]):   Flux         — 质量精修
    """
    _steps = steps if steps is not None else INFER_STEPS
    _cfg = cfg if cfg is not None else CFG_SCALE

    inferencer = FluxInferencer()

    # Phase 1: Prompt 编码
    inferencer.load_text_encoders(
        device=text_encoder_device, t5_max_length=t5_max_length,
        t5xxl_path=T5XXL_PATH, clip_path=CLIP_PATH,
    )
    inferencer.prepare_conditions(prompt)
    inferencer.free_text_encoders()

    # Phase 2: Flux + VAE
    inferencer.load(model_name, verbose, denoise_device)

    # Phase 3: DiT Flow Embedder
    fe_path = FLOW_EMBEDDER_PATH
    if no_flow_embedder or NO_FLOW_EMBEDDER:
        print("⏭️  Flow Embedder disabled (pure Flux reverse-flow).")
    elif fe_path:
        inferencer.load_flow_embedder(fe_path)
    else:
        print("⚠️  Flow Embedder not configured.")

    # Phase 4: 推理
    os.makedirs(out_dir, exist_ok=True)
    inferencer.gen_image(
        prompt, "", _steps, _cfg, seed, out_dir,
        init_image=init_image, scale=scale, switch_t=switch_t, shift=shift,
        chopping_enabled=chopping_enabled,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )


if __name__ == "__main__":
    fire.Fire(main)
