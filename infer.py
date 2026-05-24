"""
Flux img2img 超分推理脚本 — 逆流积分版本

基于 HR→LR 整流流：t=0→HR, t=1→LR
推理时从 z_lr (t=1) 逆流积分到 t=0 得到 z_hr:
  x_{t-dt} = x_t - dt · (Flux(x_t, t) + embedder(x_t, t, z_lr))

使用方法:
  python infer.py --init_image input.png --scale 2.0
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
from models.flow_embedder import create_flow_embedder
from utils.image_spliter import ImageSpliterTh


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
STEPS = _CFG.get("steps", 28)
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
CHOPPING_PCH_SIZE = _CHOPPING_CFG.get("pch_size", 1024)
CHOPPING_STRIDE_RATIO = _CHOPPING_CFG.get("stride_ratio", 0.5)
CHOPPING_EXTRA_BS = _CHOPPING_CFG.get("extra_bs", 1)
CHOPPING_WEIGHT_TYPE = _CHOPPING_CFG.get("weight_type", "Gaussian")

FLOW_EMBEDDER_PATH = _CFG.get("flow_embedder_path", None)
NO_FLOW_EMBEDDER = _CFG.get("no_flow_embedder", False)
T5_MAX_LENGTH = _CFG.get("t5_max_length", 512)

# 推理步数（逆流积分步数）
INFER_STEPS = _CFG.get("infer_steps", 28)


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
        """加载训练好的 Flow Embedder 速度校正模块。"""
        if config_path is None:
            config_path = os.path.join(_SCRIPT_DIR, "configs", "flow_embedder.yaml")
        print(f"Loading FlowEmbedder from {ckpt_path} ...")
        self.flow_embedder = create_flow_embedder(config_path)
        sd = safe_load(ckpt_path)
        self.flow_embedder.load_state_dict(sd, strict=True)
        self.flow_embedder = self.flow_embedder.to(self.denoise_device, dtype=torch.bfloat16)
        self.flow_embedder.eval()
        n = sum(p.numel() for p in self.flow_embedder.parameters()) / 1e6
        print(f"  FlowEmbedder params: {n:.2f}M")
        print("✅ FlowEmbedder loaded.")

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
    # 逆流积分采样（新核心逻辑）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reverse_flow_sampling(self, z_lr: torch.Tensor, lr_image: torch.Tensor,
                               steps: int = 28, guidance: float = 3.5,
                               shift: bool = True) -> torch.Tensor:
        """
        从 LR 逆流积分到 HR。

        流方向: t=0→HR, t=1→LR
        逆流:   从 z_lr (t=1) 开始，逐步积分到 t=0:
                 x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_i) · v(x_{t_i}, t_i)

        v(x, t) = Flux(x, t) + embedder(x, t, lr_image)

        Args:
            z_lr:     [B, 16, H, W]      spatial latent of LR (initial state)
            lr_image: [B, 3, H_img, W_img] raw LR image [-1, 1] (embedder context)
            steps:    逆流积分步数
            guidance: CFG 权重
            shift:    是否启用 time shift

        Returns:
            z_hr: [B, 16, H, W] spatial latent of HR
        """
        device = z_lr.device
        B, C, H, W = z_lr.shape
        h_pack, w_pack = H // 2, W // 2
        seq_len = h_pack * w_pack

        # ---- 准备条件 ----
        flux_dtype = torch.bfloat16
        txt = self._cached_txt.expand(B, -1, -1).to(device, dtype=flux_dtype)
        txt_ids = self._cached_txt_ids.expand(B, -1, -1).to(device)
        vec = self._cached_vec.expand(B, -1).to(device, dtype=flux_dtype)
        guidance_vec = torch.full((B,), guidance, device=device, dtype=flux_dtype)
        img_ids = self._make_img_ids(B, h_pack, w_pack, device)

        # ---- 时间步调度：从 1.0 到 0.0 ----
        timesteps = self._get_schedule(steps, seq_len, shift=shift)  # [1.0, ..., 0.0]
        self.print(f"   Reverse flow: {steps} steps, t: 1.0 → 0.0{', shift' if shift else ''}")

        # ---- 初始状态 x = z_lr (t=1) ----
        x = z_lr.clone()  # [B, 16, H, W], t=1

        # ---- 逐步逆流积分（autocast 统一 dtype，与训练保持一致）----
        step_pairs = list(zip(timesteps[:-1], timesteps[1:]))  # [(1.0, t1), (t1, t2), ..., (t_{-1}, 0.0)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t_curr, t_prev in tqdm(step_pairs, desc="Reverse flow", total=len(step_pairs), leave=False):
                # Flux 速度预测
                t_batch = torch.full((B,), t_curr, device=device)
                x_packed = self._pack(x.to(flux_dtype))
                v_flux_packed = self.model(
                    img=x_packed, img_ids=img_ids,
                    txt=txt, txt_ids=txt_ids,
                    timesteps=t_batch.to(flux_dtype), y=vec, guidance=guidance_vec,
                )
                v_flux = self._unpack(v_flux_packed.float(), h_pack, w_pack)  # [B, 16, H, W]

                # Embedder 速度校正（原始 LR 图像作为条件上下文）
                if hasattr(self, 'flow_embedder'):
                    v_corr = self.flow_embedder(x, t_batch, lr_image, v_flux)
                    v_corr = v_corr.float()
                else:
                    v_corr = torch.zeros_like(v_flux)

                # 总速度
                v_total = v_flux + v_corr

                # Euler 步: dt = t_prev - t_curr < 0 (逆流)
                dt = t_prev - t_curr
                x = x + dt * v_total

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
    # 处理 patch
    # ------------------------------------------------------------------

    def process_patch(self, image_patch, seed, prompt, neg_prompt,
                      steps, cfg_scale, shift=True):
        """处理单个 patch：VAE encode → 逆流积分 → VAE decode。

        Args:
            image_patch: [B,3,H,W] 像素 tensor [0,1]
            steps: 逆流积分步数
        """
        device = self.denoise_device
        batch_size = image_patch.shape[0]

        # 1) [0,1] → [-1,1] → VAE encode
        image_tensor = image_patch * 2.0 - 1.0  # [B, 3, H, W] raw LR image
        z_lr = self.vae_encode_tensor(image_tensor)  # [B, 16, H//8, W//8] LR latent

        # 2) 逆流积分: z_lr(t=1) → z_hr(t=0)
        #    z_lr: initial state (latent), image_tensor: raw LR for embedder context
        z_hr = self.reverse_flow_sampling(
            z_lr, image_tensor, steps=steps, guidance=cfg_scale, shift=shift,
        )

        # 3) VAE decode → [0,1]
        result = self.vae_decode_tensor(z_hr)
        return result

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def gen_image(self, prompt="", neg_prompt="", steps=None,
                  cfg_scale=CFG_SCALE, seed=SEED, out_dir=OUTDIR,
                  init_image=None, scale=1.0, shift=True,
                  chopping_enabled=False, chopping_pch_size=512,
                  chopping_stride_ratio=0.5, chopping_extra_bs=1,
                  chopping_weight_type='Gaussian'):
        """img2img 超分/增强。"""
        if init_image is None:
            raise ValueError("必须提供 init_image 参数。")

        _steps = steps if steps is not None else INFER_STEPS

        image = self._gen_img2img(
            init_image, scale, seed, prompt, neg_prompt,
            _steps, cfg_scale, shift,
            chopping_enabled, chopping_pch_size,
            chopping_stride_ratio, chopping_extra_bs,
            chopping_weight_type,
        )

        base_name = os.path.splitext(os.path.basename(init_image))[0]
        save_path = os.path.join(out_dir, f"{base_name}_sr.png")
        self.print(f"Saving to {save_path}")
        image.save(save_path)
        self.print("Done")

    def _gen_img2img(self, init_image, scale, seed, prompt, neg_prompt,
                     steps, cfg_scale, shift,
                     chopping_enabled, chopping_pch_size,
                     chopping_stride_ratio, chopping_extra_bs,
                     chopping_weight_type) -> Image.Image:
        """img2img 核心逻辑。"""
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

        idle_pch_size = chopping_pch_size

        use_chopping = (
            chopping_enabled
            and not (ori_h <= idle_pch_size and ori_w <= idle_pch_size)
        )

        if not use_chopping:
            print(f"📐 img2img 整张推理: {ori_w}x{ori_h}, {steps} steps")
            res_sr = self.process_patch(
                im_cond, seed, prompt, neg_prompt, steps, cfg_scale, shift,
            )
        else:
            stride = int(idle_pch_size * chopping_stride_ratio)
            print(f"📐 img2img 分块推理: {ori_w}x{ori_h}, pch={idle_pch_size}, stride={stride}")

            im_spliter = ImageSpliterTh(
                im_cond, pch_size=idle_pch_size, stride=stride, sf=1,
                extra_bs=chopping_extra_bs, weight_type=chopping_weight_type,
            )

            total_patches = len(im_spliter)
            patch_idx = 0
            for im_pch, index_infos in im_spliter:
                patch_idx += len(index_infos)
                print(f"   🧩 Patch {patch_idx}/{total_patches} ({im_pch.shape[-1]}x{im_pch.shape[-2]})")
                res_pch = self.process_patch(
                    im_pch, seed, prompt, neg_prompt, steps, cfg_scale, shift,
                )
                im_spliter.update(res_pch, index_infos)

            res_sr = im_spliter.gather()

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
    shift=SHIFT,
    t5_max_length=T5_MAX_LENGTH,
    chopping_enabled=CHOPPING_ENABLED,
    chopping_pch_size=CHOPPING_PCH_SIZE,
    chopping_stride_ratio=CHOPPING_STRIDE_RATIO,
    chopping_extra_bs=CHOPPING_EXTRA_BS,
    chopping_weight_type=CHOPPING_WEIGHT_TYPE,
    no_flow_embedder=False,
):
    """Flux img2img 超分推理入口（逆流积分版本）。

    流方向: HR(t=0) → LR(t=1)
    推理:   从 LR(t=1) 逆流积分到 HR(t=0)
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

    # Phase 3: Flow Embedder（速度校正模块）
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
        init_image=init_image, scale=scale, shift=shift,
        chopping_enabled=chopping_enabled,
        chopping_pch_size=chopping_pch_size,
        chopping_stride_ratio=chopping_stride_ratio,
        chopping_extra_bs=chopping_extra_bs,
        chopping_weight_type=chopping_weight_type,
    )


if __name__ == "__main__":
    fire.Fire(main)
