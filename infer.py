"""
Flux img2img 超分推理脚本
基于 black-forest-labs/flux 官方实现，适配 img2img 超分/增强

使用方法:
  1. python infer.py --init_image input.png --scale 2.0
  2. 首次运行会自动下载 flux-dev 权重（需 HF token）
  3. 修改 configs/infer.yaml 可设置默认参数
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

from safetensors.torch import load_file as safe_load
from flux.util import load_flow_model, load_t5, load_clip, load_ae
from flux.sampling import denoise, get_schedule, get_noise, unpack
from models.flow_embedder import create_flow_embedder
from utils.image_spliter import ImageSpliterTh


#################################################################################################
### 配置文件加载 & 默认参数
#################################################################################################

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "configs", "infer.yaml")


def load_config(config_path: str = None) -> dict:
    path = config_path or _DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        print(f"⚠️  配置文件 {path} 不存在，使用内置默认值")
        return {}
    with open(path, "r", encoding="utf-8") as f:
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
START_TIMESTEP = _CFG.get("start_timestep", 1.0)
SHIFT = _CFG.get("shift", True)
TEXT_ENCODER_DEVICE = _CFG.get("text_encoder_device", "cuda")
DENOISE_DEVICE = _CFG.get("denoise_device", "cuda")

_CHOPPING_CFG = _CFG.get("chopping", {})
CHOPPING_ENABLED = _CHOPPING_CFG.get("enabled", False)
CHOPPING_PCH_SIZE = _CHOPPING_CFG.get("pch_size", 1024)
CHOPPING_STRIDE_RATIO = _CHOPPING_CFG.get("stride_ratio", 0.5)
CHOPPING_EXTRA_BS = _CHOPPING_CFG.get("extra_bs", 1)
CHOPPING_WEIGHT_TYPE = _CHOPPING_CFG.get("weight_type", "Gaussian")

# Flow Embedder 路径
FLOW_EMBEDDER_PATH = _CFG.get("flow_embedder_path", None)

# 最大 T5 序列长度（dev 推荐 512，schnell 推荐 256）
T5_MAX_LENGTH = _CFG.get("t5_max_length", 512)


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

    def load(self, model_name=MODEL_NAME, verbose=False,
             denoise_device="cuda"):
        """加载 Flux 主模型 + VAE。文本编码器由 load_text_encoders 单独加载。"""
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
        """加载训练好的 Flow Embedder Δ_φ 修正模块。"""
        if config_path is None:
            config_path = os.path.join(_SCRIPT_DIR, "configs", "flow_embedder.yaml")
        print(f"Loading FlowEmbedder from {ckpt_path} ...")
        self.flow_embedder = create_flow_embedder(config_path)
        sd = safe_load(ckpt_path)
        self.flow_embedder.load_state_dict(sd, strict=True)
        self.flow_embedder = self.flow_embedder.to(self.denoise_device, dtype=torch.bfloat16)
        self.flow_embedder.eval()
        print(f"  FlowEmbedder params: {sum(p.numel() for p in self.flow_embedder.parameters())/1e6:.2f}M")
        print("✅ FlowEmbedder loaded.")

    def load_text_encoders(self, device="cuda", t5_max_length=T5_MAX_LENGTH,
                           t5xxl_path=None, clip_path=None):
        """在指定设备上加载 T5 + CLIP 文本编码器。"""
        print(f"Loading T5 ({torch.bfloat16}) -> {device}...")
        self.t5 = load_t5(device, max_length=t5_max_length, ckpt_path=t5xxl_path)
        print(f"Loading CLIP ({torch.bfloat16}) -> {device}...")
        self.clip = load_clip(device, ckpt_path=clip_path)
        print("✅ Text encoders loaded.")

    def free_text_encoders(self):
        print("Freeing text encoders...")
        del self.t5
        del self.clip
        self.t5 = None
        self.clip = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Text encoders freed.")

    def prepare_conditions(self, prompt):
        """使用 T5 + CLIP 编码 prompt，保存 txt/vec 到成员变量，不释放编码器。

        调用方应在得到编码结果后手动调用 free_text_encoders() 释放显存。
        """
        print("Encoding prompt with T5 and CLIP...")
        with torch.no_grad():
            self._prompt_bs = 1
            self._cached_txt = self.t5(prompt)   # [1, seq_len, 4096]
            self._cached_vec = self.clip(prompt)  # [1, 768]
        self._cached_txt_ids = torch.zeros(1, self._cached_txt.shape[1], 3,
                                           dtype=torch.float32)
        print("✅ Prompt encoded.")

    def do_sampling(self, packed_latent, packed_noise, img_ids,
                    txt, txt_ids, vec,
                    timesteps, guidance=4.0, start_timestep=1.0):
        """Flux 整流流去噪（支持 img2img 起始时间步）。

        Args:
            packed_latent: [B, seq_len, 64] VAE 编码后的 packed latent
            packed_noise:  [B, seq_len, 64] 噪声（与 latent 同 packed 格式）
            img_ids: RoPE 位置编码
            txt: T5 编码的文本
            txt_ids: 文本位置编码（全零）
            vec: CLIP 池化嵌入
            timesteps: 时间步调度（从 start_timestep 到 0）
            guidance: CFG 权重
            start_timestep: 起始时间 (0~1)，1.0=完全重绘，0.5=半保留原图

        Returns:
            去噪后的 packed latent [B, seq_len, 64]
        """
        device = packed_latent.device
        dtype = packed_latent.dtype

        # 初始状态：整流流插值 x = t * noise + (1-t) * latent
        if start_timestep < 1.0:
            t = timesteps[0]  # 调度已从 start_timestep 开始，time_shift 后的首值
            img = t * packed_noise + (1.0 - t) * packed_latent
            self.print(f"   start_timestep={start_timestep:.2f}, t={t:.4f}, "
                       f"steps: {len(timesteps) - 1}")
        else:
            img = packed_noise  # 纯噪声
            self.print(f"   完整去噪: {len(timesteps) - 1} steps")

        # 去噪
        result = denoise(
            self.model, img=img, img_ids=img_ids,
            txt=txt, txt_ids=txt_ids, vec=vec,
            timesteps=timesteps, guidance=guidance,
        )
        return result

    def vae_encode_tensor(self, image_tensor):
        """将像素 tensor [-1,1] 编码为 spatial latent [B,16,H,W]。"""
        self.ae = self.ae.to(self.denoise_device)
        return self.ae.encode(image_tensor)

    def vae_decode_tensor(self, latent):
        """将 spatial latent [B,16,H,W] 解码为 pixel tensor [0,1]。"""
        device = self.denoise_device
        self.ae = self.ae.to(device)
        latent = latent.to(device)
        image = self.ae.decode(latent)
        # [-1,1] → [0,1]
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        return image

    def process_patch(self, image_patch, seed, prompt, neg_prompt,
                      steps, cfg_scale, start_timestep=1.0, shift=True):
        """处理单个 patch：VAE encode → pack → 准备条件 → 去噪 → unpack → decode。

        Args:
            image_patch: [B,3,H,W] 像素 tensor [0,1]
            seed: 随机种子
            prompt: 正 prompt（字符串）
            neg_prompt: 负 prompt（字符串，未使用，Flux 不支持 CFG 负 prompt）
            steps: 采样步数
            cfg_scale: CFG 权重
            start_timestep: 起始时间步
            shift: 是否启用时间步 shift（高噪声区分配更多步数）

        Returns:
            [B,3,H,W] 像素 tensor [0,1]
        """
        device = self.denoise_device
        h_pix, w_pix = image_patch.shape[-2:]
        batch_size = image_patch.shape[0]

        # 1) [0,1] → [-1,1] 然后 VAE encode
        image_tensor = image_patch * 2.0 - 1.0
        latent = self.vae_encode_tensor(image_tensor)  # [B,16,H//8,W//8]
        b, c, h_lat, w_lat = latent.shape

        # 1.5) Flow Embedder 修正（若有）
        if hasattr(self, 'flow_embedder'):
            t_start = torch.full((b,), start_timestep, device=self.denoise_device, dtype=torch.bfloat16)
            with torch.no_grad():
                delta = self.flow_embedder(latent.to(torch.bfloat16), t_start)
            latent = latent + delta.to(latent.dtype)
            self.print(f"   FlowEmbedder applied, t={start_timestep}")

        # 2) 生成噪声（spatial 格式，与 latent 同尺寸）
        noise = get_noise(batch_size, h_pix, w_pix, device,
                          dtype=torch.bfloat16, seed=seed)

        # 3) Prepare 条件（使用预编码的 txt/vec，不重复调 T5/CLIP）
        with torch.no_grad():
            packed_latent = rearrange(
                latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
            )

            # img_ids: RoPE 位置编码（pack 后空间减半）
            h_pack, w_pack = h_lat // 2, w_lat // 2
            img_ids = torch.zeros(h_pack, w_pack, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h_pack)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w_pack)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
            img_ids = img_ids.to(device)

            # 使用预编码的 txt/vec，按需要拓展 batch
            txt = self._cached_txt
            vec = self._cached_vec
            txt_ids = self._cached_txt_ids
            if batch_size > 1:
                txt = repeat(txt, "1 ... -> bs ...", bs=batch_size)
                vec = repeat(vec, "1 ... -> bs ...", bs=batch_size)
                txt_ids = repeat(txt_ids, "1 ... -> bs ...", bs=batch_size)
            txt = txt.to(device)
            txt_ids = txt_ids.to(device)
            vec = vec.to(device)

        # 手动 pack noise（与 prepare 内部相同的 rearrange）
        packed_noise = rearrange(
            noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
        )

        # 4) 获取时间步调度（从 start_timestep 到 0，固定 steps 步）
        seq_len = packed_latent.shape[1]
        timesteps = get_schedule(steps, seq_len, start_timestep=start_timestep, shift=shift)

        # 5) 去噪（内部处理 img2img 混合）
        packed_result = self.do_sampling(
            packed_latent, packed_noise,
            img_ids, txt, txt_ids, vec,
            timesteps,
            guidance=cfg_scale,
            start_timestep=start_timestep,
        )

        # 6) Unpack → VAE decode
        spatial_result = unpack(packed_result, h_pix, w_pix)
        result = self.vae_decode_tensor(spatial_result)

        return result

    def gen_image(self, prompt="", neg_prompt="", steps=STEPS,
                  cfg_scale=CFG_SCALE, seed=SEED,
                  out_dir=OUTDIR,
                  init_image=None, scale=1.0, start_timestep=1.0, shift=True,
                  chopping_enabled=False, chopping_pch_size=512,
                  chopping_stride_ratio=0.5, chopping_extra_bs=1,
                  chopping_weight_type='Gaussian'):
        """img2img 超分/增强：输入图像 → 放大 + 增强 → 输出图像。

        输出尺寸 = 原图尺寸 × scale
          小图 → 直接整张 pipeline
          大图 → 像素空间分块推理 + 高斯加权融合
        """
        if init_image is None:
            raise ValueError("必须提供 init_image 参数。")

        image = self._gen_img2img(
            init_image, scale, seed, prompt, neg_prompt,
            steps, cfg_scale, start_timestep, shift,
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
                     steps, cfg_scale, start_timestep, shift,
                     chopping_enabled, chopping_pch_size,
                     chopping_stride_ratio, chopping_extra_bs,
                     chopping_weight_type) -> Image.Image:
        """img2img 核心逻辑。

        流程：
        1. 原图 resize 到目标尺寸 → [0,1] tensor
        2. 对齐到 8 的倍数（Flux VAE 下采样 8x，16 倍更安全）
        3. 判断是否需要 chopping
        4. crop 回精确目标尺寸
        """
        src_image = Image.open(init_image).convert("RGB")
        src_w, src_h = src_image.size

        exact_w = int(src_w * scale)
        exact_h = int(src_h * scale)

        # resize 到目标尺寸
        target_image = src_image.resize((exact_w, exact_h), Image.BICUBIC)
        im_np = np.array(target_image).astype(np.float32) / 255.0
        im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0)
        im_cond = im_cond.to(dtype=torch.bfloat16, device=self.denoise_device)

        ori_h, ori_w = im_cond.shape[-2:]

        # ---- 对齐到 16 的倍数（VAE 下采样 8x + Flux pack 2x = 16）----
        mod_pixel = 16
        h, w = im_cond.shape[-2:]
        pad_h = (math.ceil(h / mod_pixel) * mod_pixel) - h
        pad_w = (math.ceil(w / mod_pixel) * mod_pixel) - w
        if pad_h > 0 or pad_w > 0:
            im_cond = F.pad(im_cond, (0, pad_w, 0, pad_h), mode='reflect')
            self.print(f"Align pad: +({pad_w},{pad_h}) → {im_cond.shape[-1]}x{im_cond.shape[-2]}")

        idle_pch_size = chopping_pch_size

        # ---- 判断走哪条路径 ----
        use_chopping = (
            chopping_enabled
            and not (ori_h <= idle_pch_size and ori_w <= idle_pch_size)
        )

        if not use_chopping:
            # 路径 A：整张推理
            print(f"📐 img2img 整张推理: {ori_w}x{ori_h}")
            res_sr = self.process_patch(
                im_cond, seed, prompt, neg_prompt,
                steps, cfg_scale, start_timestep, shift,
            )
        else:
            # 路径 B：分块推理（ImageSpliterTh）
            stride = int(idle_pch_size * chopping_stride_ratio)
            print(f"📐 img2img 分块推理: {ori_w}x{ori_h} → aligned {im_cond.shape[-1]}x{im_cond.shape[-2]}, "
                  f"pch={idle_pch_size}, stride={stride}, sf=1")

            im_spliter = ImageSpliterTh(
                im_cond,
                pch_size=idle_pch_size,
                stride=stride,
                sf=1,
                extra_bs=chopping_extra_bs,
                weight_type=chopping_weight_type,
            )

            total_patches = len(im_spliter)
            patch_idx = 0
            for im_pch, index_infos in im_spliter:
                patch_idx += len(index_infos)
                print(f"   🧩 Processing patch {patch_idx}/{total_patches} "
                      f"({im_pch.shape[-1]}x{im_pch.shape[-2]})")

                res_pch = self.process_patch(
                    im_pch, seed, prompt, neg_prompt,
                    steps, cfg_scale, start_timestep, shift,
                )

                im_spliter.update(res_pch, index_infos)

            res_sr = im_spliter.gather()

        # ---- crop 回精确目标尺寸 ----
        res_sr = res_sr[
            :, :,
            0:ori_h,
            0:ori_w,
        ]
        self.print(f"Cropped to exact size: {res_sr.shape[-1]}x{res_sr.shape[-2]}")

        # tensor [0,1] → PIL Image
        image = torch.clamp(res_sr, 0.0, 1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().float().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        return Image.fromarray(decoded_np)


#################################################################################################
### CLI 入口
#################################################################################################


@torch.no_grad()
def main(
    config=None,
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
    start_timestep=START_TIMESTEP,
    shift=SHIFT,
    t5_max_length=T5_MAX_LENGTH,
    # ---- Chopping 参数 ----
    chopping_enabled=CHOPPING_ENABLED,
    chopping_pch_size=CHOPPING_PCH_SIZE,
    chopping_stride_ratio=CHOPPING_STRIDE_RATIO,
    chopping_extra_bs=CHOPPING_EXTRA_BS,
    chopping_weight_type=CHOPPING_WEIGHT_TYPE,
    no_flow_embedder=False,
):
    """Flux img2img 超分推理入口。

    参数优先级: CLI 参数 > configs/infer.yaml > 内置默认值。
    必须提供 --init_image 参数。

    Flux 使用 RoPE 位置编码，天然支持任意分辨率输入 ✅
    """
    # 自定义配置文件覆盖
    if config is not None:
        custom_cfg = load_config(config)
        model_name = model_name if model_name != MODEL_NAME else custom_cfg.get("model_name", model_name)
        prompt = prompt if prompt != PROMPT else custom_cfg.get("prompt", prompt)
        out_dir = out_dir if out_dir != OUTDIR else custom_cfg.get("out_dir", out_dir)
        seed = seed if seed != SEED else custom_cfg.get("seed", seed)
        verbose = verbose if verbose != VERBOSE else custom_cfg.get("verbose", verbose)
        text_encoder_device = text_encoder_device if text_encoder_device != TEXT_ENCODER_DEVICE else custom_cfg.get("text_encoder_device", text_encoder_device)
        denoise_device = denoise_device if denoise_device != DENOISE_DEVICE else custom_cfg.get("denoise_device", denoise_device)
        init_image = init_image if init_image is not None else custom_cfg.get("init_image", init_image)
        scale = scale if scale != 1.0 else custom_cfg.get("scale", scale)
        start_timestep = start_timestep if start_timestep != 1.0 else custom_cfg.get("start_timestep", start_timestep)
        shift = shift if shift != SHIFT else custom_cfg.get("shift", shift)
        t5_max_length = t5_max_length if t5_max_length != 512 else custom_cfg.get("t5_max_length", t5_max_length)
        custom_chopping = custom_cfg.get("chopping", {})
        if chopping_enabled == CHOPPING_ENABLED:
            chopping_enabled = custom_chopping.get("enabled", chopping_enabled)

    _steps = steps or STEPS
    _cfg = cfg or CFG_SCALE

    inferencer = FluxInferencer()

    # Phase 1: 加载文本编码器 → 编码 prompt → 释放编码器显存
    inferencer.load_text_encoders(device=text_encoder_device, t5_max_length=t5_max_length,
                                  t5xxl_path=T5XXL_PATH, clip_path=CLIP_PATH)
    inferencer.prepare_conditions(prompt)
    inferencer.free_text_encoders()

    # Phase 2: 加载 Flux + VAE（此时 T5/CLIP 已释放，显存只供 Flux ~12GB）
    inferencer.load(model_name, verbose, denoise_device)

    # Phase 2.5: 加载 Flow Embedder（若有）
    fe_path = FLOW_EMBEDDER_PATH
    if config is not None:
        fe_path = custom_cfg.get("flow_embedder_path", fe_path)
        if custom_cfg.get("no_flow_embedder", False):
            no_flow_embedder = True
    if fe_path and not no_flow_embedder:
        inferencer.load_flow_embedder(fe_path)
    elif no_flow_embedder:
        print("⏭️  Flow Embedder disabled.")

    # Phase 3: 推理
    os.makedirs(out_dir, exist_ok=True)

    inferencer.gen_image(
        prompt, "", _steps, _cfg, seed, out_dir,
        init_image=init_image, scale=scale, start_timestep=start_timestep, shift=shift,
        chopping_enabled=chopping_enabled,
        chopping_pch_size=chopping_pch_size,
        chopping_stride_ratio=chopping_stride_ratio,
        chopping_extra_bs=chopping_extra_bs,
        chopping_weight_type=chopping_weight_type,
    )


if __name__ == "__main__":
    fire.Fire(main)
