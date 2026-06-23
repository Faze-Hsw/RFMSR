"""
RFMSR 训练脚本 — LightningDiT 速度预测 (SD2.1 VAE)

训练目标: RFMSR 学习 Residual Flow Matching — 从 LR→HR 的残差流。

Residual Flow 路径: t=0 → HR latent, t=1 → z_lr + σ·ε
  x_t = z_hr + t·(z_lr - z_hr) + t·σ·ε
  v_gt = (z_lr - z_hr) + σ·ε

架构 (LightningDiT):
  cat(z_lr[4ch], x_t[4ch]) → [B, 8, H, W]
    → PatchEmbed(patch=2) → tokens
    → LightningDiTBlock (Self-Attn + Cross-Attn(DINOv2) + SwiGLU + AdaLN)
    → unpatchify → v [4ch]

用法:
  python train_rfmsr.py
"""

import os
import math
import argparse
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from safetensors.torch import save_file as safe_save
from tqdm import tqdm
from PIL import Image

from diffusers import AutoencoderKL
from datapipe.train_dataloader import create_train_dataloader
from models.rfmsr import create_rfmsr
from models.dinov2_encoder import create_dinov2_encoder

# CUDA 优化
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# =========================================================================
# Trainer
# =========================================================================

class RFMSRTrainer:

    def __init__(self, resume_path: str = None):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(_script_dir, "configs", "train_rfmsr.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.resume_path = resume_path
        self.device = torch.device(self.cfg["denoise_device"])
        self._setup_seed(self.cfg["training"]["seed"])

        # Residual Flow Matching
        flow_cfg = self.cfg.get("flow", {})
        self.sigma = flow_cfg.get("sigma", 1.0)
        self.time_dist = flow_cfg.get("time_dist", "uniform")
        self.lognorm_mu = flow_cfg.get("lognorm_mu", 0.0)
        self.lognorm_sigma = flow_cfg.get("lognorm_sigma", 1.0)

        # ---- 实验目录 ----
        exp = self.cfg["experiment"]
        self.exp_dir = Path(exp["save_dir"])
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.log_freq = exp["log_freq"]
        self.save_freq = exp["save_freq"]
        self.keep_last_n = exp.get("keep_last_n", 0)

        # ---- 加载模块 ----
        self._load_vae()
        self._load_dinov2()
        self._build_rfmsr()
        self._build_optimizer()
        self._build_dataloader()
        self._build_ema()

        # AMP (bf16 autocast，不使用 GradScaler)
        self.use_amp = self.cfg["training"]["use_amp"]

        # 梯度累计
        tcfg = self.cfg["training"]
        self.accumulation_steps = tcfg.get("gradient_accumulation_steps", 1)

        # 验证指标（与 cal_metrics.py 计算逻辑完全一致）
        self.val_enabled = self.cfg.get("validation", {}).get("enabled", False)
        self.lpips_fn = None
        self.psnr_metric = None
        self.ssim_metric = None
        self.dists_metric = None
        self.niqe_metric = None
        self.musiq_metric = None
        self.maniqa_metric = None
        self.clipiqa_metric = None
        if self.val_enabled:
            self._init_val_metrics()

        # 训练状态
        self.global_step = 0
        self.accumulation_count = 0

        self._print_summary()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _load_vae(self):
        """加载 SD2.1 VAE 编解码器（冻结）。"""
        vae_path = self.cfg["vae_path"]
        device = self.device

        print(f"Loading SD2.1 VAE from {vae_path} -> {device} ...")
        self.ae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        self.ae = self.ae.to(device).eval()
        self.ae.requires_grad_(False)
        print(f"✅ VAE loaded, scaling_factor={self.ae.config.scaling_factor}")

    def _load_dinov2(self):
        """加载冻结的 DINOv2 语义编码器。"""
        dv2_cfg = self.cfg.get("dinov2", {}) or {}
        self.use_dinov2 = dv2_cfg.get("enabled", False)

        if self.use_dinov2:
            model_cfg_path = self.cfg["model_config"]
            self.venc = create_dinov2_encoder(model_cfg_path, device=self.device)
        else:
            self.venc = None

    def _build_rfmsr(self):
        cfg_path = self.cfg["model_config"]
        self.rfmsr = create_rfmsr(cfg_path).to(self.device)

        # 预训练权重初始化（兼容 VOSR checkpoint），resume 时跳过
        if self.resume_path:
            print("⏭️  Skipping pretrained init (will restore from resume checkpoint)")
        else:
            pretrained_ckpt = self.cfg.get("pretrained_ckpt", None)
            if pretrained_ckpt:
                self.rfmsr.load_pretrained(pretrained_ckpt)

        n_params = sum(p.numel() for p in self.rfmsr.parameters())
        print(f"✅ RFMSR: {n_params / 1e6:.1f}M params")

    def _build_optimizer(self):
        tcfg = self.cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.rfmsr.parameters(),
            lr=tcfg["lr"],
            betas=(tcfg["adam_beta1"], tcfg["adam_beta2"]),
            weight_decay=tcfg["adam_weight_decay"],
            eps=tcfg["adam_epsilon"],
        )

    def _build_dataloader(self):
        dcfg = self.cfg["data"]
        gt_size = dcfg["gt_size"]
        assert gt_size % 16 == 0, f"gt_size={gt_size} 必须能被 16 整除"
        self.dataloader = create_train_dataloader(
            data_dir=dcfg["hr_dir"],
            config_path=dcfg["degradation_config"],
            batch_size=dcfg["batch_size"],
            num_workers=dcfg["num_workers"],
            gt_size=gt_size,
            use_hflip=dcfg.get("use_hflip", True),
            use_rot=dcfg.get("use_rot", False),
        )
        print(f"✅ DataLoader: {len(self.dataloader)} batches/epoch")

    def _build_ema(self):
        rate = self.cfg["training"].get("ema_rate", 0)
        if rate > 0:
            self.ema_rate = rate
            self.ema_state = OrderedDict(
                {k: deepcopy(v.data) for k, v in self.rfmsr.state_dict().items()}
            )
        else:
            self.ema_rate = 0
            self.ema_state = None

    def _print_summary(self):
        tcfg = self.cfg["training"]
        dcfg = self.cfg["data"]
        accum = self.accumulation_steps
        print("\n" + "=" * 60)
        print(f"Experiment      : {self.cfg['experiment']['name']}")
        print(f"Save dir        : {self.exp_dir}")
        print(f"Iterations      : {tcfg['iterations']}")
        print(f"Batch size      : {dcfg['batch_size']}")
        print(f"Accum steps     : {accum}")
        print(f"Effective batch : {dcfg['batch_size'] * accum}")
        print(f"Learning Rate   : {tcfg['lr']}")
        print(f"Weight Decay    : {tcfg['adam_weight_decay']}")
        print(f"GT size         : {dcfg['gt_size']}")
        print(f"Flow            : Residual FM (LR→HR), σ={self.sigma}, t~{self.time_dist}")
        print(f"Conditioning    : LR latent (VAE-encoded upsampled LR)" + 
              (" + DINOv2 Cross-Attn" if self.use_dinov2 else ""))
        print(f"AMP             : {self.use_amp}")
        print("=" * 60 + "\n")

    def _sample_t(self, B: int, device: torch.device) -> torch.Tensor:
        """按配置的时间分布采样 t ∈ [0,1]。

        uniform:  t ~ U(0, 1)
        lognorm:  t = sigmoid(N(μ, σ))，中间密度更高
        """
        if self.time_dist == "lognorm":
            rnd = torch.randn(B, device=device)
            t = torch.sigmoid(rnd * self.lognorm_sigma + self.lognorm_mu)
        else:
            t = torch.rand(B, device=device)
        return t

    # ------------------------------------------------------------------
    # VAE encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] float [0,1] → [B,4,H/8,W/8] SD2.1 latent (scaled)."""
        img = img.to(device=self.device, dtype=torch.bfloat16)
        img = img * 2.0 - 1.0
        return self.ae.encode(img.float()).latent_dist.sample() * self.ae.config.scaling_factor

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        device = self.device

        hr = batch["gt"].to(device)      # [B, 3, H, W]  [0, 1]
        lr = batch["lq"].to(device)      # [B, 3, H, W]  [0, 1]

        # 1. VAE encode
        z_hr = self.vae_encode(hr)       # [B, 4, H/8, W/8]
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        z_lr = self.vae_encode(lr_up)    # [B, 4, H/8, W/8]

        z_hr = z_hr.detach().float()
        z_lr = z_lr.detach().float()
        B = z_hr.shape[0]

        # 2. DINOv2 语义特征（从像素空间 LR 提取）
        venc_fea = None
        if self.use_dinov2 and self.venc is not None:
            venc_fea = self.venc(lr)

        # 3. Residual Flow: x_t = z_hr + t·(z_lr - z_hr) + t·σ·ε
        t = self._sample_t(B, device)
        t_expand = t[:, None, None, None]
        epsilon = torch.randn_like(z_hr)
        residual = z_lr - z_hr
        x_t = z_hr + t_expand * residual + t_expand * self.sigma * epsilon
        v_gt = residual + self.sigma * epsilon

        # 4. RFMSR + Loss (bf16 autocast)
        #    梯度累计时 loss 需要 / accumulation_steps，保证有效梯度不变
        loss_scale = 1.0 / self.accumulation_steps
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            v_super = self.rfmsr(x_t, t, z_lr, venc_fea=venc_fea)
            loss = F.mse_loss(v_super, v_gt) * loss_scale

        losses = {"velo": loss.item() * self.accumulation_steps}  # 上报原始 scale

        # 5. 反向传播（仅 backward，accumulation 由 train 循环管理）
        loss.backward()

        return losses

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_state is None:
            return
        for k, v in self.rfmsr.state_dict().items():
            if v.is_floating_point():
                self.ema_state[k].mul_(self.ema_rate).add_(v.data, alpha=1 - self.ema_rate)
            else:
                self.ema_state[k] = v.data.clone()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int):
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 推理权重 (safetensors)
        weights = self.ema_state if self.ema_state is not None else self.rfmsr.state_dict()
        ema_path = ckpt_dir / f"rfmsr_step{step}.safetensors"
        safe_save(weights, ema_path)

        # 完整训练状态 (torch.save)
        state = {
            "step": step,
            "rfmsr": self.rfmsr.state_dict(),
            "ema_state": self.ema_state,
            "optimizer": self.optimizer.state_dict(),
        }
        state_path = ckpt_dir / f"training_state_step{step}.pth"
        torch.save(state, state_path)
        print(f"💾 Checkpoint saved: step {step}")

        # 清理旧检查点
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(ckpt_dir)

    def _cleanup_old_checkpoints(self, ckpt_dir: Path):
        """只保留最近 N 个检查点，删除其余。"""
        import re
        # 收集所有检查点文件，按 step 分组
        pattern = re.compile(r"(rfmsr_step|training_state_step)(\d+)")
        ckpt_steps: dict[int, list[Path]] = {}
        for f in ckpt_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                s = int(m.group(2))
                ckpt_steps.setdefault(s, []).append(f)

        # 按 step 排序，删除超出 keep_last_n 的旧检查点
        sorted_steps = sorted(ckpt_steps.keys(), reverse=True)
        for step in sorted_steps[self.keep_last_n:]:
            for f in ckpt_steps[step]:
                f.unlink()
            print(f"🗑️  Removed old checkpoint: step {step}")

    # ------------------------------------------------------------------
    # 验证
    # ------------------------------------------------------------------

    def _init_val_metrics(self):
        """初始化全量验证指标（与 cal_metrics.py 计算逻辑完全一致）。"""
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="alex").to(self.device)
        except ImportError:
            print("[WARN] lpips not installed, LPIPS will be skipped")
        try:
            import pyiqa
            # Full-reference（Y 通道，与 cal_metrics.py 一致）
            self.psnr_metric = pyiqa.create_metric(
                "psnr", test_y_channel=True, color_space="ycbcr", device=self.device)
            self.ssim_metric = pyiqa.create_metric(
                "ssim", test_y_channel=True, color_space="ycbcr", device=self.device)
            self.dists_metric = pyiqa.create_metric("dists", device=self.device)
            # No-reference
            self.niqe_metric = pyiqa.create_metric("niqe", device=self.device)
            self.musiq_metric = pyiqa.create_metric("musiq", device=self.device)
            self.maniqa_metric = pyiqa.create_metric("maniqa", device=self.device)
            self.clipiqa_metric = pyiqa.create_metric("clipiqa", device=self.device)
            print("✅ Validation metrics initialized: PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA")
        except ImportError:
            print("[WARN] pyiqa not installed, all FR/NR metrics will be skipped")

    @torch.no_grad()
    def validate(self, step: int):
        """验证：推理 test_lq → 计算 PSNR/SSIM/LPIPS/DISTS/NIQE/MUSIQ/MANIQA/CLIPIQA → 保存 SR 图片。"""
        self.rfmsr.eval()

        # ---- EMA swap: 验证时使用 EMA 权重，与 save_checkpoint 导出的推理权重一致 ----
        orig_state = None
        if self.ema_state is not None:
            orig_state = OrderedDict({k: v.data.clone() for k, v in self.rfmsr.state_dict().items()})
            self.rfmsr.load_state_dict(self.ema_state)

        val_cfg = self.cfg.get("validation", {})
        lq_dir = Path(val_cfg.get("lq_dir", "assets/validate_lq"))
        gt_dir = Path(val_cfg.get("gt_dir", "assets/validate_gt"))
        val_steps = val_cfg.get("steps", 15)
        val_scale = val_cfg.get("scale", 4.0)
        val_seed = val_cfg.get("seed", 42)
        max_images = val_cfg.get("max_images", 0)

        out_dir = self.exp_dir / "validation" / f"step_{step:08d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        SCALE = self.ae.config.scaling_factor
        MOD_PIXEL = 16

        lq_paths = sorted(lq_dir.glob("*.png"))
        pairs = []
        for lp in lq_paths:
            gp = gt_dir / lp.name
            if gp.exists():
                pairs.append((lp, gp))
        if max_images > 0:
            pairs = pairs[:max_images]
        total = len(pairs)
        if total == 0:
            print(f"[Val @ step {step}] No paired images found")
            if orig_state is not None:
                self.rfmsr.load_state_dict(orig_state)
            self.rfmsr.train()
            return

        # 全量指标收集（与 cal_metrics.py 一致）
        psnr_vals, ssim_vals, lpips_vals, dists_vals = [], [], [], []
        niqe_vals, musiq_vals, maniqa_vals, clipiqa_vals = [], [], [], []

        for lq_path, gt_path in tqdm(pairs, desc=f"Val@{step}", leave=False):
            # ---- 加载 & resize LR ----
            src = Image.open(lq_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            exact_w = int(src.size[0] * val_scale)
            exact_h = int(src.size[1] * val_scale)
            target = src.resize((exact_w, exact_h), Image.BICUBIC)
            ori_h, ori_w = target.size[1], target.size[0]  # PIL (w,h) → (h,w)

            im_np = np.array(target).astype(np.float32) / 255.0
            im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0)
            im_cond = im_cond.to(dtype=torch.bfloat16, device=self.device)

            # ---- 对齐到 16 倍数 ----
            h, w = im_cond.shape[-2:]
            pad_h = (math.ceil(h / MOD_PIXEL) * MOD_PIXEL) - h
            pad_w = (math.ceil(w / MOD_PIXEL) * MOD_PIXEL) - w
            if pad_h > 0 or pad_w > 0:
                im_cond = F.pad(im_cond, (0, pad_w, 0, pad_h), mode="reflect")

            # ---- VAE 编码 ----
            image_tensor = im_cond * 2.0 - 1.0
            z_lr = self.ae.encode(image_tensor.float()).latent_dist.sample() * SCALE

            # ---- DINOv2 特征 ----
            venc_fea = None
            if self.use_dinov2 and self.venc is not None:
                venc_fea = self.venc(im_cond.float())

            # ---- 逆流积分 (t=1 → t=0) ----
            B_v, C_v, H_v, W_v = z_lr.shape
            timesteps = torch.linspace(1.0, 0.0, val_steps + 1, device=self.device)
            generator = torch.Generator(device=self.device).manual_seed(val_seed)
            x = z_lr + self.sigma * torch.randn(
                B_v, C_v, H_v, W_v, generator=generator, device=self.device
            )

            for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
                t_batch = torch.full((B_v,), t_curr, device=self.device)
                dt = t_prev - t_curr
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    v = self.rfmsr(x, t_batch, z_lr, venc_fea=venc_fea).float()
                x = x + dt * v

            # ---- VAE 解码 ----
            latent = x / SCALE
            sr_decoded = self.ae.decode(latent).sample
            sr_decoded = torch.clamp((sr_decoded + 1.0) / 2.0, 0.0, 1.0)
            sr_decoded = sr_decoded[:, :, 0:ori_h, 0:ori_w]

            # ---- 保存 SR ----
            sr_np = (sr_decoded[0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            sr_np = np.moveaxis(sr_np, 0, 2)
            Image.fromarray(sr_np).save(out_dir / lq_path.name)

            # ---- 准备 GT tensor（所有 FR 指标共用） ----
            gt_np = np.array(gt_img).astype(np.float32) / 255.0
            gt_tensor = torch.from_numpy(np.moveaxis(gt_np, 2, 0)).unsqueeze(0)
            gt_tensor = gt_tensor.to(self.device)

            # ---- Full-Reference（与 cal_metrics.py 完全一致） ----
            if self.psnr_metric is not None:
                psnr_vals.append(self.psnr_metric(sr_decoded, gt_tensor).mean().item())
            if self.ssim_metric is not None:
                ssim_vals.append(self.ssim_metric(sr_decoded, gt_tensor).mean().item())
            if self.dists_metric is not None:
                dists_vals.append(self.dists_metric(sr_decoded, gt_tensor).mean().item())

            # ---- LPIPS (expects [-1, 1]) ----
            if self.lpips_fn is not None:
                gt_norm = (gt_tensor - 0.5) / 0.5
                sr_norm = (sr_decoded - 0.5) / 0.5
                lpips_vals.append(self.lpips_fn(gt_norm, sr_norm).mean().item())

            # ---- No-Reference（与 cal_metrics.py 完全一致） ----
            if self.niqe_metric is not None:
                niqe_vals.append(self.niqe_metric(sr_decoded).mean().item())
            if self.musiq_metric is not None:
                musiq_vals.append(self.musiq_metric(sr_decoded).mean().item())
            if self.maniqa_metric is not None:
                maniqa_vals.append(self.maniqa_metric(sr_decoded).mean().item())
            if self.clipiqa_metric is not None:
                clipiqa_vals.append(self.clipiqa_metric(sr_decoded).mean().item())

        # ---- 打印结果（与 cal_metrics.py 格式一致） ----
        print(f"\n[Val @ step {step}] images={total}")
        if psnr_vals:
            print(f"  PSNR (Y):      {np.mean(psnr_vals):>8.2f} dB")
        if ssim_vals:
            print(f"  SSIM (Y):      {np.mean(ssim_vals):>8.4f}")
        if lpips_vals:
            print(f"  LPIPS (Alex):  {np.mean(lpips_vals):>8.4f}")
        if dists_vals:
            print(f"  DISTS:         {np.mean(dists_vals):>8.4f}")
        if niqe_vals:
            print(f"  NIQE:          {np.mean(niqe_vals):>8.4f}")
        if musiq_vals:
            print(f"  MUSIQ:         {np.mean(musiq_vals):>8.4f}")
        if maniqa_vals:
            print(f"  MANIQA:        {np.mean(maniqa_vals):>8.4f}")
        if clipiqa_vals:
            print(f"  CLIPIQA:       {np.mean(clipiqa_vals):>8.4f}")
        print(f"  SR saved to: {out_dir}\n")

        # ---- 恢复原始权重 ----
        if orig_state is not None:
            self.rfmsr.load_state_dict(orig_state)

        self.rfmsr.train()

    # ------------------------------------------------------------------
    # Checkpoint 续
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.rfmsr.load_state_dict(ckpt["rfmsr"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        # overrides the old hyperparameters in the checkpoint with the current config
        tcfg = self.cfg["training"]
        for pg in self.optimizer.param_groups:
            pg["lr"]        = tcfg["lr"]
            pg["betas"]     = (tcfg["adam_beta1"], tcfg["adam_beta2"])
            pg["weight_decay"] = tcfg["adam_weight_decay"]
            pg["eps"]       = tcfg["adam_epsilon"]
        if ckpt.get("ema_state"):
            self.ema_state = ckpt["ema_state"]
        self.global_step = ckpt["step"]
        print(f"✅ Resumed from step {self.global_step}")

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------

    def train(self):
        tcfg = self.cfg["training"]
        total_iters = tcfg["iterations"]
        grad_clip = tcfg["gradient_clip"]
        accum_steps = self.accumulation_steps
        self.rfmsr.train()

        data_iter = iter(self.dataloader)
        pbar = tqdm(
            total=total_iters, initial=self.global_step, desc="Train", unit="step",
            bar_format="{desc} [{n:>6d}/{total_fmt}] {percentage:3.0f}% |{bar}| {postfix} [{rate_fmt}]",
        )

        ema_velo = 0.0
        ema_cnt = 0
        accum_velo = 0.0
        self.accumulation_count = 0

        try:
            # 首轮 zero_grad
            self.optimizer.zero_grad()

            while self.global_step < total_iters:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # 前向 + 反向（loss 已在 train_step 内按 accum_steps 缩放）
                loss_dict = self.train_step(batch)
                self.accumulation_count += 1

                # 累计统计（日志用）
                ema_velo += loss_dict.get("velo", 0)
                ema_cnt += 1
                # 当次累计步内的 velo 累积（进度条显示用）
                accum_velo += loss_dict.get("velo", 0)

                # 累计步数达到 → optimizer step
                if self.accumulation_count >= accum_steps:
                    # 梯度裁剪
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.rfmsr.parameters(), grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # EMA 更新（每个有效 step 一次）
                    self._update_ema()

                    self.global_step += 1
                    # 进度条显示当前累计步内的平均 velo
                    step_avg = accum_velo / self.accumulation_count
                    pbar.set_postfix(**{"velo": f"{step_avg:.4f}"})
                    self.accumulation_count = 0
                    accum_velo = 0.0
                    pbar.update(1)

                    # 日志 & 保存
                    if self.global_step % self.log_freq == 0:
                        avg_v = ema_velo / max(ema_cnt, 1)
                        lr = self.optimizer.param_groups[0]['lr']
                        print(f"\n[step {self.global_step}/{total_iters}]  velo={avg_v:.6f}  lr={lr:.2e}")
                        ema_velo = 0.0
                        ema_cnt = 0

                    if self.global_step % self.save_freq == 0:
                        self.save_checkpoint(self.global_step)
                        if self.val_enabled:
                            try:
                                self.validate(self.global_step)
                            except Exception as e:
                                print(f"\n[WARN] Validation failed at step {self.global_step}: {e}")

            self.save_checkpoint(self.global_step)
            pbar.close()
            print("Training finished!")
        except KeyboardInterrupt:
            pbar.close()
            print("\nInterrupted, saving checkpoint ...")
            self.save_checkpoint(self.global_step)
            print("Checkpoint saved on interrupt.")


# =========================================================================
# CLI 入口
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RFMSR (LightningDiT)")
    parser.add_argument("--resume", type=str, default=None, help="Path to training state checkpoint")
    args = parser.parse_args()

    trainer = RFMSRTrainer(resume_path=args.resume)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
