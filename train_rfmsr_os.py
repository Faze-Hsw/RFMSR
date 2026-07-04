"""
RFMSR 一步训练脚本 — T=1 直接生成，端到端损失

训练目标: RFMSR 从 T=1 (z_lr + σ·ε) 一步预测 z_pred → z_hr
损失组合:
  - L2 损失 (潜空间): MSE(z_pred, z_hr)
  - GAN 损失 (潜空间): PatchGAN/UNet 判别器区分 z_pred vs z_hr
  - LPIPS 损失 (图像空间): VAE decode 后在像素空间计算感知损失

用法:
  python train_rfmsr_os.py
  python train_rfmsr_os.py --resume experiments/rfmsr_os/checkpoints/training_state_stepXXXXX.pth
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
from utils import GANLoss, LPIPSLoss, create_discriminator

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Suppress checkpoint warning for frozen VAE decoder (params don't need grad)
import warnings
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")


# =========================================================================
# Trainer
# =========================================================================

class RFMSROneStepTrainer:

    def __init__(self, resume_path: str = None):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(_script_dir, "configs", "train_rfmsr_os.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.resume_path = resume_path
        self.device = torch.device(self.cfg["denoise_device"])
        self._setup_seed(self.cfg["training"]["seed"])

        # Flow
        flow_cfg = self.cfg.get("flow", {})
        self.sigma = flow_cfg.get("sigma", 1.0)

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
        self._build_generator()
        self._build_discriminator()
        self._build_losses()
        self._build_optimizers()
        self._build_dataloader()
        self._build_ema()

        # AMP
        self.use_amp = self.cfg["training"]["use_amp"]

        # 梯度累计
        tcfg = self.cfg["training"]
        self.accumulation_steps = tcfg.get("gradient_accumulation_steps", 1)
        self.grad_clip = tcfg.get("gradient_clip", 1.0)

        # GAN 交替训练
        disc_cfg = self.cfg["discriminator"]
        self.disc_update_freq = disc_cfg.get("update_freq", 2)

        # 验证
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
        vae_path = self.cfg["vae_path"]
        device = self.device
        print(f"Loading SD2.1 VAE from {vae_path} -> {device} ...")
        self.ae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
        self.ae = self.ae.to(device).eval()
        self.ae.requires_grad_(False)
        print(f"✅ VAE loaded, scaling_factor={self.ae.config.scaling_factor}")

    def _load_dinov2(self):
        dv2_cfg = self.cfg.get("dinov2", {}) or {}
        self.use_dinov2 = dv2_cfg.get("enabled", False)
        if self.use_dinov2:
            model_cfg_path = self.cfg["model_config"]
            self.venc = create_dinov2_encoder(model_cfg_path, device=self.device)
        else:
            self.venc = None

    def _build_generator(self):
        cfg_path = self.cfg["model_config"]
        self.rfmsr = create_rfmsr(cfg_path).to(self.device)

        if self.resume_path:
            print("⏭️  Skipping pretrained init (will restore from resume checkpoint)")
        else:
            pretrained_ckpt = self.cfg.get("pretrained_ckpt", None)
            if pretrained_ckpt:
                self.rfmsr.load_pretrained(pretrained_ckpt)

        n_params = sum(p.numel() for p in self.rfmsr.parameters())
        print(f"✅ Generator: {n_params / 1e6:.1f}M params")

    def _build_discriminator(self):
        disc_cfg = self.cfg["discriminator"]
        self.discriminator = create_discriminator(
            disc_type=disc_cfg.get("type", "patch"),
            input_nc=4,                          # 潜空间 4 通道
            ndf=disc_cfg.get("ndf", 64),
            n_layers=disc_cfg.get("n_layers", 3),
            norm_type=disc_cfg.get("norm_type", "spectral"),
        ).to(self.device)
        n_disc = sum(p.numel() for p in self.discriminator.parameters()) / 1e3
        print(f"✅ Discriminator ({disc_cfg.get('type','patch')}): {n_disc:.1f}K params")

    def _sample_t(self, B: int) -> torch.Tensor:
        """随机采样 t ∈ [0,1] 用于速度监督训练。"""
        return torch.rand(B, device=self.device)

    def _build_losses(self):
        loss_cfg = self.cfg.get("loss", {})

        self.velo_weight = loss_cfg.get("velo_weight", 0.0)
        self.l2_weight = loss_cfg.get("l2_weight", 1.0)
        self.lpips_weight = loss_cfg.get("lpips_weight", 1.0)
        self.gan_weight = loss_cfg.get("gan_weight", 0.1)

        # LPIPS 分块解码：避免 VAE decode + LPIPS backbone 大 batch OOM
        self.lpips_chunk_size = loss_cfg.get("lpips_chunk_size", None)

        self.gan_loss_fn = GANLoss(
            gan_type=loss_cfg.get("gan_type", "hinge"),
            loss_weight=1.0,
        ).to(self.device)

        self.lpips_loss_fn = LPIPSLoss(
            loss_weight=1.0,
            net_type=loss_cfg.get("lpips_net_type", "alex"),
        ).to(self.device)

    def _build_optimizers(self):
        tcfg = self.cfg["training"]
        disc_cfg = self.cfg["discriminator"]

        # Generator optimizer
        self.optimizer_g = torch.optim.AdamW(
            self.rfmsr.parameters(),
            lr=tcfg["lr"],
            betas=(tcfg["adam_beta1"], tcfg["adam_beta2"]),
            weight_decay=tcfg["adam_weight_decay"],
            eps=tcfg["adam_epsilon"],
        )

        # Discriminator optimizer
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=disc_cfg.get("lr", 1e-4),
            betas=(tcfg["adam_beta1"], tcfg["adam_beta2"]),
            weight_decay=0,
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
        loss_cfg = self.cfg.get("loss", {})
        disc_cfg = self.cfg["discriminator"]
        accum = self.accumulation_steps
        print("\n" + "=" * 60)
        print(f"Experiment      : {self.cfg['experiment']['name']}")
        print(f"Save dir        : {self.exp_dir}")
        print(f"Iterations      : {tcfg['iterations']}")
        print(f"Batch size      : {dcfg['batch_size']}")
        print(f"Accum steps     : {accum}")
        print(f"Effective batch : {dcfg['batch_size'] * accum}")
        print(f"Learning Rate   : G={tcfg['lr']},  D={disc_cfg.get('lr',1e-4)}")
        print(f"GT size         : {dcfg['gt_size']}")
        print(f"Flow            : One-step T=1 → 0,  σ={self.sigma}")
        print(f"Conditioning    : LR latent + {'DINOv2' if self.use_dinov2 else 'none'}")
        use_velo = self.velo_weight > 0
        print(f"Velocity supv   : {'on' if use_velo else 'off'}"
              + (' (ground-truth)' if use_velo else ''))
        print(f"Loss            : "
              + (f"Velo(w={self.velo_weight}) + " if use_velo else '')
              + f"L2(w={self.l2_weight}) + "
              + f"LPIPS(w={self.lpips_weight}) + "
              + f"GAN(w={self.gan_weight}, {loss_cfg.get('gan_type','hinge')})")
        print(f"Discriminator   : {disc_cfg.get('type','patch')}, "
              f"update_freq={self.disc_update_freq}")
        print(f"AMP             : {self.use_amp}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] float [0,1] → [B,4,H/8,W/8] SD2.1 latent (scaled)."""
        img = img.to(device=self.device, dtype=torch.bfloat16)
        img = img * 2.0 - 1.0
        return self.ae.encode(img.float()).latent_dist.sample() * self.ae.config.scaling_factor

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """[B,4,H/8,W/8] scaled latent → [B,3,H,W] float [0,1].
        调用方按需自行包裹 torch.no_grad()。"""
        s = self.ae.config.scaling_factor
        decoded = self.ae.decode((latent / s).float()).sample
        return torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

    def vae_decode_checkpointed(self, latent: torch.Tensor) -> torch.Tensor:
        """Gradient-checkpointed VAE decode for LPIPS 训练。
        用 compute 换 memory：不保存 VAE decoder 的中间激活，
        backward 时重新计算 forward，省 80%+ VAE decoder 显存。"""
        s = self.ae.config.scaling_factor
        latent = latent / s

        def _decode(l):
            return self.ae.decode(l).sample

        decoded = torch.utils.checkpoint.checkpoint(
            _decode, latent.float(), use_reentrant=False
        )
        return torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict, train_disc: bool) -> dict:
        """
        一步训练: RFMSR 从 T=1 直接预测终点。

        train_disc=True  → 训练判别器
        train_disc=False → 训练生成器
        """
        device = self.device

        hr = batch["gt"].to(device)
        lr = batch["lq"].to(device)

        # 1. VAE encode
        z_hr = self.vae_encode(hr).detach().float()   # [B, 4, H/8, W/8]
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        z_lr = self.vae_encode(lr_up).detach().float()
        B = z_hr.shape[0]

        # 2. DINOv2
        venc_fea = None
        if self.use_dinov2 and self.venc is not None:
            venc_fea = self.venc(lr)

        # 3. T=1 初始化: x_1 = z_lr + σ·ε
        epsilon = torch.randn_like(z_hr)
        x_1 = z_lr + self.sigma * epsilon
        t_ones = torch.ones(B, device=device)

        # 4. 前向: v = f(x_1, t=1, z_lr)  →  z_pred = x_1 - v
        losses = {}
        inv_accum = 1.0 / self.accumulation_steps

        if train_disc:
            # ================ 判别器步 ================
            self.rfmsr.eval()
            self.discriminator.train()

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    v = self.rfmsr(x_1, t_ones, z_lr, venc_fea=venc_fea).float()
                z_pred = x_1 - v

            z_real = z_hr.detach()
            z_fake = z_pred.detach()

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                d_real = self.discriminator(z_real)
                d_fake = self.discriminator(z_fake)

                d_loss_real = self.gan_loss_fn(d_real, target_is_real=True, is_disc=True)
                d_loss_fake = self.gan_loss_fn(d_fake, target_is_real=False, is_disc=True)
                loss = (d_loss_real + d_loss_fake) * 0.5 * inv_accum

            losses["D_real"] = d_loss_real.item() * inv_accum
            losses["D_fake"] = d_loss_fake.item() * inv_accum
            losses["D"] = (d_loss_real.item() + d_loss_fake.item()) * 0.5 * inv_accum

            self.rfmsr.train()

        else:
            # ================ 生成器步 ================
            self.rfmsr.train()
            self.discriminator.eval()

            # ---- ① T=1 一步推理 (autocast 仅管 rfmsr 前向) ----
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                v = self.rfmsr(x_1, t_ones, z_lr, venc_fea=venc_fea)
                z_pred = x_1.to(v.dtype) - v

            # ★ loss 全部在 autocast 外, 统一 fp32 —— 彻底杜绝 backward 类型冲突
            z_pred = z_pred.float()
            z_hr_fp = z_hr.float() if z_hr.dtype != torch.float32 else z_hr

            total = torch.zeros((), device=device)
            if self.l2_weight > 0:
                l2_loss = F.mse_loss(z_pred, z_hr_fp)
                total = total + self.l2_weight * l2_loss
           
            if self.gan_weight > 0:
                d_fake = self.discriminator(z_pred)
                gan_g_loss = self.gan_loss_fn(d_fake, target_is_real=True, is_disc=False)
                total = total + self.gan_weight * gan_g_loss

            # ---- ② Ground-truth 速度监督: v_true = z_lr - z_hr + σ·ε ----
            if self.velo_weight > 0:
                t = self._sample_t(B)
                t_expand = t[:, None, None, None]
                residual = z_lr - z_hr
                epsilon_rf = torch.randn_like(z_hr)
                x_t = z_hr + t_expand * residual + t_expand * self.sigma * epsilon_rf
                # 解析速度 (Residual Flow Matching ground-truth)
                v_true = residual + self.sigma * epsilon_rf

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    v_pred_rf = self.rfmsr(x_t, t, z_lr, venc_fea=venc_fea)
                velo_loss = F.mse_loss(v_pred_rf.float(), v_true.float())
                total = total + self.velo_weight * velo_loss
                losses["Velo"] = velo_loss.item() * inv_accum

            # ---- ③ LPIPS (图像空间, 分块解码, gradient-checkpointed) ----
            if self.lpips_weight > 0:
                chunk_sz = self.lpips_chunk_size or B
                lpips_loss = torch.zeros((), device=device)
                for i in range(0, B, chunk_sz):
                    z_pred_i = z_pred[i:i + chunk_sz]
                    z_hr_i   = z_hr_fp[i:i + chunk_sz]
                    sr_pred_i = self.vae_decode_checkpointed(z_pred_i)
                    with torch.no_grad():
                        sr_gt_i = self.vae_decode(z_hr_i).float()
                    lpips_loss = lpips_loss + self.lpips_loss_fn(sr_pred_i, sr_gt_i)
                    del sr_pred_i
                n_chunks = max(int(math.ceil(B / chunk_sz)), 1)
                lpips_loss = lpips_loss / n_chunks
                total = total + self.lpips_weight * lpips_loss
                losses["LPIPS"] = lpips_loss.item() * inv_accum
                torch.cuda.empty_cache()

            loss = total * inv_accum

            if self.l2_weight > 0:
                losses["L2"] = l2_loss.item() * inv_accum
            if self.gan_weight > 0:
                losses["GAN_G"] = gan_g_loss.item() * inv_accum
            losses["total"] = total.item() * inv_accum

        # 5. 反向传播
        loss.backward()

        return losses

    # ------------------------------------------------------------------
    # EMA (仅对生成器)
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
        ema_path = ckpt_dir / f"rfmsr_os_step{step}.safetensors"
        safe_save(weights, ema_path)

        # 完整训练状态 (torch.save)
        state = {
            "step": step,
            "rfmsr": self.rfmsr.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "ema_state": self.ema_state,
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
        }
        state_path = ckpt_dir / f"training_state_step{step}.pth"
        torch.save(state, state_path)
        print(f"💾 Checkpoint saved: step {step}")

        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints(ckpt_dir)

    def _cleanup_old_checkpoints(self, ckpt_dir: Path):
        import re
        pattern = re.compile(r"(rfmsr_os_step|training_state_step)(\d+)")
        ckpt_steps: dict[int, list[Path]] = {}
        for f in ckpt_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                s = int(m.group(2))
                ckpt_steps.setdefault(s, []).append(f)

        sorted_steps = sorted(ckpt_steps.keys(), reverse=True)
        for step in sorted_steps[self.keep_last_n:]:
            for f in ckpt_steps[step]:
                f.unlink()
            print(f"🗑️  Removed old checkpoint: step {step}")

    # ------------------------------------------------------------------
    # 验证 (一步推理)
    # ------------------------------------------------------------------

    def _init_val_metrics(self):
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net="alex").to(self.device)
        except ImportError:
            print("[WARN] lpips not installed")
        try:
            import pyiqa
            self.psnr_metric = pyiqa.create_metric(
                "psnr", test_y_channel=True, color_space="ycbcr", device=self.device)
            self.ssim_metric = pyiqa.create_metric(
                "ssim", test_y_channel=True, color_space="ycbcr", device=self.device)
            self.dists_metric = pyiqa.create_metric("dists", device=self.device)
            self.niqe_metric = pyiqa.create_metric("niqe", device=self.device)
            self.musiq_metric = pyiqa.create_metric("musiq", device=self.device)
            self.maniqa_metric = pyiqa.create_metric("maniqa", device=self.device)
            self.clipiqa_metric = pyiqa.create_metric("clipiqa", device=self.device)
            print("✅ Validation metrics: PSNR, SSIM, LPIPS, DISTS, NIQE, MUSIQ, MANIQA, CLIPIQA")
        except ImportError:
            print("[WARN] pyiqa not installed")

    @torch.no_grad()
    def _validate_inference(self, z_lr, venc_fea, val_seed, n_steps):
        """执行 N 步 Euler 积分推理, 返回解码后的 [0,1] 图像 tensor。"""
        B_v = z_lr.shape[0]
        generator = torch.Generator(device=self.device).manual_seed(val_seed)
        x = z_lr + self.sigma * torch.randn(
            B_v, *z_lr.shape[1:], generator=generator, device=self.device
        )
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = 1.0 - k * dt
            t_tensor = torch.full((B_v,), t, device=self.device)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                v = self.rfmsr(x, t_tensor, z_lr, venc_fea=venc_fea).float()
            x = x - v * dt   # Euler step: x_{t-dt} = x_t - v·dt
        # VAE decode
        SCALE = self.ae.config.scaling_factor
        decoded = self.ae.decode((x / SCALE).float()).sample
        return torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)

    def _compute_val_metrics(self, sr_decoded, gt_tensor, metrics_dict):
        """将当前图片的指标 append 到对应 list。"""
        if self.psnr_metric is not None:
            metrics_dict["psnr"].append(self.psnr_metric(sr_decoded, gt_tensor).mean().item())
        if self.ssim_metric is not None:
            metrics_dict["ssim"].append(self.ssim_metric(sr_decoded, gt_tensor).mean().item())
        if self.dists_metric is not None:
            metrics_dict["dists"].append(self.dists_metric(sr_decoded, gt_tensor).mean().item())
        if self.lpips_fn is not None:
            gt_norm = (gt_tensor - 0.5) / 0.5
            sr_norm = (sr_decoded - 0.5) / 0.5
            metrics_dict["lpips"].append(self.lpips_fn(gt_norm, sr_norm).mean().item())
        if self.niqe_metric is not None:
            metrics_dict["niqe"].append(self.niqe_metric(sr_decoded).mean().item())
        if self.musiq_metric is not None:
            metrics_dict["musiq"].append(self.musiq_metric(sr_decoded).mean().item())
        if self.maniqa_metric is not None:
            metrics_dict["maniqa"].append(self.maniqa_metric(sr_decoded).mean().item())
        if self.clipiqa_metric is not None:
            metrics_dict["clipiqa"].append(self.clipiqa_metric(sr_decoded).mean().item())

    @torch.no_grad()
    def validate(self, step: int):
        """验证：同时跑 1-step 和多步推理, 对比性能。"""
        self.rfmsr.eval()

        orig_state = None
        if self.ema_state is not None:
            orig_state = OrderedDict({k: v.data.clone() for k, v in self.rfmsr.state_dict().items()})
            self.rfmsr.load_state_dict(self.ema_state)

        val_cfg = self.cfg.get("validation", {})
        lq_dir = Path(val_cfg.get("lq_dir", "assets/validate_lq"))
        gt_dir = Path(val_cfg.get("gt_dir", "assets/validate_gt"))
        val_scale = val_cfg.get("scale", 4.0)
        val_seed = val_cfg.get("seed", 42)
        max_images = val_cfg.get("max_images", 0)
        val_steps = val_cfg.get("val_steps", [1, 15])  # [1-step, multi-step]

        out_base = self.exp_dir / "validation" / f"step_{step:08d}"
        MOD_PIXEL = 16
        SCALE = self.ae.config.scaling_factor

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

        all_metrics = {steps: {k: [] for k in ["psnr","ssim","lpips","dists","niqe","musiq","maniqa","clipiqa"]}
                       for steps in val_steps}

        for n_steps in val_steps:
            out_dir = out_base / f"{n_steps}step"
            out_dir.mkdir(parents=True, exist_ok=True)
            desc = f"Val@{step}({n_steps}-step)"

            for lq_path, gt_path in tqdm(pairs, desc=desc, leave=False):
                src = Image.open(lq_path).convert("RGB")
                gt_img = Image.open(gt_path).convert("RGB")
                exact_w = int(src.size[0] * val_scale)
                exact_h = int(src.size[1] * val_scale)
                target = src.resize((exact_w, exact_h), Image.BICUBIC)
                ori_h, ori_w = target.size[1], target.size[0]

                im_np = np.array(target).astype(np.float32) / 255.0
                im_cond = torch.from_numpy(np.moveaxis(im_np, 2, 0)).unsqueeze(0)
                im_cond = im_cond.to(dtype=torch.bfloat16, device=self.device)

                h, w = im_cond.shape[-2:]
                pad_h = (math.ceil(h / MOD_PIXEL) * MOD_PIXEL) - h
                pad_w = (math.ceil(w / MOD_PIXEL) * MOD_PIXEL) - w
                if pad_h > 0 or pad_w > 0:
                    im_cond = F.pad(im_cond, (0, pad_w, 0, pad_h), mode="reflect")

                image_tensor = im_cond * 2.0 - 1.0
                z_lr = self.ae.encode(image_tensor.float()).latent_dist.sample() * SCALE

                venc_fea = None
                if self.use_dinov2 and self.venc is not None:
                    venc_fea = self.venc(im_cond.float())

                sr_decoded = self._validate_inference(z_lr, venc_fea, val_seed, n_steps)
                sr_decoded = sr_decoded[:, :, 0:ori_h, 0:ori_w]

                # Save
                sr_np = (sr_decoded[0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                sr_np = np.moveaxis(sr_np, 0, 2)
                Image.fromarray(sr_np).save(out_dir / lq_path.name)

                # Metrics
                gt_np = np.array(gt_img).astype(np.float32) / 255.0
                gt_tensor = torch.from_numpy(np.moveaxis(gt_np, 2, 0)).unsqueeze(0)
                gt_tensor = gt_tensor.to(self.device)
                self._compute_val_metrics(sr_decoded, gt_tensor, all_metrics[n_steps])

        # ---- Report (side-by-side) ----
        metric_names = [
            ("PSNR (Y)", "psnr", "8.2f", "dB"),
            ("SSIM (Y)", "ssim", "8.4f", ""),
            ("LPIPS-Alex", "lpips", "8.4f", ""),
            ("DISTS", "dists", "8.4f", ""),
            ("NIQE", "niqe", "8.4f", ""),
            ("MUSIQ", "musiq", "8.4f", ""),
            ("MANIQA", "maniqa", "8.4f", ""),
            ("CLIPIQA", "clipiqa", "8.4f", ""),
        ]
        header = f"{'Metric':<14s}"
        for n in val_steps:
            header += f"  {n}-step     "
        lines = [f"[Val @ step {step}] images={total}", header, "-" * (14 + 13 * len(val_steps))]
        for display_name, key, fmt, unit in metric_names:
            vals = all_metrics.get(val_steps[0], {}).get(key, [])
            if not vals:
                continue
            row = f"  {display_name:<12s}"
            for n in val_steps:
                m = np.mean(all_metrics[n][key]) if all_metrics[n][key] else float("nan")
                row += f"  {m:{fmt}}"
                if unit:
                    row += f" {unit}" + " " * max(0, 5 - len(unit))
            lines.append(row)
        lines.append(f"\n  SR saved to: {out_base}")
        print("\n" + "\n".join(lines))

        if orig_state is not None:
            self.rfmsr.load_state_dict(orig_state)
        self.rfmsr.train()

    # ------------------------------------------------------------------
    # 续训
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.rfmsr.load_state_dict(ckpt["rfmsr"])
        self.optimizer_g.load_state_dict(ckpt["optimizer_g"])

        # Override hyperparameters from current config
        tcfg = self.cfg["training"]
        for pg in self.optimizer_g.param_groups:
            pg["lr"] = tcfg["lr"]
            pg["betas"] = (tcfg["adam_beta1"], tcfg["adam_beta2"])
            pg["weight_decay"] = tcfg["adam_weight_decay"]
            pg["eps"] = tcfg["adam_epsilon"]

        if "discriminator" in ckpt:
            self.discriminator.load_state_dict(ckpt["discriminator"])
        if "optimizer_d" in ckpt:
            self.optimizer_d.load_state_dict(ckpt["optimizer_d"])
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
        accum_steps = self.accumulation_steps

        self.rfmsr.train()
        self.discriminator.train()

        data_iter = iter(self.dataloader)
        pbar = tqdm(
            total=total_iters, initial=self.global_step, desc="1-Step",
            unit="step",
            bar_format="{desc} [{n:>6d}/{total_fmt}] {percentage:3.0f}% |{bar}| {postfix} [{rate_fmt}]",
        )

        # EMA 统计
        ema_velo = 0.0; ema_l2 = 0.0; ema_lpips = 0.0
        ema_gan_g = 0.0; ema_gan_d = 0.0
        ema_cnt_g = 0; ema_cnt_d = 0

        # 独立的 G/D 梯度累计计数器
        g_count = 0; d_count = 0
        # G/D 交替用 batch 级计数器 (不受 accumulation 影响; 续训从 0 重启)
        batch_idx = 0

        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        last_log_step = self.global_step
        last_save_step = self.global_step

        try:
            while self.global_step < total_iters:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # G/D 交替: batch 级 (不受 accumulation 影响)
                train_d = (self.gan_weight > 0 and batch_idx % self.disc_update_freq == 0)
                batch_idx += 1

                loss_dict = self.train_step(batch, train_disc=train_d)

                # EMA & 累计
                if train_d:
                    ema_gan_d += loss_dict.get("D", 0)
                    ema_cnt_d += 1
                    d_count += 1
                else:
                    ema_velo += loss_dict.get("Velo", 0)
                    ema_l2 += loss_dict.get("L2", 0)
                    ema_lpips += loss_dict.get("LPIPS", 0)
                    ema_gan_g += loss_dict.get("GAN_G", 0)
                    ema_cnt_g += 1
                    g_count += 1

                # ---- G/D 独立梯度累计 ----
                stepped = False

                if g_count >= accum_steps:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.rfmsr.parameters(), self.grad_clip)
                    self.optimizer_g.step()
                    self.optimizer_g.zero_grad()
                    self.optimizer_d.zero_grad()
                    self._update_ema()
                    g_count = 0
                    self.global_step += 1
                    stepped = True

                if d_count >= accum_steps:
                    self.optimizer_d.step()
                    self.optimizer_d.zero_grad()
                    d_count = 0
                    self.global_step += 1
                    stepped = True

                if stepped:
                    # 进度条
                    postfix = {}
                    if ema_cnt_g > 0:
                        if ema_velo > 0:
                            postfix["Velo"] = f"{ema_velo/max(ema_cnt_g,1):.4f}"
                        postfix["L2"] = f"{ema_l2/max(ema_cnt_g,1):.4f}"
                        postfix["LPIPS"] = f"{ema_lpips/max(ema_cnt_g,1):.4f}"
                        postfix["G_GAN"] = f"{ema_gan_g/max(ema_cnt_g,1):.4f}"
                    if ema_cnt_d > 0:
                        postfix["D"] = f"{ema_gan_d/max(ema_cnt_d,1):.4f}"
                    pbar.set_postfix(**postfix)
                    pbar.update(1)

                    # 日志 (避免 G/D 连续 step 时重复打印)
                    if self.global_step // self.log_freq > last_log_step // self.log_freq:
                        last_log_step = self.global_step
                        lr_g = self.optimizer_g.param_groups[0]['lr']
                        lr_d = self.optimizer_d.param_groups[0]['lr']
                        parts = [f"step {self.global_step}/{total_iters}"]
                        if ema_cnt_g > 0:
                            if ema_velo > 0:
                                parts.append(f"Velo={ema_velo/max(ema_cnt_g,1):.4f}")
                            parts.append(f"L2={ema_l2/max(ema_cnt_g,1):.4f}")
                            parts.append(f"LPIPS={ema_lpips/max(ema_cnt_g,1):.4f}")
                            parts.append(f"G_GAN={ema_gan_g/max(ema_cnt_g,1):.4f}")
                        if ema_cnt_d > 0:
                            parts.append(f"D={ema_gan_d/max(ema_cnt_d,1):.4f}")
                        parts.append(f"lr_g={lr_g:.2e} lr_d={lr_d:.2e}")
                        print("\n[" + "]  [".join(parts))
                        ema_velo = ema_l2 = ema_lpips = ema_gan_g = ema_gan_d = 0.0
                        ema_cnt_g = ema_cnt_d = 0

                    # 保存 (避免 G/D 连续 step 时重复保存)
                    if self.global_step // self.save_freq > last_save_step // self.save_freq:
                        last_save_step = self.global_step
                        self.save_checkpoint(self.global_step)
                        if self.val_enabled:
                            try:
                                self.validate(self.global_step)
                            except Exception as e:
                                print(f"\n[WARN] Validation failed: {e}")

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
    parser = argparse.ArgumentParser(description="Train RFMSR One-Step (T=1→0, L2+LPIPS+GAN)")
    parser.add_argument("--resume", type=str, default=None, help="Path to training state checkpoint")
    args = parser.parse_args()

    trainer = RFMSROneStepTrainer(resume_path=args.resume)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
