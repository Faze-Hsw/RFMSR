"""
Flow Embedder Δ_φ 训练脚本

训练目标: 一步 Euler 重构损失 + 可选 GAN 对抗损失
Loss = MSE(z_pred, z_HR) + w_max * t * GAN_loss(z_pred, z_HR)
其中 z_pred = x_t - t * v_θ(x_t, t), x_t = (1-t) * (z_LR + Δ_φ(z_LR, t)) + t * ε

用法:
  python train_flow_embedder.py --config configs/train_flow_embedder.yaml
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import random
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange, repeat
from torch.amp import GradScaler, autocast
from safetensors.torch import save_file as safe_save
from tqdm import tqdm

from datapipe.train_dataloader import create_train_dataloader
from flux.util import load_flow_model, load_t5, load_clip, load_ae
from models.flow_embedder import FlowEmbedder, create_flow_embedder
from models.latent_discriminator import LatentDiscriminator, hinge_d_loss, gen_loss

try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# =========================================================================
# Trainer
# =========================================================================

class FlowEmbedderTrainer:

    def __init__(self):
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(_script_dir, "configs", "train_flow_embedder.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(self.cfg["flux"]["denoise_device"])
        self._setup_seed(self.cfg["training"]["seed"])

        # ---- 实验目录 ----
        exp = self.cfg["experiment"]
        self.exp_dir = Path(exp["save_dir"])
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.log_freq = exp["log_freq"]
        self.save_freq = exp["save_freq"]

        # ---- 梯度检查点开关（需在 _load_flux 前设置） ----
        self.use_gradient_checkpointing = self.cfg["training"].get("use_gradient_checkpointing", True)
        self.gradient_checkpointing_chunk = self.cfg["training"].get("gradient_checkpointing_chunk", 3)

        # ---- 加载各模块 ----
        self._load_flux()
        self._encode_prompt()
        self._build_embedder()
        self._build_optimizer()
        self._build_dataloader()
        self._build_ema()

        # AMP
        self.use_amp = self.cfg["training"]["use_amp"]
        self.scaler = GradScaler() if self.use_amp else None

        self._build_discriminator()
        self._build_lpips()

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

    def _load_flux(self):
        """加载冻结的 Flux DiT + VAE，T5+CLIP 后续单独加载并释放。"""
        flux_cfg = self.cfg["flux"]
        name = flux_cfg["model_name"]
        device = self.device

        print(f"Loading Flux '{name}' -> {device} ...")
        self.flux = load_flow_model(name, device=device, verbose=False)
        self.flux.requires_grad_(False)
        # 保持 train() 模式：让 Flux.forward 中的 self.training=True，
        # 否则梯段检查点的条件 self.training and self.gradient_checkpointing 永不为真
        self.flux.train()

        print(f"Loading VAE -> {device} ...")
        self.ae = load_ae(name, device=device)
        self.ae.eval()
        self.ae.requires_grad_(False)

        self.guidance = flux_cfg["guidance"]
        self.flux.gradient_checkpointing = self.use_gradient_checkpointing
        self.flux.gradient_checkpointing_chunk = self.gradient_checkpointing_chunk
        d_groups = (19 + self.gradient_checkpointing_chunk - 1) // self.gradient_checkpointing_chunk
        s_groups = (38 + self.gradient_checkpointing_chunk - 1) // self.gradient_checkpointing_chunk
        print(f"  training={self.flux.training} | grad_cp={self.flux.gradient_checkpointing} | chunk={self.flux.gradient_checkpointing_chunk} | {d_groups}D+{s_groups}S groups")
        print(f"  Flux params: {sum(p.numel() for p in self.flux.parameters())/1e9:.2f}B → {sum(p.numel() for p in self.flux.parameters())*2/1e9:.2f} GB (bf16)")
        print("✅ Flux + VAE loaded (frozen).")

    def _encode_prompt(self):
        """加载 T5+CLIP → 编码固定 prompt → 缓存到 denoise_device → 释放编码器。"""
        flux_cfg = self.cfg["flux"]
        te_device = flux_cfg.get("text_encoder_device", "cuda")  # 保持字符串，与 infer.py 一致
        prompt = flux_cfg["prompt"]
        t5_max_length = flux_cfg.get("t5_max_length", 512)
        weights = flux_cfg.get("weights", {})
        t5_path = weights.get("t5xxl")
        clip_path = weights.get("clip")

        print(f"Loading T5 + CLIP -> {te_device} ...")
        t5 = load_t5(te_device, max_length=t5_max_length, ckpt_path=t5_path)
        clip = load_clip(te_device, ckpt_path=clip_path)

        print(f"Encoding prompt: '{prompt[:60]}...'")
        with torch.no_grad():
            self.cached_txt = t5(prompt).to(self.device)       # [1, seq, 4096]
            self.cached_vec = clip(prompt).to(self.device)      # [1, 768]
        self.cached_txt_ids = torch.zeros(
            1, self.cached_txt.shape[1], 3, dtype=torch.float32, device=self.device,
        )

        # 释放显存
        del t5, clip
        torch.cuda.empty_cache()
        print("✅ Prompt encoded, T5+CLIP released.")

    def _build_embedder(self):
        """创建可训练的 Flow Embedder Δ_φ。"""
        cfg_path = self.cfg["model_config"]
        self.embedder = create_flow_embedder(cfg_path).to(self.device)
        n_params = sum(p.numel() for p in self.embedder.parameters())
        print(f"✅ FlowEmbedder created: {n_params / 1e6:.2f}M params")

    def _build_optimizer(self):
        tcfg = self.cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.embedder.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )

    def _build_dataloader(self):
        dcfg = self.cfg["data"]
        gt_size = dcfg["gt_size"]
        assert gt_size % 16 == 0, (
            f"gt_size={gt_size} 必须能被 16 整除 "
            f"(VAE 8x 下采样 + pack 2x 重排)"
        )
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
                {k: deepcopy(v.data) for k, v in self.embedder.state_dict().items()}
            )
        else:
            self.ema_rate = 0
            self.ema_state = None

    def _print_summary(self):
        print("\n" + "=" * 60)
        print(f"Experiment : {self.cfg['experiment']['name']}")
        print(f"Save dir   : {self.exp_dir}")
        print(f"Iterations : {self.cfg['training']['iterations']}")
        print(f"Batch size : {self.cfg['data']['batch_size']}")
        print(f"LR         : {self.cfg['training']['lr']}")
        logit_mean = self.cfg['training'].get('logit_normal_mean', 0.0)
        logit_std = self.cfg['training'].get('logit_normal_std', 1.0)
        print(f"LogitNorm  : μ={logit_mean}, σ={logit_std}")
        print(f"GT size    : {self.cfg['data']['gt_size']}")
        print(f"AMP        : {self.use_amp}")
        print(f"GradCP     : {self.use_gradient_checkpointing} | chunk={self.gradient_checkpointing_chunk}", end="")
        if self.use_gradient_checkpointing and hasattr(self, 'flux'):
            d_groups = (19 + self.gradient_checkpointing_chunk - 1) // self.gradient_checkpointing_chunk
            s_groups = (38 + self.gradient_checkpointing_chunk - 1) // self.gradient_checkpointing_chunk
            print(f" | {d_groups}D+{s_groups}S groups", end="")
        print()
        print(f"EMA rate   : {self.ema_rate}")
        # 判别器 & 损失权重
        dcfg = self.cfg.get("discriminator", {})
        if dcfg.get("enabled", False) and self.discriminator is not None:
            dp = sum(p.numel() for p in self.discriminator.parameters())
            print(f"Discriminator: {dp/1e6:.2f}M params, lr={dcfg.get('lr', 5e-5)}, w_max={dcfg.get('w_max', 0.1)}")
        else:
            print("Discriminator: disabled")
        print(f"Loss weights : L2=1.0, LPIPS={self.lpips_weight}, GAN(w_max)={dcfg.get('w_max', 0)} * t")
        if dcfg.get("enabled"):
            print(f"  GAN warmup : {dcfg.get('dis_init_iterations', 0)} steps (generator L2+LPIPS only)")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # VAE encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] float [0,1] → [B,16,H/8,W/8] latent."""
        img = img.to(device=self.device, dtype=torch.bfloat16)
        img = img * 2.0 - 1.0           # [0,1] → [-1,1]
        return self.ae.encode(img)       # [B,16,H/8,W/8]

    # ------------------------------------------------------------------
    # Pack / Unpack / img_ids — 与 Flux infer.py 保持一致
    # ------------------------------------------------------------------

    @staticmethod
    def pack(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def make_img_ids(b: int, h_lat: int, w_lat: int, device: torch.device) -> torch.Tensor:
        img_ids = torch.zeros(h_lat, w_lat, 3, device=device)
        img_ids[..., 1] = torch.arange(h_lat, device=device)[:, None]
        img_ids[..., 2] = torch.arange(w_lat, device=device)[None, :]
        return repeat(img_ids, "h w c -> b (h w) c", b=b)

    # ------------------------------------------------------------------
    # 单步 Flux 前向（保留 x_t 的梯度图）
    # ------------------------------------------------------------------

    @staticmethod
    def _flux_forward_no_args(
        flux, x_t_packed, img_ids, txt, txt_ids, t_vec, vec, guidance_vec,
    ) -> torch.Tensor:
        """纯位置参数包装，供 checkpoint 调用。"""
        return flux(
            img=x_t_packed, img_ids=img_ids,
            txt=txt, txt_ids=txt_ids,
            timesteps=t_vec, y=vec, guidance=guidance_vec,
        )

    def flux_velocity(
        self, x_t_packed: torch.Tensor, t: torch.Tensor,
        img_ids: torch.Tensor, bs: int,
    ) -> torch.Tensor:
        """冻结 Flux 单步前向，返回 velocity 预测 [B, seq, 64]。

        梯度检查点已在 Flux.forward 内部按 block 分组实现，
        这里直接调用 Flux 即可。
        """
        # 对齐到 Flux 权重的 dtype (bf16)，移出 autocast 后不再自动转换
        flux_dtype = x_t_packed.dtype
        txt = self.cached_txt.expand(bs, -1, -1).to(self.device, dtype=flux_dtype)
        txt_ids = self.cached_txt_ids.expand(bs, -1, -1).to(self.device)
        vec = self.cached_vec.expand(bs, -1).to(self.device, dtype=flux_dtype)

        t_vec = t.to(self.device, dtype=flux_dtype)
        guidance_vec = torch.full((bs,), self.guidance, device=self.device, dtype=flux_dtype)

        return self._flux_forward_no_args(
            self.flux, x_t_packed, img_ids, txt, txt_ids, t_vec, vec, guidance_vec,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        tcfg = self.cfg["training"]
        dcfg = self.cfg.get("discriminator", {})
        device = self.device

        hr = batch["gt"].to(device)      # [B,3,512,512] [0,1]
        lr = batch["lq"].to(device)      # [B,3,128,128] [0,1]
        bs = hr.shape[0]

        # 1. VAE encode (no grad)
        z_hr = self.vae_encode(hr)       # [B,16,64,64]
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        z_lr = self.vae_encode(lr_up)    # [B,16,64,64]

        z_hr = z_hr.detach().float()
        z_lr = z_lr.detach().float()

        # 2. 时间步 t
        logit_mean = tcfg.get("logit_normal_mean", 0.0)
        logit_std = tcfg.get("logit_normal_std", 1.0)
        t = torch.sigmoid(torch.randn(bs, device=device) * logit_std + logit_mean)

        # ══════════════════════════════════════════
        # Generator 前向
        # ══════════════════════════════════════════
        with autocast(device_type="cuda", enabled=self.use_amp):
            eps_pred = self.embedder(z_lr, t)               # [B,16,64,64] 预测噪声
            t_expand = t[:, None, None, None]
            # 用预测噪声替代随机高斯构造整流流初始状态
            x_t = (1.0 - t_expand) * z_lr + t_expand * eps_pred

        x_t_packed = self.pack(x_t.to(torch.bfloat16))
        _, _, h_lat, w_lat = z_hr.shape
        h_pack, w_pack = h_lat // 2, w_lat // 2
        img_ids = self.make_img_ids(bs, h_pack, w_pack, device)
        v_pred = self.flux_velocity(x_t_packed, t, img_ids, bs)

        # 3. 一步 Euler 重构 z_pred = x_t - t * v_pred
        z_pred_packed = x_t_packed - t[:, None, None].to(torch.bfloat16) * v_pred

        # 4. Unpack → 潜空间 L2（跳过 VAE decode，更快且稳定）
        z_pred_spatial = rearrange(
            z_pred_packed, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_pack, w=w_pack, ph=2, pw=2,
        )

        loss_l2 = F.mse_loss(z_pred_spatial.float(), z_hr.float())
        losses = {"l2": loss_l2.item()}

        # LPIPS（可选，需要解码到像素空间）
        if self.lpips_weight > 0 and self.lpips_loss is not None:
            sr_img = torch.utils.checkpoint.checkpoint(
                lambda z: self.ae.decode(z), z_pred_spatial.to(torch.bfloat16),
                use_reentrant=False,
            )
            with torch.no_grad():
                hr_img = self.ae.decode(z_hr.to(torch.bfloat16))
            # LPIPS 需要 [0, 1] 输入
            sr_norm = (sr_img.float() + 1.0) / 2.0
            hr_norm = (hr_img.float() + 1.0) / 2.0
            loss_lpips = self.lpips_loss(sr_norm, hr_norm).mean()
        else:
            loss_lpips = torch.zeros(1, device=device)

        # 5. GAN 生成器损失（若有判别器且过预热期）
        loss_gan = torch.zeros(1, device=device)
        dis_enabled = (
            self.discriminator is not None
            and self.global_step >= dcfg.get("dis_init_iterations", 0)
        )
        if dis_enabled:
            logits_fake = self.discriminator(z_pred_spatial.to(torch.float32))
            loss_gan_raw = gen_loss(logits_fake)
            losses["gan"] = loss_gan_raw.item()
            w_max = dcfg.get("w_max", 0.1)
            gan_weight = w_max * t.mean()
            loss_gan = loss_gan_raw * gan_weight

        loss = loss_l2 + self.lpips_weight * loss_lpips + loss_gan
        if self.lpips_weight > 0:
            losses["lpips"] = loss_lpips.item()

        # 5. Generator 反向传播
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if tcfg["gradient_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.embedder.parameters(), tcfg["gradient_clip"],
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if tcfg["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.embedder.parameters(), tcfg["gradient_clip"],
                )
            self.optimizer.step()

        # 6. Discriminator 训练（每步都训）
        if dis_enabled:
            z_hr_clamp = z_hr.float().clamp(-10, 10)
            logits_real = self.discriminator(z_hr_clamp)
            logits_fake = self.discriminator(
                z_pred_spatial.detach().float().clamp(-10, 10),
            )
            loss_d = hinge_d_loss(logits_real, logits_fake)

            self.opt_dis.zero_grad()
            if self.scaler_dis is not None:
                self.scaler_dis.scale(loss_d).backward()
                self.scaler_dis.step(self.opt_dis)
                self.scaler_dis.update()
            else:
                loss_d.backward()
                self.opt_dis.step()
            losses["d"] = loss_d.item()

        # 7. EMA
        self._update_ema()

        return losses

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_state is None:
            return
        for k, v in self.embedder.state_dict().items():
            if v.is_floating_point():
                self.ema_state[k].mul_(self.ema_rate).add_(v.data, alpha=1 - self.ema_rate)
            else:
                self.ema_state[k] = v.data.clone()

    # ------------------------------------------------------------------
    # Discriminator
    # ------------------------------------------------------------------

    def _build_discriminator(self):
        dcfg = self.cfg.get("discriminator", {})
        if not dcfg.get("enabled", False):
            self.discriminator = None
            return
        params = dcfg.get("params", {})
        self.discriminator = LatentDiscriminator(**params).to(self.device)
        self.discriminator.train()
        self.opt_dis = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=dcfg.get("lr", 5e-5),
            weight_decay=dcfg.get("weight_decay", 1e-3),
        )
        self.scaler_dis = GradScaler() if self.use_amp else None
        dp = sum(p.numel() for p in self.discriminator.parameters())
        print(f"✅ Discriminator: {dp/1e6:.2f}M params")

    # ------------------------------------------------------------------
    # LPIPS
    # ------------------------------------------------------------------

    def _build_lpips(self):
        lcfg = self.cfg.get("lpips", {})
        weight = lcfg.get("weight", 0.0)
        self.lpips_weight = weight
        self.lpips_loss = None
        if weight <= 0:
            return
        if not _LPIPS_AVAILABLE:
            raise ImportError(
                "LPIPS weight > 0 but lpips is not installed. "
                "Run: pip install lpips"
            )
        self.lpips_loss = lpips.LPIPS(net=lcfg.get("net", "vgg"))
        self.lpips_loss.to(self.device)
        self.lpips_loss.eval()
        for p in self.lpips_loss.parameters():
            p.requires_grad_(False)
        print(f"✅ LPIPS ({lcfg.get('net', 'vgg')}) loaded, weight={weight}")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int):
        ckpt_dir = self.exp_dir / "checkpoints"

        # 推理权重（safetensors，仅含 tensor）
        weights = self.ema_state if self.ema_state is not None else self.embedder.state_dict()
        ema_path = ckpt_dir / f"flow_embedder_step{step}.safetensors"
        safe_save(weights, ema_path)

        # 完整训练状态（含 optimizer/scaler，仍需 torch.save）
        state = {
            "step": step,
            "embedder": self.embedder.state_dict(),
            "ema_state": self.ema_state,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }
        if self.discriminator is not None:
            state["discriminator"] = self.discriminator.state_dict()
            state["opt_dis"] = self.opt_dis.state_dict()
            if self.scaler_dis is not None:
                state["scaler_dis"] = self.scaler_dis.state_dict()
        state_path = ckpt_dir / f"training_state_step{step}.pth"
        torch.save(state, state_path)
        print(f"💾 Checkpoint saved: step {step}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.embedder.load_state_dict(ckpt["embedder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scaler and ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("ema_state"):
            self.ema_state = ckpt["ema_state"]
        if self.discriminator is not None and ckpt.get("discriminator"):
            self.discriminator.load_state_dict(ckpt["discriminator"])
            self.opt_dis.load_state_dict(ckpt["opt_dis"])
            if self.scaler_dis is not None and ckpt.get("scaler_dis"):
                self.scaler_dis.load_state_dict(ckpt["scaler_dis"])
        self.global_step = ckpt["step"]
        print(f"✅ Resumed from step {self.global_step}")

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------

    def train(self):
        tcfg = self.cfg["training"]
        total_iters = tcfg["iterations"]
        self.embedder.train()

        data_iter = iter(self.dataloader)

        pbar = tqdm(
            total=total_iters,
            initial=self.global_step,
            desc="Train",
            unit="step",
            bar_format="{desc} [{n:>6d}/{total_fmt}] {percentage:3.0f}% |{bar}| {postfix} [{rate_fmt}]",
        )

        # 累计损失，每 log_freq 步输出一次平均值
        ema_l2 = 0.0
        ema_lpips = 0.0
        ema_gan = 0.0
        ema_d = 0.0
        ema_cnt = 0

        try:
            while self.global_step < total_iters:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                loss_dict = self.train_step(batch)
                self.global_step += 1

                ema_l2 += loss_dict.get("l2", 0)
                ema_lpips += loss_dict.get("lpips", 0)
                ema_gan += loss_dict.get("gan", 0)
                ema_d += loss_dict.get("d", 0)
                ema_cnt += 1

                # 进度条显示当前步 loss
                postfix = {"l2": f"{loss_dict.get('l2', 0):.4f}"}
                if "gan" in loss_dict:
                    postfix["gan"] = f"{loss_dict['gan']:.4f}"
                if "d" in loss_dict:
                    postfix["d"] = f"{loss_dict['d']:.4f}"
                if "lpips" in loss_dict:
                    postfix["lpips"] = f"{loss_dict['lpips']:.4f}"
                pbar.set_postfix(**postfix)
                pbar.update(1)

                # 每 log_freq 步输出平均损失
                if self.global_step % self.log_freq == 0:
                    avg_l2 = ema_l2 / max(ema_cnt, 1)
                    avg_lpips = ema_lpips / max(ema_cnt, 1)
                    avg_gan = ema_gan / max(ema_cnt, 1)
                    avg_d = ema_d / max(ema_cnt, 1)
                    lr = self.optimizer.param_groups[0]['lr']

                    parts = [f"l2={avg_l2:.6f}"]
                    if ema_lpips > 0:
                        parts.append(f"lpips={avg_lpips:.6f}")
                    if ema_gan != 0:
                        parts.append(f"gan={avg_gan:.6f}")
                    if ema_d != 0:
                        parts.append(f"d={avg_d:.6f}")
                    print(
                        f"\n[step {self.global_step}/{total_iters}] "
                        + "  ".join(parts)
                        + f"  lr={lr:.2e}"
                    )

                    ema_l2 = 0.0
                    ema_lpips = 0.0
                    ema_gan = 0.0
                    ema_d = 0.0
                    ema_cnt = 0

                if self.global_step % self.save_freq == 0:
                    self.save_checkpoint(self.global_step)

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
    parser = argparse.ArgumentParser(description="Train Flow Embedder Δ_φ")
    parser.add_argument("--resume", type=str, default=None, help="Path to training state checkpoint")
    args = parser.parse_args()

    trainer = FlowEmbedderTrainer()
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
