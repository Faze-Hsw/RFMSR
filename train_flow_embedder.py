"""
Flow Embedder 训练脚本 — LightningDiT 速度预测

训练目标: DiT Embedder 学习从噪声到 HR 的流匹配速度，以 LR latent 为条件。

流路径: t=0 → HR latent, t=1 → 纯高斯噪声 ε
  x_t = (1-t)·z_hr + t·ε
  v_gt = ε - z_hr

架构 (VOSR LightningDiT, ~0.35B):
  cat(z_lr[16ch], x_t[16ch]) → [B, 32, H, W]
    → PatchEmbed(patch=2) → tokens
    → 28× LightningDiTBlock (Self-Attn + RoPE + QKNorm + SwiGLU + AdaLN)
    → unpatchify → v [16ch]

训推一致:
  Phase 1 (高噪声): DiT Embedder 从纯噪声锚定结构
  Phase 2 (低噪声): Flux 精修质量

用法:
  python train_flow_embedder.py
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from safetensors.torch import save_file as safe_save
from tqdm import tqdm

from datapipe.train_dataloader import create_train_dataloader
from flux.util import load_ae
from models.dit_flow_embedder import create_dit_flow_embedder
from models.dinov2_encoder import create_dinov2_encoder


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
        self.keep_last_n = exp.get("keep_last_n", 0)

        # ---- 加载模块 ----
        self._load_vae()
        self._load_dinov2()
        self._build_embedder()
        self._build_optimizer()
        self._build_dataloader()
        self._build_ema()

        # AMP
        self.use_amp = self.cfg["training"]["use_amp"]
        self.scaler = GradScaler() if self.use_amp else None

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
        """只加载 VAE 编解码器（训练不需要 Flux DiT / T5 / CLIP）。"""
        flux_cfg = self.cfg["flux"]
        name = flux_cfg["model_name"]
        device = self.device

        print(f"Loading VAE '{name}' -> {device} ...")
        self.ae = load_ae(name, device=device)
        self.ae.eval()
        self.ae.requires_grad_(False)
        print("✅ VAE loaded.")

    def _load_dinov2(self):
        """加载冻结的 DINOv2 语义编码器。"""
        dv2_cfg = self.cfg.get("dinov2", {}) or {}
        self.use_dinov2 = dv2_cfg.get("enabled", False)

        if self.use_dinov2:
            model_cfg_path = self.cfg["model_config"]
            self.venc = create_dinov2_encoder(model_cfg_path, device=self.device)
        else:
            self.venc = None

    def _build_embedder(self):
        cfg_path = self.cfg["model_config"]
        self.embedder = create_dit_flow_embedder(cfg_path).to(self.device)
        n_params = sum(p.numel() for p in self.embedder.parameters())
        print(f"✅ LightningDiT Embedder: {n_params / 1e6:.1f}M (~0.35B) params")

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
                {k: deepcopy(v.data) for k, v in self.embedder.state_dict().items()}
            )
        else:
            self.ema_rate = 0
            self.ema_state = None

    def _print_summary(self):
        tcfg = self.cfg["training"]
        print("\n" + "=" * 60)
        print(f"Experiment      : {self.cfg['experiment']['name']}")
        print(f"Save dir        : {self.exp_dir}")
        print(f"Iterations      : {tcfg['iterations']}")
        print(f"Batch size      : {self.cfg['data']['batch_size']}")
        print(f"LR              : {tcfg['lr']}")
        print(f"GT size         : {self.cfg['data']['gt_size']}")
        print(f"Flow            : Noise→HR (HR@t=0, Noise@t=1)")
        print(f"Architecture    : LightningDiT")
        print(f"Conditioning    : LR latent (VAE-encoded upsampled LR)" + 
              (" + DINOv2 Cross-Attn" if self.use_dinov2 else ""))
        print(f"AMP             : {self.use_amp}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # VAE encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def vae_encode(self, img: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] float [0,1] → [B,16,H/8,W/8] latent."""
        img = img.to(device=self.device, dtype=torch.bfloat16)
        img = img * 2.0 - 1.0
        return self.ae.encode(img)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        tcfg = self.cfg["training"]
        device = self.device

        hr = batch["gt"].to(device)      # [B, 3, H, W]  [0, 1]
        lr = batch["lq"].to(device)      # [B, 3, H, W]  [0, 1]
        bs = hr.shape[0]

        # 1. VAE encode
        z_hr = self.vae_encode(hr)       # [B, 16, H/8, W/8]
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        z_lr = self.vae_encode(lr_up)    # [B, 16, H/8, W/8]  ← LR latent 条件

        z_hr = z_hr.detach().float()
        z_lr = z_lr.detach().float()
        B, C, H, W = z_hr.shape

        # 2. DINOv2 语义特征（从像素空间 LR 提取）
        venc_fea = None
        if self.use_dinov2 and self.venc is not None:
            venc_fea = self.venc(lr)  # list of [B, N, enc_dim]

        # 3. 噪声→HR 去噪流: x_t = (1-t)·z_hr + t·ε
        t = torch.rand(B, device=device)  # [B] ∈ [0,1]
        t_expand = t[:, None, None, None]
        epsilon = torch.randn_like(z_hr)   # [B, 16, H, W]  纯高斯噪声
        x_t = (1.0 - t_expand) * z_hr + t_expand * epsilon  # [B, 16, H, W]

        # 真值速度: 从噪声推往 HR
        v_gt = epsilon - z_hr  # [B, 16, H, W]

        # 4. DiT Embedder + Loss (LR latent 条件 + DINOv2 语义)
        with autocast(device_type="cuda", enabled=self.use_amp):
            v_super = self.embedder(x_t, t, z_lr, venc_fea=venc_fea)
            loss = F.mse_loss(v_super, v_gt)

        losses = {"velo": loss.item()}

        # 4. 反向传播
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if tcfg["gradient_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.embedder.parameters(), tcfg["gradient_clip"]
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if tcfg["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.embedder.parameters(), tcfg["gradient_clip"]
                )
            self.optimizer.step()

        # EMA 更新
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
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int):
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 推理权重 (safetensors)
        weights = self.ema_state if self.ema_state is not None else self.embedder.state_dict()
        ema_path = ckpt_dir / f"flow_embedder_step{step}.safetensors"
        safe_save(weights, ema_path)

        # 完整训练状态 (torch.save)
        state = {
            "step": step,
            "embedder": self.embedder.state_dict(),
            "ema_state": self.ema_state,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
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
        pattern = re.compile(r"(flow_embedder_step|training_state_step)(\d+)")
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

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.embedder.load_state_dict(ckpt["embedder"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scaler and ckpt.get("scaler"):
            self.scaler.load_state_dict(ckpt["scaler"])
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
        self.embedder.train()

        data_iter = iter(self.dataloader)
        pbar = tqdm(
            total=total_iters, initial=self.global_step, desc="Train", unit="step",
            bar_format="{desc} [{n:>6d}/{total_fmt}] {percentage:3.0f}% |{bar}| {postfix} [{rate_fmt}]",
        )

        ema_velo = 0.0
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

                ema_velo += loss_dict.get("velo", 0)
                ema_cnt += 1

                pbar.set_postfix(**{"velo": f"{loss_dict.get('velo', 0):.4f}"})
                pbar.update(1)

                if self.global_step % self.log_freq == 0:
                    avg_v = ema_velo / max(ema_cnt, 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"\n[step {self.global_step}/{total_iters}]  velo={avg_v:.6f}  lr={lr:.2e}")
                    ema_velo = 0.0
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
    parser = argparse.ArgumentParser(description="Train DiT Flow Embedder (LightningDiT)")
    parser.add_argument("--resume", type=str, default=None, help="Path to training state checkpoint")
    args = parser.parse_args()

    trainer = FlowEmbedderTrainer()
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
