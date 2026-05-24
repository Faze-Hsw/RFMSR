"""
Flow Embedder 速度校正训练脚本

训练目标: 在 HR→LR 整流流路径上，embedder 学习校正冻结 Flux 的速度预测。

流路径: t=0 → HR, t=1 → LR
  x_t = (1-t)·z_hr + t·z_lr
  v_gt = z_lr - z_hr

训练:  v_total = Flux(x_t, t).detach() + embedder(x_t, t, z_lr)
       loss = MSE(v_total, z_lr - z_hr)

核心优势:
  - Flux 完全冻结 + detach，梯度零穿透 12B 模型
  - 直接监督速度场，embedder 收到满强度梯度
  - 训推一致：训练学速度，推理积分速度

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

        # ---- 加载模块 ----
        self._load_flux()
        self._encode_prompt()
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

    def _load_flux(self):
        """加载冻结的 Flux DiT + VAE。"""
        flux_cfg = self.cfg["flux"]
        name = flux_cfg["model_name"]
        device = self.device

        print(f"Loading Flux '{name}' -> {device} ...")
        self.flux = load_flow_model(name, device=device, verbose=False)
        self.flux.requires_grad_(False)
        self.flux.eval()  # Flux 冻结，不需要 train()（无 dropout，梯度不穿透）

        print(f"Loading VAE -> {device} ...")
        self.ae = load_ae(name, device=device)
        self.ae.eval()
        self.ae.requires_grad_(False)

        self.guidance = flux_cfg["guidance"]

        n_params = sum(p.numel() for p in self.flux.parameters())
        print(f"  Flux params: {n_params / 1e9:.2f}B (frozen, bf16)")
        print("✅ Flux + VAE loaded.")

    def _encode_prompt(self):
        """编码固定 prompt → 缓存，释放 T5/CLIP。"""
        flux_cfg = self.cfg["flux"]
        te_device = flux_cfg.get("text_encoder_device", "cuda")
        prompt = flux_cfg["prompt"]
        t5_max_length = flux_cfg.get("t5_max_length", 512)
        weights = flux_cfg.get("weights", {})

        print(f"Loading T5 + CLIP -> {te_device} ...")
        t5 = load_t5(te_device, max_length=t5_max_length, ckpt_path=weights.get("t5xxl"))
        clip = load_clip(te_device, ckpt_path=weights.get("clip"))

        print(f"Encoding prompt: '{prompt[:60]}...'")
        with torch.no_grad():
            self.cached_txt = t5(prompt).to(self.device)
            self.cached_vec = clip(prompt).to(self.device)
        self.cached_txt_ids = torch.zeros(
            1, self.cached_txt.shape[1], 3, dtype=torch.float32, device=self.device
        )

        del t5, clip
        torch.cuda.empty_cache()
        print("✅ Prompt encoded, T5+CLIP released.")

    def _build_embedder(self):
        cfg_path = self.cfg["model_config"]
        self.embedder = create_flow_embedder(cfg_path).to(self.device)
        n_params = sum(p.numel() for p in self.embedder.parameters())
        print(f"✅ FlowEmbedder: {n_params / 1e6:.2f}M params")

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
        print("\n" + "=" * 60)
        print(f"Experiment      : {self.cfg['experiment']['name']}")
        print(f"Save dir        : {self.exp_dir}")
        print(f"Iterations      : {self.cfg['training']['iterations']}")
        print(f"Batch size      : {self.cfg['data']['batch_size']}")
        print(f"LR              : {self.cfg['training']['lr']}")
        print(f"GT size         : {self.cfg['data']['gt_size']}")
        print(f"Flow direction  : HR(t=0) → LR(t=1)")
        print(f"Training target : velocity MSE (Flux detach, zero grad-penetration)")
        print(f"AMP             : {self.use_amp}")
        print(f"Loss            : velocity MSE = 1.0")
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
    # Pack / Unpack / img_ids
    # ------------------------------------------------------------------

    @staticmethod
    def pack(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def unpack(x_packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return rearrange(x_packed, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                         h=h, w=w, ph=2, pw=2)

    @staticmethod
    def make_img_ids(b: int, h_lat: int, w_lat: int, device: torch.device) -> torch.Tensor:
        img_ids = torch.zeros(h_lat, w_lat, 3, device=device)
        img_ids[..., 1] = torch.arange(h_lat, device=device)[:, None]
        img_ids[..., 2] = torch.arange(w_lat, device=device)[None, :]
        return repeat(img_ids, "h w c -> b (h w) c", b=b)

    # ------------------------------------------------------------------
    # Flux 前向（detach，梯度不穿透）
    # ------------------------------------------------------------------

    @staticmethod
    def _flux_forward_no_args(flux, x_t_packed, img_ids, txt, txt_ids, t_vec, vec, guidance_vec):
        return flux(
            img=x_t_packed, img_ids=img_ids,
            txt=txt, txt_ids=txt_ids,
            timesteps=t_vec, y=vec, guidance=guidance_vec,
        )

    def flux_velocity(self, x_t_packed: torch.Tensor, t: torch.Tensor,
                      img_ids: torch.Tensor, bs: int) -> torch.Tensor:
        """冻结 Flux 前向，返回速度预测。gradient 不穿透 Flux。"""
        flux_dtype = x_t_packed.dtype
        txt = self.cached_txt.expand(bs, -1, -1).to(self.device, dtype=flux_dtype)
        txt_ids = self.cached_txt_ids.expand(bs, -1, -1).to(self.device)
        vec = self.cached_vec.expand(bs, -1).to(self.device, dtype=flux_dtype)
        t_vec = t.to(self.device, dtype=flux_dtype)
        guidance_vec = torch.full((bs,), self.guidance, device=self.device, dtype=flux_dtype)

        v = self._flux_forward_no_args(
            self.flux, x_t_packed, img_ids, txt, txt_ids, t_vec, vec, guidance_vec,
        )
        return v.detach()  # ⬅ 关键：detach，梯度零穿透 Flux

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
        z_lr = self.vae_encode(lr_up)    # [B, 16, H/8, W/8]

        z_hr = z_hr.detach().float()
        z_lr = z_lr.detach().float()
        B, C, H, W = z_hr.shape
        h_pack, w_pack = H // 2, W // 2

        # LR image [-1, 1] for embedder conditioning (raw image space, not latent)
        lr_raw = lr_up * 2.0 - 1.0  # [0,1] → [-1,1], [B, 3, H_pix, W_pix]

        # 2. 采样时间 t ~ U(0, 1)，构造流路径上的点 x_t
        t = torch.rand(B, device=device)  # [B] ∈ [0,1]
        t_expand = t[:, None, None, None]
        x_t = (1.0 - t_expand) * z_hr + t_expand * z_lr  # [B, 16, H, W]

        # 真值速度
        v_gt = z_lr - z_hr  # [B, 16, H, W]  沿直线恒定

        # 3. Flux + Embedder + Loss（统一 AMP 域）
        with autocast(device_type="cuda", enabled=self.use_amp):
            # Flux 速度预测（detach, 零梯度穿透）
            x_t_packed = self.pack(x_t.to(torch.bfloat16))
            img_ids = self.make_img_ids(bs, h_pack, w_pack, device)
            v_flux_packed = self.flux_velocity(x_t_packed, t, img_ids, bs)
            v_flux = self.unpack(v_flux_packed.float(), h_pack, w_pack)

            # Flux 对 SR latent 可能产出 NaN/极端值，净化后再进 embedder
            v_flux = torch.nan_to_num(v_flux, nan=0.0, posinf=10.0, neginf=-10.0)
            v_flux = v_flux.clamp(-20.0, 20.0)

            # Embedder 速度校正（原始 LR 图像作为条件上下文）
            v_corr = self.embedder(x_t, t, lr_raw, v_flux)

            # 总速度 = Flux + 校正
            v_total = v_flux + v_corr

            # 速度回归损失
            loss = F.mse_loss(v_total, v_gt)

        losses = {"velo": loss.item()}

        # 4. 反向传播（梯度只到 embedder）
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
    parser = argparse.ArgumentParser(description="Train Flow Embedder (Velocity Correction)")
    parser.add_argument("--resume", type=str, default=None, help="Path to training state checkpoint")
    args = parser.parse_args()

    trainer = FlowEmbedderTrainer()
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
