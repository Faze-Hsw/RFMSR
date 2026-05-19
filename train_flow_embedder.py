"""
Flow Embedder Δ_φ 训练脚本

训练目标: 让冻结 Flux 在 FlowEmbedder 修正后的嵌入点上预测出正确的 velocity
Loss = MSE( v_θ(x_t, t) , ε - z_HR )
其中 x_t = (1-t) * (z_LR + Δ_φ(z_LR, t)) + t * ε

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

    def __init__(self, config_path: str):
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
        device = self.device

        hr = batch["gt"].to(device)      # [B,3,512,512] [0,1]
        lr = batch["lq"].to(device)      # [B,3,128,128] [0,1]
        bs = hr.shape[0]

        # 1. VAE encode (no grad)
        z_hr = self.vae_encode(hr)       # [B,16,64,64]
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        z_lr = self.vae_encode(lr_up)    # [B,16,64,64]

        # detach，后面 Δ_φ 接收 z_lr 作为输入但不需要 VAE 的梯度
        z_hr = z_hr.detach().float()
        z_lr = z_lr.detach().float()

        # 2. 采样时间步 t（Logit-Normal 分布，与 SD3/Flux 预训练一致）和噪声 ε
        logit_mean = tcfg.get("logit_normal_mean", 0.0)
        logit_std = tcfg.get("logit_normal_std", 1.0)
        t = torch.sigmoid(torch.randn(bs, device=device) * logit_std + logit_mean)
        eps = torch.randn_like(z_hr)

        # 3. Flow Embedder 前向（在 AMP 下，它对精度敏感）
        with autocast(device_type="cuda", enabled=self.use_amp):
            delta = self.embedder(z_lr, t)               # [B,16,64,64]
            z_corrected = z_lr + delta

            # 4. 构造嵌入点 x_t
            t_expand = t[:, None, None, None]              # [B,1,1,1]
            x_t = (1.0 - t_expand) * z_corrected + t_expand * eps

        # 5. Flux 前向（Flux 权重已冻结 + bf16，AMP 无帮助反而可能引起精度问题）
        x_t_packed = self.pack(x_t.to(torch.bfloat16))
        _, _, h_lat, w_lat = z_hr.shape
        img_ids = self.make_img_ids(bs, h_lat // 2, w_lat // 2, device)

        v_pred = self.flux_velocity(x_t_packed, t, img_ids, bs)

        # 6. 目标 velocity
        v_target = eps - z_hr                          # [B,16,64,64]
        v_target_packed = self.pack(v_target.to(torch.bfloat16))

        # 7. Loss（NaN 诊断）
        loss = F.mse_loss(v_pred.float(), v_target_packed.float())
        if torch.isnan(loss):
            for name, t in [("delta", delta), ("x_t", x_t), ("v_pred", v_pred), ("v_target", v_target)]:
                has_nan = torch.isnan(t).any().item()
                has_inf = torch.isinf(t).any().item()
                print(f"  ⚠️  {name}: nan={has_nan} inf={has_inf}  min={t.min().item():.3f} max={t.max().item():.3f}")

        # 8. Backward
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

        # 9. EMA
        self._update_ema()

        return {"loss": loss.item()}

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
            total=total_iters,
            initial=self.global_step,
            desc="Train",
            unit="step",
            bar_format="{desc} [{n:>6d}/{total_fmt}] {percentage:3.0f}% |{bar}| {postfix} [{rate_fmt}]",
        )

        # 累计损失，每 log_freq 步输出一次平均值
        loss_ema = 0.0
        loss_cnt = 0

        try:
            while self.global_step < total_iters:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                loss_dict = self.train_step(batch)
                self.global_step += 1

                loss_ema += loss_dict["loss"]
                loss_cnt += 1

                # 进度条显示当前步 loss
                pbar.set_postfix(loss=f"{loss_dict['loss']:.6f}")
                pbar.update(1)

                # 每 log_freq 步输出平均损失
                if self.global_step % self.log_freq == 0:
                    avg_loss = loss_ema / loss_cnt
                    lr = self.optimizer.param_groups[0]['lr']
                    print(
                        f"\n[step {self.global_step}/{total_iters}] "
                        f"avg_loss={avg_loss:.6f}  lr={lr:.2e}"
                    )
                    loss_ema = 0.0
                    loss_cnt = 0

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
    parser.add_argument("--config", type=str, default="configs/train_flow_embedder.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to training state checkpoint")
    args = parser.parse_args()

    trainer = FlowEmbedderTrainer(args.config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
