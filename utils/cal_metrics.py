"""
计算 SR 结果与 GT 的 PSNR / SSIM / LPIPS / DISTS / NIQE / MUSIQ / FID / MANIQA / CLIPIQA。

配对逻辑：SR 文件名 == GT 文件名（纯文件名匹配，不递归子目录）。

用法:
    python utils/cal_metrics.py --gt_dir testdata/RealSR/HR --sr_dir outputs/RealSR

依赖:
    pip install pyiqa lpips
"""

import os
import sys
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def scan_flat(dir_path: Path, exts: list[str]) -> list[Path]:
    """扫描目录下所有图片（不递归），按文件名排序。"""
    files = []
    for ext in exts:
        for pattern in [f"*.{ext}", f"*.{ext.upper()}"]:
            files.extend(dir_path.glob(pattern))
    files = sorted(set(files), key=lambda p: p.name.lower())
    return files


class PairDataset(Dataset):
    """通过文件名将 SR 与 GT 一一配对。"""

    def __init__(self, sr_dir: str, gt_dir: str, exts: list[str] = None):
        if exts is None:
            exts = ["png", "jpg", "jpeg", "bmp", "webp"]

        sr_dir = Path(sr_dir)
        gt_dir = Path(gt_dir)

        self.sr_files = scan_flat(sr_dir, exts)
        self.pairs = []

        for sr_path in self.sr_files:
            gt_path = gt_dir / sr_path.name
            if gt_path.exists():
                self.pairs.append((str(sr_path), str(gt_path)))

        if not self.pairs:
            raise RuntimeError(
                f"No paired SR-GT images found.\n"
                f"  SR dir: {sr_dir}\n  GT dir: {gt_dir}\n"
                f"  Ensure filenames match exactly."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import cv2
        import numpy as np

        sr_path, gt_path = self.pairs[idx]

        sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)

        if sr_img is None or gt_img is None:
            raise RuntimeError(f"Cannot read: {sr_path} or {gt_path}")

        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Ensure same size
        if sr_img.shape != gt_img.shape:
            h = min(sr_img.shape[0], gt_img.shape[0])
            w = min(sr_img.shape[1], gt_img.shape[1])
            sr_img = sr_img[:h, :w, :]
            gt_img = gt_img[:h, :w, :]

        sr_tensor = torch.from_numpy(sr_img.copy()).permute(2, 0, 1)  # [3, H, W]
        gt_tensor = torch.from_numpy(gt_img.copy()).permute(2, 0, 1)

        return {
            "sr": sr_tensor,
            "gt": gt_tensor,
            "sr_path": sr_path,
            "gt_path": gt_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Calculate SR metrics with GT reference")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground truth (HR) image directory")
    parser.add_argument("--sr_dir", type=str, required=True, help="SR result image directory")
    parser.add_argument("--bs", type=int, default=8, help="Batch size")
    parser.add_argument("--log_name", type=str, default="metrics.log", help="Log filename")
    parser.add_argument("--test_y_channel", action="store_true", default=True,
                        help="Use Y channel for PSNR/SSIM")
    parser.add_argument("--no_y_channel", action="store_true",
                        help="Use RGB for PSNR/SSIM instead of Y channel")
    args = parser.parse_args()

    test_y = not args.no_y_channel

    # ---- Output ----
    log_path = Path(args.sr_dir).parent / args.log_name
    log_lines = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"GT dir: {args.gt_dir}")
    log(f"SR dir: {args.sr_dir}")
    log(f"Log:    {log_path}")

    # ---- Dataset ----
    dataset = PairDataset(args.sr_dir, args.gt_dir)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)
    log(f"Paired images: {len(dataset)}")

    # ---- Metrics ----
    import pyiqa
    import lpips

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psnr_metric = pyiqa.create_metric("psnr", test_y_channel=test_y, color_space="ycbcr", device=device)
    ssim_metric = pyiqa.create_metric("ssim", test_y_channel=test_y, color_space="ycbcr", device=device)
    dists_metric = pyiqa.create_metric("dists", device=device)

    # NR metrics (on SR only)
    niqe_metric = pyiqa.create_metric("niqe", device=device)
    musiq_metric = pyiqa.create_metric("musiq", device=device)
    maniqa_metric = pyiqa.create_metric("maniqa", device=device)
    clipiqa_metric = pyiqa.create_metric("clipiqa", device=device)

    # LPIPS
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    # FID
    fid_metric = pyiqa.create_metric("fid", device=device)

    metrics_sum = {
        "PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0,
        "DISTS": 0.0, "NIQE": 0.0,
        "MUSIQ": 0.0, "MANIQA": 0.0, "CLIPIQA": 0.0,
    }

    # ---- Compute ----
    log("Computing metrics...")
    for batch in tqdm(dataloader, desc="Metrics", unit="batch"):
        im_sr = batch["sr"].to(device)   # [B, 3, H, W]  [0, 1]
        im_gt = batch["gt"].to(device)
        B = im_sr.shape[0]

        # Full-reference
        current_psnr = psnr_metric(im_sr, im_gt).mean().item()
        current_ssim = ssim_metric(im_sr, im_gt).mean().item()
        current_dists = dists_metric(im_sr, im_gt).mean().item()

        # LPIPS expects [-1, 1]
        im_gt_norm = (im_gt - 0.5) / 0.5
        im_sr_norm = (im_sr - 0.5) / 0.5
        current_lpips = loss_fn_alex(im_gt_norm, im_sr_norm).mean().item()

        # No-reference (on SR only)
        current_niqe = niqe_metric(im_sr).mean().item()
        current_musiq = musiq_metric(im_sr).mean().item()
        current_maniqa = maniqa_metric(im_sr).mean().item()
        current_clipiqa = clipiqa_metric(im_sr).mean().item()

        metrics_sum["PSNR"] += current_psnr * B
        metrics_sum["SSIM"] += current_ssim * B
        metrics_sum["LPIPS"] += current_lpips * B
        metrics_sum["DISTS"] += current_dists * B
        metrics_sum["NIQE"] += current_niqe * B
        metrics_sum["MUSIQ"] += current_musiq * B
        metrics_sum["MANIQA"] += current_maniqa * B
        metrics_sum["CLIPIQA"] += current_clipiqa * B

    N = len(dataset)
    for k in metrics_sum:
        metrics_sum[k] /= N

    # FID (directory-level)
    try:
        fid = fid_metric(args.sr_dir, args.gt_dir)
        metrics_sum["FID"] = fid
    except Exception as e:
        log(f"[WARN] FID failed: {e}")
        metrics_sum["FID"] = float("nan")

    # ---- Report ----
    log("")
    log("=" * 45)
    log("  Mean Metrics")
    log("=" * 45)
    log(f"  PSNR (Y):      {metrics_sum['PSNR']:>8.2f} dB")
    log(f"  SSIM (Y):      {metrics_sum['SSIM']:>8.4f}")
    log(f"  LPIPS (Alex):  {metrics_sum['LPIPS']:>8.4f}")
    log(f"  DISTS:         {metrics_sum['DISTS']:>8.4f}")
    log(f"  NIQE:         {metrics_sum['NIQE']:>8.4f}")
    log(f"  MUSIQ:        {metrics_sum['MUSIQ']:>8.4f}")
    log(f"  MANIQA:       {metrics_sum['MANIQA']:>8.4f}")
    log(f"  CLIPIQA:      {metrics_sum['CLIPIQA']:>8.4f}")
    log(f"  FID:           {metrics_sum['FID']:>8.2f}")
    log("=" * 45)

    # Write log
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
