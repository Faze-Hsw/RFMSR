<div align="center">

# RFMSR: Residual Flow Matching for Image Super-Resolution

<p align="center"><i>Residual Flow Matching in SD2.1 VAE Latent Space with LightningDiT.</i></p>


[![HF-Model](https://img.shields.io/badge/🤗%20RFMSR-HuggingFace-FCC624.svg)](https://huggingface.co/CSWRY/RFMSR)

</div>

<p align="center">
  <img src="assets/overview.png" alt="RFMSR overview" width="100%">
</p>
<p align="center"><em>Overview of the RFMSR framework. <b>Forward:</b> the residual flow progressively injects noise into the HQ latent along the residual path (HQ → LR). <b>Reverse:</b> the network learns to denoise and recover the HQ latent from the LR condition, yielding the SR result.</em></p>


## News

- **2026-07** — Initial release: training & inference code, one-step and multi-step checkpoints.


## Preparation

### Hardware Requirements

- **GPU VRAM**: >= 16 GB (recommended >= 24 GB for training)
- **Disk**: >= 10 GB (model weights ~5.5 GB + training data)
- Recommended: A100 / RTX 4090 / 3090

### Installation

**Python >= 3.10** (recommended 3.10 ~ 3.12)

```bash
# 0. Create virtual environment (recommended)
conda create -n rfmsr python=3.12 -y
conda activate rfmsr

# 1. Install PyTorch (CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install other dependencies
pip install -r requirements.txt
```

### Pretrained Weights

Download pretrained weights into `ckpts/`:

| File | Size | Description |
|------|------|-------------|
| `rfmsr.safetensors` | 1.81 GB | Multi-step Flow Matching checkpoint |
| `rfmsr_os.safetensors` | 1.81 GB | One-step distillation checkpoint (recommended) |
| `sd21_lwdecoder.pth` | 50 MB | SD2.1 lightweight decoder |
| `stable-diffusion-2-1-base/` | — | SD2.1 VAE (auto-download via diffusers) |

**Option A: HuggingFace (recommended)**

Model weights are available at [CSWRY/RFMSR](https://huggingface.co/CSWRY/RFMSR) on HuggingFace.

```bash
# Download ckpts from HuggingFace
huggingface-cli login
huggingface-cli download CSWRY/RFMSR ckpts/ --local-dir . --local-dir-use-symlinks False
```

**Option B: Manual placement**

```
ckpts/
├── rfmsr.safetensors
├── rfmsr_os.safetensors
├── sd21_lwdecoder.pth
├── stable-diffusion-2-1-base/
└── VOSR_0.5B_ms/
    └── checkpoints/
        └── ema_model.safetensors   # optional, for training init
```

### Data Preparation

Training data (`traindata/`): place HR images (DIV2K, Flickr2K, etc.) directly under this directory.

Validation data (`assets/validate_gt/`, `assets/validate_lq/`): paired 512x512 HR/LR images for validation during training.

Test data (`testdata/`): LSDIR, ImageNet512, and RealSR benchmarks for evaluation.


## Training

### Stage 1: Multi-step Flow Matching

Train the LightningDiT with Residual Flow Matching (velocity prediction):

```bash
python train_rfmsr.py
```

Key configuration in `configs/train_rfmsr.yaml`:
- 10,000 iterations, batch size 32, GT crop 512x512
- Learning rate 5e-5, EMA rate 0.999, bf16 AMP
- RealESRGAN degradation pipeline (blur + noise + JPEG + resize)
- Pretrained init from VOSR checkpoint (optional)

Residual FM formulation:
- Flow path: `x_t = z_hr + t * (z_lr - z_hr) + t * sigma * epsilon`
- Target velocity: `v_gt = (z_lr - z_hr) + sigma * epsilon`
- Network input: `cat(z_lr[4ch], x_t[4ch]) + DINOv2 Cross-Attention`

### Stage 2: One-Step Distillation (L2 + LPIPS + GAN)

Distill the multi-step model into a one-step generator with perceptual and adversarial losses:

```bash
python train_rfmsr_os.py
```

Key configuration in `configs/train_rfmsr_os.yaml`:
- Losses: velocity supervision + LPIPS (VGG) + Hinge GAN
- PatchGAN discriminator in 4-channel latent space
- Student initialized from `ckpts/rfmsr.safetensors`
- Validation runs both 1-step and 15-step inference for comparison


## Inference

### Quick Start

```bash
# One-step inference (fast, recommended)
python infer_rfmsr.py --input input.png --steps 1

# Multi-step inference (higher quality)
python infer_rfmsr.py --input input.png --steps 15

# Process a folder
python infer_rfmsr.py --input ./test_images/ --scale 4.0 --steps 15
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Input image or directory |
| `--rfmsr_path` | `ckpts/rfmsr_os.safetensors` | Path to RFMSR checkpoint |
| `--scale` | 4.0 | Upscale factor |
| `--steps` | 15 | Reverse integration steps (1 for one-step) |
| `--flow_sigma` | 1.0 | Noise standard deviation |
| `--seed` | 42 | Random seed |
| `--color_correction` | wavelet | Color correction: `adain` / `wavelet` / `ycbcr` / `none` |
| `--chopping` | True | Tiled inference for large images |
| `--tile_size` | 512 | Pixel-space tile size |
| `--tile_stride` | 256 | Sliding window stride |
| `--vae_path` | `ckpts/stable-diffusion-2-1-base` | SD2.1 VAE path |
| `--denoise_device` | cuda | Device for RFMSR + VAE |
| `--output` | outputs | Output directory |

### One-Step vs Multi-Step

| Mode | Steps | Speed | Quality | Use Case |
|------|-------|-------|---------|----------|
| One-step (`rfmsr_os`) | 1 | Fast | Good | Real-time / batch processing |
| Multi-step (`rfmsr`) | 15 | Slower | Best | Maximum quality |


## Model Architecture

RFMSR uses a **LightningDiT** backbone operating in SD2.1 VAE latent space:

```
Input: cat(z_lr[4ch], x_t[4ch]) → [B, 8, H, W]
  ├── PatchEmbed (patch_size=2) → tokens [B, N, 1024]
  ├── TimestepEmbedder (sinusoidal + MLP)
  ├── LightningDiTBlock × 28
  │     ├── Self-Attention + QK-Norm + RoPE
  │     ├── Cross-Attention (DINOv2 semantic features)
  │     ├── SwiGLU FFN
  │     └── AdaLN (time-conditioned scale/shift)
  └── FinalLayer → unpatchify → v [B, 4, H, W]
```

Key design choices:
- RoPE positional encoding for spatial awareness
- RMSNorm for stable training
- DINOv2 frozen encoder (ViT-B) injects semantic features via Cross-Attention
- SwiGLU activation in FFN layers
- AdaLN modulation conditioned on timestep `t`

Dependencies:
- `diffusers` — SD2.1 VAE encoder/decoder (frozen)
- `timm` — PatchEmbed
- `torch.hub` — DINOv2 (facebookresearch/dinov2)
- `basicsr` — RealESRGAN degradation


## Evaluation Metrics

The training script evaluates the following metrics during validation:

- **PSNR** / **SSIM** — distortion-based metrics
- **LPIPS** (Alex) — perceptual similarity
- **DISTS** — deep image structure and texture similarity
- **NIQE** — no-reference image quality
- **MUSIQ** / **MANIQA** / **CLIPIQA** — transformer-based IQA


## Visual Comparisons

Below are qualitative comparisons on benchmark datasets. Each figure shows the low-quality (LQ) input, ground-truth high-quality (HQ) image, and results from competing methods (SeeSR, VOSR, ResShift, InvSR, OSEDiff) alongside RFMSR in both multi-step (15-step) and one-step modes. RFMSR preserves fine textures, text clarity, and natural details while avoiding hallucination and artifacts.

### Portrait

<p align="center">
  <img src="assets/comparison_1.png" alt="RFMSR portrait comparison" width="100%">
</p>
<p align="center"><em>Portrait super-resolution (4×). RFMSR-15 and RFMSR-1 faithfully recover facial details without over-smoothing or hallucinated artifacts.</em></p>

### Text & Logo

<p align="center">
  <img src="assets/comparison_2.png" alt="RFMSR text comparison" width="100%">
</p>
<p align="center"><em>Text super-resolution (4×). RFMSR preserves sharp letter edges and correct glyph shapes, while one-step baselines (VOSR-1, InvSR-1, OSEDiff-1) introduce distortion or miss strokes.</em></p>

<p align="center">
  <img src="assets/comparison_6.png" alt="RFMSR text comparison 2" width="100%">
</p>
<p align="center"><em>Another text example (4×). RFMSR-1 achieves comparable fidelity to the 15-step model, demonstrating strong one-step distillation.</em></p>

### Landscape & Nature

<p align="center">
  <img src="assets/comparison_4.png" alt="RFMSR landscape comparison" width="100%">
</p>
<p align="center"><em>Landscape super-resolution (4×). RFMSR recovers fine rock and grass textures without the blurriness or over-sharpening seen in competing methods.</em></p>


## Output

- Output files: `{input_name}_rfmsr.png`
- Output directory: defaults to `outputs/` (configurable via `--output`)
- Output size: `ceil(W × scale) × ceil(H × scale)`


## Contact

For questions or collaboration, please open an issue on GitHub.

## Citation

If you find this work useful, please cite:

```bibtex
@article{rfmsr2026,
  title={RFMSR: Residual Flow Matching for Image Super-Resolution},
  author={},
  journal={},
  year={2026}
}
```
