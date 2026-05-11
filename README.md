# Flux SR

基于 Black Forest Labs [FLUX.1](https://github.com/black-forest-labs/flux) 的 img2img 图像超分/增强工具。

## 环境配置

### 硬件要求

- **GPU 显存**：≥16GB（flux-dev 约 22GB，flux-schnell 约 14GB）
- **磁盘空间**：≥40GB（模型权重约 34GB）
- 推荐 A100 / RTX 4090 / 3090

### 安装

```bash
# 1. 进入项目目录
cd d:/Projects/flux_sr

# 2. 安装依赖
pip install -r requirements.txt

# 3. HuggingFace 认证
#    需要接受 Black Forest Labs 的模型协议
#    访问 https://huggingface.co/black-forest-labs/FLUX.1-dev 并接受许可
huggingface-cli login

#    或者设置环境变量：
#    set HF_TOKEN=hf_xxxxxxxxxx

# 4. （可选）配置镜像加速
#    在 infer.py 中已默认设置 HF_ENDPOINT=https://hf-mirror.com
#    手动修改或删除该设置即可切换
```

### 手动下载权重

完整运行需要以下 4 个组件：

| 组件 | 类型 | 下载地址 | 自动下载？ |
|------|------|---------|-----------|
| **flux1-dev.safetensors** | Flux Transformer | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | ✅ |
| **ae.safetensors** | VAE | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | ✅ |
| **google/t5-v1_1-xxl** | T5 文本编码器 | HuggingFace | ✅ (transformers) |
| **openai/clip-vit-large-patch14** | CLIP 文本编码器 | HuggingFace | ✅ (transformers) |

**手动放置路径**（如需跳过自动下载）：

```
checkpoints/
└── black-forest-labs_FLUX.1-dev/
    ├── flux1-dev.safetensors   # ~23GB
    └── ae.safetensors          # ~300MB
```

**环境变量覆盖**（完全自定义路径）：

```bash
# 不放在 checkpoints 目录，直接指向已下载的文件
set FLUX_MODEL=D:/models/flux1-dev.safetensors
set FLUX_AE=D:/models/ae.safetensors
```

**文本编码器说明**：
- T5-XXL 和 CLIP 通过 `transformers` 库自动下载到 `~/.cache/huggingface/`
- 如果需要离线使用，需提前下载对应模型到 transformers 缓存目录
- 或使用 `HF_HUB_OFFLINE=1` + 预下载的缓存

## 使用方法

### 快速开始

```bash
# 模型权重会自动下载并缓存到 ./checkpoints/ 目录

# 小图增强（scale=1 保持原尺寸）
python infer.py --init_image input.png

# 2x 超分
python infer.py --init_image input.png --scale 2.0

# 精调参数
python infer.py --init_image input.png --scale 4.0 --cfg 3.0 --steps 28 --seed 42
```

### 分块推理（处理大图）

当输入图像尺寸较大时（如 2K/4K），必须启用分块推理：

```bash
python infer.py --init_image large_image.jpg --scale 1.0 ^
    --chopping_enabled true ^
    --chopping_pch_size 1024 ^
    --chopping_stride_ratio 0.5
```

### 使用自定义配置文件

修改 `configs/infer.yaml` 设置默认参数，然后：

```bash
python infer.py --config configs/infer.yaml --init_image input.png
```

### 模型变体选择

| 模型 | 特点 | 步数 | 推荐用途 |
|------|------|------|---------|
| `flux-dev` | 高质量，Guidance 蒸馏 | 28~50 | 首选超分 |
| `flux-schnell` | 4 步快速推理 | 4 | 快速预览 |
| `flux-dev-canny` | Canny 边缘控制 | 28 | 结构保持超分 |
| `flux-dev-depth` | 深度图控制 | 28 | 立体感增强 |

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--init_image` | (必填) | 输入图像路径 |
| `--scale` | 1.0 | 放大倍数，输出尺寸 = 原图 × scale |
| `--start_timestep` | 1.0 | 起始时间步 (0~1)，1.0=完全重绘，0.5=半保留原图结构 |
| `--cfg` | 3.5 | CFG 引导权重（dev 推荐 2.5~4.0，schnell 忽略） |
| `--steps` | 28 | 采样步数（dev 推荐 28~50，schnell 固定 4） |
| `--seed` | 42 | 随机种子（相同种子 + 相同输入 = 可复现结果） |
| `--model_name` | flux-dev | 模型名称 |
| `--t5_max_length` | 512 | T5 最大序列长度（dev=512，schnell=256） |
| `--out_dir` | outputs | 输出目录（自动创建） |
| `--verbose` | false | 打印详细日志 |
| `--text_encoder_device` | cuda | T5+CLIP 所在设备（省显存可设为 "cpu"） |
| `--denoise_device` | cuda | Flux+VAE 所在设备 |
| **Chopping：** | | |
| `--chopping_enabled` | false | 启用像素空间分块推理 |
| `--chopping_pch_size` | 1024 | 分块边长（需为 16 倍数） |
| `--chopping_stride_ratio` | 0.5 | 滑窗步长比例 |
| `--chopping_extra_bs` | 1 | 并行处理 patch 数 |
| `--chopping_weight_type` | Gaussian | 融合权重类型 |

## 输出

- 输出文件：`{输入文件名}_sr.png`
- 输出目录：默认为 `outputs/`（通过 `--out_dir` 修改）
- 输出尺寸：`ceil(原图宽 × scale) × ceil(原图高 × scale)`
