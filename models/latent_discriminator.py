"""
Latent Discriminator — 基于 InvSR UNet2DConditionDiscriminator 架构
在 Flux 潜空间（16 通道）上做多尺度真伪判别，条件化时间步 + 文本。

包含 GAN 损失函数：
  - hinge_d_loss: 判别器 Hinge 损失
  - gen_loss: 生成器非饱和损失
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# 工具组件
# ═══════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """Sinusoidal time embedding + MLP."""
    def __init__(self, dim, freq_shift=0):
        super().__init__()
        self.dim = dim
        self.freq_shift = freq_shift
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(half) / half))

    def forward(self, t):
        """t: [B] float timesteps in [0, 1]."""
        freqs = self.freqs.to(t.device)  # [half]
        t_emb = t[:, None] * freqs[None, :]  # [B, half]
        t_emb = torch.cat([t_emb.sin(), t_emb.cos()], dim=-1)  # [B, dim]
        return t_emb


class TimestepMLP(nn.Module):
    """Timestep embedding → MLP → output vec."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embed = TimestepEmbedding(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.mlp(self.embed(t))


class ResnetBlock2D(nn.Module):
    """Conv-Norm-SiLU block with optional time embedding injection."""
    def __init__(self, in_ch, out_ch, time_emb_ch=None, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-5)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = None
        if time_emb_ch is not None:
            self.time_mlp = nn.Linear(time_emb_ch, out_ch * 2)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, time_emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_mlp is not None and time_emb is not None:
            scale_shift = self.time_mlp(F.silu(time_emb))[:, :, None, None]
            scale, shift = scale_shift.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SpatialTransformer(nn.Module):
    """Single-layer transformer with self-attn + cross-attn + FFN."""
    def __init__(self, dim, n_heads, cross_dim=None):
        super().__init__()
        assert dim % n_heads == 0, f"{dim} not divisible by {n_heads}"
        self.head_dim = dim // n_heads
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim) if cross_dim else None

        # Self-attention
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # Cross-attention
        if cross_dim:
            self.to_kv = nn.Linear(cross_dim, dim * 2, bias=False)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )

    def _attention(self, q, k, v):
        B, L, _ = q.shape
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, L, -1)

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Self-attention
        residual = x
        x = self.norm1(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        x = residual + self.to_out(self._attention(q, k, v))

        # Cross-attention
        if self.norm3 is not None and context is not None:
            residual = x
            x = self.norm2(x)
            q = self.to_q(x)
            kv = self.to_kv(context)
            k, v = kv.chunk(2, dim=-1)
            x = residual + self.to_out(self._attention(q, k, v))
        else:
            x = self.norm2(x)
            x = x + self.ff(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


# ═══════════════════════════════════════════════
# 编解码 Block
# ═══════════════════════════════════════════════

class DownBlock2D(nn.Module):
    """ResNet blocks + downsampling."""
    def __init__(self, in_ch, out_ch, num_layers, time_emb_ch, add_down=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(ResnetBlock2D(
                in_ch if i == 0 else out_ch, out_ch, time_emb_ch,
            ))
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if add_down else None

    def forward(self, x, time_emb):
        for block in self.blocks:
            x = block(x, time_emb)
        skip = x
        if self.down is not None:
            x = self.down(x)
        return x, [skip]


class CrossAttnDownBlock2D(nn.Module):
    """ResNet + attention + downsampling."""
    def __init__(self, in_ch, out_ch, num_layers, time_emb_ch, cross_dim, n_heads, add_down=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(ResnetBlock2D(
                in_ch if i == 0 else out_ch, out_ch, time_emb_ch,
            ))
            self.attentions.append(SpatialTransformer(out_ch, n_heads, cross_dim))
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if add_down else None

    def forward(self, x, time_emb, context):
        for block, attn in zip(self.blocks, self.attentions):
            x = block(x, time_emb)
            x = attn(x, context)
        skip = [x]
        if self.down is not None:
            x = self.down(x)
        return x, skip


class UpBlock2D(nn.Module):
    """ResNet blocks + upsampling."""
    def __init__(self, in_ch, out_ch, num_layers, time_emb_ch, add_up=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(ResnetBlock2D(
                in_ch if i == 0 else out_ch, out_ch, time_emb_ch,
            ))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if add_up else None

    def forward(self, x, time_emb):
        for block in self.blocks:
            x = block(x, time_emb)
        if self.up is not None:
            x = self.up(x)
        return x


class CrossAttnUpBlock2D(nn.Module):
    """ResNet + attention + upsampling."""
    def __init__(self, in_ch, out_ch, num_layers, time_emb_ch, cross_dim, n_heads, add_up=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(ResnetBlock2D(
                in_ch if i == 0 else out_ch, out_ch, time_emb_ch,
            ))
            self.attentions.append(SpatialTransformer(out_ch, n_heads, cross_dim))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) if add_up else None

    def forward(self, x, time_emb, context):
        for block, attn in zip(self.blocks, self.attentions):
            x = block(x, time_emb)
            x = attn(x, context)
        if self.up is not None:
            x = self.up(x)
        return x


class UNetMidBlock2DCrossAttn(nn.Module):
    """Middle block: ResNet + attention + ResNet."""
    def __init__(self, in_ch, time_emb_ch, cross_dim, n_heads):
        super().__init__()
        self.block1 = ResnetBlock2D(in_ch, in_ch, time_emb_ch)
        self.attn = SpatialTransformer(in_ch, n_heads, cross_dim)
        self.block2 = ResnetBlock2D(in_ch, in_ch, time_emb_ch)

    def forward(self, x, time_emb, context):
        x = self.block1(x, time_emb)
        x = self.attn(x, context)
        x = self.block2(x, time_emb)
        return x


# ═══════════════════════════════════════════════
# 主模型
# ═══════════════════════════════════════════════

class LatentDiscriminator(nn.Module):
    """
    基于 UNet 的多尺度条件判别器，在 Flux 潜空间操作。

    Args:
        in_channels:      输入通道数（Flux VAE = 16）
        block_out_channels: 各级通道数 [128, 256, 512]
        layers_per_block:   各级 ResNet 层数 [1, 2, 2]
        norm_num_groups:    GroupNorm 分组数
        cross_attention_dim: 文本嵌入维度（T5 = 4096，0=不使用文本）
        time_emb_dim:      时间嵌入 MLP 维度
        attention_head_dim: 每级注意力头维度（list），与 InvSR 一致
        vec_dim:           CLIP pooled 维度（=768，0=不使用）
    """
    def __init__(
        self,
        in_channels=16,
        block_out_channels=(128, 256, 512),
        layers_per_block=(1, 2, 2),
        norm_num_groups=32,
        cross_attention_dim=4096,
        time_emb_dim=256,
        attention_head_dim=(8, 16, 16),
        vec_dim=768,
    ):
        super().__init__()
        n_levels = len(block_out_channels)
        assert n_levels == len(layers_per_block)
        assert n_levels == len(attention_head_dim)

        # 时间嵌入
        self.time_embed = TimestepMLP(time_emb_dim, time_emb_dim)

        # CLIP vec 注入（投影到 time_emb 维度并相加）
        self.vec_proj = None
        if vec_dim > 0:
            self.vec_proj = nn.Sequential(
                nn.Linear(vec_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

        # 文本压缩（T5 4096 → 1024）
        self.hidden_compress = None
        self.cross_dim = cross_attention_dim
        if cross_attention_dim > 1024:
            self.hidden_compress = nn.Linear(cross_attention_dim, 1024)
            self.cross_dim = 1024

        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.feature_in = nn.Conv2d(in_channels, norm_num_groups, 3, padding=1)

        # 每级注意力头数 = 通道数 // 每头维度（与 InvSR 一致）
        def _n_heads(level_idx):
            return block_out_channels[level_idx] // attention_head_dim[level_idx]

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        for i in range(n_levels):
            in_ch = block_out_channels[i-1] if i > 0 else block_out_channels[0]
            in_ch = in_ch + norm_num_groups  # 拼接 feature 后
            out_ch = block_out_channels[i]
            n_layers = layers_per_block[i]
            is_last = (i == n_levels - 1)
            use_cross = (self.cross_dim > 0) and (i >= 1)  # 第 2、3 级用 cross-attn

            if use_cross:
                block = CrossAttnDownBlock2D(
                    in_ch, out_ch, n_layers, time_emb_dim, self.cross_dim, _n_heads(i),
                    add_down=not is_last,
                )
            else:
                block = DownBlock2D(
                    in_ch, out_ch, n_layers, time_emb_dim,
                    add_down=not is_last,
                )
            self.down_blocks.append(block)

        mid_ch = block_out_channels[-1]
        self.mid_block = None
        if self.cross_dim > 0:
            self.mid_block = UNetMidBlock2DCrossAttn(mid_ch, time_emb_dim, self.cross_dim, _n_heads(-1))

        # 输出头（中间尺度）
        self.out_blocks = nn.ModuleList()
        self.out_blocks.append(self._make_out_head(mid_ch))

        # 上采样路径（out_ch 决定哪级，头数对应本级）
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(block_out_channels))
        rev_layers = list(reversed(layers_per_block))
        rev_heads = list(reversed([_n_heads(i) for i in range(n_levels)]))
        for i in range(n_levels - 1):
            in_ch = rev_channels[i]
            out_ch = rev_channels[i + 1]
            n_layers = rev_layers[i + 1]
            is_last = (i == n_levels - 2)
            use_cross = (self.cross_dim > 0) and (i < n_levels - 2)

            if use_cross:
                block = CrossAttnUpBlock2D(
                    in_ch, out_ch, n_layers, time_emb_dim, self.cross_dim, rev_heads[i],
                    add_up=not is_last,
                )
            else:
                block = UpBlock2D(
                    in_ch, out_ch, n_layers, time_emb_dim,
                    add_up=not is_last,
                )
            self.up_blocks.append(block)
            self.out_blocks.append(self._make_out_head(out_ch))

    def _make_out_head(self, ch):
        return nn.Sequential(
            nn.GroupNorm(32, ch, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(ch, 1, 3, padding=1),
        )

    def forward(self, sample, timestep, encoder_hidden_states=None, vec=None):
        """
        Args:
            sample: [B, 16, H, W] Flux latent
            timestep: [B] float in [0, 1]
            encoder_hidden_states: [B, seq, 4096] T5 embeddings or None
            vec: [B, 768] CLIP pooled embedding or None
        Returns:
            list of [B, 1, h, w] logit maps at different scales
        """
        # 0) 统一 cast 输入到 float（模型是 float32）
        sample = sample.float()
        timestep = timestep.float()
        context = encoder_hidden_states.float() if encoder_hidden_states is not None else None
        vec = vec.float() if vec is not None else None

        # 1) 文本压缩
        if context is not None and self.hidden_compress is not None:
            context = self.hidden_compress(context)

        # 2) 时间嵌入 + CLIP vec 注入
        time_emb = self.time_embed(timestep)  # [B, time_emb_dim]
        if vec is not None and self.vec_proj is not None:
            time_emb = time_emb + self.vec_proj(vec)

        # 2) 预处理
        inputs = sample
        sample = self.conv_in(sample)
        inputs_feature = self.feature_in(inputs)  # [B, norm_num_groups, H, W]

        # 3) 下采样 + 记录跳跃连接
        skips = []
        for block in self.down_blocks:
            inputs_down = F.interpolate(inputs_feature, size=sample.shape[-2:], mode="bilinear", align_corners=False)
            x_in = torch.cat([sample, inputs_down], dim=1)
            if isinstance(block, CrossAttnDownBlock2D):
                sample, res = block(x_in, time_emb, context)
            else:
                sample, res = block(x_in, time_emb)
            skips.append(res[0])

        # 4) 中间块
        if self.mid_block is not None:
            sample = self.mid_block(sample, time_emb, context)

        # 5) 第 0 个输出头（最粗尺度）
        out = [self.out_blocks[0](sample)]

        # 6) 上采样 + 多尺度输出
        for i, block in enumerate(self.up_blocks):
            if isinstance(block, CrossAttnUpBlock2D):
                sample = block(sample, time_emb, context)
            else:
                sample = block(sample, time_emb)
            out.append(self.out_blocks[i + 1](sample))

        return out


# ═══════════════════════════════════════════════
# GAN 损失函数
# ═══════════════════════════════════════════════

def hinge_d_loss(logits_real, logits_fake):
    """判别器 Hinge 损失。支持多尺度 logits 列表。"""
    if not isinstance(logits_real, list):
        logits_real, logits_fake = [logits_real], [logits_fake]

    loss = 0.0
    for lr, lf in zip(logits_real, logits_fake):
        loss_real = F.relu(1.0 - lr)
        loss_fake = F.relu(1.0 + lf)
        loss = loss + (loss_real + loss_fake).mean(dim=list(range(1, lr.ndim))).mean()

    return loss / len(logits_real)


def gen_loss(logits_fake):
    """生成器非饱和 GAN 损失。支持多尺度 logits 列表。"""
    if not isinstance(logits_fake, list):
        logits_fake = [logits_fake]

    loss = 0.0
    for lf in logits_fake:
        loss = loss - lf.mean(dim=list(range(1, lf.ndim))).mean()

    return loss / len(logits_fake)
