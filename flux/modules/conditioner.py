import glob
import os

import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)
from safetensors.torch import load_file as load_sft


def _from_pretrained_local_first(cls, *args, **kwargs):
    """同官方 from_pretrained，但跳过模型权重下载（只下小文件）。"""
    version = args[0] if args else kwargs.get("pretrained_model_name_or_path", "")
    from huggingface_hub import snapshot_download
    snapshot_download(
        version,
        allow_patterns=["tokenizer*", "config*", "spiece*", "special_tokens*",
                        "vocab*", "merges*", "added_tokens*", "*.json"],
        ignore_patterns=["*.bin", "*.safetensors", "*.h5", "model*"],
    )
    return cls.from_pretrained(*args, **kwargs, local_files_only=True)


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, ckpt_path: str | None = None,
                 device: str | torch.device = "cuda", **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # 1) Tokenizer — 优先本地缓存，没有再联网下载
        tok_cls = CLIPTokenizer if self.is_clip else T5Tokenizer
        self.tokenizer = _from_pretrained_local_first(tok_cls, version, max_length=max_length)

        # 2) Config — 同上
        config = _from_pretrained_local_first(AutoConfig, version)

        # ⭐ 优化：在 meta device 上创建模型，避免 ~44GB CPU float32 随机初始化
        with torch.device("meta"):
            if self.is_clip:
                self.hf_module: CLIPTextModel = CLIPTextModel._from_config(config.text_config)
            else:
                self.hf_module: T5EncoderModel = T5EncoderModel._from_config(config)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

        # 3) 从本地 safetensors 覆盖权重（直接加载到目标设备）
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_local_weights(ckpt_path, device)

    def _load_local_weights(self, ckpt_path: str, device: str = "cpu"):
        """从本地 safetensors 文件加载权重（支持单文件或分片目录）"""
        if os.path.isdir(ckpt_path):
            files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
            if not files:
                print(f"  ⚠️  未在 {ckpt_path} 中找到 safetensors 文件")
                return
            state_dict = {}
            for f in files:
                state_dict.update(load_sft(f, device=device))
        else:
            state_dict = load_sft(ckpt_path, device=device)

        # 本地 safetensors 的 key 可能带 text_model. 前缀，统一去掉
        if self.is_clip:
            state_dict = {k.removeprefix("text_model."): v for k, v in state_dict.items()}

        # ⭐ assign=True：直接以 state_dict 的 tensor 作为模型参数，跳过额外拷贝
        missing, unexpected = self.hf_module.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            print(f"  ⚠️  本地权重加载缺少 {len(missing)} 个 key（部分未用）")
            for k in missing[:10]:
                print(f"     缺少: {k}")
        if unexpected:
            print(f"  ⚠️  本地权重有 {len(unexpected)} 个额外 key（已忽略）")
            for k in unexpected[:10]:
                print(f"     多余: {k}")
        print(f"  ✅ 从本地 safetensors 加载权重完成: {ckpt_path}")

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()
