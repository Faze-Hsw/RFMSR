import glob
import os

from torch import Tensor, nn
from transformers import (
    AutoConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)
from safetensors.torch import load_file as load_sft


class HFEmbedder(nn.Module):
    def __init__(self, version: str, max_length: int, ckpt_path: str | None = None, **hf_kwargs):
        super().__init__()
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # 1) Tokenizer（小文件，自动缓存/下载）
        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)

        # 2) 只加载架构 config（小文件），不下载大权重
        config = AutoConfig.from_pretrained(version)
        if self.is_clip:
            self.hf_module: CLIPTextModel = CLIPTextModel._from_config(config)
        else:
            self.hf_module: T5EncoderModel = T5EncoderModel._from_config(config)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

        # 3) 从本地 safetensors 覆盖权重（核心：大权重不走网络）
        if ckpt_path is not None and os.path.exists(ckpt_path):
            self._load_local_weights(ckpt_path)

    def _load_local_weights(self, ckpt_path: str):
        """从本地 safetensors 文件加载权重（支持单文件或分片目录）"""
        if os.path.isdir(ckpt_path):
            files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
            if not files:
                print(f"  ⚠️  未在 {ckpt_path} 中找到 safetensors 文件")
                return
            state_dict = {}
            for f in files:
                state_dict.update(load_sft(f, device="cpu"))
        else:
            state_dict = load_sft(ckpt_path, device="cpu")

        missing, unexpected = self.hf_module.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  ⚠️  本地权重加载缺少 {len(missing)} 个 key（部分未用）")
        if unexpected:
            print(f"  ⚠️  本地权重有 {len(unexpected)} 个额外 key（已忽略）")
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
