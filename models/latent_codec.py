"""Unified VAE latent codec.

Normalizes the two latent conventions into one "normalized latent" space:

    sd2   scalar scaling_factor, 4 channels, weights under <path>/vae
    qwen  per-channel latents_mean / latents_std, 16 channels, <path> is the VAE root

    encode(img)     pixels [-1, 1]     -> normalized latent
    decode(latent)  normalized latent  -> pixels [-1, 1] (not clamped)

Use denormalize / decode_vae when the two halves are needed separately
(e.g. gradient-checkpointed decoding).
"""

import torch

_AE_TYPES = ("sd2", "qwen")


class LatentCodec:
    """Single interface over the supported VAEs."""

    def __init__(self, ae_type: str = "sd2", path: str = None,
                 device: "str | torch.device" = "cuda", subfolder: str = "vae"):
        if ae_type not in _AE_TYPES:
            raise ValueError(f"Unknown ae_type={ae_type!r}, expected one of {_AE_TYPES}")
        if not path:
            raise ValueError("LatentCodec requires a VAE weight path")

        self.ae_type = ae_type
        self.path = path
        self.device = torch.device(device) if isinstance(device, str) else device

        if ae_type == "qwen":
            from models.qwenimage_vae2d import AutoencoderKLQwenImage2D

            self.vae = AutoencoderKLQwenImage2D.from_pretrained(path)
        else:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(path, subfolder=subfolder)

        self.vae = self.vae.to(self.device).eval()
        self.vae.requires_grad_(False)

        if ae_type == "qwen":
            self._mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1)
            self._std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1)
            self.latent_channels = int(self._mean.numel())
        else:
            self._scale = float(self.vae.config.scaling_factor)
            self.latent_channels = int(self.vae.config.latent_channels)

    def __repr__(self) -> str:
        return (
            f"LatentCodec(type={self.ae_type}, channels={self.latent_channels}, "
            f"path={self.path})"
        )

    # ---- latent space conversion ----

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        if self.ae_type == "qwen":
            return (z - self._mean.to(z.device, z.dtype)) / self._std.to(z.device, z.dtype)
        return z * self._scale

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if self.ae_type == "qwen":
            return z * self._std.to(z.device, z.dtype) + self._mean.to(z.device, z.dtype)
        return z / self._scale

    # ---- raw VAE calls ----

    def encode_vae(self, img: torch.Tensor) -> torch.Tensor:
        """Pixels [-1, 1] -> sampled VAE latent (not normalized)."""
        return self.vae.encode(img).latent_dist.sample()

    def decode_vae(self, z: torch.Tensor) -> torch.Tensor:
        """VAE latent -> pixels [-1, 1] (qwen clamps internally, sd2 does not)."""
        out = self.vae.decode(z)
        return out.sample if hasattr(out, "sample") else out[0]

    # ---- public API ----

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.encode_vae(img))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode_vae(self.denormalize(latent))


def create_latent_codec(ae_cfg: dict, device: "str | torch.device" = "cuda") -> LatentCodec:
    """Build a codec from the `ae` section of a backbone config."""
    ae_cfg = ae_cfg or {}
    return LatentCodec(
        ae_type=ae_cfg.get("type", "sd2"),
        path=ae_cfg.get("path"),
        device=device,
        subfolder=ae_cfg.get("subfolder") or "vae",
    )
