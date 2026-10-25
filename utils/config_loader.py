"""Experiment config loading: merge backbone spec, resolve paths, validate.

Layering:
    configs/backbones/<name>.yaml   architecture + VAE + DINOv2, replace as a whole
    configs/train_rfmsr*.yaml       data / optimizer / loss / logging, references a backbone

The returned dict is flat: `model_config` points at the resolved backbone file, `ae`
holds the VAE spec, and `vae_path` follows `ae.path` unless explicitly overridden.
Callers branch on `ae.type` and channel counts, never on file names or paths.
"""

import os

import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_AE_TYPES = ("sd2", "qwen")

_DINOV2_DIMS = {"dinov2s": 384, "dinov2b": 768, "dinov2l": 1024, "dinov2g": 1536}


def project_root() -> str:
    return _PROJECT_ROOT


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path, root=None):
    """Resolve a relative path against root, then cwd, then the project root."""
    if not path:
        return path
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    for base in (root, os.getcwd(), _PROJECT_ROOT):
        if not base:
            continue
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(_PROJECT_ROOT, path))


def _merge(base: dict, over: dict) -> dict:
    """Deep merge with `over` taking precedence."""
    out = dict(base or {})
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def validate_config(cfg: dict) -> dict:
    """Fail early on half-edited configs: channel counts and DINOv2 width must agree."""
    ae = cfg.get("ae") or {}
    arch = cfg.get("dit_arch") or {}

    ae_type = ae.get("type")
    if ae_type is not None and ae_type not in _AE_TYPES:
        raise ValueError(f"ae.type={ae_type!r} is invalid, expected one of {_AE_TYPES}")

    channels = ae.get("latent_channels")
    if channels and arch:
        out_ch = arch.get("out_channels")
        in_ch = arch.get("in_channels")
        if out_ch is not None and int(out_ch) != int(channels):
            raise ValueError(
                f"dit_arch.out_channels={out_ch} does not match ae.latent_channels={channels}"
            )
        if in_ch is not None and int(in_ch) != 2 * int(channels):
            raise ValueError(
                f"dit_arch.in_channels={in_ch} should be 2 x ae.latent_channels={channels}"
            )

    dv2 = cfg.get("dinov2") or {}
    enc_type, enc_dim = dv2.get("enc_type"), dv2.get("enc_dim")
    expected = _DINOV2_DIMS.get(enc_type)
    if expected is not None and enc_dim is not None and int(enc_dim) != expected:
        raise ValueError(
            f"dinov2.enc_dim={enc_dim} does not match enc_type={enc_type} (expected {expected})"
        )

    return cfg


def load_config(path, root=None, backbone=None, verbose: bool = False) -> dict:
    """Load an experiment config and merge the backbone file it references.

    Args:
        path: experiment config (e.g. configs/train_rfmsr.yaml) or a backbone file.
        root: base directory for relative paths, usually the script directory.
        backbone: backbone file that overrides the config's `backbone:` field
            (the --backbone flag of the training / inference scripts).
    """
    cfg_path = resolve_path(path, root)
    if not os.path.exists(cfg_path):
        print(f"[config] config not found, using built-in defaults: {cfg_path}")
        return {}

    cfg = _read_yaml(cfg_path)
    if backbone:
        cfg["backbone"] = backbone
    backbone_ref = cfg.pop("backbone", None)

    if backbone_ref:
        backbone_path = resolve_path(backbone_ref, root)
        spec = _read_yaml(backbone_path)
        cfg = _merge(spec, cfg)          # experiment overrides, backbone supplies the rest
        cfg["model_config"] = backbone_path
        cfg["backbone_path"] = backbone_path
        cfg["backbone_name"] = spec.get("name") or os.path.splitext(
            os.path.basename(backbone_path)
        )[0]
    elif cfg.get("model_config"):
        # also accept a plain model_config entry (e.g. set from the CLI)
        backbone_path = resolve_path(cfg["model_config"], root)
        cfg["model_config"] = backbone_path
        if os.path.exists(backbone_path):
            spec = _read_yaml(backbone_path)
            cfg = _merge(spec, cfg)
            cfg["backbone_path"] = backbone_path
            cfg["backbone_name"] = spec.get("name") or os.path.splitext(
                os.path.basename(backbone_path)
            )[0]

    ae = dict(cfg.get("ae") or {})
    if ae.get("path"):
        ae["path"] = resolve_path(ae["path"], root)
    if ae:
        cfg["ae"] = ae

    # vae_path is a legacy field; it mirrors ae.path unless set explicitly
    if cfg.get("vae_path"):
        cfg["vae_path"] = resolve_path(cfg["vae_path"], root)
    else:
        cfg["vae_path"] = ae.get("path")

    validate_config(cfg)

    if verbose:
        print(
            f"[config] {os.path.basename(cfg_path)} -> backbone="
            f"{cfg.get('backbone_name')} ae={ae.get('type')} "
            f"channels={ae.get('latent_channels')}"
        )
    return cfg
