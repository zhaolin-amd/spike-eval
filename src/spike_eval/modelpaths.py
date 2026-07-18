"""Model-path policy: read pretrained snapshots from a shared cache first, and download
anything missing to a writable scratch dir (never the shared cache). Adapted from
paper_reprise.modelpaths (env prefix SPIKE_EVAL_).

The shared cache is laid out as `<MODEL_BASE>/<org>/<model>` snapshot dirs (config.json +
weights), NOT the HF hub `models--org--name` layout — so a bare id is mapped to its local
snapshot path when present, else returned unchanged (repo downloads it, into scratch).
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_MODEL_BASE = "/group/amdneuralopt/huggingface/pretrained_models"


def model_base() -> Path:
    """Shared read-mostly pretrained-model cache root (env SPIKE_EVAL_MODEL_BASE)."""
    return Path(os.environ.get("SPIKE_EVAL_MODEL_BASE", _DEFAULT_MODEL_BASE))


def download_dir() -> Path:
    """Scratch dir missing models download into (env SPIKE_EVAL_DOWNLOAD_DIR, else
    /scratch/$USER/pretrained_models)."""
    env = os.environ.get("SPIKE_EVAL_DOWNLOAD_DIR")
    if env:
        return Path(env)
    user = os.environ.get("USER") or "shared"
    return Path(f"/scratch/{user}/pretrained_models")


def resolve_model(model_id: str) -> str:
    """Map a model id to its local snapshot path under the shared cache when present,
    else return it unchanged. Absolute paths / `..` segments are returned verbatim; a
    snapshot counts only if `<base>/<id>/config.json` exists."""
    if not model_id:
        return model_id
    if model_id.startswith("/") or ".." in Path(model_id).parts:
        return model_id
    candidate = model_base() / model_id
    if (candidate / "config.json").is_file():
        return str(candidate)
    return model_id


def hf_env_overlay() -> dict:
    """Env vars so HF downloads land in scratch (created here), not $HOME or the
    read-only shared cache. HF_HOME only defaulted if unset."""
    dl = download_dir()
    dl.mkdir(parents=True, exist_ok=True)
    overlay: dict = {"HF_HUB_CACHE": str(dl)}
    if not os.environ.get("HF_HOME"):
        overlay["HF_HOME"] = str(dl.parent / "cache" / "huggingface")
    return overlay
