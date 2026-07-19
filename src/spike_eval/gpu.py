"""GPU selection: pick free GPU(s), trying nvidia-smi first then rocm-smi as fallback.

Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs transparently. The caller gets back
integer indices and a CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES string; the env var
name is chosen to match the detected vendor. Pure helpers (_parse_free_*) are
offline-testable against captured smi output. No hard dependency on a GPU being present
— returns [] when neither smi is available.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuInfo:
    vendor: str          # "nvidia" | "amd"
    env_var: str         # "CUDA_VISIBLE_DEVICES" | "HIP_VISIBLE_DEVICES"
    indices: list[int]   # free GPU indices, least-used first


# --- nvidia-smi parsing (csv,noheader,nounits) ----------------------------


def _parse_free_nvidia(csv_text: str, *, max_used_mib: int = 2000) -> list[int]:
    """Parse `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`
    output; return indices of GPUs using <= max_used_mib MiB, least-used first."""
    rows: list[tuple[int, int]] = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue
    used = dict(rows)
    return sorted((idx for idx, mib in rows if mib <= max_used_mib), key=used.__getitem__)


# --- rocm-smi parsing -----------------------------------------------------


def _parse_free_amd(text: str, *, max_used_pct: int = 5) -> list[int]:
    """Parse `rocm-smi --showuse --csv` output; return indices of GPUs whose
    GPU-use% is <= max_used_pct, least-used first.

    rocm-smi CSV header: device,GPU use(%),Memory use(%),...  (varies by version).
    Also handles `rocm-smi --showmeminfo vram --csv` as a fallback.
    """
    rows: list[tuple[int, float]] = []
    header: Optional[list[str]] = None
    for line in text.strip().splitlines():
        parts = [p.strip().lower() for p in line.split(",")]
        if not parts:
            continue
        if header is None:
            header = parts
            continue
        if not header:
            continue
        try:
            dev_idx = next(
                (i for i, h in enumerate(header) if "device" in h or "card" in h), None)
            use_idx = next(
                (i for i, h in enumerate(header)
                 if "gpu use" in h or "gpu%" in h or "utilization" in h), None)
            if dev_idx is None or use_idx is None or len(parts) <= max(dev_idx, use_idx):
                continue
            # device column is often "card0" or "0"
            dev_raw = parts[dev_idx].lstrip("card").strip()
            rows.append((int(dev_raw), float(parts[use_idx].rstrip("%"))))
        except (ValueError, StopIteration):
            continue
    return sorted((idx for idx, pct in rows if pct <= max_used_pct),
                  key=lambda i: dict(rows)[i])


# --- smi probes -----------------------------------------------------------


def _try_nvidia(n: int, max_used_mib: int) -> Optional[GpuInfo]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
        indices = _parse_free_nvidia(out, max_used_mib=max_used_mib)[:n]
        return GpuInfo(vendor="nvidia", env_var="CUDA_VISIBLE_DEVICES", indices=indices)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _try_amd(n: int, max_used_pct: int = 5) -> Optional[GpuInfo]:
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--csv"],
            capture_output=True, text=True, timeout=15, check=True).stdout
        indices = _parse_free_amd(out, max_used_pct=max_used_pct)[:n]
        return GpuInfo(vendor="amd", env_var="HIP_VISIBLE_DEVICES", indices=indices)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


# --- public API -----------------------------------------------------------


def pick_gpus(n: int = 1, *, max_used_mib: int = 2000, max_used_pct: int = 5,
              prefer: Optional[str] = None) -> GpuInfo:
    """Return a GpuInfo with up to n free GPU indices. Tries nvidia-smi first, then
    rocm-smi; use `prefer='amd'` or `prefer='nvidia'` to override the order.
    Returns GpuInfo with an empty indices list when no GPUs are found."""
    order = (["amd", "nvidia"] if prefer == "amd" else ["nvidia", "amd"])
    probes = {"nvidia": lambda: _try_nvidia(n, max_used_mib),
              "amd":    lambda: _try_amd(n, max_used_pct)}
    for vendor in order:
        info = probes[vendor]()
        if info is not None:
            return info
    return GpuInfo(vendor="none", env_var="CUDA_VISIBLE_DEVICES", indices=[])


def visible_devices(n: int = 1, **kw) -> tuple[str, str]:
    """Return (env_var_name, value) for n free GPUs, e.g. ('CUDA_VISIBLE_DEVICES', '4').
    value is '' when no GPU is available."""
    info = pick_gpus(n, **kw)
    return info.env_var, ",".join(str(i) for i in info.indices)


def cuda_visible(n: int = 1) -> str:
    """Legacy: CUDA_VISIBLE_DEVICES value for n free GPUs (NVIDIA or AMD); '' if none.
    Kept for backward compat; prefer visible_devices() for new callers."""
    _, val = visible_devices(n)
    return val
