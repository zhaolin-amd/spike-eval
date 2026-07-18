"""GPU selection: pick the least-used GPU(s) via nvidia-smi so runs land on free cards.

Pure-ish: `_parse_free` is offline-testable against captured nvidia-smi output; `pick_gpus`
shells out. No hard dependency on a GPU being present (returns [] then).
"""
from __future__ import annotations

import subprocess


def _parse_free(csv_text: str, *, max_used_mib: int = 2000) -> list[int]:
    """Given `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits`
    output, return indices of GPUs using <= max_used_mib MiB, least-used first."""
    rows: list[tuple[int, int]] = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((int(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue
    free = sorted((idx for idx, used in rows if used <= max_used_mib),
                  key=lambda idx: dict(rows)[idx])
    return free


def pick_gpus(n: int = 1, *, max_used_mib: int = 2000) -> list[int]:
    """Return up to n free GPU indices (least-used first), or [] if nvidia-smi is
    unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return _parse_free(out, max_used_mib=max_used_mib)[:n]


def cuda_visible(n: int = 1) -> str:
    """A CUDA_VISIBLE_DEVICES value for n free GPUs (e.g. "4" or "4,5"); "" if none."""
    return ",".join(str(i) for i in pick_gpus(n))
