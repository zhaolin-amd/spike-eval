"""Offline tests for gpu.py NVIDIA + AMD parsing (no smi subprocess)."""
from spike_eval.gpu import GpuInfo, _parse_free_amd, _parse_free_nvidia, pick_gpus

# --- nvidia-smi CSV (index, memory.used MiB) ---
NVIDIA_CSV = """0, 41699
1, 0
2, 46055
3, 0
4, 0
"""


def test_nvidia_parse_free():
    free = _parse_free_nvidia(NVIDIA_CSV, max_used_mib=2000)
    assert 1 in free and 3 in free and 4 in free
    assert 0 not in free and 2 not in free


def test_nvidia_parse_least_used_first():
    free = _parse_free_nvidia(NVIDIA_CSV, max_used_mib=2000)
    # All three free have 0 MiB; order should be by index (stable)
    assert free == sorted(free)


def test_nvidia_parse_empty():
    assert _parse_free_nvidia("") == []


# --- rocm-smi --showuse --csv ---
ROCM_CSV = """device,GPU use(%),Memory use(%)
card0,0,12
card1,95,80
card2,3,15
card3,0,5
"""

ROCM_CSV_ALT = """device,GPU%,MemoryUse%
0,0,10
1,100,90
2,1,5
"""


def test_amd_parse_free():
    free = _parse_free_amd(ROCM_CSV, max_used_pct=5)
    assert 0 in free and 2 in free and 3 in free
    assert 1 not in free


def test_amd_parse_alt_header():
    free = _parse_free_amd(ROCM_CSV_ALT, max_used_pct=5)
    assert 0 in free and 2 in free
    assert 1 not in free


def test_amd_parse_empty():
    assert _parse_free_amd("") == []


# --- pick_gpus fallback behaviour (monkeypatched) ---

def test_pick_gpus_returns_amd_when_nvidia_absent(monkeypatch):
    monkeypatch.setattr("spike_eval.gpu._try_nvidia", lambda n, mib: None)
    monkeypatch.setattr(
        "spike_eval.gpu._try_amd",
        lambda n, pct: GpuInfo(vendor="amd", env_var="HIP_VISIBLE_DEVICES", indices=[0, 2]),
    )
    info = pick_gpus(1)
    assert info.vendor == "amd"
    assert info.env_var == "HIP_VISIBLE_DEVICES"
    assert info.indices == [0, 2]


def test_pick_gpus_prefers_nvidia_by_default(monkeypatch):
    monkeypatch.setattr(
        "spike_eval.gpu._try_nvidia",
        lambda n, mib: GpuInfo(vendor="nvidia", env_var="CUDA_VISIBLE_DEVICES",
                               indices=[4]),
    )
    monkeypatch.setattr(
        "spike_eval.gpu._try_amd",
        lambda n, pct: GpuInfo(vendor="amd", env_var="HIP_VISIBLE_DEVICES", indices=[0]),
    )
    info = pick_gpus(1)
    assert info.vendor == "nvidia"


def test_pick_gpus_empty_when_none(monkeypatch):
    monkeypatch.setattr("spike_eval.gpu._try_nvidia", lambda n, mib: None)
    monkeypatch.setattr("spike_eval.gpu._try_amd", lambda n, pct: None)
    info = pick_gpus(1)
    assert info.indices == []
    assert info.vendor == "none"
