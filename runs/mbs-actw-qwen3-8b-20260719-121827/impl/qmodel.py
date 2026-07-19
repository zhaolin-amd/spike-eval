"""Wrap a HuggingFace causal-LM's linear layers with MXFP4 fake-quant.

Direct-cast, calibration-free (paper Setups): every linear layer's WEIGHT is
quantized once at load time, and its input ACTIVATION is quantized on every forward
pass. Quantizes all QKVO + FFN projections (the transformer's Linear layers);
leaves the embedding and the lm_head in full precision, matching the paper's
"we quantize all linear layers" for the compute-bound GEMMs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

from mxquant import METHODS, fake_quant

# Quark path (for MXFP4-Quark[-Even] which delegate to Quark's own kernel).
# Override with env QUARK_ROOT on nodes where the checkout lives elsewhere.
_QUARK_ROOT = os.environ.get("QUARK_ROOT", "/home/zhaolin/code/Quark")
if _QUARK_ROOT not in sys.path:
    sys.path.insert(0, _QUARK_ROOT)


def _load_quark_triton_qdq():
    """Load Quark's qdq_mxfp4_triton directly from the triton.py file, bypassing
    mx/__init__.py which unconditionally imports the HIP extension (failing on this node
    because the JIT-compiled hw_emulation kernel is missing). The triton file itself
    has no hw_emulation dependency and works fine on NVIDIA GPUs."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_quark_mx_triton",
        os.path.join(_QUARK_ROOT, "quark/torch/kernel/mx/triton.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.qdq_mxfp4_triton


class QuarkQuantLinear(nn.Module):
    """Drop-in for nn.Linear using Quark's qdq_mxfp4_triton (block_size=32).

    FP4 element rounding is the same as our MXFP4-OCP (ties-up). The difference is
    in the E8M0 **scale calculation**:
      - Our MXFP4-OCP: SF = 2^floor(log2(8/amax)) → maps amax to (4,8], allows overflow
      - Quark 'even':  SF = 2^floor(log2(round_even(amax)) - max_exp) → amax first
        rounded to nearest-even before the floor, giving a slightly different scale."""

    _qdq = None   # class-level cache — loaded once on first instantiation

    def __init__(self, lin: nn.Linear):
        super().__init__()
        if QuarkQuantLinear._qdq is None:
            QuarkQuantLinear._qdq = _load_quark_triton_qdq()
        qdq = QuarkQuantLinear._qdq
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        with torch.no_grad():
            wq = qdq(lin.weight.data.to(torch.bfloat16),
                     scale_calculation_mode="even").to(lin.weight.dtype)
        self.weight = nn.Parameter(wq, requires_grad=False)
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdq = QuarkQuantLinear._qdq
        xq = qdq(x.to(torch.bfloat16), scale_calculation_mode="even").to(x.dtype)
        return F.linear(xq, self.weight, self.bias)


class QuantLinear(nn.Module):
    """A drop-in for nn.Linear: pre-quantized weight + per-forward activation quant."""

    def __init__(self, lin: nn.Linear, weight_mbs: str, act_mbs: str,
                 oas: bool, ocp: bool = False, ocp_block: int = 32,
                 oas_block: int = 16, inner: str = "oas", macro_block: int = 128,
                 mbs_bits: int | None = None, act_amax: torch.Tensor | None = None):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.act_mbs = act_mbs
        self.oas = oas
        self.ocp = ocp
        self.ocp_block = ocp_block
        self.oas_block = oas_block
        self.inner = inner
        self.macro_block = macro_block
        self.mbs_bits = mbs_bits
        # SpikeEval idea: when act_amax (per-input-channel amax, shape (in_features,)) is
        # given AND the weight uses the dynamic search, weight the search SSE by amax^2
        # (activation-aware MBS). None -> paper's plain-SSE search. Activations are never
        # reweighted (they use static MBS).
        chan_w = None
        if act_amax is not None and weight_mbs == "dynamic":
            chan_w = act_amax.to(torch.float32) ** 2
        with torch.no_grad():
            wq = fake_quant(lin.weight.data.to(torch.float32),
                            mbs=weight_mbs, oas=oas, ocp=ocp,
                            ocp_block=ocp_block,
                            oas_block=oas_block, inner=inner,
                            macro_block=macro_block, mbs_bits=mbs_bits,
                            chan_w=chan_w).to(lin.weight.dtype)
        self.weight = nn.Parameter(wq, requires_grad=False)
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant(x, mbs=self.act_mbs, oas=self.oas, ocp=self.ocp,
                        ocp_block=self.ocp_block, oas_block=self.oas_block,
                        inner=self.inner, macro_block=self.macro_block,
                        mbs_bits=self.mbs_bits)
        return F.linear(xq, self.weight, self.bias)


def _quantizable(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    # Skip the output head (not one of the paper's compute-bound body GEMMs).
    if name.split(".")[-1] == "lm_head" or name == "lm_head":
        return False
    # Only quantize when both dims allow exact 1x128 / 1x16 blocking.
    return module.in_features % 128 == 0


def quantize_model_(model: nn.Module, method: str,
                    act_amax_map: dict | None = None) -> int:
    """In-place: replace eligible nn.Linear with the appropriate quantized layer. Returns count.

    act_amax_map: optional {layer_name: per-input-channel amax tensor} for the SpikeEval
    activation-aware idea (MXFP4-MBS-H-ACTW). When given, each QuantLinear's dynamic weight
    search is weighted by that layer's amax^2. None -> unchanged paper behavior.
    """
    targets = [(n, m) for n, m in model.named_modules() if _quantizable(n, m)]
    if method == "MXFP4-Quark":
        # Delegate entirely to Quark's qdq_mxfp4 (block_size=32, even rounding).
        for name, lin in targets:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            child = name.rsplit(".", 1)[-1]
            ql = QuarkQuantLinear(lin).to(next(lin.parameters()).device)
            setattr(parent, child, ql)
    else:
        cfg = METHODS[method]
        for name, lin in targets:
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            child = name.rsplit(".", 1)[-1]
            aa = act_amax_map.get(name) if act_amax_map else None
            ql = QuantLinear(lin, weight_mbs=cfg["weight_mbs"],
                             act_mbs=cfg["act_mbs"], oas=cfg["oas"],
                             ocp=cfg.get("ocp", False),
                             ocp_block=cfg.get("ocp_block", 32),
                             oas_block=cfg.get("oas_block", 16),
                             inner=cfg.get("inner", "oas"),
                             macro_block=cfg.get("macro_block", 128),
                             mbs_bits=cfg.get("mbs_bits", None),
                             act_amax=aa)
            ql = ql.to(next(lin.parameters()).device)
            setattr(parent, child, ql)
    return len(targets)
