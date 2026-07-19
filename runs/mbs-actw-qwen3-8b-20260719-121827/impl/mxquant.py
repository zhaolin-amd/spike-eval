"""Fake-quant core for the enhanced MXFP4 formats of arXiv 2603.08713v1.

Implements, as numerically-faithful quantize->dequantize (no custom kernel needed;
the paper's CUTLASS/SM100 kernels only affect SPEED, not the numbers):

  * FP4 (E2M1) round-to-nearest with clamp at Fmax = 6.0
  * MXFP4-OCP (baseline): block_size=32, E8M0 OCP scale (maps amax to (4,8])
  * MXFP4-16: the SAME MX/OCP scale (maps amax to (4,8], allows overflow) but at
    block size 16 -- the paper's "MX-style scaling at block size 16" (paper 4.1
    "Quantization Block Granularity"), realized via the NVFP4 pipeline with the
    scales constrained to powers of two.
  * Overflow-Aware Scaling (OAS): map block absmax to (3.5, 7] instead of (3, 6]
    (paper 4.2)

  PITFALL (do not repeat): the non-saturating (3, 6] scale (SF = 2^floor(log2(6/amax)))
  is an INGREDIENT OF OAS (paper 4.2), NOT of plain MXFP4-16. MXFP4-16 must use the
  (4,8] overflow scale, same as OCP. If MXFP4-16 is (wrongly) given the (3,6] scale it
  already realizes most of OAS's benefit, so it reproduces the paper's *OAS* numbers
  (e.g. Qwen3-8B wikitext ppl ~13.65) instead of its own (~15.15), and OAS then appears
  to add almost nothing.
  * Macro Block Scaling (MBS), 1x128 macro block, 8-bit mantissa refinement factor
    (paper 4.3):
      - Static  (MBS-S): factor from top-8 mantissa bits of 6/absmax128 (eq. 1)
      - Dynamic (MBS-D): LUT/search over 16 mantissa slots minimizing macro-block SSE

Everything operates on the LAST dim (the contraction / K dim). For a weight (N, K)
that is the input-feature dim; for an activation (T, K) it is the hidden dim. All
Qwen3-8B linear in-features are multiples of 128, so blocking is exact (no padding).

The numbers in the paper (Table 2) are NOT read here; correctness is pinned by
independent closed-form checks in test_mxquant.py.
"""
from __future__ import annotations

import torch

FP4_MAX = 6.0
# Positive FP4 (E2M1) representable magnitudes.
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# Midpoints between consecutive levels -> round-to-nearest via bucketize.
_FP4_MIDS = ((_FP4_LEVELS[1:] + _FP4_LEVELS[:-1]) / 2.0)  # 7 boundaries

_TINY = 1e-30


def quant_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round each element to the nearest FP4 (E2M1) value, clamped to [-6, 6].

    Nearest with ties rounding up (bucketize on midpoints); ties are vanishingly
    rare for real activations/weights and have no measurable downstream effect.
    """
    mids = _FP4_MIDS.to(v.device, v.dtype)
    levels = _FP4_LEVELS.to(v.device, v.dtype)
    sign = torch.sign(v)
    a = v.abs()
    idx = torch.bucketize(a, mids, right=True)  # 0..7
    return sign * levels[idx]


def _pow2_scale_oas(amax: torch.Tensor, oas: bool) -> torch.Tensor:
    """E8M0 power-of-two scale SF for a block with the given absmax.

    Standard: SF = 2^floor(log2(6/amax)) so amax*SF in (3, 6] (masking the mantissa
    bits of 6/amax == truncating to its power of two). OAS: if amax*SF <= 3.5, double
    SF (one more power of two), mapping amax to (6, 7] -> overall (3.5, 7].

    Computed via frexp so the exponent is exact (no log2 rounding hazard at powers
    of two). Returns SF with the same shape as amax; zero-absmax blocks get SF = 1.
    """
    amax = amax.clamp_min(_TINY)
    target = FP4_MAX / amax
    # frexp: target = mant * 2^exp, mant in [0.5, 1)  ->  floor(log2(target)) = exp - 1
    _mant, exp = torch.frexp(target)
    e = (exp - 1).to(torch.float32)
    sf = torch.exp2(e)
    if oas:
        scaled_amax = amax * sf
        e = torch.where(scaled_amax <= 3.5, e + 1.0, e)
        sf = torch.exp2(e)
    return sf


_OCP_REF = 8.0  # OCP maps amax to (4, 8] using 8 as scale reference


def _pow2_scale_ocp(amax: torch.Tensor) -> torch.Tensor:
    """E8M0 power-of-two scale per the OCP MXFP4 spec: SF = 2^floor(log2(8/amax)).

    Maps block absmax to (4, 8]: when amax > 6, overflow occurs (values clamped to ±6).
    The paper notes this is the key difference from the enhanced methods — MXFP4-OCP
    allows ~15% of blocks to saturate (paper §4.2).
    """
    amax = amax.clamp_min(_TINY)
    target = _OCP_REF / amax
    _mant, exp = torch.frexp(target)
    e = (exp - 1).to(torch.float32)
    return torch.exp2(e)


def _quant_blocks(x: torch.Tensor, block_size: int, sf: torch.Tensor) -> torch.Tensor:
    """Generic MXFP4 quantize->dequantize given a pre-computed per-block SF tensor.

    sf shape: (*lead, k//block_size, 1). Returns dequantized x, same shape as x.
    """
    *lead, k = x.shape
    xb = x.reshape(*lead, k // block_size, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    q = quant_fp4(xb * sf)
    deq = torch.where(amax > 0, q / sf, torch.zeros_like(q))
    return deq.reshape(*lead, k)


def _quant_blocks16(x: torch.Tensor, oas: bool, block_size: int = 16) -> torch.Tensor:
    """MXFP4 quantize->dequantize with a 1x{block_size} E8M0(+OAS) scale.

    block_size=16 → MXFP4-16-OAS / MBS (paper default).
    block_size=32 → Quark-OAS (OAS scale at group-size=32)."""
    *lead, k = x.shape
    assert k % block_size == 0, f"last dim {k} not divisible by {block_size}"
    xb = x.reshape(*lead, k // block_size, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    sf = _pow2_scale_oas(amax, oas)
    q = quant_fp4(xb * sf)
    deq = torch.where(amax > 0, q / sf, torch.zeros_like(q))
    return deq.reshape(*lead, k)


def _quant_blocks_ocp(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """MX/OCP-style quant: E8M0 OCP scale (maps amax to (4,8], allows overflow) at the
    given block size. block_size=32 -> MXFP4-OCP; block_size=16 -> MXFP4-16 (paper 4.1,
    "MX-style scaling at block size 16")."""
    *lead, k = x.shape
    assert k % block_size == 0, f"last dim {k} not divisible by {block_size}"
    xb = x.reshape(*lead, k // block_size, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    sf = _pow2_scale_ocp(amax)
    q = quant_fp4(xb * sf)
    deq = torch.where(amax > 0, q / sf, torch.zeros_like(q))
    return deq.reshape(*lead, k)


def _float32_bits(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).contiguous().view(torch.int32)


def _mbs_factor_static(xr: torch.Tensor, mbs_bits: int = 8) -> torch.Tensor:
    """MBS-Static factor (1 + m/2^mbs_bits) per 1x128 macro block (paper eq. 1).

    xr: (..., n_macro, 128). Returns (..., n_macro, 1) in [1, 2).
    Extracts the top `mbs_bits` mantissa bits of the ideal scale 6/amax128 (float32
    mantissa is 23 bits, so mbs_bits<=23). mbs_bits=8 -> paper default (m8/256);
    mbs_bits=16 -> finer factor storage (m16/65536).
    """
    assert 1 <= mbs_bits <= 23, f"mbs_bits must be in [1, 23], got {mbs_bits}"
    amax = xr.abs().amax(dim=-1, keepdim=True)  # (..., n_macro, 1)
    sf_full = FP4_MAX / amax.clamp_min(_TINY)
    bits = _float32_bits(sf_full)
    shift = 23 - mbs_bits                       # mantissa top-bit alignment
    mask = ((1 << mbs_bits) - 1) << shift       # top mbs_bits of the 23-bit mantissa
    m = (bits & mask) >> shift                  # 0 .. 2^mbs_bits - 1
    factor = 1.0 + m.to(torch.float32) / float(1 << mbs_bits)
    return torch.where(amax > 0, factor, torch.ones_like(factor))


_QUARK_EVEN_QDQ = None


def _quant_blocks_quark_even(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Inner MXFP4 quant->dequant using Quark's own `qdq_mxfp4_triton` with
    `scale_calculation_mode="even"` (E8M0 scale with amax rounded to nearest-even
    before the floor; block=32). Drop-in replacement for `_quant_blocks16` when the
    inner scale should be Quark's *even* rounding instead of the paper's OAS.

    `block_size` is accepted for signature parity; Quark's MXFP4 group size is 32."""
    global _QUARK_EVEN_QDQ
    if _QUARK_EVEN_QDQ is None:
        from qmodel import _load_quark_triton_qdq  # lazy; reuses qmodel._QUARK_ROOT
        _QUARK_EVEN_QDQ = _load_quark_triton_qdq()
    out = _QUARK_EVEN_QDQ(x.to(torch.bfloat16), scale_calculation_mode="even")
    return out.to(x.dtype)


def _mbs_factor_dynamic(xr: torch.Tensor, oas: bool, n_slots: int = 16,
                        oas_block: int = 16, inner: str = "oas",
                        mbs_bits: int | None = None,
                        chan_w: torch.Tensor | None = None) -> torch.Tensor:
    """MBS-Dynamic factor per 1x128 macro block: pick the mantissa slot minimizing
    the macro-block sum of squared quantization error (paper 4.3.3).

    xr: (..., n_macro, 128). Returns (..., n_macro, 1) in [1, 2).
    oas_block: the inner OAS block size (16 for MXFP4-16 family, 32 for Quark family).
    inner: "oas" (default, OAS scale) or "quark_even" (Quark's even-rounded scale).

    chan_w: optional per-position weight of shape (n_macro, mb), broadcast over the lead
      (output-row) dims. When given, the search minimizes the *weighted* SSE
      `sum_k chan_w[k]*(deq_k - w_k)^2` instead of the plain SSE. This is the
      activation-aware SpikeEval idea: chan_w = input-channel amax^2, so the slot is
      chosen to protect the weight columns that see the largest activations (output
      error ~ (dW . x)^2). chan_w=None reproduces the paper's plain-SSE search
      bit-exactly (weight is a no-op multiplier of 1).

    Search resolution:
      - mbs_bits is None (default): single pass over `n_slots` candidates
        (1 + j/n_slots), j in [0, n_slots). This is the paper's 16-slot LUT search.
      - mbs_bits set: coarse-to-fine search reaching 2^mbs_bits factor resolution
        (1 + m/2^mbs_bits) in ceil(mbs_bits/8) passes of <=256 candidates each,
        so 16-bit is 2 passes (512 evals) instead of an infeasible 65536.
    j/m=0 -> factor 1.0 (== no MBS), so the result never increases error vs inner-only.
    """
    lead = xr.shape[:-1]
    mb = xr.shape[-1]  # macro-block size (128 by default, configurable e.g. 64)

    def _sse(c):
        # c: scalar or (*lead, 1) tensor of per-macro-block factors
        xs = (xr * c).reshape(*xr.shape[:-2], -1)
        if inner == "quark_even":
            q = _quant_blocks_quark_even(xs).reshape(*lead, mb)
        else:
            q = _quant_blocks16(xs, oas, oas_block).reshape(*lead, mb)
        deq = q / c
        sq = (deq - xr) ** 2
        if chan_w is not None:
            sq = sq * chan_w
        return sq.sum(dim=-1, keepdim=True)

    if mbs_bits is None:
        best_sse = xr.new_full((*lead, 1), float("inf"))
        best_factor = xr.new_ones((*lead, 1))
        for j in range(n_slots):
            c = 1.0 + j / n_slots
            sse = _sse(c)
            better = sse < best_sse
            best_sse = torch.where(better, sse, best_sse)
            best_factor = torch.where(better, xr.new_full((*lead, 1), c), best_factor)
        return best_factor

    # coarse-to-fine: accumulate the mantissa integer `best_m` bit-group by bit-group,
    # each pass fixing the next (up to) 8 mantissa bits by minimizing per-block SSE.
    assert 1 <= mbs_bits <= 23, f"mbs_bits must be in [1, 23], got {mbs_bits}"
    best_m = xr.new_zeros((*lead, 1))   # integer mantissa fixed so far
    bits_done = 0
    while bits_done < mbs_bits:
        step = min(8, mbs_bits - bits_done)
        n = 1 << step
        denom = float(1 << (bits_done + step))
        best_sse = xr.new_full((*lead, 1), float("inf"))
        best_k = xr.new_zeros((*lead, 1))
        for k in range(n):
            m_try = best_m * n + k                     # (*lead,1) tensor
            c = 1.0 + m_try / denom
            sse = _sse(c)
            better = sse < best_sse
            best_sse = torch.where(better, sse, best_sse)
            best_k = torch.where(better, m_try, best_k)
        best_m = best_k
        bits_done += step
    return 1.0 + best_m / float(1 << mbs_bits)


def fake_quant(x: torch.Tensor, mbs: str = "none", oas: bool = True,
               ocp: bool = False, ocp_block: int = 32,
               oas_block: int = 16, inner: str = "oas",
               macro_block: int = 128, mbs_bits: int | None = None,
               chan_w: torch.Tensor | None = None) -> torch.Tensor:
    """Direct-cast MXFP4 fake-quant of `x` along its last dim.

    ocp=True -> MX/OCP (4,8] overflow scale at `ocp_block`; mbs/oas/inner ignored.
    ocp=False -> enhanced path; `inner` selects the per-block scale:
        inner="oas"        -> paper OAS scale (maps amax to (3.5, 7]), block=`oas_block`
        inner="quark_even" -> Quark's `qdq_mxfp4` even-rounded E8M0 scale (block=32)
      oas_block=16 -> MXFP4-16 family (paper default); oas_block=32 -> Quark group-size.
      mbs: "none"|"static"|"dynamic" — a 1x`macro_block` macro-block factor on top.
      macro_block: MBS macro-block size (paper default 128; e.g. 64 for finer grouping).
        Must be a multiple of the inner block size (oas_block / 32).
      mbs_bits: MBS factor mantissa precision. None -> paper defaults (static m8/8-bit,
        dynamic 16-slot LUT). Set (e.g. 8 or 16) -> both static/dynamic use that many
        mantissa bits (dynamic via coarse-to-fine search).
      chan_w: optional per-input-channel weight vector of shape (k,) for the dynamic
        search objective (SpikeEval activation-aware idea). Reshaped to
        (k//macro_block, macro_block) and passed as the weighted-SSE weights. Only used
        when mbs="dynamic"; None -> paper's plain-SSE search (bit-exact baseline).
    Returns a tensor of the same shape/dtype as x.
    """
    orig_dtype = x.dtype
    xf = x.to(torch.float32)

    if ocp:
        return _quant_blocks_ocp(xf, ocp_block).to(orig_dtype)

    def _inner(t):
        if inner == "quark_even":
            return _quant_blocks_quark_even(t)
        return _quant_blocks16(t, oas, oas_block)

    if mbs == "none":
        return _inner(xf).to(orig_dtype)

    *lead, k = xf.shape
    assert k % macro_block == 0, \
        f"last dim {k} not divisible by {macro_block} (MBS macro block)"
    xr = xf.reshape(*lead, k // macro_block, macro_block)
    if mbs == "static":
        factor = _mbs_factor_static(xr, mbs_bits=mbs_bits if mbs_bits else 8)
    elif mbs == "dynamic":
        cw = None
        if chan_w is not None:
            cw = chan_w.to(xf.device, torch.float32).reshape(k // macro_block, macro_block)
        factor = _mbs_factor_dynamic(xr, oas, oas_block=oas_block, inner=inner,
                                     mbs_bits=mbs_bits, chan_w=cw)
    else:
        raise ValueError(f"unknown mbs mode: {mbs!r}")

    xs = (xr * factor).reshape(*lead, k)
    q = _inner(xs).reshape(*lead, k // macro_block, macro_block)
    deq = q / factor
    return deq.reshape(*lead, k).to(orig_dtype)


# Method registry: how each Table-2 row maps to fake_quant kwargs (weight & activation).
# All rows quantize BOTH weights and activations (paper Setups).
# ocp=True -> OCP path (block_size=32, OCP scale); ocp=False -> enhanced path (block_size=16).
METHODS = {
    "MXFP4-OCP":    {"weight_mbs": "none", "act_mbs": "none", "oas": False, "ocp": True,  "ocp_block": 32},
    # MXFP4-16 = MX/OCP (4,8] overflow scale at block 16 (paper 4.1). It uses the SAME
    # scale as OCP, NOT the (3,6]/OAS scale — see the PITFALL note in the module docstring.
    "MXFP4-16":     {"weight_mbs": "none", "act_mbs": "none", "oas": False, "ocp": True,  "ocp_block": 16},
    "MXFP4-16-OAS": {"weight_mbs": "none", "act_mbs": "none", "oas": True,  "ocp": False},
    "MXFP4-MBS-S":  {"weight_mbs": "static",  "act_mbs": "static",  "oas": True, "ocp": False},
    # MBS-Hybrid: Dynamic for weights, Static for activations (paper default).
    "MXFP4-MBS-H":  {"weight_mbs": "dynamic", "act_mbs": "static",  "oas": True, "ocp": False},
    # SpikeEval idea: MBS-H whose *dynamic weight* search minimizes an input-amax^2-weighted
    # SSE (activation-aware). Same kwargs as MXFP4-MBS-H; per-layer chan_w (amax^2) is
    # injected by quantize_model_ from a calibration file. Activation path unchanged.
    "MXFP4-MBS-H-ACTW": {"weight_mbs": "dynamic", "act_mbs": "static", "oas": True, "ocp": False},
    # Quark-OAS: OAS at group-size=32 (matching Quark's MXFP4 kernel block size).
    "MXFP4-Quark-OAS":   {"weight_mbs": "none",    "act_mbs": "none",   "oas": True, "ocp": False, "oas_block": 32},
    # MBS-H (dynamic weight / static activation, 1x128 macro block) on top of Quark's own
    # MXFP4 kernel: the inner per-32 scale is Quark's even-rounded E8M0 (qdq_mxfp4 "even").
    "MXFP4-Quark-MBS-H": {"weight_mbs": "dynamic", "act_mbs": "static", "oas": False, "ocp": False, "oas_block": 32, "inner": "quark_even"},
    # Same as Quark-MBS-H but with a finer 1x64 MBS macro block (default is 1x128).
    "MXFP4-Quark-MBS-H-64": {"weight_mbs": "dynamic", "act_mbs": "static", "oas": False, "ocp": False, "oas_block": 32, "inner": "quark_even", "macro_block": 64},
    # MBS factor mantissa-precision ablation (macro-block 128). Both weight (dynamic,
    # coarse-to-fine) and activation (static) MBS factors use `mbs_bits` mantissa bits.
    "MXFP4-Quark-MBS-H-8bit":  {"weight_mbs": "dynamic", "act_mbs": "static", "oas": False, "ocp": False, "oas_block": 32, "inner": "quark_even", "macro_block": 128, "mbs_bits": 8},
    "MXFP4-Quark-MBS-H-16bit": {"weight_mbs": "dynamic", "act_mbs": "static", "oas": False, "ocp": False, "oas_block": 32, "inner": "quark_even", "macro_block": 128, "mbs_bits": 16},
}


def qsnr_db(x: torch.Tensor, xq: torch.Tensor) -> float:
    """Quantization SNR in dB (signal power over error power) -- diagnostic only."""
    x = x.to(torch.float32)
    xq = xq.to(torch.float32)
    sig = (x ** 2).mean().clamp_min(_TINY)
    err = ((x - xq) ** 2).mean().clamp_min(_TINY)
    return float(10.0 * torch.log10(sig / err))
