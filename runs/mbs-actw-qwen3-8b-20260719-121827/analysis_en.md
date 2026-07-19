# SpikeEval — Activation-aware (amax²-weighted) MBS-Dynamic search

**Verdict: NEUTRAL** (real, consistent improvement, but the end-to-end PPL gain lands just
below the significance threshold).

## Idea
Base repo: `paper-reprise/runs/OAS-MBS-2603.08713-20260709-150131/impl` (MXFP4 quantization
of Qwen3-8B, "Unveiling the Potential of Quantization with MXFP4", arXiv 2603.08713).

The paper's **MBS-Dynamic** chooses, for each 1×128 weight macro-block, the mantissa-slot
refinement factor `c ∈ [1,2)` that minimizes the **unweighted** sum of squared quantization
error `Σ_k (deq_k − w_k)²` (`mxquant._mbs_factor_dynamic`).

The idea keeps the search but changes the objective to an **input-amax²-weighted** SSE

```
Σ_k a_k² · (deq_k − w_k)²,   a_k = amax of input-channel k (calibration)
```

Rationale: a Linear's output error is `Σ_k Δw_k · x_k`, so the expected output MSE is
`Σ_k Δw_k² E[x_k²] + (cross terms)`. Weighting each channel's weight-error by `a_k²`
(a cheap proxy for `E[x_k²]`) steers the search to protect the columns that matter most for
the output — an AWQ-flavored MBS. The activation path is unchanged (still static MBS);
`a_k = const` reduces to the paper's search **bit-exactly**.

Calibration is the only new cost: the base method is calibration-free direct-cast, so we add
one forward pass on **pileval, num_calib_data=1** (512 tokens) to collect per-input-channel
amax for the 252 quantized linears.

## What was implemented (surgical, zero-blast-radius diff)
- `_mbs_factor_dynamic(..., chan_w)` — optional per-position weight in the search objective.
- `fake_quant(..., chan_w)` and `QuantLinear(..., act_amax)` thread `amax²` to the **weight**
  dynamic search only.
- `collect_act_amax.py` — pileval-1-sample per-input-channel amax → `act_amax.pt`.
- New method `MXFP4-MBS-H-ACTW`; `chan_w=None` is the pristine `MXFP4-MBS-H` path.

## Correctness (hard gate — all pass)
- **Degenerate equivalence:** uniform `chan_w` → `MBS-H-ACTW` == `MBS-H` bit-exact
  (tensor-level and layer-level).
- **Closed-form cross-check:** the selected slot equals an independent brute-force argmin of
  the weighted objective over all 16 slots.
- **Mechanism:** the weighted search provably lowers the amax²-weighted error and is not a
  no-op. Base regression suite (`test_mxquant.py`) still 12/12.

## Results
Controlled, same run dir / env / lm-eval call. Baseline reproduces the paper-reprise
reference (13.05) exactly → the delta is clean.

| tier | metric | baseline `MBS-H` | idea `MBS-H-ACTW` | delta |
|---|---|---|---|---|
| **L1 proxy** | full-model output NMSE (pileval-1) | 2.331e-3 | 1.510e-3 | **−35.2%**, 252/252 layers ↓ |
| **L2** | Qwen3-8B wikitext word_perplexity | 13.0535 | 13.0155 | **−0.0380** |

- **L1** is a strong, unambiguous mechanism win: on *real* (quantized) activations — including
  cross-channel terms the idea does not directly optimize — the weighted search cuts every
  single layer's output reconstruction error, −35% aggregate. Largest gains in the MLP
  `up_proj`/`down_proj` (most skewed activation profiles); smallest in attention `o_proj`,
  but never negative.
- **L2**: PPL improves by 0.038, consistent with L1's direction, but **< min_delta = 0.05**.

## Why NEUTRAL despite a 35% layer-error cut
1. **Only weights are reweighted.** Activations still use static MBS; their quant error is a
   floor the weight search cannot touch.
2. **Low headroom in the MBS factor itself.** The dynamic factor is a coarse 16-slot
   refinement on top of an already-good OAS scale. The base paper's own 8-bit-vs-16-bit
   factor-precision ablation found little headroom here; a better *objective* helps, but the
   ceiling is low (compare: halving the MBS macro-block 128→64 moved PPL by −0.41).
3. **Layer-MSE → PPL attenuation.** Per-layer output-error reductions partially wash out
   through the network; a 35% local cut is a much smaller global logit/PPL effect.

## Honest take & cheap next steps (not run — would surface scale first)
The idea is **directionally validated and correct**: it robustly reduces the true output
error it targets, and improves PPL, just sub-threshold at the paper-simple 1-sample setting.
It sits in the same class as the sibling `gptq-bias-correction` example (+0.0409, also
NEUTRAL). Natural cheap levers if a WIN is wanted: (a) more calibration data (num_calib_data
8–128 gives a less noisy `a_k` than a single 512-token sample); (b) also weight the *static
activation* factor, or use dynamic activations; (c) `E[x²]` (mean-square) instead of `amax²`
as the channel weight — amax is outlier-driven and noisier at 1 sample.
