# mbs-actw-qwen3-8b — SpikeEval run

**Idea:** activation-aware MBS-Dynamic — the paper's 1×128 weight macro-block factor search
minimizes an **input-amax²-weighted** SSE `Σ a_k²(deq_k−w_k)²` instead of plain SSE.
**Target/baseline:** `paper-reprise/.../OAS-MBS-2603.08713` impl, method `MXFP4-MBS-H`.
**Verdict:** **NEUTRAL** (see `analysis_en.md` / `analysis_zh.md`, machine result in `results.json`).

| tier | baseline `MBS-H` | idea `MBS-H-ACTW` | delta |
|---|---|---|---|
| L1 output-NMSE (pileval-1) | 2.331e-3 | 1.510e-3 | −35.2% (252/252 layers) |
| L2 wikitext word_perplexity | 13.0535 | 13.0155 | −0.0380 (min_delta 0.05) |

## Layout
- `impl/` — copy of the base impl + the surgical diff:
  - `mxquant.py` (`_mbs_factor_dynamic`/`fake_quant` gain `chan_w`), `qmodel.py`
    (`QuantLinear.act_amax`, `quantize_model_(act_amax_map=...)`, method `MXFP4-MBS-H-ACTW`)
  - `collect_act_amax.py` — pileval-1 per-input-channel amax → `act_amax.pt`
  - `l1_proxy.py` — per-layer output-reconstruction NMSE, baseline vs idea
  - `eval_one.py` — recognizes `*-mbs-h-actw-ppl`, loads `ACT_AMAX_PATH`
  - `test_actw.py` (4/4) + `test_mxquant.py` regression (12/12)
- `env` → base run's uv env (torch 2.11 / transformers 5.13 / lm_eval 0.4.12).

## Reproduce
```bash
cd impl
export PAPER_REPRISE_MODEL=<Qwen3-8B snapshot>
export HF_HOME=/group/amdneuralopt/huggingface HF_HUB_OFFLINE=1
PY=../env/bin/python
CUDA_VISIBLE_DEVICES="" $PY test_mxquant.py && CUDA_VISIBLE_DEVICES="" $PY test_actw.py   # gates
CUDA_VISIBLE_DEVICES=4 $PY collect_act_amax.py --num-calib-data 1 --seqlen 512 --out act_amax.pt
CUDA_VISIBLE_DEVICES=4 $PY l1_proxy.py
CUDA_VISIBLE_DEVICES=4 $PY eval_one.py qwen3-8b-mxfp4-mbs-h-ppl        # baseline
ACT_AMAX_PATH=act_amax.pt CUDA_VISIBLE_DEVICES=5 $PY eval_one.py qwen3-8b-mxfp4-mbs-h-actw-ppl  # idea
```

**Environment:** CUDA 13.0 / torch 2.11.0+cu130 / transformers 5.13.0 / lm_eval 0.4.12,
1× NVIDIA H200. Baseline reproduces the paper-reprise reference PPL 13.05 → controlled delta.
