# Analysis — GPTQ bias-correction (headless)

**Verdict: ⛔ BLOCKED** — correctness gate failed — idea not equivalent/valid

## Measured
- metric: `perplexity` (lower is better)
- baseline: `None`
- idea: `None`
- delta: `n/a`
- deciding tier: `None`

## Gates
- correctness gate: FAIL
- eval-infra sanity: FAIL

## Claim
- On facebook/opt-125m at W4, GPTQ + (headless-implemented) bias-correction lowers WikiText2 perplexity vs vanilla GPTQ by at least min_delta, same calibration/seqlen.

- min_delta (real win): `0.05` ; tolerance (noise): `0.02`

## Extension point
- `opt.py` :: `opt_sequential` (hook)
