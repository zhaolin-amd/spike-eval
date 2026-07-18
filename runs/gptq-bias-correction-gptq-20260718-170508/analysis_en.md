# Analysis — GPTQ bias-correction

**Verdict: ➖ NEUTRAL** — delta 0.0408745 within noise band [-0.02, 0.05)

## Measured
- metric: `perplexity` (lower is better)
- baseline: `30.537620544433594`
- idea: `30.496746063232422`
- delta: `+0.0408745 (lower better)`
- deciding tier: `L2_tiny`

## Gates
- correctness gate: PASS
- eval-infra sanity: PASS

## Claim
- On facebook/opt-125m at W4 (per-channel, groupsize -1), GPTQ + bias-correction lowers WikiText2 perplexity vs vanilla GPTQ by at least min_delta, same calibration/seqlen.

- min_delta (real win): `0.05` ; tolerance (noise): `0.02`

## Extension point
- `opt.py` :: `opt_sequential` (hook)
