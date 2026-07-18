# GPTQ bias-correction

Validation of a new-algorithm idea against the baseline in `https://github.com/IST-DASLab/gptq`.

## Verdict: ➖ NEUTRAL
delta 0.0408745 within noise band [-0.02, 0.05)

| item | value |
|---|---|
| metric | `perplexity` (lower better) |
| baseline | `30.537620544433594` |
| idea | `30.496746063232422` |
| delta | `+0.0408745 (lower better)` |
| deciding tier | `L2_tiny` |
| correctness gate | PASS |
| eval-infra sanity | PASS |
