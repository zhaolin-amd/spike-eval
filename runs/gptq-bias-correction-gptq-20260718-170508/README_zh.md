# GPTQ bias-correction

在 `https://github.com/IST-DASLab/gptq` 上验证新算法 idea 是否胜过原 baseline。

## 结论:➖ NEUTRAL
delta 0.0408745 within noise band [-0.02, 0.05)

| 项 | 值 |
|---|---|
| 指标 | `perplexity`（越低越好）|
| baseline | `30.537620544433594` |
| idea | `30.496746063232422` |
| delta | `+0.0408745 (lower better)` |
| 判定台阶 | `L2_tiny` |
| 正确性 gate | 通过 |
| eval-infra sanity | 通过 |
