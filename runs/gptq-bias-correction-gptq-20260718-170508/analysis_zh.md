# 分析 — GPTQ bias-correction

**结论:➖ NEUTRAL** — delta 0.0408745 within noise band [-0.02, 0.05)

## 实测
- 指标:`perplexity`（越低越好）
- baseline:`30.537620544433594`
- idea:`30.496746063232422`
- delta:`+0.0408745 (lower better)`
- 判定台阶:`L2_tiny`

## Gate
- 正确性 gate:通过
- eval-infra sanity:通过

## 命题
- On facebook/opt-125m at W4 (per-channel, groupsize -1), GPTQ + bias-correction lowers WikiText2 perplexity vs vanilla GPTQ by at least min_delta, same calibration/seqlen.

- min_delta（真实 win 阈值):`0.05`;tolerance（噪声带):`0.02`

## Extension point
- `opt.py` :: `opt_sequential` (hook)
