# 分析 — GPTQ bias-correction (headless)

**结论:⛔ BLOCKED** — correctness gate failed — idea not equivalent/valid

## 实测
- 指标:`perplexity`（越低越好）
- baseline:`None`
- idea:`None`
- delta:`n/a`
- 判定台阶:`None`

## Gate
- 正确性 gate:未通过
- eval-infra sanity:未通过

## 命题
- On facebook/opt-125m at W4, GPTQ + (headless-implemented) bias-correction lowers WikiText2 perplexity vs vanilla GPTQ by at least min_delta, same calibration/seqlen.

- min_delta（真实 win 阈值):`0.05`;tolerance（噪声带):`0.02`

## Extension point
- `opt.py` :: `opt_sequential` (hook)
