# SpikeEval —— 激活感知（amax² 加权）的 MBS-Dynamic 搜索

**判定：NEUTRAL**（改善真实且一致，但端到端 PPL 增益刚好低于显著性阈值）。

## Idea
Base repo：`paper-reprise/runs/OAS-MBS-2603.08713-20260709-150131/impl`（Qwen3-8B 的 MXFP4
量化，论文 "Unveiling the Potential of Quantization with MXFP4"，arXiv 2603.08713）。

论文的 **MBS-Dynamic** 对每个 1×128 权重 macro-block，**搜索** mantissa slot 得到细化因子
`c ∈ [1,2)`，目标是最小化**未加权**的量化误差平方和 `Σ_k (deq_k − w_k)²`
（`mxquant._mbs_factor_dynamic`）。

本 idea 保留搜索，但把目标换成 **input-amax² 加权**的 SSE：

```
Σ_k a_k² · (deq_k − w_k)²，   a_k = 第 k 个 input channel 的 amax（校准得到）
```

动机：Linear 的输出误差为 `Σ_k Δw_k · x_k`，因此期望输出 MSE ≈
`Σ_k Δw_k² E[x_k²] + (交叉项)`。用 `a_k²`（`E[x_k²]` 的廉价代理）给每个 channel 的权重误差
加权，能把搜索导向"对输出最重要"的那些列 —— 即 AWQ 风格的 MBS。激活路径不变（仍是 static
MBS）；`a_k = 常数` 时**逐比特**退化为论文的搜索。

唯一新增成本是校准：base 方法是免校准 direct-cast，这里加一次 **pileval, num_calib_data=1**
（512 tokens）前向，为 252 个被量化的 linear 收集 per-input-channel amax。

## 实现（外科式、零爆炸半径 diff）
- `_mbs_factor_dynamic(..., chan_w)` —— 搜索目标里加可选的逐位置权重。
- `fake_quant(..., chan_w)` 和 `QuantLinear(..., act_amax)` 只把 `amax²` 透传给**权重**的
  dynamic 搜索。
- `collect_act_amax.py` —— pileval 1 sample 的 per-input-channel amax → `act_amax.pt`。
- 新 method `MXFP4-MBS-H-ACTW`；`chan_w=None` 即原始 `MXFP4-MBS-H` 路径。

## 正确性（硬性 gate —— 全部通过）
- **退化等价：** uniform `chan_w` → `MBS-H-ACTW` == `MBS-H` 逐比特一致（tensor 级 + layer 级）。
- **闭式交叉验证：** 选中的 slot 等于对 16 个 slot 独立暴力枚举加权目标的 argmin。
- **机制验证：** 加权搜索确实降低 amax² 加权误差且非 no-op。base 回归套件
  （`test_mxquant.py`）仍 12/12。

## 结果
受控对比（同 run dir / 同 env / 同 lm-eval 调用）。baseline 逐点复现 paper-reprise 参考值
（13.05）→ delta 干净可信。

| 层级 | 指标 | baseline `MBS-H` | idea `MBS-H-ACTW` | delta |
|---|---|---|---|---|
| **L1 proxy** | 全模型输出 NMSE (pileval-1) | 2.331e-3 | 1.510e-3 | **−35.2%**，252/252 层 ↓ |
| **L2** | Qwen3-8B wikitext word_perplexity | 13.0535 | 13.0155 | **−0.0380** |

- **L1** 是明确无歧义的机制性胜利：在**真实**（量化后）激活上 —— 含 idea 并未直接优化的
  交叉项 —— 加权搜索让**每一层**的输出重构误差都下降，整体 −35%。改善最大的是 MLP 的
  `up_proj`/`down_proj`（激活分布最不均匀），attention `o_proj` 改善最小但从不为负。
- **L2**：PPL 改善 0.038，方向与 L1 一致，但 **< min_delta = 0.05**。

## 为什么层误差降 35% 却仍是 NEUTRAL
1. **只对权重加权。** 激活仍用 static MBS，其量化误差是权重搜索碰不到的地板。
2. **MBS 因子本身空间就小。** dynamic 因子只是在已经很好的 OAS scale 之上做 16 档粗调；
   base 论文自己的 8-bit vs 16-bit 因子精度消融就发现这里几乎没余量 —— 换更好的**目标**有
   帮助，但天花板低（对比：MBS macro-block 128→64 能让 PPL 动 −0.41）。
3. **层 MSE → PPL 的衰减。** 逐层输出误差的下降会在网络里部分抵消；局部降 35% 对应到全局
   logit/PPL 的效应要小得多。

## 诚实结论 & 便宜的后续（未跑 —— 会先报规模再做）
这个 idea **方向正确且实现无误**：它稳健地降低了自己所针对的真实输出误差，也确实改善了
PPL，只是在"论文简化版 1 sample"设定下刚好没过阈值。它和 sibling 的 `gptq-bias-correction`
例子（+0.0409，同样 NEUTRAL）属于同一档。若想推成 WIN，便宜的杠杆有：(a) 增加校准样本
（num_calib_data 8–128 比单条 512-token 样本得到的 `a_k` 更稳）；(b) 同时给 **static 激活**
因子加权，或改用 dynamic 激活；(c) 用 `E[x²]`（均方）而非 `amax²` 作为 channel 权重 —— amax
受离群点主导，在 1 sample 下更噪。
