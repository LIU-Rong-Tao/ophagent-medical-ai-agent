# v0.7.1b 随机种子敏感性检查

## 目的

v0.7.1b 主分析使用固定随机种子 `seed=42`。为了确认 random gate-only baseline 与 image-clustered bootstrap 的 Monte Carlo 随机性不会改变主结论，额外使用 `seed=123`、`seed=2026`、`seed=3407` 重复运行完整评估流程。

该分析只作为稳健性检查，不用于更改主指标、选择结果或调整结论阈值。

## 主比较

- Target：VTDR miss
- Budget：Top20%
- Method：`gated_severe_prob_mass_only`
- Comparator：`random_gate_only_expected`
- Metric：`mean_delta_event_recall`
- 解释：`gated_severe_prob_mass_only - random_gate_only`

## 结果

| seed | dataset | mean_delta_event_recall | ci95_low_delta_event_recall | ci95_high_delta_event_recall | win_rate_delta_gt_0 |
| --- | --- | --- | --- | --- | --- |
| 42 | IDRiD_data | 0.3385 | 0.2195 | 0.4742 | 1.0000 |
| 42 | MESSIDOR2 | 0.6268 | 0.5003 | 0.7369 | 1.0000 |
| 123 | IDRiD_data | 0.3392 | 0.2229 | 0.4761 | 1.0000 |
| 123 | MESSIDOR2 | 0.6266 | 0.5001 | 0.7363 | 1.0000 |
| 2026 | IDRiD_data | 0.3408 | 0.2242 | 0.4779 | 1.0000 |
| 2026 | MESSIDOR2 | 0.6306 | 0.5108 | 0.7345 | 1.0000 |
| 3407 | IDRiD_data | 0.3370 | 0.2179 | 0.4762 | 1.0000 |
| 3407 | MESSIDOR2 | 0.6309 | 0.5131 | 0.7364 | 1.0000 |

## 结论

四个固定随机种子下，两个外部数据集的 `mean_delta_event_recall` 均为正，95% CI 下界均大于 0，`win_rate_delta_gt_0` 均为 1.0。

因此，v0.7.1b 的 primary comparison 结论对 Monte Carlo 随机种子不敏感。

## 展示口径

可以写：

> 额外随机种子敏感性检查显示，主比较结论在 seed=42、123、2026、3407 下保持一致。

不能写：

> 不同随机种子证明临床有效性。

本结果仍限于公共数据集回顾性 grade-based proxy、当前六个 frozen APTOS backbones 和 image-level external stress test。

## Comparator 口径说明

- `random_gate_only`：2000 次随机抽样，用于估计 gate-only baseline 的随机分布。
- `random_gate_only_expected`：primary bootstrap 中使用的期望对照，避免每次 bootstrap 又叠加随机抽样噪声。
