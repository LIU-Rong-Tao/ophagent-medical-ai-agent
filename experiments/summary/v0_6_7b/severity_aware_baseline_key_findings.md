# v0.6.7b Severity-aware Baseline Ablation

## Goal

This ablation checks whether the original OphAgent combined rule is more stable than single severity-aware ranking signals for dangerous undergrading events.

The ranking stage does not use true labels. True labels are used only for posterior evaluation.

## Ranking methods

- `ophagent_combined`: original v0.6.7 `review_priority_rank`.
- `severe_prob_mass_only`: sort by `P(Severe) + P(PDR)`.
- `gated_severe_prob_mass_only`: prioritize `pred_grade <= 2`, then sort by severe probability mass.
- `expected_grade_only`: sort by expected grade from class probabilities.
- `expected_gap_only`: sort by `expected_grade - pred_grade`.
- `top2_more_severe_only`: prioritize samples whose top-2 grade is more severe than top-1.

## Main event mean summary

| clinical_event | review_budget | ranking_method | mean_recall | mean_lift | total_captured | total_dangerous | total_residual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large_undergrading | 10% | gated_severe_prob_mass_only | 43.4% | 4.34x | 114 | 263 | 149 |
| large_undergrading | 10% | expected_gap_only | 42.5% | 4.25x | 115 | 263 | 148 |
| large_undergrading | 10% | ophagent_combined | 39.0% | 3.90x | 104 | 263 | 159 |
| large_undergrading | 10% | top2_more_severe_only | 38.0% | 3.80x | 100 | 263 | 163 |
| large_undergrading | 10% | expected_grade_only | 8.1% | 0.81x | 24 | 263 | 239 |
| large_undergrading | 10% | severe_prob_mass_only | 6.3% | 0.63x | 19 | 263 | 244 |
| large_undergrading | 20% | expected_gap_only | 66.1% | 3.31x | 177 | 263 | 86 |
| large_undergrading | 20% | top2_more_severe_only | 65.3% | 3.26x | 173 | 263 | 90 |
| large_undergrading | 20% | gated_severe_prob_mass_only | 64.0% | 3.20x | 169 | 263 | 94 |
| large_undergrading | 20% | ophagent_combined | 58.7% | 2.94x | 156 | 263 | 107 |
| large_undergrading | 20% | severe_prob_mass_only | 44.4% | 2.22x | 117 | 263 | 146 |
| large_undergrading | 20% | expected_grade_only | 42.1% | 2.11x | 112 | 263 | 151 |
| large_undergrading | 30% | top2_more_severe_only | 78.2% | 2.61x | 207 | 263 | 56 |
| large_undergrading | 30% | expected_gap_only | 76.4% | 2.55x | 203 | 263 | 60 |
| large_undergrading | 30% | gated_severe_prob_mass_only | 76.1% | 2.54x | 200 | 263 | 63 |
| large_undergrading | 30% | ophagent_combined | 72.2% | 2.41x | 192 | 263 | 71 |
| large_undergrading | 30% | severe_prob_mass_only | 64.3% | 2.14x | 170 | 263 | 93 |
| large_undergrading | 30% | expected_grade_only | 55.4% | 1.85x | 145 | 263 | 118 |
| vision_threatening_dr_miss | 10% | gated_severe_prob_mass_only | 52.3% | 5.23x | 204 | 391 | 187 |
| vision_threatening_dr_miss | 10% | ophagent_combined | 40.5% | 4.05x | 158 | 391 | 233 |
| vision_threatening_dr_miss | 10% | top2_more_severe_only | 38.1% | 3.81x | 150 | 391 | 241 |
| vision_threatening_dr_miss | 10% | expected_gap_only | 36.3% | 3.63x | 142 | 391 | 249 |
| vision_threatening_dr_miss | 10% | expected_grade_only | 8.6% | 0.86x | 37 | 391 | 354 |
| vision_threatening_dr_miss | 10% | severe_prob_mass_only | 7.8% | 0.78x | 34 | 391 | 357 |
| vision_threatening_dr_miss | 20% | gated_severe_prob_mass_only | 75.9% | 3.80x | 298 | 391 | 93 |
| vision_threatening_dr_miss | 20% | top2_more_severe_only | 65.7% | 3.29x | 259 | 391 | 132 |
| vision_threatening_dr_miss | 20% | ophagent_combined | 61.4% | 3.07x | 241 | 391 | 150 |
| vision_threatening_dr_miss | 20% | expected_gap_only | 57.4% | 2.87x | 226 | 391 | 165 |
| vision_threatening_dr_miss | 20% | severe_prob_mass_only | 52.0% | 2.60x | 204 | 391 | 187 |
| vision_threatening_dr_miss | 20% | expected_grade_only | 49.9% | 2.49x | 197 | 391 | 194 |
| vision_threatening_dr_miss | 30% | gated_severe_prob_mass_only | 89.7% | 2.99x | 351 | 391 | 40 |
| vision_threatening_dr_miss | 30% | top2_more_severe_only | 80.3% | 2.68x | 316 | 391 | 75 |
| vision_threatening_dr_miss | 30% | severe_prob_mass_only | 77.2% | 2.57x | 303 | 391 | 88 |
| vision_threatening_dr_miss | 30% | ophagent_combined | 74.6% | 2.49x | 293 | 391 | 98 |
| vision_threatening_dr_miss | 30% | expected_grade_only | 72.6% | 2.42x | 285 | 391 | 106 |
| vision_threatening_dr_miss | 30% | expected_gap_only | 71.1% | 2.37x | 279 | 391 | 112 |

## Winner count by backbone

| clinical_event | review_budget | ranking_method | best_backbone_count |
| --- | --- | --- | --- |
| large_undergrading | 10% | expected_gap_only | 4 |
| large_undergrading | 10% | gated_severe_prob_mass_only | 2 |
| large_undergrading | 20% | expected_gap_only | 4 |
| large_undergrading | 20% | gated_severe_prob_mass_only | 1 |
| large_undergrading | 20% | ophagent_combined | 1 |
| large_undergrading | 30% | expected_gap_only | 3 |
| large_undergrading | 30% | gated_severe_prob_mass_only | 2 |
| large_undergrading | 30% | top2_more_severe_only | 1 |
| vision_threatening_dr_miss | 10% | gated_severe_prob_mass_only | 5 |
| vision_threatening_dr_miss | 10% | ophagent_combined | 1 |
| vision_threatening_dr_miss | 20% | gated_severe_prob_mass_only | 6 |
| vision_threatening_dr_miss | 30% | gated_severe_prob_mass_only | 6 |

## 中文结论

本次 v0.6.7b 消融实验的核心结论是：`ophagent_combined` 是有效的，但在加入更直接的 severity-aware baselines（严重程度感知基线）后，它并不是危险低估任务上的最优排序规则。该结果说明，v0.6.7 中观察到的危险错误富集能力，并不完全来自完整的手工组合分数，而主要可以由模型输出中的严重程度相关概率信号解释。

本次消融不只观察 Top20% 复核预算，而是在 Top5%、Top10%、Top20%、Top30% 多个复核预算下进行评估。其中，Top20% 作为主展示预算，用于和 v0.6.7 的展示口径保持一致；Top10% 和 Top30% 用于观察排序信号在不同复核负担下是否稳定；Top5% 作为极低复核预算下的补充结果，不作为主要结论来源。

对 `large_undergrading`（大幅低估）而言，Top20% 复核预算下，`expected_gap_only` 表现最好，mean recall 达到 66.1%，共捕获 177 / 263 个大幅低估样本，自动放行区残余 86 个；而原始 `ophagent_combined` 的 mean recall 为 58.7%，捕获 156 / 263 个，残余 107 个。从 winner count 看，`expected_gap_only` 在 Top10%、Top20%、Top30% 下分别达到 4 / 6、4 / 6、3 / 6 个 backbone 最优，说明它是跨 backbone 较稳定的核心信号。但在不同预算下，`gated_severe_prob_mass_only` 和 `top2_more_severe_only` 也具有竞争力，因此不能简单说某一个信号在所有复核预算下绝对最优。更稳妥的解释是：大幅低估主要与 `expected_grade - pred_grade` 有关，即模型 Top1 虽然预测偏轻，但完整概率分布的期望严重程度更高。

对 `vision_threatening_dr_miss`（威胁视力级 DR 漏检）而言，`gated_severe_prob_mass_only` 的优势更稳定。Top10%、Top20%、Top30% 下，它的 mean recall 均为最高；其中 Top20% 复核预算下，mean recall 达到 75.9%，共捕获 298 / 391 个样本，自动放行区残余 93 个；而原始 `ophagent_combined` 的 mean recall 为 61.4%，捕获 241 / 391 个，残余 150 个。从 winner count 看，`gated_severe_prob_mass_only` 在 Top20% 和 Top30% 下均为 6 / 6 个 backbone 最优，Top10% 下为 5 / 6 个 backbone 最优。这说明对重症漏检而言，`pred_grade <= 2` 条件下的 `P(Severe) + P(PDR)` 是最关键的排序信号。

因此，v0.6.7b 对 v0.6.7 的故事线进行了修正：项目价值不应表述为“某个手工 combined 分数最优”，而应表述为“方向敏感的危险错误可以通过模型输出中的严重程度感知信号被显著富集”。不同 clinical dangerous events 可能需要不同排序信号：大幅低估更适合 `expected_gap_only`，威胁视力级 DR 漏检更适合 `gated_severe_prob_mass_only`。

这也提示后续医院真实数据阶段不应固定依赖一个通用风险分数，而应根据具体危险错误类型选择或组合对应的 post-hoc risk signals。`ophagent_combined` 可以作为第一版透明审计规则保留，但 v0.6.7b 表明，更简单、更可解释的 severity-aware ranking 可能是后续临床审计工作流中更稳的方向。

## 结论检查清单

1. `ophagent_combined` 是否仍然是 `large_undergrading` 和 `vision_threatening_dr_miss` 上最稳定的排序方法？
2. 如果不是，哪个简单的 severity-aware signal（严重程度感知信号）解释了主要增益？
3. 不同 clinical dangerous events（临床危险错误类型）是否需要不同的排序信号？
