# v0.7.2 Metric-Sensitivity Audit

本版本是 secondary metric audit，不替代 v0.7.1b 的 primary analysis。

目的：检查外部 DR review-ranking 结论是否依赖单一评价口径，尤其是 AURC、AUGRC、partial AUGRC 与固定 Top20% 工作点之间是否一致。

## 指标口径

- 高 risk score 表示更应优先复核。
- AURC：selective risk-coverage 曲线面积。
- AUGRC：generalized risk-coverage 曲线面积。
- partial AUGRC 0.70–0.90：coverage 0.70–0.90 区间的归一化 generalized risk 面积，对应 Top10%–Top30% 复核预算范围。
- Top20% event recall / residual event count：固定复核预算下的工作点指标。

## 排名变化概览

| dataset | event | n_rows | n_changed_aurc_vs_augrc | n_changed_augrc_vs_partial |
| --- | --- | --- | --- | --- |
| IDRiD_data | general_error | 36 | 12 | 29 |
| IDRiD_data | large_undergrading | 36 | 20 | 30 |
| IDRiD_data | vtdr_miss | 36 | 2 | 17 |
| MESSIDOR2 | general_error | 36 | 11 | 23 |
| MESSIDOR2 | large_undergrading | 36 | 13 | 12 |
| MESSIDOR2 | vtdr_miss | 36 | 6 | 9 |

## VTDR miss 下的 Top-ranked methods

| dataset | backbone | method | auroc_error | aurc | augrc | partial_augrc_70_90 | top20_event_recall | top20_residual_event_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDRiD_data | convnext_tiny | gated_severe_prob_mass_only | 0.896196 | 0.0574202 | 0.0429352 | 0.0853026 | 0.695652 | 7 |
| IDRiD_data | retfound_mae_cfp_official_like | gated_severe_prob_mass_only | 0.888205 | 0.0638943 | 0.0500047 | 0.10908 | 0.52 | 12 |
| IDRiD_data | swin_tiny | gated_severe_prob_mass_only | 0.884436 | 0.0830688 | 0.0630125 | 0.127743 | 0.586207 | 12 |
| IDRiD_data | vit_b_imagenet | gated_severe_prob_mass_only | 0.873127 | 0.0715071 | 0.0558017 | 0.122629 | 0.538462 | 12 |
| IDRiD_data | vit_b_official_like | gated_severe_prob_mass_only | 0.80116 | 0.0860336 | 0.0626826 | 0.131796 | 0.375 | 15 |
| IDRiD_data | vit_l_official_like | gated_severe_prob_mass_only | 0.880985 | 0.0361381 | 0.0300217 | 0.0641884 | 0.647059 | 6 |
| MESSIDOR2 | convnext_tiny | gated_severe_prob_mass_only | 0.950642 | 0.00443888 | 0.00372096 | 0.00519019 | 0.888889 | 3 |
| MESSIDOR2 | retfound_mae_cfp_official_like | gated_severe_prob_mass_only | 0.879846 | 0.00934344 | 0.00686724 | 0.0097569 | 0.807692 | 5 |
| MESSIDOR2 | swin_tiny | gated_severe_prob_mass_only | 0.938875 | 0.00513866 | 0.00470406 | 0.00754095 | 0.862069 | 4 |
| MESSIDOR2 | vit_b_imagenet | gated_severe_prob_mass_only | 0.89716 | 0.00831052 | 0.00659978 | 0.0112189 | 0.821429 | 5 |
| MESSIDOR2 | vit_b_official_like | gated_severe_prob_mass_only | 0.860404 | 0.0128068 | 0.00879187 | 0.0146287 | 0.758621 | 7 |
| MESSIDOR2 | vit_l_official_like | gated_severe_prob_mass_only | 0.915794 | 0.0047383 | 0.00402456 | 0.0061263 | 0.857143 | 3 |

## 主要发现

- general_error 和 large_undergrading 的方法排名在 AURC / AUGRC / partial AUGRC 下存在较明显变化，说明总体错误排序与方向敏感危险错误排序不应混用同一个结论。当前 `n_changed` 仅表示排名发生变化，尚未区分微小数值差异、并列排名或 top-1 方法是否真正改变。
- 在核心事件 VTDR miss 上，`gated_severe_prob_mass_only` 在 12/12 个 dataset-backbone 组合中均取得 AURC、AUGRC 和 partial_AUGRC_70_90 的第一排名。
- 在固定 Top20% 工作点下，`gated_severe_prob_mass_only` 在 11/12 个 dataset-backbone 组合中达到第一排名（含并列第一）；剩余组合中排名第三。
- 该结果说明，在当前两个外部 DR 公共数据集和六个冻结 backbone 上，`gated_severe_prob_mass_only` 对 grade-based VTDR miss 的排序优势不局限于 Top20% 单点，而是在 AURC、AUGRC 和 coverage 0.70–0.90 的 partial AUGRC 下保持一致。
- 该结果应解释为跨评价口径的描述性一致性证据，而不是独立统计显著性、临床效用或真实医生工作流有效性的证明。

## 机制解释边界

- `gated_severe_prob_mass_only` 是事件特异性、结构对齐的排序信号：VTDR miss 定义为 `true_grade >= 3 and pred_grade < 3`，该方法也显式利用 `pred_grade <= 2` gate 和 severe-class probability mass。因此，本结果不能直接推广为通用 failure detector。
- 更严谨的解释是：即使 top-1 落在非重症类别，输出分布中的 severe-class probability mass 仍保留与 VTDR miss 排序有关的信息。
- partial_AUGRC_70_90 第一表示 coverage 0.70–0.90 区间的积分表现最好，不保证该区间内每一个具体复核预算点都排名第一。

## 解释边界

- AUGRC 不是临床效用指标，也不是临床安全证明。
- AURC 不是错误指标；AUGRC 是补充敏感性评价。
- 跨 backbone 比较会混合基础分类准确率与 risk score 排序能力，因此主要作为 descriptive audit。
- 本结果仍是 image-level retrospective audit，不是 patient-level clinical workflow validation。
