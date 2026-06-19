# v0.7.1 External DR Review Ranking Summary

## 版本定位

本结果基于 v0.7.1 frozen checkpoint direct external inference 输出，评估外部 DR 数据上的危险错误富集与自动放行区残余风险。

本阶段不使用 IDRiD_data / MESSIDOR2 train 或 val 训练，不根据外部结果重新选择 primary target、ranking signal 或 review budget。

## 事件定义

- `large_undergrading = true_grade - pred_grade >= 2`
- `vision_threatening_dr_miss = true_grade >= 3 and pred_grade < 3`
- `dangerous_undergrading = large_undergrading OR vision_threatening_dr_miss`

## 冻结协议

- `large_undergrading`: primary ranking signal = `expected_gap_only`, primary budget = Top20%
- `vision_threatening_dr_miss`: primary ranking signal = `gated_severe_prob_mass_only`, primary budget = Top20%
- `dangerous_undergrading`: secondary composite target，仅作为补充分析

## 事件数量

| dataset | backbone | n | large_undergrading | large_undergrading_rate | vision_threatening_dr_miss | vision_threatening_dr_miss_rate | dangerous_undergrading | dangerous_undergrading_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDRiD_data | convnext_tiny | 103 | 12 | 0.1165 | 23 | 0.2233 | 25 | 0.2427 |
| IDRiD_data | retfound_mae_cfp_official_like | 103 | 12 | 0.1165 | 25 | 0.2427 | 26 | 0.2524 |
| IDRiD_data | swin_tiny | 103 | 18 | 0.1748 | 29 | 0.2816 | 34 | 0.3301 |
| IDRiD_data | vit_b_imagenet | 103 | 16 | 0.1553 | 26 | 0.2524 | 28 | 0.2718 |
| IDRiD_data | vit_b_official_like | 103 | 13 | 0.1262 | 24 | 0.2330 | 26 | 0.2524 |
| IDRiD_data | vit_l_official_like | 103 | 7 | 0.0680 | 17 | 0.1650 | 18 | 0.1748 |
| MESSIDOR2 | convnext_tiny | 526 | 99 | 0.1882 | 27 | 0.0513 | 115 | 0.2186 |
| MESSIDOR2 | retfound_mae_cfp_official_like | 526 | 79 | 0.1502 | 26 | 0.0494 | 97 | 0.1844 |
| MESSIDOR2 | swin_tiny | 526 | 77 | 0.1464 | 29 | 0.0551 | 97 | 0.1844 |
| MESSIDOR2 | vit_b_imagenet | 526 | 101 | 0.1920 | 28 | 0.0532 | 112 | 0.2129 |
| MESSIDOR2 | vit_b_official_like | 526 | 117 | 0.2224 | 29 | 0.0551 | 124 | 0.2357 |
| MESSIDOR2 | vit_l_official_like | 526 | 81 | 0.1540 | 21 | 0.0399 | 94 | 0.1787 |

## Primary Top20% 结果

| dataset | backbone | target | ranking_method | n | top_k | total_event | captured_event | residual_event | base_event_rate | flagged_event_rate | event_recall | enrichment_ratio | low_risk_npv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDRiD_data | convnext_tiny | large_undergrading | expected_gap_only | 103 | 21 | 12 | 4 | 8 | 0.1165 | 0.1905 | 0.3333 | 1.6349 | 0.9024 |
| IDRiD_data | retfound_mae_cfp_official_like | large_undergrading | expected_gap_only | 103 | 21 | 12 | 7 | 5 | 0.1165 | 0.3333 | 0.5833 | 2.8611 | 0.9390 |
| IDRiD_data | swin_tiny | large_undergrading | expected_gap_only | 103 | 21 | 18 | 5 | 13 | 0.1748 | 0.2381 | 0.2778 | 1.3624 | 0.8415 |
| IDRiD_data | vit_b_imagenet | large_undergrading | expected_gap_only | 103 | 21 | 16 | 8 | 8 | 0.1553 | 0.3810 | 0.5000 | 2.4524 | 0.9024 |
| IDRiD_data | vit_b_official_like | large_undergrading | expected_gap_only | 103 | 21 | 13 | 4 | 9 | 0.1262 | 0.1905 | 0.3077 | 1.5092 | 0.8902 |
| IDRiD_data | vit_l_official_like | large_undergrading | expected_gap_only | 103 | 21 | 7 | 3 | 4 | 0.0680 | 0.1429 | 0.4286 | 2.1020 | 0.9512 |
| MESSIDOR2 | convnext_tiny | large_undergrading | expected_gap_only | 526 | 106 | 99 | 47 | 52 | 0.1882 | 0.4434 | 0.4747 | 2.3558 | 0.8762 |
| MESSIDOR2 | retfound_mae_cfp_official_like | large_undergrading | expected_gap_only | 526 | 106 | 79 | 20 | 59 | 0.1502 | 0.1887 | 0.2532 | 1.2563 | 0.8595 |
| MESSIDOR2 | swin_tiny | large_undergrading | expected_gap_only | 526 | 106 | 77 | 39 | 38 | 0.1464 | 0.3679 | 0.5065 | 2.5134 | 0.9095 |
| MESSIDOR2 | vit_b_imagenet | large_undergrading | expected_gap_only | 526 | 106 | 101 | 34 | 67 | 0.1920 | 0.3208 | 0.3366 | 1.6705 | 0.8405 |
| MESSIDOR2 | vit_b_official_like | large_undergrading | expected_gap_only | 526 | 106 | 117 | 49 | 68 | 0.2224 | 0.4623 | 0.4188 | 2.0782 | 0.8381 |
| MESSIDOR2 | vit_l_official_like | large_undergrading | expected_gap_only | 526 | 106 | 81 | 24 | 57 | 0.1540 | 0.2264 | 0.2963 | 1.4703 | 0.8643 |
| IDRiD_data | convnext_tiny | vision_threatening_dr_miss | gated_severe_prob_mass_only | 103 | 21 | 23 | 16 | 7 | 0.2233 | 0.7619 | 0.6957 | 3.4120 | 0.9146 |
| IDRiD_data | retfound_mae_cfp_official_like | vision_threatening_dr_miss | gated_severe_prob_mass_only | 103 | 21 | 25 | 13 | 12 | 0.2427 | 0.6190 | 0.5200 | 2.5505 | 0.8537 |
| IDRiD_data | swin_tiny | vision_threatening_dr_miss | gated_severe_prob_mass_only | 103 | 21 | 29 | 17 | 12 | 0.2816 | 0.8095 | 0.5862 | 2.8752 | 0.8537 |
| IDRiD_data | vit_b_imagenet | vision_threatening_dr_miss | gated_severe_prob_mass_only | 103 | 21 | 26 | 14 | 12 | 0.2524 | 0.6667 | 0.5385 | 2.6410 | 0.8537 |
| IDRiD_data | vit_b_official_like | vision_threatening_dr_miss | gated_severe_prob_mass_only | 103 | 21 | 24 | 9 | 15 | 0.2330 | 0.4286 | 0.3750 | 1.8393 | 0.8171 |
| IDRiD_data | vit_l_official_like | vision_threatening_dr_miss | gated_severe_prob_mass_only | 103 | 21 | 17 | 11 | 6 | 0.1650 | 0.5238 | 0.6471 | 3.1737 | 0.9268 |
| MESSIDOR2 | convnext_tiny | vision_threatening_dr_miss | gated_severe_prob_mass_only | 526 | 106 | 27 | 24 | 3 | 0.0513 | 0.2264 | 0.8889 | 4.4109 | 0.9929 |
| MESSIDOR2 | retfound_mae_cfp_official_like | vision_threatening_dr_miss | gated_severe_prob_mass_only | 526 | 106 | 26 | 21 | 5 | 0.0494 | 0.1981 | 0.8077 | 4.0080 | 0.9881 |
| MESSIDOR2 | swin_tiny | vision_threatening_dr_miss | gated_severe_prob_mass_only | 526 | 106 | 29 | 25 | 4 | 0.0551 | 0.2358 | 0.8621 | 4.2778 | 0.9905 |
| MESSIDOR2 | vit_b_imagenet | vision_threatening_dr_miss | gated_severe_prob_mass_only | 526 | 106 | 28 | 23 | 5 | 0.0532 | 0.2170 | 0.8214 | 4.0761 | 0.9881 |
| MESSIDOR2 | vit_b_official_like | vision_threatening_dr_miss | gated_severe_prob_mass_only | 526 | 106 | 29 | 22 | 7 | 0.0551 | 0.2075 | 0.7586 | 3.7645 | 0.9833 |
| MESSIDOR2 | vit_l_official_like | vision_threatening_dr_miss | gated_severe_prob_mass_only | 526 | 106 | 21 | 18 | 3 | 0.0399 | 0.1698 | 0.8571 | 4.2534 | 0.9929 |

## 解释边界

- 当前外部分类迁移表现存在域迁移压力，尤其 MESSIDOR2 上多模型预测分布偏向 0 类。
- 因此本结果应解释为 frozen APTOS checkpoints 在外部 DR 数据上的错误富集与 residual risk analysis。
- 若分类迁移不足，不能将 ranking 结果强称为临床泛化成功验证。
- Top-K 使用 `ceil(n * budget)`，因此 IDRiD_data Top20% 为 21 张，MESSIDOR2 Top20% 为 106 张。
