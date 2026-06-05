# v0.6.6 Pre-review Risk Table

本表仅根据模型输出信号生成预审风险排序，不使用真实标签参与排序。

## 输入字段识别

- case_id: `None`
- image_path: `image_path`
- pred_label: `pred_label`
- confidence: `confidence`
- top2_label: `None`
- top2_confidence: `None`
- margin: `None`
- entropy: `None`
- prob_columns: `['prob_No DR', 'prob_Mild DR', 'prob_Moderate DR', 'prob_Severe DR', 'prob_Proliferative DR']`

## 排序阶段未使用但输入中存在的后验字段

- `correct`
- `true_label`

## 风险排序预览

| review_priority_rank | case_id | pred_label | confidence | top2_label | top2_confidence | margin | entropy_norm | severe_prob_mass | pre_review_risk_score | pre_review_risk_level | risk_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | a76b69e443ce | Moderate DR | 0.3552 | Proliferative DR | 0.2429 | 0.1123 | 0.9400 | 0.3491 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 2 | a11bf2edd470 | Mild DR | 0.3630 | Moderate DR | 0.2836 | 0.0794 | 0.9162 | 0.2616 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 3 | bca2bdc15fc5 | Moderate DR | 0.3329 | Proliferative DR | 0.3299 | 0.0030 | 0.8859 | 0.5255 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 4 | 3dbc90c7ee7d | Moderate DR | 0.3473 | Proliferative DR | 0.3461 | 0.0013 | 0.8645 | 0.4910 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 5 | 1e4b3b823b95 | Moderate DR | 0.3481 | Proliferative DR | 0.3171 | 0.0310 | 0.8587 | 0.5447 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 6 | 253e96488cfb | Mild DR | 0.4006 | Proliferative DR | 0.2676 | 0.1331 | 0.8576 | 0.3033 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 7 | 67c03349bb31 | Moderate DR | 0.4093 | Proliferative DR | 0.2648 | 0.1445 | 0.8463 | 0.3234 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 8 | 03676c71ed1b | Moderate DR | 0.3704 | Proliferative DR | 0.2990 | 0.0714 | 0.8420 | 0.3587 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 9 | ee059945b08a | Moderate DR | 0.3926 | Severe DR | 0.2781 | 0.1146 | 0.8330 | 0.5189 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 10 | af133a85ea0c | Mild DR | 0.3542 | Moderate DR | 0.3336 | 0.0206 | 0.8188 | 0.2810 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 11 | eb32a815f78c | Moderate DR | 0.4187 | Proliferative DR | 0.3415 | 0.0771 | 0.8072 | 0.3942 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 12 | 365f8c01d994 | Moderate DR | 0.4040 | Proliferative DR | 0.3789 | 0.0250 | 0.8009 | 0.4568 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 13 | 6e68e742f5bc | Mild DR | 0.3936 | Moderate DR | 0.2787 | 0.1150 | 0.8007 | 0.2984 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 14 | 7f60f2a083d3 | Moderate DR | 0.3760 | Proliferative DR | 0.3407 | 0.0352 | 0.7997 | 0.3936 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 15 | 735836b1ffa6 | Moderate DR | 0.4488 | Proliferative DR | 0.3306 | 0.1182 | 0.7963 | 0.4317 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 16 | dbb2c63f6f08 | Moderate DR | 0.3290 | Proliferative DR | 0.3112 | 0.0178 | 0.7941 | 0.6201 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 17 | 82bb8a01935f | Moderate DR | 0.3936 | Proliferative DR | 0.3195 | 0.0741 | 0.7902 | 0.3754 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 18 | cd93a472e5cd | Moderate DR | 0.3640 | Proliferative DR | 0.3136 | 0.0504 | 0.7889 | 0.5861 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 19 | d10d315f123f | Moderate DR | 0.4484 | Proliferative DR | 0.3073 | 0.1411 | 0.7866 | 0.3854 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 20 | eeb231c3ef1f | Mild DR | 0.3644 | Moderate DR | 0.3537 | 0.0108 | 0.7863 | 0.2643 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 21 | f460608cf4cc | Moderate DR | 0.3915 | Proliferative DR | 0.3599 | 0.0316 | 0.7747 | 0.3885 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 22 | ee1ec90b980f | Moderate DR | 0.4091 | Proliferative DR | 0.3920 | 0.0171 | 0.7733 | 0.4844 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 23 | b60dbf9f0744 | Moderate DR | 0.4584 | Proliferative DR | 0.3322 | 0.1262 | 0.7730 | 0.4430 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 24 | f69835dc7c50 | Moderate DR | 0.4170 | Severe DR | 0.3115 | 0.1055 | 0.7669 | 0.5380 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 25 | d0ffa0425ef1 | Moderate DR | 0.4617 | Proliferative DR | 0.3345 | 0.1273 | 0.7640 | 0.4478 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 26 | a56729de89e9 | Moderate DR | 0.4162 | Proliferative DR | 0.3868 | 0.0293 | 0.7639 | 0.4186 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 27 | 8c7c26c52a6c | Moderate DR | 0.4408 | Proliferative DR | 0.3652 | 0.0756 | 0.7636 | 0.4734 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 28 | 3ee4841936ef | Moderate DR | 0.4391 | Proliferative DR | 0.3352 | 0.1039 | 0.7630 | 0.3865 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 29 | 475c7ded0f7a | Moderate DR | 0.4408 | Proliferative DR | 0.3617 | 0.0791 | 0.7521 | 0.4071 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 30 | 40140a925c43 | Moderate DR | 0.4210 | Proliferative DR | 0.3566 | 0.0644 | 0.7506 | 0.5304 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 31 | 76516f828d88 | No DR | 0.3392 | Moderate DR | 0.2252 | 0.1140 | 0.9268 | 0.2353 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 32 | 6efa36d59ada | Moderate DR | 0.3859 | Proliferative DR | 0.2126 | 0.1733 | 0.9222 | 0.3517 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 33 | 2f5c9cdfb333 | Moderate DR | 0.4128 | Severe DR | 0.2155 | 0.1973 | 0.9025 | 0.3841 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 34 | 4b1001050f1d | No DR | 0.3540 | Mild DR | 0.3227 | 0.0313 | 0.8829 | 0.1678 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 35 | b4b04d81acbb | No DR | 0.3703 | Mild DR | 0.3020 | 0.0682 | 0.8800 | 0.1688 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 36 | 4a589edaea60 | Mild DR | 0.4021 | Moderate DR | 0.2683 | 0.1338 | 0.8750 | 0.1748 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 37 | ca30a97e9d13 | Mild DR | 0.4154 | Moderate DR | 0.2507 | 0.1647 | 0.8515 | 0.2716 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 38 | a7b7dc8788b9 | Moderate DR | 0.4240 | Proliferative DR | 0.2688 | 0.1552 | 0.8469 | 0.3066 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 39 | 3ddb86eb530e | Moderate DR | 0.4866 | Severe DR | 0.2060 | 0.2806 | 0.8451 | 0.3559 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 40 | 3b018e8b7303 | Moderate DR | 0.4886 | Severe DR | 0.2101 | 0.2785 | 0.8405 | 0.3600 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 41 | d144144a2f3f | Mild DR | 0.4096 | Moderate DR | 0.3000 | 0.1096 | 0.8392 | 0.2443 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 42 | 0415fc68b176 | Moderate DR | 0.4475 | Severe DR | 0.2468 | 0.2008 | 0.8282 | 0.4488 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 43 | 2b074afdf626 | Mild DR | 0.4618 | Proliferative DR | 0.2645 | 0.1973 | 0.8172 | 0.3156 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 44 | 48c49f662f7d | Moderate DR | 0.4787 | Proliferative DR | 0.2179 | 0.2607 | 0.8143 | 0.3305 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 45 | af831c158744 | Mild DR | 0.3835 | Moderate DR | 0.3412 | 0.0424 | 0.8091 | 0.2340 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 46 | 7663aba8d762 | Moderate DR | 0.4242 | Proliferative DR | 0.2724 | 0.1518 | 0.7966 | 0.3195 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 47 | 4189d4e631ec | Moderate DR | 0.4827 | Severe DR | 0.2206 | 0.2620 | 0.7926 | 0.4325 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 48 | a3d2a0c4cd17 | Moderate DR | 0.4884 | Proliferative DR | 0.2578 | 0.2306 | 0.7925 | 0.4025 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 49 | 52230bbef30e | Moderate DR | 0.4738 | Proliferative DR | 0.2776 | 0.1962 | 0.7897 | 0.3397 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 50 | 91cbe1c775ef | Mild DR | 0.4043 | Moderate DR | 0.3554 | 0.0489 | 0.7855 | 0.2034 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
