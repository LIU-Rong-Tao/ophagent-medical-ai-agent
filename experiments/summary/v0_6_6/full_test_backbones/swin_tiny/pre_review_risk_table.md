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
| 1 | eb1d37b71fd1 | Moderate DR | 0.4150 | Severe DR | 0.3222 | 0.0928 | 0.7286 | 0.5591 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 2 | 070f67572d03 | Moderate DR | 0.4080 | Proliferative DR | 0.3555 | 0.0525 | 0.6753 | 0.5899 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 3 | a476fd984005 | Moderate DR | 0.4548 | Severe DR | 0.3351 | 0.1197 | 0.6597 | 0.5438 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 4 | 55fd453001cc | Moderate DR | 0.4867 | Proliferative DR | 0.3762 | 0.1105 | 0.6584 | 0.3990 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 5 | f3b6b7ca1eb1 | Moderate DR | 0.5252 | Proliferative DR | 0.3528 | 0.1724 | 0.6473 | 0.4385 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 6 | 4c60b10a3a6a | Moderate DR | 0.4798 | Severe DR | 0.4500 | 0.0298 | 0.5613 | 0.5193 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 7 | 15f440753916 | Moderate DR | 0.5422 | Severe DR | 0.4208 | 0.1214 | 0.5116 | 0.4568 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 8 | 08a3875063c3 | Moderate DR | 0.5405 | Proliferative DR | 0.4234 | 0.1171 | 0.5102 | 0.4585 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 9 | 40140a925c43 | Moderate DR | 0.6078 | Severe DR | 0.3584 | 0.2494 | 0.4930 | 0.3901 | 7 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe;confident_but_close_decision |
| 10 | 698d6e422a80 | Moderate DR | 0.4990 | Severe DR | 0.4972 | 0.0018 | 0.4459 | 0.5002 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 11 | 06024377d573 | Moderate DR | 0.5693 | Severe DR | 0.2673 | 0.3020 | 0.6172 | 0.4258 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 12 | 66d2ca47aa44 | Moderate DR | 0.5556 | Severe DR | 0.3449 | 0.2108 | 0.5921 | 0.4373 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 13 | 4818672273af | Moderate DR | 0.5767 | Proliferative DR | 0.3086 | 0.2681 | 0.5914 | 0.3124 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 14 | 674057ab250c | Moderate DR | 0.6047 | Severe DR | 0.3130 | 0.2918 | 0.5857 | 0.3687 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 15 | 3fa4f4d77177 | Moderate DR | 0.5926 | Severe DR | 0.3494 | 0.2432 | 0.5427 | 0.3958 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 16 | 929cd3867815 | No DR | 0.3638 | Mild DR | 0.3634 | 0.0004 | 0.6952 | 0.0053 | 5 | high | low_margin_boundary;moderate_entropy;second_choice_more_severe |
| 17 | 8f318a978844 | Severe DR | 0.4316 | Proliferative DR | 0.2987 | 0.1329 | 0.6885 | 0.7302 | 5 | high | low_margin_boundary;moderate_entropy;second_choice_more_severe |
| 18 | 365f8c01d994 | No DR | 0.4537 | Moderate DR | 0.3371 | 0.1166 | 0.6692 | 0.0045 | 5 | high | low_margin_boundary;moderate_entropy;second_choice_more_severe |
| 19 | efff2f1a35f5 | Moderate DR | 0.6748 | Proliferative DR | 0.2021 | 0.4727 | 0.5763 | 0.2934 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 20 | bca2bdc15fc5 | Moderate DR | 0.6371 | Proliferative DR | 0.2707 | 0.3664 | 0.5512 | 0.3568 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 21 | b9127e38d9b9 | Moderate DR | 0.6490 | Proliferative DR | 0.2643 | 0.3847 | 0.5461 | 0.3419 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 22 | 222f3ee3a1e8 | Moderate DR | 0.6444 | Proliferative DR | 0.2963 | 0.3481 | 0.5305 | 0.3373 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 23 | c1799a6f5c65 | Moderate DR | 0.6997 | Severe DR | 0.2112 | 0.4885 | 0.5295 | 0.2792 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 24 | 6fe67482bfae | Moderate DR | 0.6660 | Proliferative DR | 0.2780 | 0.3880 | 0.4923 | 0.3332 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 25 | ab1c20a94f3f | Moderate DR | 0.7149 | Proliferative DR | 0.2432 | 0.4717 | 0.4653 | 0.2566 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 26 | 51d0034d177d | Moderate DR | 0.7090 | Severe DR | 0.2369 | 0.4721 | 0.4640 | 0.2903 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 27 | e811f39a1243 | Moderate DR | 0.6803 | Severe DR | 0.2914 | 0.3889 | 0.4608 | 0.3125 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 28 | 2f42e20db938 | Moderate DR | 0.6670 | Severe DR | 0.3060 | 0.3610 | 0.4558 | 0.3322 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 29 | 1a90fad9ffa2 | Moderate DR | 0.7331 | Proliferative DR | 0.2286 | 0.5045 | 0.4455 | 0.2563 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 30 | 299086c6d1b5 | Moderate DR | 0.7273 | Severe DR | 0.2294 | 0.4979 | 0.4423 | 0.2713 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 31 | e83d315d8f98 | Moderate DR | 0.6954 | Severe DR | 0.2832 | 0.4123 | 0.4314 | 0.3042 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 32 | 71a39c660432 | Moderate DR | 0.7406 | Severe DR | 0.2251 | 0.5155 | 0.4225 | 0.2582 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 33 | 3a122851e526 | Moderate DR | 0.6919 | Severe DR | 0.3020 | 0.3899 | 0.4060 | 0.3049 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 34 | d10d315f123f | Proliferative DR | 0.3607 | Moderate DR | 0.3490 | 0.0117 | 0.8108 | 0.5354 | 4 | medium | low_margin_boundary;high_entropy |
| 35 | a3706ce27869 | Mild DR | 0.4708 | Moderate DR | 0.2969 | 0.1739 | 0.6820 | 0.0089 | 4 | medium | moderate_margin_boundary;moderate_entropy;second_choice_more_severe |
| 36 | 537e50fdf22e | Severe DR | 0.5300 | Proliferative DR | 0.3076 | 0.2223 | 0.6240 | 0.8376 | 4 | medium | moderate_margin_boundary;moderate_entropy;second_choice_more_severe |
| 37 | 27e4c800a449 | Mild DR | 0.5262 | Moderate DR | 0.3681 | 0.1580 | 0.6186 | 0.1016 | 4 | medium | moderate_margin_boundary;moderate_entropy;second_choice_more_severe |
| 38 | fa3e544a7401 | Moderate DR | 0.7076 | Proliferative DR | 0.1024 | 0.6052 | 0.6051 | 0.1902 | 4 | medium | moderate_entropy;weak_severe_undergrading_signal;weak_second_choice_more_severe |
| 39 | fce93caa4758 | Mild DR | 0.6260 | Moderate DR | 0.2192 | 0.4068 | 0.5859 | 0.1535 | 4 | medium | weak_severe_undergrading_signal;second_choice_more_severe |
| 40 | db4ed1e07aa3 | No DR | 0.4799 | Mild DR | 0.4619 | 0.0180 | 0.5656 | 0.0158 | 4 | medium | low_margin_boundary;second_choice_more_severe |
| 41 | f7fec8935126 | Moderate DR | 0.7140 | Proliferative DR | 0.1309 | 0.5832 | 0.5553 | 0.2537 | 4 | medium | potential_severe_undergrading_signal;weak_second_choice_more_severe |
| 42 | 12ce6a1a1f31 | Mild DR | 0.4881 | Moderate DR | 0.4748 | 0.0133 | 0.5347 | 0.0305 | 4 | medium | low_margin_boundary;second_choice_more_severe |
| 43 | 79540be95177 | Moderate DR | 0.6970 | Severe DR | 0.1575 | 0.5395 | 0.5186 | 0.3011 | 4 | medium | potential_severe_undergrading_signal;weak_second_choice_more_severe |
| 44 | 15cd5f52d300 | Moderate DR | 0.6990 | Proliferative DR | 0.1811 | 0.5179 | 0.5100 | 0.2999 | 4 | medium | potential_severe_undergrading_signal;weak_second_choice_more_severe |
| 45 | cb0cc98d7e35 | Mild DR | 0.6051 | Moderate DR | 0.3697 | 0.2353 | 0.4893 | 0.0098 | 4 | medium | moderate_margin_boundary;second_choice_more_severe;confident_but_close_decision |
| 46 | 6253f23229b1 | Moderate DR | 0.7310 | Severe DR | 0.2107 | 0.5203 | 0.4857 | 0.2367 | 4 | medium | weak_severe_undergrading_signal;second_choice_more_severe |
| 47 | 76cfe8967f7d | Moderate DR | 0.7310 | Severe DR | 0.2107 | 0.5203 | 0.4857 | 0.2367 | 4 | medium | weak_severe_undergrading_signal;second_choice_more_severe |
| 48 | dd19428c3d29 | Mild DR | 0.4937 | Moderate DR | 0.4905 | 0.0032 | 0.4829 | 0.0146 | 4 | medium | low_margin_boundary;second_choice_more_severe |
| 49 | d8cdb7d7283a | Mild DR | 0.5128 | Moderate DR | 0.4751 | 0.0376 | 0.4723 | 0.0111 | 4 | medium | low_margin_boundary;second_choice_more_severe |
| 50 | 1632c4311fc9 | Mild DR | 0.5474 | Moderate DR | 0.4400 | 0.1074 | 0.4719 | 0.0089 | 4 | medium | low_margin_boundary;second_choice_more_severe |
