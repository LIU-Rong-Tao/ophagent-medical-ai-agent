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
| 1 | 0e75d51152fc | No DR | 0.2967 | Moderate DR | 0.2727 | 0.0240 | 0.9571 | 0.2564 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 2 | ea15a290eb96 | No DR | 0.2793 | Mild DR | 0.2671 | 0.0122 | 0.9560 | 0.3256 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 3 | 253e96488cfb | Mild DR | 0.2838 | Moderate DR | 0.2556 | 0.0283 | 0.9504 | 0.3645 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 4 | 55034b1dbff2 | Moderate DR | 0.2904 | Proliferative DR | 0.2600 | 0.0304 | 0.9459 | 0.3628 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 5 | 6d7d26025122 | Moderate DR | 0.3082 | Proliferative DR | 0.2702 | 0.0379 | 0.9456 | 0.4763 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 6 | 35aa7f5c2ec0 | Mild DR | 0.3053 | Moderate DR | 0.2765 | 0.0288 | 0.9450 | 0.3287 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 7 | 1b8ad0afe9fb | Moderate DR | 0.2814 | Proliferative DR | 0.2530 | 0.0283 | 0.9382 | 0.4681 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 8 | 8bc6716c2238 | Moderate DR | 0.3460 | Severe DR | 0.2155 | 0.1304 | 0.9361 | 0.3998 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 9 | 7e0598cc88a0 | Mild DR | 0.3498 | Proliferative DR | 0.2588 | 0.0909 | 0.9318 | 0.3502 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 10 | 495255c7492f | Mild DR | 0.3353 | Moderate DR | 0.2735 | 0.0618 | 0.9197 | 0.2852 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 11 | 6ea07d19b4ce | Moderate DR | 0.3596 | Proliferative DR | 0.2863 | 0.0733 | 0.9190 | 0.3940 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 12 | 8f318a978844 | Moderate DR | 0.3292 | Proliferative DR | 0.2738 | 0.0555 | 0.9148 | 0.5044 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 13 | 3ddb86eb530e | Moderate DR | 0.3569 | Severe DR | 0.2329 | 0.1240 | 0.9146 | 0.4037 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 14 | 3b018e8b7303 | Moderate DR | 0.3639 | Severe DR | 0.2358 | 0.1281 | 0.9109 | 0.4050 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 15 | c3a82acb7d7a | Moderate DR | 0.3538 | Proliferative DR | 0.2665 | 0.0873 | 0.9097 | 0.3642 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 16 | 27e4c800a449 | Mild DR | 0.3013 | Proliferative DR | 0.2963 | 0.0050 | 0.9053 | 0.3976 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 17 | c0f15fe3b4b7 | Moderate DR | 0.3278 | Proliferative DR | 0.3129 | 0.0150 | 0.9046 | 0.4994 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 18 | 8a759f94613a | Moderate DR | 0.3627 | Proliferative DR | 0.2558 | 0.1068 | 0.9044 | 0.3770 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 19 | cd9e2190c73f | Moderate DR | 0.3828 | Proliferative DR | 0.2466 | 0.1361 | 0.9006 | 0.4081 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 20 | e32a359be36d | Moderate DR | 0.2965 | Proliferative DR | 0.2960 | 0.0005 | 0.8989 | 0.5392 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 21 | 59f3f70abddd | Mild DR | 0.3687 | Proliferative DR | 0.2641 | 0.1046 | 0.8776 | 0.3239 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 22 | a76b69e443ce | Moderate DR | 0.3508 | Proliferative DR | 0.3045 | 0.0463 | 0.8723 | 0.5150 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 23 | ae8424cdb029 | Moderate DR | 0.3776 | Severe DR | 0.2605 | 0.1171 | 0.8687 | 0.5025 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 24 | d10d315f123f | Moderate DR | 0.4125 | Proliferative DR | 0.2715 | 0.1409 | 0.8630 | 0.4073 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 25 | a6d45de20e4d | Moderate DR | 0.3630 | Proliferative DR | 0.3487 | 0.0142 | 0.8527 | 0.5104 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 26 | 8bad12d70368 | Moderate DR | 0.3504 | Proliferative DR | 0.2780 | 0.0724 | 0.8483 | 0.5548 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 27 | 01eb826f6467 | Moderate DR | 0.4164 | Proliferative DR | 0.2911 | 0.1253 | 0.8444 | 0.4097 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 28 | 4189d4e631ec | Moderate DR | 0.3751 | Proliferative DR | 0.3029 | 0.0722 | 0.8319 | 0.5260 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 29 | 5b644a403e1f | Moderate DR | 0.4117 | Proliferative DR | 0.3111 | 0.1006 | 0.8294 | 0.4839 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 30 | 5152bf091152 | Moderate DR | 0.4302 | Proliferative DR | 0.3134 | 0.1168 | 0.8239 | 0.4236 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 31 | 39134907127a | Moderate DR | 0.4335 | Proliferative DR | 0.3134 | 0.1201 | 0.8150 | 0.4637 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 32 | 857230f64a2e | Moderate DR | 0.3602 | Proliferative DR | 0.3559 | 0.0043 | 0.8034 | 0.5680 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 33 | 498f143c0374 | Moderate DR | 0.3725 | Proliferative DR | 0.3135 | 0.0590 | 0.7991 | 0.5661 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 34 | 5265dc9acdf8 | Moderate DR | 0.4256 | Proliferative DR | 0.3421 | 0.0835 | 0.7991 | 0.4744 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 35 | 1a90fad9ffa2 | Moderate DR | 0.4394 | Proliferative DR | 0.3014 | 0.1380 | 0.7946 | 0.4850 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 36 | 1e4b3b823b95 | Moderate DR | 0.4127 | Severe DR | 0.3210 | 0.0917 | 0.7930 | 0.5215 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 37 | 613028ede6a0 | Moderate DR | 0.3668 | Severe DR | 0.3332 | 0.0337 | 0.7855 | 0.5835 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 38 | 91cf56d3d1af | Moderate DR | 0.4094 | Proliferative DR | 0.2777 | 0.1317 | 0.7678 | 0.5473 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 39 | 537e50fdf22e | Moderate DR | 0.4181 | Severe DR | 0.3392 | 0.0790 | 0.7542 | 0.5393 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 40 | 290ecdba359f | Moderate DR | 0.4258 | Severe DR | 0.2887 | 0.1371 | 0.7538 | 0.5378 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 41 | ee74c3b177e0 | Moderate DR | 0.4105 | Severe DR | 0.3271 | 0.0835 | 0.7537 | 0.5528 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 42 | 57ce57a8cfb0 | Mild DR | 0.3761 | Proliferative DR | 0.2128 | 0.1633 | 0.9305 | 0.2985 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 43 | e12b67835e03 | No DR | 0.3757 | Moderate DR | 0.2079 | 0.1678 | 0.9141 | 0.3621 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 44 | 6e68e742f5bc | Mild DR | 0.3530 | Moderate DR | 0.3115 | 0.0415 | 0.8940 | 0.2323 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 45 | d2c2f02bb313 | Mild DR | 0.4215 | Proliferative DR | 0.2412 | 0.1804 | 0.8872 | 0.3063 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 46 | 4fa26d065ad3 | Mild DR | 0.4110 | Moderate DR | 0.2521 | 0.1589 | 0.8868 | 0.2508 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 47 | 69591ebb198d | Moderate DR | 0.4391 | Proliferative DR | 0.2125 | 0.2266 | 0.8606 | 0.3123 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 48 | a790a3b36390 | Moderate DR | 0.4097 | Proliferative DR | 0.2558 | 0.1539 | 0.8576 | 0.4649 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 49 | 63d217b059b6 | Moderate DR | 0.4375 | Proliferative DR | 0.2193 | 0.2182 | 0.8574 | 0.4210 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 50 | a3475dc3ac80 | Mild DR | 0.4204 | Moderate DR | 0.2621 | 0.1583 | 0.8558 | 0.2538 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
