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
| 1 | 7e0598cc88a0 | Moderate DR | 0.3810 | Severe DR | 0.2947 | 0.0863 | 0.8106 | 0.5072 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 2 | 01eb826f6467 | Moderate DR | 0.4133 | Proliferative DR | 0.3029 | 0.1104 | 0.7835 | 0.5035 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 3 | 78bcdffb8785 | Moderate DR | 0.4473 | Severe DR | 0.3450 | 0.1023 | 0.7387 | 0.4119 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 4 | 222f3ee3a1e8 | Moderate DR | 0.4813 | Proliferative DR | 0.3787 | 0.1026 | 0.6777 | 0.4680 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 5 | e93394175a19 | Moderate DR | 0.4246 | Proliferative DR | 0.3340 | 0.0906 | 0.6695 | 0.5748 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 6 | cd3fd04d72f5 | Moderate DR | 0.4723 | Proliferative DR | 0.3689 | 0.1034 | 0.6343 | 0.5267 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 7 | 76cfe8967f7d | Moderate DR | 0.3221 | Mild DR | 0.3068 | 0.0153 | 0.8366 | 0.2876 | 7 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal |
| 8 | 6253f23229b1 | Moderate DR | 0.3221 | Mild DR | 0.3068 | 0.0153 | 0.8366 | 0.2876 | 7 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal |
| 9 | 3fa4f4d77177 | Moderate DR | 0.4969 | Severe DR | 0.2807 | 0.2161 | 0.6932 | 0.4792 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 10 | 1dfbede13143 | Moderate DR | 0.4553 | Severe DR | 0.2919 | 0.1634 | 0.6851 | 0.5363 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 11 | 9b32e8ef0ca0 | Moderate DR | 0.5432 | Severe DR | 0.2552 | 0.2879 | 0.6824 | 0.4211 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 12 | 86b3a7929bec | Moderate DR | 0.5366 | Proliferative DR | 0.2738 | 0.2628 | 0.6318 | 0.4611 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 13 | 362c4a96cebb | Moderate DR | 0.5633 | Severe DR | 0.3131 | 0.2502 | 0.6240 | 0.4146 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 14 | 2c2aa057afc5 | Moderate DR | 0.5565 | Proliferative DR | 0.2687 | 0.2878 | 0.6148 | 0.4427 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 15 | 63c3c571b8ee | Moderate DR | 0.5231 | Severe DR | 0.3414 | 0.1818 | 0.6142 | 0.4748 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 16 | c9f0dc2c8b43 | Moderate DR | 0.5528 | Proliferative DR | 0.2805 | 0.2723 | 0.6141 | 0.4464 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 17 | 9fab29e69a6b | Moderate DR | 0.5186 | Severe DR | 0.4039 | 0.1147 | 0.5749 | 0.4759 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 18 | 8bc6716c2238 | Moderate DR | 0.3916 | No DR | 0.2695 | 0.1221 | 0.9025 | 0.2053 | 6 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal |
| 19 | 4661006f3ba6 | Moderate DR | 0.5369 | Proliferative DR | 0.2112 | 0.3257 | 0.7262 | 0.2813 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 20 | 63d217b059b6 | Moderate DR | 0.5469 | Severe DR | 0.2233 | 0.3235 | 0.7227 | 0.2818 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 21 | 13063d1bc4ea | Moderate DR | 0.5990 | Severe DR | 0.2137 | 0.3853 | 0.6826 | 0.2853 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 22 | a8652b2de23f | Moderate DR | 0.5945 | Severe DR | 0.2216 | 0.3729 | 0.6731 | 0.3021 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 23 | 5a03fe3ed15c | Moderate DR | 0.5983 | Severe DR | 0.2040 | 0.3943 | 0.6685 | 0.3425 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 24 | d66ccb75ada1 | Moderate DR | 0.5641 | Severe DR | 0.2414 | 0.3227 | 0.6562 | 0.2634 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 25 | 735836b1ffa6 | Moderate DR | 0.5855 | Severe DR | 0.2512 | 0.3343 | 0.6372 | 0.2720 | 6 | high | moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 26 | 82bb8a01935f | Moderate DR | 0.6062 | Proliferative DR | 0.3258 | 0.2804 | 0.5521 | 0.3785 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 27 | a80dab8eddf4 | Moderate DR | 0.5692 | Severe DR | 0.3708 | 0.1984 | 0.5517 | 0.4190 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 28 | 77baa08a1345 | Moderate DR | 0.5713 | Proliferative DR | 0.3729 | 0.1983 | 0.5335 | 0.4265 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 29 | 9b0eb9f41da4 | Moderate DR | 0.5542 | Severe DR | 0.3917 | 0.1625 | 0.5317 | 0.4451 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 30 | 03676c71ed1b | Moderate DR | 0.6043 | Proliferative DR | 0.3383 | 0.2660 | 0.5315 | 0.3896 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 31 | 7356dd08b0ae | Moderate DR | 0.5697 | Severe DR | 0.3776 | 0.1921 | 0.5281 | 0.4289 | 6 | high | moderate_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 32 | e2a47a74e6e1 | Severe DR | 0.4172 | Proliferative DR | 0.2592 | 0.1580 | 0.8033 | 0.6764 | 5 | high | moderate_margin_boundary;high_entropy;second_choice_more_severe |
| 33 | 0bf37ca3156a | Severe DR | 0.4597 | Proliferative DR | 0.2573 | 0.2023 | 0.7692 | 0.7170 | 5 | high | moderate_margin_boundary;high_entropy;second_choice_more_severe |
| 34 | 0369f3efe69b | Mild DR | 0.4721 | Moderate DR | 0.3894 | 0.0828 | 0.6605 | 0.1304 | 5 | high | low_margin_boundary;moderate_entropy;second_choice_more_severe |
| 35 | a3475dc3ac80 | Mild DR | 0.4604 | Moderate DR | 0.4192 | 0.0412 | 0.6572 | 0.1150 | 5 | high | low_margin_boundary;moderate_entropy;second_choice_more_severe |
| 36 | 6d7d26025122 | Mild DR | 0.5115 | Moderate DR | 0.3670 | 0.1446 | 0.6504 | 0.0970 | 5 | high | low_margin_boundary;moderate_entropy;second_choice_more_severe |
| 37 | a2ddabee14e9 | Mild DR | 0.6131 | Proliferative DR | 0.2310 | 0.3821 | 0.6213 | 0.2489 | 5 | high | moderate_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 38 | d0ffa0425ef1 | Moderate DR | 0.6256 | Proliferative DR | 0.1958 | 0.4298 | 0.6166 | 0.3502 | 5 | high | moderate_entropy;potential_severe_undergrading_signal;weak_second_choice_more_severe |
| 39 | b8f1b30877db | Moderate DR | 0.6592 | Severe DR | 0.2175 | 0.4417 | 0.5874 | 0.2991 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 40 | b60dbf9f0744 | Moderate DR | 0.6282 | Severe DR | 0.2488 | 0.3794 | 0.5849 | 0.3574 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 41 | 37de05ef12a5 | Moderate DR | 0.6143 | Severe DR | 0.2381 | 0.3762 | 0.5786 | 0.3844 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 42 | 8596a24a14bd | Moderate DR | 0.6620 | Severe DR | 0.2332 | 0.4287 | 0.5729 | 0.2938 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 43 | 0180bfa26c0b | Moderate DR | 0.6478 | Proliferative DR | 0.2489 | 0.3989 | 0.5726 | 0.3256 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 44 | 06024377d573 | Moderate DR | 0.6216 | Severe DR | 0.2873 | 0.3343 | 0.5641 | 0.3668 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 45 | 66d2ca47aa44 | Moderate DR | 0.6147 | Severe DR | 0.3140 | 0.3007 | 0.5562 | 0.3647 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 46 | 18323d8f2470 | Moderate DR | 0.6131 | Severe DR | 0.3085 | 0.3046 | 0.5418 | 0.3848 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 47 | 15f440753916 | Moderate DR | 0.6235 | Severe DR | 0.3228 | 0.3007 | 0.5306 | 0.3546 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 48 | ce207b69ff37 | Moderate DR | 0.6712 | Severe DR | 0.2232 | 0.4480 | 0.5266 | 0.3274 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 49 | bf7047dc683c | Moderate DR | 0.6588 | Severe DR | 0.2787 | 0.3801 | 0.5249 | 0.3210 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
| 50 | ee059945b08a | Moderate DR | 0.7083 | Severe DR | 0.2525 | 0.4557 | 0.4648 | 0.2763 | 5 | high | potential_severe_undergrading_signal;second_choice_more_severe |
