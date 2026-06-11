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
| 1 | 6253f23229b1 | Moderate DR | 0.3172 | Severe DR | 0.2897 | 0.0275 | 0.8635 | 0.4051 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 2 | 76cfe8967f7d | Moderate DR | 0.3172 | Severe DR | 0.2897 | 0.0275 | 0.8635 | 0.4051 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 3 | 7e0598cc88a0 | Mild DR | 0.3486 | Proliferative DR | 0.3452 | 0.0035 | 0.8428 | 0.4017 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 4 | fce93caa4758 | Mild DR | 0.3552 | Moderate DR | 0.2802 | 0.0750 | 0.8292 | 0.3511 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 5 | 6c3745a222da | Moderate DR | 0.4240 | Proliferative DR | 0.3338 | 0.0903 | 0.8164 | 0.4337 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 6 | eadc57064154 | Moderate DR | 0.4240 | Proliferative DR | 0.3338 | 0.0902 | 0.8164 | 0.4337 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 7 | 57ce57a8cfb0 | Mild DR | 0.4111 | Moderate DR | 0.2969 | 0.1142 | 0.8125 | 0.2651 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 8 | e10190a9d52f | Moderate DR | 0.3701 | Proliferative DR | 0.2933 | 0.0768 | 0.8062 | 0.5464 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 9 | d271d3a2b552 | Moderate DR | 0.4106 | Severe DR | 0.3133 | 0.0973 | 0.8011 | 0.4823 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 10 | 08a3875063c3 | Moderate DR | 0.3462 | Proliferative DR | 0.3087 | 0.0375 | 0.7665 | 0.6107 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 11 | 8f318a978844 | Moderate DR | 0.3870 | Severe DR | 0.3671 | 0.0200 | 0.7639 | 0.5483 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 12 | 6a91eb157f47 | Moderate DR | 0.4282 | Severe DR | 0.3317 | 0.0965 | 0.7629 | 0.5031 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 13 | a11bf2edd470 | Mild DR | 0.3699 | Moderate DR | 0.2136 | 0.1563 | 0.9433 | 0.2590 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 14 | 7e9458de5707 | Mild DR | 0.4281 | Proliferative DR | 0.2406 | 0.1875 | 0.8300 | 0.3146 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 15 | 8a759f94613a | Mild DR | 0.4015 | Moderate DR | 0.3169 | 0.0846 | 0.7987 | 0.2386 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 16 | 0ad7f631dedb | Mild DR | 0.4160 | Moderate DR | 0.3617 | 0.0543 | 0.7632 | 0.2133 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 17 | 1e4b3b823b95 | Moderate DR | 0.4732 | Severe DR | 0.3121 | 0.1611 | 0.7623 | 0.4522 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 18 | 71f6a6e4620a | Mild DR | 0.4109 | Moderate DR | 0.3835 | 0.0274 | 0.7598 | 0.1874 | 8 | high | low_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 19 | 79540be95177 | Moderate DR | 0.4872 | Proliferative DR | 0.2404 | 0.2468 | 0.7581 | 0.4363 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 20 | 0415fc68b176 | Moderate DR | 0.5094 | Severe DR | 0.2097 | 0.2998 | 0.7522 | 0.4115 | 8 | high | moderate_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 21 | 910bfd38e2f5 | Moderate DR | 0.4288 | Severe DR | 0.3243 | 0.1045 | 0.7392 | 0.5314 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 22 | 61bbc11fe503 | Moderate DR | 0.4376 | Severe DR | 0.3509 | 0.0867 | 0.7362 | 0.5083 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 23 | b9127e38d9b9 | Moderate DR | 0.4154 | Severe DR | 0.3389 | 0.0765 | 0.7297 | 0.5546 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 24 | 2c2aa057afc5 | Moderate DR | 0.4255 | Severe DR | 0.3781 | 0.0474 | 0.7151 | 0.5396 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 25 | 3fa4f4d77177 | Moderate DR | 0.4518 | Severe DR | 0.3756 | 0.0763 | 0.7091 | 0.4957 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 26 | bf7047dc683c | Moderate DR | 0.4706 | Severe DR | 0.3630 | 0.1076 | 0.6985 | 0.4851 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 27 | 3b185ac445d0 | Moderate DR | 0.4807 | Severe DR | 0.4353 | 0.0454 | 0.6106 | 0.4854 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 28 | 51d780864365 | Moderate DR | 0.5077 | Severe DR | 0.3996 | 0.1081 | 0.6103 | 0.4726 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 29 | 26453eb7e989 | Moderate DR | 0.4887 | Severe DR | 0.4360 | 0.0527 | 0.6010 | 0.4692 | 8 | high | low_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 30 | 55fd453001cc | Moderate DR | 0.3879 | Mild DR | 0.3185 | 0.0694 | 0.8241 | 0.2811 | 7 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal |
| 31 | 4818672273af | Moderate DR | 0.3882 | Mild DR | 0.3147 | 0.0735 | 0.8208 | 0.2863 | 7 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal |
| 32 | a6d45de20e4d | Mild DR | 0.5128 | Moderate DR | 0.2479 | 0.2650 | 0.7697 | 0.2138 | 7 | high | moderate_margin_boundary;high_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 33 | 7c90ab025331 | Moderate DR | 0.4525 | Proliferative DR | 0.2616 | 0.1908 | 0.7441 | 0.5051 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 34 | 530d78467615 | Mild DR | 0.4675 | Moderate DR | 0.3577 | 0.1098 | 0.7413 | 0.1548 | 7 | high | low_margin_boundary;moderate_entropy;weak_severe_undergrading_signal;second_choice_more_severe |
| 35 | d0ffa0425ef1 | Moderate DR | 0.5026 | Severe DR | 0.2698 | 0.2328 | 0.7248 | 0.4450 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 36 | 423abbaa5fad | Moderate DR | 0.5144 | Severe DR | 0.2905 | 0.2240 | 0.7167 | 0.4161 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 37 | c0202976c670 | Moderate DR | 0.5291 | Severe DR | 0.3043 | 0.2248 | 0.6889 | 0.4173 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 38 | 290ecdba359f | Moderate DR | 0.5256 | Severe DR | 0.3179 | 0.2077 | 0.6735 | 0.4372 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 39 | 94ef1d14597f | Moderate DR | 0.5129 | Severe DR | 0.3411 | 0.1718 | 0.6720 | 0.4462 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 40 | cd3fd04d72f5 | Moderate DR | 0.5193 | Severe DR | 0.3112 | 0.2081 | 0.6702 | 0.4554 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 41 | eb1d37b71fd1 | Moderate DR | 0.5551 | Severe DR | 0.2978 | 0.2573 | 0.6653 | 0.3996 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 42 | 2cbfc6182ba2 | Moderate DR | 0.5676 | Severe DR | 0.3067 | 0.2609 | 0.6472 | 0.3595 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 43 | 6e92b1c5ac8e | Moderate DR | 0.5581 | Severe DR | 0.3216 | 0.2365 | 0.6428 | 0.3902 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 44 | 9c52b87d01f1 | Moderate DR | 0.5581 | Severe DR | 0.3216 | 0.2365 | 0.6428 | 0.3902 | 7 | high | moderate_margin_boundary;moderate_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 45 | 4189d4e631ec | Moderate DR | 0.5361 | Severe DR | 0.4012 | 0.1348 | 0.5727 | 0.4351 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 46 | c3acf47700ea | Moderate DR | 0.5146 | Severe DR | 0.4427 | 0.0719 | 0.5392 | 0.4704 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 47 | e83d315d8f98 | Moderate DR | 0.5345 | Severe DR | 0.4227 | 0.1118 | 0.5380 | 0.4462 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 48 | df3adfd6ba36 | Moderate DR | 0.5245 | Severe DR | 0.4482 | 0.0763 | 0.5089 | 0.4600 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 49 | e4730ddde408 | Moderate DR | 0.5348 | Severe DR | 0.4369 | 0.0979 | 0.5072 | 0.4562 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
| 50 | c7c0470bcf87 | Moderate DR | 0.5315 | Severe DR | 0.4415 | 0.0899 | 0.5065 | 0.4586 | 7 | high | low_margin_boundary;potential_severe_undergrading_signal;second_choice_more_severe |
