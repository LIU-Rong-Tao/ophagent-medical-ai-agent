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
| 1 | ab1c20a94f3f | Moderate DR | 0.3042 | Proliferative DR | 0.2921 | 0.0121 | 0.9082 | 0.4984 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 2 | 2a5a8b744f08 | Mild DR | 0.3820 | Moderate DR | 0.2982 | 0.0838 | 0.8798 | 0.2579 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 3 | d10d315f123f | Moderate DR | 0.3408 | Proliferative DR | 0.3123 | 0.0286 | 0.8795 | 0.4584 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 4 | ff77e8e5b5f3 | Moderate DR | 0.3774 | Proliferative DR | 0.3031 | 0.0743 | 0.8583 | 0.4581 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 5 | d93b61dc8f64 | Moderate DR | 0.3754 | Severe DR | 0.2992 | 0.0762 | 0.8527 | 0.5065 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 6 | 63b4d030b016 | Moderate DR | 0.3794 | Proliferative DR | 0.3098 | 0.0695 | 0.8515 | 0.4399 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 7 | fecf4c5ae84b | Moderate DR | 0.3833 | Proliferative DR | 0.3379 | 0.0454 | 0.8426 | 0.4512 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 8 | 3402124408ea | Moderate DR | 0.4202 | Proliferative DR | 0.2762 | 0.1441 | 0.8332 | 0.4567 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 9 | eb1d37b71fd1 | Moderate DR | 0.3629 | Proliferative DR | 0.3214 | 0.0415 | 0.8181 | 0.5617 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 10 | 8596a24a14bd | Moderate DR | 0.4084 | Proliferative DR | 0.2654 | 0.1431 | 0.8102 | 0.5131 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 11 | dbb2c63f6f08 | Moderate DR | 0.3695 | Proliferative DR | 0.3114 | 0.0581 | 0.8098 | 0.5648 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 12 | 3dbc90c7ee7d | Moderate DR | 0.3823 | Proliferative DR | 0.2913 | 0.0910 | 0.8093 | 0.5470 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 13 | 711d1480d2e3 | Moderate DR | 0.3767 | Proliferative DR | 0.3214 | 0.0553 | 0.8091 | 0.5474 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 14 | ff8a0b45c789 | Moderate DR | 0.4157 | Proliferative DR | 0.2672 | 0.1484 | 0.8085 | 0.5038 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 15 | 60e269e3e188 | Moderate DR | 0.4237 | Proliferative DR | 0.2938 | 0.1300 | 0.8018 | 0.4921 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 16 | 1c5ad36fb799 | Moderate DR | 0.4099 | Proliferative DR | 0.3327 | 0.0773 | 0.7978 | 0.5111 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 17 | a28bfb772f50 | Moderate DR | 0.4444 | Proliferative DR | 0.3010 | 0.1434 | 0.7967 | 0.4662 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 18 | 37c4dfe03aba | Moderate DR | 0.4050 | Proliferative DR | 0.3183 | 0.0866 | 0.7965 | 0.5298 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 19 | 698d6e422a80 | Moderate DR | 0.3838 | Proliferative DR | 0.3153 | 0.0685 | 0.7942 | 0.5594 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 20 | 36b5b3c9fb32 | Moderate DR | 0.3767 | Severe DR | 0.2909 | 0.0858 | 0.7939 | 0.5682 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 21 | a8652b2de23f | Moderate DR | 0.3946 | Proliferative DR | 0.3545 | 0.0401 | 0.7918 | 0.5292 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 22 | 15f440753916 | Moderate DR | 0.3552 | Proliferative DR | 0.3273 | 0.0279 | 0.7912 | 0.5920 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 23 | c9f0dc2c8b43 | Moderate DR | 0.4067 | Proliferative DR | 0.3211 | 0.0855 | 0.7905 | 0.5291 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 24 | 8c7c26c52a6c | Moderate DR | 0.3921 | Proliferative DR | 0.2946 | 0.0975 | 0.7879 | 0.5548 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 25 | 7c90ab025331 | Moderate DR | 0.4298 | Proliferative DR | 0.3029 | 0.1269 | 0.7878 | 0.5057 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 26 | 2f5c9cdfb333 | Moderate DR | 0.4013 | Severe DR | 0.3243 | 0.0770 | 0.7871 | 0.5413 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 27 | cd93a472e5cd | Moderate DR | 0.3697 | Proliferative DR | 0.3272 | 0.0425 | 0.7871 | 0.5798 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 28 | 2b21d293fdf2 | Moderate DR | 0.4010 | Proliferative DR | 0.2988 | 0.1022 | 0.7869 | 0.5416 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 29 | b60dbf9f0744 | Moderate DR | 0.4165 | Proliferative DR | 0.3024 | 0.1141 | 0.7863 | 0.5235 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 30 | c1799a6f5c65 | Moderate DR | 0.4278 | Proliferative DR | 0.2898 | 0.1381 | 0.7839 | 0.5123 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 31 | bca2bdc15fc5 | Moderate DR | 0.3620 | Proliferative DR | 0.3108 | 0.0513 | 0.7822 | 0.5926 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 32 | 8aab201c0691 | Moderate DR | 0.4127 | Proliferative DR | 0.2995 | 0.1132 | 0.7818 | 0.5315 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 33 | 18323d8f2470 | Moderate DR | 0.3753 | Proliferative DR | 0.3127 | 0.0626 | 0.7808 | 0.5771 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 34 | 6efa36d59ada | Moderate DR | 0.3786 | Severe DR | 0.3037 | 0.0749 | 0.7777 | 0.5773 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 35 | f901d460517c | Moderate DR | 0.3987 | Severe DR | 0.2940 | 0.1046 | 0.7773 | 0.5552 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 36 | ff52392372d3 | Moderate DR | 0.4330 | Proliferative DR | 0.3004 | 0.1326 | 0.7733 | 0.5111 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 37 | 06024377d573 | Moderate DR | 0.4305 | Proliferative DR | 0.2893 | 0.1412 | 0.7711 | 0.5191 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 38 | 66d2ca47aa44 | Moderate DR | 0.3896 | Severe DR | 0.3036 | 0.0860 | 0.7705 | 0.5694 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 39 | fc4d69128e7c | Moderate DR | 0.3786 | Proliferative DR | 0.2942 | 0.0844 | 0.7697 | 0.5827 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 40 | 3b018e8b7303 | Moderate DR | 0.4597 | Severe DR | 0.3258 | 0.1339 | 0.7667 | 0.4718 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 41 | 1da25637859b | Moderate DR | 0.4290 | Proliferative DR | 0.3109 | 0.1181 | 0.7666 | 0.5213 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 42 | 3ddb86eb530e | Moderate DR | 0.4596 | Severe DR | 0.3260 | 0.1336 | 0.7666 | 0.4720 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 43 | 299086c6d1b5 | Moderate DR | 0.3842 | Severe DR | 0.3542 | 0.0300 | 0.7662 | 0.5711 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 44 | 9a28d4e8aef0 | Moderate DR | 0.4249 | Proliferative DR | 0.3124 | 0.1125 | 0.7660 | 0.5287 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 45 | ef7a4ed8d5d1 | Moderate DR | 0.4424 | Severe DR | 0.3153 | 0.1271 | 0.7653 | 0.5047 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 46 | 1dfbede13143 | Moderate DR | 0.4274 | Proliferative DR | 0.3237 | 0.1037 | 0.7643 | 0.5220 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 47 | ff344e5c9341 | Moderate DR | 0.4438 | Proliferative DR | 0.3448 | 0.0991 | 0.7632 | 0.4891 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 48 | c3acf47700ea | Moderate DR | 0.3788 | Proliferative DR | 0.3238 | 0.0549 | 0.7624 | 0.5848 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 49 | 0415fc68b176 | Moderate DR | 0.3664 | Proliferative DR | 0.3021 | 0.0643 | 0.7614 | 0.6007 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
| 50 | b9127e38d9b9 | Moderate DR | 0.4052 | Proliferative DR | 0.3202 | 0.0850 | 0.7612 | 0.5549 | 9 | high | low_margin_boundary;high_entropy;potential_severe_undergrading_signal;second_choice_more_severe |
