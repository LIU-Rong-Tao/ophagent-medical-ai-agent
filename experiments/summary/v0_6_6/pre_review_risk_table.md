# v0.6.6 Pre-review Risk Table

本表仅根据模型输出信号生成预审风险排序，不使用真实标签参与排序。

## 输入字段识别

- case_id: `case_id`
- image_path: `image_path`
- pred_label: `pred_label`
- confidence: `confidence`
- top2_label: `top2_label`
- top2_confidence: `top2_confidence`
- margin: `margin`
- entropy: `None`
- prob_columns: `[]`

## 排序阶段未使用但输入中存在的后验字段

- `correct`
- `gt_label`

## 风险排序预览

| review_priority_rank | case_id | pred_label | confidence | top2_label | top2_confidence | margin | entropy_norm | severe_prob_mass | pre_review_risk_score | pre_review_risk_level | risk_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6c3745a222da | Moderate DR | 0.4240 | Proliferative DR | 0.3338 | 0.0902 |  |  | 4 | medium | low_margin_boundary;second_choice_more_severe |
| 2 | b9127e38d9b9 | Moderate DR | 0.4154 | Severe DR | 0.3389 | 0.0765 |  |  | 4 | medium | low_margin_boundary;second_choice_more_severe |
| 3 | d9bbdc33db83 | Moderate DR | 0.6026 | Severe DR | 0.3330 | 0.2697 |  |  | 3 | medium | moderate_margin_boundary;second_choice_more_severe |
| 4 | ca30a97e9d13 | Mild DR | 0.5609 | Moderate DR | 0.2941 | 0.2667 |  |  | 3 | medium | moderate_margin_boundary;second_choice_more_severe |
| 5 | 686ed1dbae20 | Mild DR | 0.6598 | Moderate DR | 0.2416 | 0.4182 |  |  | 2 | low | second_choice_more_severe |
| 6 | 383e72af1955 | Moderate DR | 0.6438 | Severe DR | 0.2586 | 0.3851 |  |  | 2 | low | second_choice_more_severe |
| 7 | e93394175a19 | Severe DR | 0.6931 | Proliferative DR | 0.1666 | 0.5265 |  |  | 1 | low | weak_second_choice_more_severe |
| 8 | e52ed5c29c5e | Moderate DR | 0.6790 | Severe DR | 0.1725 | 0.5065 |  |  | 1 | low | weak_second_choice_more_severe |
| 9 | bba38f2294a3 | Proliferative DR | 0.6316 | Severe DR | 0.3486 | 0.2830 |  |  | 1 | low | moderate_margin_boundary |
| 10 | e1c02f6c3362 | No DR | 0.9995 | Mild DR | 0.0005 | 0.9991 |  |  | 0 | low | routine_low_risk |
| 11 | d39752cb6e57 | No DR | 0.9992 | Mild DR | 0.0008 | 0.9984 |  |  | 0 | low | routine_low_risk |
| 12 | c9e697117f3f | No DR | 0.9991 | Mild DR | 0.0009 | 0.9982 |  |  | 0 | low | routine_low_risk |
| 13 | 247e98aba610 | Proliferative DR | 0.8574 | Mild DR | 0.0723 | 0.7852 |  |  | 0 | low | routine_low_risk |
| 14 | 07929d32b5b3 | Mild DR | 0.7889 | Moderate DR | 0.0928 | 0.6961 |  |  | 0 | low | routine_low_risk |
| 15 | 58184d6fd087 | Mild DR | 0.7045 | No DR | 0.2168 | 0.4876 |  |  | 0 | low | routine_low_risk |
