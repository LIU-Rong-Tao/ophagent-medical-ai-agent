# v0.7.1b Go/No-Go Summary

## 输入

- Predictions: `experiments/summary/v0_7_1/external_dr_direct_inference_predictions.csv`
- Prediction rows: 3774
- Image identifier column: `image_key`
- Target: `VTDR miss = true_grade >= 3 and pred_grade < 3`
- Primary budget: Top20%
- Primary comparison: `gated_severe_prob_mass_only` vs `random_gate_only`

## 数据集与事件规模

| dataset    | n_unique_images | n_prediction_rows | n_backbones | event_count_min_per_backbone | event_count_mean_per_backbone | event_count_max_per_backbone |
| ---------- | --------------- | ----------------- | ----------- | ---------------------------- | ----------------------------- | ---------------------------- |
| IDRiD_data | 103             | 618               | 6           | 17                           | 24.0000                       | 29                           |
| MESSIDOR2  | 526             | 3156              | 6           | 21                           | 26.6667                       | 29                           |

## Primary comparison: Top20% recall difference

| dataset    | mean_delta_event_recall | ci95_low_delta_event_recall | ci95_high_delta_event_recall | win_rate_delta_gt_0 | mean_delta_residual_event_count | ci95_low_delta_residual_event_count | ci95_high_delta_residual_event_count | mean_delta_residual_event_rate | ci95_low_delta_residual_event_rate | ci95_high_delta_residual_event_rate |
| ---------- | ----------------------- | --------------------------- | ---------------------------- | ------------------- | ------------------------------- | ----------------------------------- | ------------------------------------ | ------------------------------ | ---------------------------------- | ----------------------------------- |
| IDRiD_data | 0.3385                  | 0.2195                      | 0.4742                       | 1.0000              | 7.8787                          | 5.3939                              | 10.2196                              | 0.0961                         | 0.0658                             | 0.1246                              |
| MESSIDOR2  | 0.6268                  | 0.5003                      | 0.7369                       | 1.0000              | 16.7396                         | 11.0143                             | 23.0891                              | 0.0399                         | 0.0262                             | 0.0550                              |

## Go/No-Go 判断

- 结论等级：**强证据**
- 判断依据：两个外部数据集差值点估计均为正，MESSIDOR2 的 95% CI 不跨 0，IDRiD_data 方向一致。

## 展示措辞

- 使用：**未进入优先复核区的残余危险事件**
- 不使用：自动放行区
- 病例页标注：公共数据集回顾性 grade-based proxy 示例，不是患者级临床判断。

## 边界

- 公共数据集回顾性评估。
- grade-based proxy，不是医生定义的患者级临床终点。
- 外部数据未用于重新拟合、重新选特征或重新标准化。
- 本结果不是临床部署验证。
