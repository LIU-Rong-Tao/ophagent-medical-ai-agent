# v0.6.6 Pre-review Risk Ranking Evaluation

本报告使用真实标签进行后验验证。真实标签不参与预审风险排序。

## Overall

- Total samples: 15
- Total errors: 4
- Overall error rate: 0.2667
- Total severe underestimation cases: 3

## Top-K Evaluation

| k | review_fraction | error_count | error_rate | enrichment_ratio | severe_underestimate_count | severe_underestimate_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.13333333333333333 | 1 | 0.5 | 1.875 | 1 | 0.3333333333333333 |
| 3 | 0.2 | 1 | 0.3333333333333333 | 1.25 | 1 | 0.3333333333333333 |
| 4 | 0.26666666666666666 | 1 | 0.25 | 0.9375 | 1 | 0.3333333333333333 |
| 10 | 0.6666666666666666 | 4 | 0.4 | 1.5 | 3 | 1.0 |
| 15 | 1.0 | 4 | 0.26666666666666666 | 1.0 | 3 | 1.0 |

## Risk Group Error Rate

| risk_level | n | error_count | error_rate | severe_underestimate_count |
| --- | --- | --- | --- | --- |
| medium | 4 | 1 | 0.25 | 1 |
| low | 11 | 3 | 0.2727272727272727 | 2 |

## Interpretation

- 如果 high risk 组错误率明显高于 overall error rate，说明预审风险排序具备初步有效性。
- 如果 Top 10% / 20% 的 enrichment ratio > 1，说明该排序优于随机抽样。
- 如果 severe_underestimate_recall 在较小 review_fraction 下较高，说明该规则对重症低估风险有价值。
- 如果上述趋势不明显，说明需要引入校准、TTA uncertainty、多模型 disagreement 或图像质量评分。
