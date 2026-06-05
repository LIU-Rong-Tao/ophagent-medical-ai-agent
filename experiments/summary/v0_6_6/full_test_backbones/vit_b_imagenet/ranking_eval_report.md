# v0.6.6 Pre-review Risk Ranking Evaluation

本报告使用真实标签进行后验验证。真实标签不参与预审风险排序。

## Overall

- Total samples: 1100
- Total errors: 200
- Overall error rate: 0.1818
- Total severe underestimation cases: 58

## Top-K Evaluation

| k | review_fraction | error_count | error_rate | enrichment_ratio | severe_underestimate_count | severe_underestimate_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.00909090909090909 | 6 | 0.6 | 3.3 | 5 | 0.08620689655172414 |
| 20 | 0.01818181818181818 | 9 | 0.45 | 2.475 | 6 | 0.10344827586206896 |
| 50 | 0.045454545454545456 | 22 | 0.44 | 2.42 | 14 | 0.2413793103448276 |
| 110 | 0.1 | 53 | 0.4818181818181818 | 2.65 | 23 | 0.39655172413793105 |
| 220 | 0.2 | 103 | 0.4681818181818182 | 2.575 | 32 | 0.5517241379310345 |
| 330 | 0.3 | 146 | 0.44242424242424244 | 2.4333333333333336 | 36 | 0.6206896551724138 |

## Risk Group Error Rate

| risk_level | n | error_count | error_rate | severe_underestimate_count |
| --- | --- | --- | --- | --- |
| high | 53 | 24 | 0.4528301886792453 | 16 |
| medium | 75 | 36 | 0.48 | 12 |
| low | 972 | 140 | 0.1440329218106996 | 30 |

## Interpretation

- 如果 high risk 组错误率明显高于 overall error rate，说明预审风险排序具备初步有效性。
- 如果 Top 10% / 20% 的 enrichment ratio > 1，说明该排序优于随机抽样。
- 如果 severe_underestimate_recall 在较小 review_fraction 下较高，说明该规则对重症低估风险有价值。
- 如果上述趋势不明显，说明需要引入校准、TTA uncertainty、多模型 disagreement 或图像质量评分。
