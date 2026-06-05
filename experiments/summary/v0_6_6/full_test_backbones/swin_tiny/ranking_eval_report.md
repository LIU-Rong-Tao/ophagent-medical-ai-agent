# v0.6.6 Pre-review Risk Ranking Evaluation

本报告使用真实标签进行后验验证。真实标签不参与预审风险排序。

## Overall

- Total samples: 1100
- Total errors: 188
- Overall error rate: 0.1709
- Total severe underestimation cases: 62

## Top-K Evaluation

| k | review_fraction | error_count | error_rate | enrichment_ratio | severe_underestimate_count | severe_underestimate_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.00909090909090909 | 5 | 0.5 | 2.925531914893617 | 5 | 0.08064516129032258 |
| 20 | 0.01818181818181818 | 9 | 0.45 | 2.6329787234042556 | 6 | 0.0967741935483871 |
| 50 | 0.045454545454545456 | 26 | 0.52 | 3.042553191489362 | 16 | 0.25806451612903225 |
| 110 | 0.1 | 51 | 0.4636363636363636 | 2.7127659574468086 | 23 | 0.3709677419354839 |
| 220 | 0.2 | 106 | 0.4818181818181818 | 2.8191489361702127 | 29 | 0.46774193548387094 |
| 330 | 0.3 | 141 | 0.42727272727272725 | 2.5 | 37 | 0.5967741935483871 |

## Risk Group Error Rate

| risk_level | n | error_count | error_rate | severe_underestimate_count |
| --- | --- | --- | --- | --- |
| high | 33 | 17 | 0.5151515151515151 | 14 |
| medium | 63 | 28 | 0.4444444444444444 | 9 |
| low | 1004 | 143 | 0.14243027888446216 | 39 |

## Interpretation

- 如果 high risk 组错误率明显高于 overall error rate，说明预审风险排序具备初步有效性。
- 如果 Top 10% / 20% 的 enrichment ratio > 1，说明该排序优于随机抽样。
- 如果 severe_underestimate_recall 在较小 review_fraction 下较高，说明该规则对重症低估风险有价值。
- 如果上述趋势不明显，说明需要引入校准、TTA uncertainty、多模型 disagreement 或图像质量评分。
