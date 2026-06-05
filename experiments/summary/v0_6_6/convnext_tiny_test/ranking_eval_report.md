# v0.6.6 Pre-review Risk Ranking Evaluation

本报告使用真实标签进行后验验证。真实标签不参与预审风险排序。

## Overall

- Total samples: 1100
- Total errors: 205
- Overall error rate: 0.1864
- Total severe underestimation cases: 65

## Top-K Evaluation

| k | review_fraction | error_count | error_rate | enrichment_ratio | severe_underestimate_count | severe_underestimate_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.00909090909090909 | 8 | 0.8 | 4.2926829268292686 | 7 | 0.1076923076923077 |
| 20 | 0.01818181818181818 | 12 | 0.6 | 3.219512195121951 | 9 | 0.13846153846153847 |
| 50 | 0.045454545454545456 | 25 | 0.5 | 2.6829268292682924 | 21 | 0.3230769230769231 |
| 110 | 0.1 | 51 | 0.4636363636363636 | 2.4878048780487805 | 33 | 0.5076923076923077 |
| 220 | 0.2 | 87 | 0.39545454545454545 | 2.1219512195121952 | 43 | 0.6615384615384615 |
| 330 | 0.3 | 133 | 0.403030303030303 | 2.16260162601626 | 53 | 0.8153846153846154 |

## Risk Group Error Rate

| risk_level | n | error_count | error_rate | severe_underestimate_count |
| --- | --- | --- | --- | --- |
| high | 132 | 58 | 0.4393939393939394 | 36 |
| medium | 148 | 48 | 0.32432432432432434 | 14 |
| low | 820 | 99 | 0.12073170731707317 | 15 |

## Interpretation

- 如果 high risk 组错误率明显高于 overall error rate，说明预审风险排序具备初步有效性。
- 如果 Top 10% / 20% 的 enrichment ratio > 1，说明该排序优于随机抽样。
- 如果 severe_underestimate_recall 在较小 review_fraction 下较高，说明该规则对重症低估风险有价值。
- 如果上述趋势不明显，说明需要引入校准、TTA uncertainty、多模型 disagreement 或图像质量评分。
