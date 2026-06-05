# v0.6.6 Pre-review Risk Ranking Evaluation

本报告使用真实标签进行后验验证。真实标签不参与预审风险排序。

## Overall

- Total samples: 1100
- Total errors: 219
- Overall error rate: 0.1991
- Total severe underestimation cases: 62

## Top-K Evaluation

| k | review_fraction | error_count | error_rate | enrichment_ratio | severe_underestimate_count | severe_underestimate_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.00909090909090909 | 4 | 0.4 | 2.009132420091324 | 2 | 0.03225806451612903 |
| 20 | 0.01818181818181818 | 11 | 0.55 | 2.762557077625571 | 5 | 0.08064516129032258 |
| 50 | 0.045454545454545456 | 25 | 0.5 | 2.5114155251141552 | 12 | 0.1935483870967742 |
| 110 | 0.1 | 48 | 0.43636363636363634 | 2.191780821917808 | 21 | 0.3387096774193548 |
| 220 | 0.2 | 80 | 0.36363636363636365 | 1.82648401826484 | 41 | 0.6612903225806451 |
| 330 | 0.3 | 110 | 0.3333333333333333 | 1.67427701674277 | 50 | 0.8064516129032258 |

## Risk Group Error Rate

| risk_level | n | error_count | error_rate | severe_underestimate_count |
| --- | --- | --- | --- | --- |
| high | 329 | 109 | 0.331306990881459 | 50 |
| medium | 163 | 60 | 0.36809815950920244 | 9 |
| low | 608 | 50 | 0.08223684210526316 | 3 |

## Interpretation

- 如果 high risk 组错误率明显高于 overall error rate，说明预审风险排序具备初步有效性。
- 如果 Top 10% / 20% 的 enrichment ratio > 1，说明该排序优于随机抽样。
- 如果 severe_underestimate_recall 在较小 review_fraction 下较高，说明该规则对重症低估风险有价值。
- 如果上述趋势不明显，说明需要引入校准、TTA uncertainty、多模型 disagreement 或图像质量评分。
