# v0.6.6 Pre-review Risk Ranking Evaluation

本报告使用真实标签进行后验验证。真实标签不参与预审风险排序。

## Overall

- Total samples: 1100
- Total errors: 221
- Overall error rate: 0.2009
- Total severe underestimation cases: 84

## Top-K Evaluation

| k | review_fraction | error_count | error_rate | enrichment_ratio | severe_underestimate_count | severe_underestimate_recall |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.00909090909090909 | 6 | 0.6 | 2.986425339366516 | 2 | 0.023809523809523808 |
| 20 | 0.01818181818181818 | 12 | 0.6 | 2.986425339366516 | 6 | 0.07142857142857142 |
| 50 | 0.045454545454545456 | 27 | 0.54 | 2.6877828054298645 | 16 | 0.19047619047619047 |
| 110 | 0.1 | 54 | 0.4909090909090909 | 2.4434389140271495 | 32 | 0.38095238095238093 |
| 220 | 0.2 | 96 | 0.43636363636363634 | 2.171945701357466 | 54 | 0.6428571428571429 |
| 330 | 0.3 | 143 | 0.43333333333333335 | 2.1568627450980395 | 66 | 0.7857142857142857 |

## Risk Group Error Rate

| risk_level | n | error_count | error_rate | severe_underestimate_count |
| --- | --- | --- | --- | --- |
| high | 265 | 112 | 0.4226415094339623 | 66 |
| medium | 198 | 70 | 0.35353535353535354 | 14 |
| low | 637 | 39 | 0.061224489795918366 | 4 |

## Interpretation

- 如果 high risk 组错误率明显高于 overall error rate，说明预审风险排序具备初步有效性。
- 如果 Top 10% / 20% 的 enrichment ratio > 1，说明该排序优于随机抽样。
- 如果 severe_underestimate_recall 在较小 review_fraction 下较高，说明该规则对重症低估风险有价值。
- 如果上述趋势不明显，说明需要引入校准、TTA uncertainty、多模型 disagreement 或图像质量评分。
