# v0.7.1 外部 DR 复核排序关键发现

## 版本定位

v0.7.1 使用 v0.7.0 冻结的 APTOS-trained checkpoints，直接在 IDRiD_data / MESSIDOR2 test split 上推理，并在不使用外部 train / val 训练或调参的前提下评估复核排序信号。

本阶段不是外部数据重训，也不是临床泛化成功验证；结果应解释为 frozen APTOS checkpoints 在外部 DR 数据上的错误富集与 residual risk analysis。

## 分类迁移背景

外部直接推理显示模型存在明显域迁移压力。

- IDRiD_data 上最高 accuracy 为 0.4563，最高 QWK 约 0.5835。
- MESSIDOR2 上 accuracy 约 0.56–0.61，但 macro-F1 仅约 0.27–0.33，且多模型预测分布明显偏向 0 类。

因此，review ranking 结果必须在分类迁移不足的背景下解释，不能包装为分类模型已经充分泛化后的临床验证。

## Primary target 1：large_undergrading

事件定义：

- large_undergrading = true_grade - pred_grade >= 2

冻结 ranking signal：

- expected_gap_only = expected_grade - pred_grade

Top20% 结果显示，expected_gap_only 在 IDRiD_data / MESSIDOR2 上具有一定危险低估富集能力，但外部迁移不稳定。

- IDRiD_data：event recall 约 0.2778–0.5833，enrichment ratio 约 1.36–2.86。
- MESSIDOR2：event recall 约 0.2532–0.5065，enrichment ratio 约 1.26–2.51。

结论：expected_gap_only 对 large_undergrading 仍优于随机复核，但外部稳定性弱于 vision-threatening miss 目标。该结果适合作为危险低估的辅助复核信号，而不应被解释为强泛化结论。

## Primary target 2：vision_threatening_dr_miss

事件定义：

- vision_threatening_dr_miss = true_grade >= 3 and pred_grade < 3

冻结 ranking signal：

- gated_severe_prob_mass_only = P(grade 3) + P(grade 4), gated by pred_grade <= 2

Top20% 结果显示，gated_severe_prob_mass_only 在两个外部数据集上均表现出重症漏检排序趋势。

- IDRiD_data：event recall 约 0.3750–0.6957，enrichment ratio 约 1.84–3.41。
- MESSIDOR2：event recall 约 0.7586–0.8889，enrichment ratio 约 3.76–4.41。
- MESSIDOR2 上 low-risk NPV 约 0.9833–0.9929，说明 Top20% 复核后自动放行区残余 vision-threatening miss 较少。

结论：即使外部分类迁移存在压力，gated_severe_prob_mass_only 仍能在外部 DR 数据上稳定富集 vision-threatening miss。这是 v0.7.1 最强的外部错误富集证据。

## 主要结论

v0.7.1 支持以下结论：

1. frozen APTOS checkpoints 在 IDRiD_data / MESSIDOR2 上存在域迁移压力，分类泛化本身并不充分。
2. 在分类迁移不充分的背景下，severity-aware review ranking 仍能发现一部分高风险错误。
3. gated_severe_prob_mass_only 对 vision-threatening miss 的外部富集能力明显强于 expected_gap_only 对 large undergrading 的外部富集能力。
4. v0.6.x 中“不同临床风险目标需要不同 ranking signal”的结论在外部数据上得到初步支持。
5. 当前结果应称为 external frozen-checkpoint error enrichment / residual risk analysis，而不是 clinical deployment validation。

## 后续建议

v0.7.1 后续可继续补充：

- Top10% / Top30% risk-coverage curve；
- per-image across-backbone aggregation；
- residual miss case gallery；
- external review ranking 与分类迁移强弱之间的关系；
- v0.7.2 若进行目标数据重训，必须处理 IDRiD_data 内部 train/test md5 重复问题。
