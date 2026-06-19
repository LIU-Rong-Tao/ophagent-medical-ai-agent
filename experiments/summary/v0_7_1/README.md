# v0.7.1 External DR Direct Inference and Review Ranking

## 目录定位

本目录保存 v0.7.1 外部 DR 直接推理与复核排序评估结果。

v0.7.1 使用 v0.7.0 冻结的 APTOS-trained checkpoints，直接在 IDRiD_data / MESSIDOR2 test split 上推理，并基于推理概率评估复核排序信号的危险错误富集能力。

本目录结果不代表外部数据重训，也不代表临床部署验证。

## 文件说明

### 直接推理结果

- `external_dr_direct_inference_predictions.csv`
  - 六个 frozen backbones 在 IDRiD_data / MESSIDOR2 test split 上的逐图像预测结果。
  - 包含 `prob_0` 到 `prob_4`、`confidence`、`margin`、`entropy`、`expected_grade`、`severe_prob_mass`、`expected_gap`、`gated_severe_prob_mass` 等字段。

- `external_dr_classification_metrics.csv`
  - 外部分类迁移指标。
  - 包含 accuracy、macro-F1、weighted-F1、QWK 和 per-class recall。

- `external_dr_confusion_matrix.csv`
  - 外部分类混淆矩阵。

- `external_dr_direct_inference_summary.md`
  - 外部直接推理摘要。

### 复核排序结果

- `external_dr_review_ranking_metrics.csv`
  - Top10% / Top20% / Top30% 复核预算下的 ranking 指标。
  - 包含 event recall、flagged event rate、base event rate、enrichment ratio、residual event count、low-risk NPV 等。

- `external_dr_review_ranking_table.csv`
  - Top-K flagged 样本与 target event residual 样本的紧凑表格。

- `external_dr_review_ranking_summary.md`
  - 复核排序评估摘要。

- `external_dr_review_ranking_key_findings.md`
  - v0.7.1 外部复核排序关键发现。

## 主要结论

外部分类迁移存在明显域迁移压力，尤其 MESSIDOR2 上多模型预测分布偏向 0 类。因此，本版本结果应解释为 external frozen-checkpoint error enrichment / residual risk analysis。

在该背景下：

- `expected_gap_only` 对 `large_undergrading` 有一定富集能力，但外部迁移不稳定。
- `gated_severe_prob_mass_only` 对 `vision_threatening_dr_miss` 显示更明确的外部排序趋势，是 v0.7.1 最强结果。

## 解释边界

本目录结果不能用于声称模型已完成临床泛化验证。若分类迁移不足，ranking 结果应作为 failure analysis 和 residual risk analysis，而不是 deployment validation。

## v0.7.1b 后续补充说明

v0.7.1 的排序结果是在 random gate-only 对照和 clustered CI 之前形成的外部压力测试结果。

v0.7.1b 后续补充显示：在 Top20% 预算下，`gated_severe_prob_mass_only` 相比 `random_gate_only_expected`，为 VTDR miss 提供了额外排序信息。该结论限于当前公共数据集、grade-based proxy 和六个 frozen APTOS backbones。
