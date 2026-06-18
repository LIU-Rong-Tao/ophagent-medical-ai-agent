# v0.6.8 Learned Deferral Score

This experiment evaluates a lightweight supervised learned deferral score for clinical-risk-aware Top-K review ranking.

## Scope

- Exploratory internal grouped cross-validation.
- 1100 unique images, producing 6600 backbone-specific prediction records.

## 重要计数口径

本目录中的 263、391、430 等事件总数，均指 6600 条骨干网络特异预测记录（backbone-specific prediction records）中的事件记录数，不是独立患者数，也不是唯一图像（unique images）数量。

180/263、297/391 等 captured/total 结果同样是按预测记录统计的捕获结果，不是患者级或图像级捕获结果。

后续如果做置信区间或显著性检验，应以 `image_key` 作为聚类单位，而不能把 6600 条预测记录当作相互独立样本。
- Pooled-backbone training with per-backbone test reporting.
- Not an independent clinical validation.
- Not an unseen-backbone generalization experiment.

## Split

- `StratifiedGroupKFold`
- `groups = image_key`
- 5 folds
- `shuffle=True`
- `random_state=42`

## Primary Targets

- `large_undergrading`
- `vision_threatening_dr_miss`

## Secondary Target

- `dangerous_undergrading`

## Learned Model

- L2 Logistic Regression
- `class_weight=balanced`
- fixed hyperparameters
- `decision_function` used as learned review score

## Main Metrics

- Top10% / Top20% / Top30% recall
- captured / total
- residual dangerous error count
- precision@K
- lift vs random
- AUROC / AUPRC as auxiliary metrics

## Outputs

- `learned_deferral_tradeoff.csv`
- `learned_deferral_cv_summary.csv`
- `learned_deferral_fold_diagnostics.csv`
- `learned_deferral_winner_count.csv`
- `learned_deferral_auc_summary.csv`
- `learned_deferral_fold_predictions.csv`
- `learned_deferral_key_findings.md`
