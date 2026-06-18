# v0.6.8 Key Findings

## Experiment Scope

- This is an exploratory internal grouped cross-validation experiment.
- The model is trained with pooled-backbone prediction records and reported with per-backbone test performance.
- This is not an unseen-backbone generalization experiment.
- The dataset contains 1100 unique images and 6600 backbone-specific prediction records.

## 重要计数口径

本目录中的 263、391、430 等事件总数，均指 6600 条骨干网络特异预测记录（backbone-specific prediction records）中的事件记录数，不是独立患者数，也不是唯一图像（unique images）数量。

180/263、297/391 等 captured/total 结果同样是按预测记录统计的捕获结果，不是患者级或图像级捕获结果。

后续如果做置信区间或显著性检验，应以 `image_key` 作为聚类单位，而不能把 6600 条预测记录当作相互独立样本。
- Top-K review metrics are the primary endpoint; AUROC/AUPRC are auxiliary metrics.

## Target: `large_undergrading`

- Top10% best mean recall: `learned_logistic` (mean recall across backbones = 0.4614, mean lift = 4.5336).
- Top20% best mean recall: `expected_gap_only` (mean recall across backbones = 0.6752, mean lift = 3.3474).
- Top30% best mean recall: `top2_more_severe_only` (mean recall across backbones = 0.7720, mean lift = 2.5578).

## Target: `vision_threatening_dr_miss`

- Top10% best mean recall: `gated_severe_prob_mass_only` (mean recall across backbones = 0.5280, mean lift = 5.1423).
- Top20% best mean recall: `gated_severe_prob_mass_only` (mean recall across backbones = 0.7591, mean lift = 3.7451).
- Top30% best mean recall: `learned_logistic` (mean recall across backbones = 0.9073, mean lift = 2.9967).

## Target: `dangerous_undergrading`

- Top10% best mean recall: `gated_severe_prob_mass_only` (mean recall across backbones = 0.4872, mean lift = 4.8268).
- Top20% best mean recall: `gated_severe_prob_mass_only` (mean recall across backbones = 0.7072, mean lift = 3.5201).
- Top30% best mean recall: `gated_severe_prob_mass_only` (mean recall across backbones = 0.8347, mean lift = 2.7653).

## Boundary Notes

- `learned_logistic` uses `decision_function` as a learned review score, not a calibrated probability.
- `StratifiedGroupKFold` only approximately balances positive events because one image can have different event labels across backbones.
- Fold balance should be checked in `learned_deferral_fold_diagnostics.csv`.
- Future confidence intervals or significance tests should use image_key-clustered resampling, not row-wise independent bootstrap.
