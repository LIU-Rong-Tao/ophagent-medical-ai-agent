# v0.6.8b Robustness and Mechanism Audit

本目录保存 v0.6.8b 的稳健性与机制审计结果。

## 目录用途

v0.6.8b 不引入新模型，主要检查 v0.6.8 learned deferral score 的稳定性和机制边界。

本目录包含三类分析：

1. paired image-key clustered bootstrap；
2. Top20% 捕获重叠分析；
3. repeated split sensitivity 与 Logistic 系数稳定性分析。

## 实验边界

- 本实验仍然属于 APTOS 内部分析，不是独立外部验证。
- bootstrap 使用 `image_key` 作为聚类单位。
- 评价口径与 v0.6.8 对齐：pooled-backbone training，per-backbone test reporting。
- repeated split sensitivity 只检查内部 grouped CV 划分敏感性，不代表外部泛化能力。
- Logistic 系数只作为探索性机制线索，不能解释为因果贡献。

## 输出文件

- `paired_cluster_bootstrap_weighted_equivalence_check.csv`：暴力复制版与权重版 clustered bootstrap 的等价性验证。
- `paired_cluster_bootstrap_top20_replicates.csv`：Top20% clustered bootstrap 逐次结果。
- `paired_cluster_bootstrap_top20_summary.csv`：Top20% clustered bootstrap 汇总结果。
- `top20_capture_overlap_summary.csv`：Top20% 捕获重叠汇总。
- `top20_capture_overlap_cases.csv`：Top20% 捕获重叠 case 明细。
- `logistic_coefficients_by_fold.csv`：v0.6.8 原始 fold 下的 Logistic 标准化系数。
- `logistic_coefficients_summary.csv`：原始 fold 系数稳定性汇总。
- `repeated_split_per_backbone_metrics.csv`：repeated split 下逐 backbone 指标。
- `repeated_split_metrics_by_seed.csv`：repeated split 下逐 seed 汇总指标。
- `repeated_split_cv_summary.csv`：repeated split 指标稳定性汇总。
- `repeated_split_coefficients_by_seed_fold.csv`：repeated split 下逐 seed / fold 系数。
- `repeated_split_coefficient_summary.csv`：repeated split 系数稳定性汇总。
- `robustness_mechanism_key_findings.md`：中文关键发现。
