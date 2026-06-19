# v0.7.2 Metric-Sensitivity Audit

本目录记录 OphAgent v0.7.2 的 secondary metric audit。

本版本只读取已有外部 DR direct inference predictions，不重跑模型，不替代 v0.7.1b primary conclusion。

核心问题：AURC、AUGRC、partial AUGRC 0.70–0.90 与固定 Top20% 工作点是否给出一致或冲突的方法排序。

输出文件：

- `v072_metric_sensitivity_audit_table.csv`
- `v072_method_rank_comparison.csv`
- `v072_metric_sensitivity_key_findings.md`

边界：AUGRC 是 failure-ranking / selective-classification 的补充评价，不是临床效用指标，不证明临床部署安全。

## 摘要

v0.7.2 显示，在当前两个外部 DR 公共数据集和六个冻结 backbone 上，`gated_severe_prob_mass_only` 对 grade-based VTDR miss 在 AURC、AUGRC 和 partial_AUGRC_70_90 下均保持第一排名，说明 v0.7.1b 的主趋势不局限于 Top20% 单点；但该结果仍属于描述性一致性证据，不构成独立统计显著性、临床效用或真实医生工作流验证。
