# v0.6.8 Learned Deferral Score

本目录保存 v0.6.8 的轻量监督式复核排序实验结果。

## 目录用途

v0.6.8 评估一个 Logistic Regression learned review score，验证它能否基于模型输出后的风险信号，形成临床风险感知的 Top-K 复核排序。

这个目录主要用于保存：

- 复核排序评价结果；
- fold 诊断结果；
- AUROC / AUPRC 辅助指标；
- learned score 的 out-of-fold 预测表；
- 简要中文结论文档。

## 实验边界

- 本实验是 APTOS 内部探索性 grouped cross-validation。
- 分组字段为 `image_key`，用于避免同一张图像的不同 backbone 预测同时进入训练集和测试集。
- 训练方式是 pooled-backbone training。
- 汇报方式是 per-backbone test performance。
- 本实验不是独立临床验证。
- 本实验不是外部验证。
- 本实验不是 unseen-backbone generalization。

## 重要计数口径

本目录中的事件总数，例如 263、391、430，均指 6600 条骨干网络特异预测记录中的事件记录数，不是独立患者数，也不是唯一图像数量。

180/263、297/391 等 captured / total 结果同样是按预测记录统计的捕获结果，不是患者级或图像级捕获结果。

后续如果做置信区间或显著性检验，应以 `image_key` 作为聚类单位，而不能把 6600 条预测记录当作相互独立样本。

## 输出文件

- `learned_deferral_tradeoff.csv`：逐 fold / target / ranking method / review budget 的完整评价结果。
- `learned_deferral_cv_summary.csv`：交叉验证聚合结果。
- `learned_deferral_fold_diagnostics.csv`：fold 正例分布和分组诊断。
- `learned_deferral_winner_count.csv`：不同预算下的最佳方法统计。
- `learned_deferral_auc_by_fold.csv`：逐 fold AUROC / AUPRC。
- `learned_deferral_auc_summary.csv`：AUROC / AUPRC 聚合结果。
- `learned_deferral_fold_predictions.csv`：out-of-fold learned score 和派生特征。
- `learned_deferral_key_findings.md`：中文关键结论。
