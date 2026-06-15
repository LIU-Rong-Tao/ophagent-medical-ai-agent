# OphAgent v0.6.6：预审风险排序与选择性复核验证

v0.6.6 完成了 OphAgent 预审风险排序的完整验证，并将分析范围从 demo 样例扩展到多个 backbone 的完整测试集输出。

本阶段关注的问题是：

在真实标签不参与排序的前提下，能否仅基于模型输出信号，为人工复核生成优先级队列，并评估这些输出信号对错误样本的富集能力。

需要明确区分：

- 排序阶段：不使用真实标签、正确性字段、错误类型或后验临床标签；
- 后验评估阶段：使用真实标签计算错误发现能力和选择性复核指标。

因此，v0.6.6 的定位是：

面向眼科模型输出的低侵入式预审排序、general failure detection 与 selective review 评估框架，而不是一个已经证明最优的新风险排序算法。

## 1. 已完成内容

v0.6.6 已完成以下内容：

- 无真实标签参与排序的泄漏审计；
- sanitized rebuild 检查；
- 输入顺序打乱稳定性检查；
- 强基线对比；
- AUROC-Error；
- AUPR-Error；
- Top-K Error Precision / Recall / Lift；
- Risk-Coverage；
- AURC。

## 2. 主要产物

核心说明文件：

- `notes/v0.6.6_pre_review_risk_ranking_design.md`
- `notes/v0.6.6_leakage_audit.md`
- `notes/v0.6.6_protocol_freeze.md`
- `notes/v0.6.6_error_detection_results.md`

核心脚本：

- `scripts/build_pre_review_risk_table.py`
- `scripts/compare_pre_review_ranking_baselines.py`

核心结果：

- `experiments/summary/v0_6_6/full_test_backbones/baseline_ranking_comparison.csv`
- `experiments/summary/v0_6_6/full_test_backbones/baseline_ranking_comparison.md`
- `experiments/summary/v0_6_6/full_test_backbones/risk_coverage_curve.csv`

说明：

- CSV 为完整逐样本结果；
- Markdown 表格仅作为部分 backbone 的人工阅读预览，不作为完整结果清单；
- 公开结果中的图像路径已脱敏为文件名，避免暴露本地绝对路径或类别目录。


## 3. 当前结论

模型输出信号能够支持低侵入式错误富集和选择性复核分析。

但是，当前 OphAgent combined 规则并没有稳定超过 1-MSP、margin、entropy 和 uncertainty rank fusion 等经典不确定性强基线。

因此，v0.6.6 不能表述为“提出了最优风险排序器”。更合适的表述是：

OphAgent v0.6.6 构建并冻结了一个面向眼科模型输出的预审排序、错误发现和选择性复核评估框架；实验结果显示经典输出不确定性信号在该任务中非常强，当前人工组合规则更适合作为可解释规则候选，而非性能最优排序分数。

## 4. 边界说明

v0.6.6 只分析 general error detection，即普通分类错误：

- `is_error = pred_label != true_label`

v0.6.6 不包含以下 clinical residual risk 分析：

- 高置信错误；
- severe underestimation；
- undergrading / overgrading；
- clinical cost matrix；
- 医生定义的 dangerous misclassification；
- OCT urgency tier。

这些内容留到 v0.6.7 单独处理。
