# OphAgent v0.6.7：选择性复核后的临床残余风险审计
## 版本定位
v0.6.7 延续 v0.6.6 的选择性复核框架，但不再只关注整体错误发现。
本版本关注：在固定医生复核预算下，选择性复核策略能否减少自动放行区中的临床危险错误；如果不能，哪些危险错误仍会残留，复核代价是多少。
## 与 v0.6.6 的关系
v0.6.6 证明模型输出信号可以用于 general failure detection 和 selective review 评估，但 OphAgent combined 没有稳定超过 confidence、margin、entropy、uncertainty rank fusion 等强基线。
v0.6.7 进一步把评价目标从 general error 转向 clinical-risk proxy，重点分析方向敏感的危险错误。
## 核心设计
排序阶段仍然不使用真实标签。
v0.6.7 读取 v0.6.6 的无标签预审排序表，再在后验评估阶段通过 image filename 合并原始 test_predictions.csv 中的 true_grade。
排序方法包括 confidence_only、margin_only、entropy_only、uncertainty_rank_fusion 和 ophagent_combined。
后验评估事件包括 general_error、large_undergrading、referable_dr_miss、vision_threatening_dr_miss 和 high_confidence_vision_threatening_miss。
## 主要结果文件
- clinical_event_definitions.csv：clinical-risk proxy 定义
- clinical_event_cases.csv：合并 true_grade 后的逐样本事件表
- clinical_event_counts.csv：各 backbone 的危险错误数量
- review_burden_tradeoff.csv：不同复核预算下的捕获、残留和复核负担
- best_method_count_summary.csv：不同事件和预算下最优方法统计
- best_method_by_event_budget_backbone.csv：逐 backbone 最优方法表
- residual_dangerous_cases.csv：复核后仍残留在自动放行区的危险错误样本
- merge_checks.csv：v0.6.6 排序表与原始预测表的对齐检查
## 初步发现
在 general_error 上，经典 uncertainty baseline 仍然更强，OphAgent combined 不是整体错误发现的最优排序器。
但在方向敏感的危险错误上，OphAgent combined 表现出更稳定的优势。
对于 vision_threatening_dr_miss，OphAgent combined 在 5%、10%、20% 复核预算下均为 6/6 个 backbone 最优，在 30% 预算下为 4/6 个 backbone 最优。
对于 large_undergrading，OphAgent combined 在 5%、10%、20% 复核预算下均为 6/6 个 backbone 最优，在 30% 预算下为 3/6 个 backbone 最优。
## 当前解释
v0.6.7 的初步结果说明：整体错误发现能力不等于临床危险错误控制能力。
OphAgent combined 虽然不是 general error detection 的最强方法，但其方向敏感规则可能更适合捕获重症低估、跨级低估等 clinical-risk proxy 错误。
## 复现方式
运行：python scripts/evaluate_clinical_residual_risk.py
该脚本会重新生成 v0.6.7 的核心 CSV 结果。
## 解释边界
v0.6.7 不证明真实临床安全性。
本版本是在 APTOS DR grading 上，用分级方向和转诊阈值构造 clinical-risk proxy，评估选择性复核策略在危险错误残留方面的表现。
更准确的定位是：clinical-risk-aware residual error auditing framework。
后续医院数据阶段需要使用医生定义的 dangerous pair、urgency tier 或 clinical cost matrix 替代当前 proxy。
