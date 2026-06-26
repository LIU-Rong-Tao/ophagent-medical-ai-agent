# v0.8.2b 输出文件说明

## 1. 主结论文件

### v082b_key_findings.md

v0.8.2b 的关键发现文档。  
采用预算匹配口径，总结 30% / 50% expert budget 下的性能、成本和 fixed risk pool 覆盖。

### v082b_final_summary.md

v0.8.2b 最终阶段总结。  
用于后续汇报、README 摘要和外部验证前的协议冻结说明。

## 2. 性能-成本-风险综合表

### controlled_performance_risk_summary.csv

全量综合表。  
合并 controlled protocol 的性能、成本、random/oracle 对照和 fixed risk pool coverage。

这是后续分析的总表。

包含：

- dense baseline；
- single-scout routing；
- multi-scout routing；
- 0.2 / 0.3 / 0.4 / 0.5 expert budget；
- dense expert reference。

### all_budget_performance_risk_best.csv

全预算最佳 policy 表。  
包含核心 routing protocols 在 0.2 / 0.3 / 0.4 / 0.5 budget 下的最佳 policy。

用途：

- 查看完整 budget curve；
- 判断 protocol 随 budget 变化的趋势；
- 不只局限于 30% / 50%。

### budget_matched_performance_risk_best.csv

预算匹配主表。  
只保留核心协议在 30% 和 50% budget 下的最佳 policy。

用途：

- 主报告优先引用；
- 避免拿 30% 和 50% 直接混比；
- 同预算下比较性能、成本和风险覆盖。

### budget_matched_performance_risk_all_policies.csv

预算匹配完整 policy 表。  
包含核心协议在 30% 和 50% budget 下的所有 policy。

用途：

- 检查 policy 敏感性；
- 验证 best policy 是否稳定。

### budget_30_performance_risk_best.csv

30% expert budget 专用表。  
每个核心 protocol 只保留 accuracy 最优 policy。

用途：

- 分析 efficiency-oriented operating point；
- 比较低预算下 single-scout 与 multi-scout 的性能/风险差异。

当前主要结论：

- multi-scout 在 Acc 和 referable miss 覆盖上最强；
- ConvNeXt single-scout 在 large undergrading 和 severe/PDR miss 覆盖上更强。

### budget_50_performance_risk_best.csv

50% expert budget 专用表。  
每个核心 protocol 只保留 accuracy 最优 policy。

用途：

- 分析 safety-oriented operating point；
- 比较高复核预算下不同 scout 对风险事件的偏置。

当前主要结论：

- ConvNeXt / Swin single-scout 是更清晰的 safety-oriented protocol；
- ConvNeXt 更偏 large undergrading / referable miss；
- Swin 更偏 severe/PDR miss；
- multi-scout 50% 风险覆盖均衡，但 Acc 不超过 single-scout，成本更高。

## 3. 性能原始表

### controlled_protocol_results.csv

受控协议完整性能表。  
包含 main、screening、ablation 等全部 controlled protocols。

### controlled_protocol_main_results.csv

受控协议性能主表。  
包含 dense baseline、single-scout routing、multi-scout routing 和 dense expert reference。

### controlled_protocol_best_per_protocol.csv

每个 protocol 的最佳性能行。

### controlled_protocol_cost_frontier.csv

受控协议 cost-performance frontier。

### controlled_protocol_key_findings.md

早期 key findings。  
已被 `v082b_key_findings.md` 替代，保留作为历史记录。

## 4. 固定风险池表

### fixed_risk_pool_coverage.csv

固定风险池覆盖完整表。  
用于跨 protocol 比较同一批风险样本的覆盖情况。

这是风险分析的主要原始表，不应删除。

固定风险池包括：

- large_undergrading_union_pool；
- referable_dr_miss_union_pool；
- severe_pdr_miss_union_pool。

### fixed_risk_pool_best_by_event.csv

按风险事件排序的 best rows。  
用于诊断风险覆盖上限，但不应单独作为主结论，因为它容易偏向高 budget。

## 5. 自身风险事件诊断表

### controlled_risk_event_coverage.csv

按每个 protocol 自身 base/scout 产生的风险事件计算覆盖。  
该表分母会随 protocol 改变，只用于诊断，不用于跨 protocol 主比较。

### controlled_risk_event_main_coverage.csv

`controlled_risk_event_coverage.csv` 的 main protocol 子集。

### controlled_risk_event_best_by_event.csv

按自身风险事件分母排序的 best rows。  
只用于诊断，不用于主结论。

## 6. 已替代或不建议主用的文件

### fixed_risk_pool_target_protocols.csv

已由以下文件替代：

- budget_matched_performance_risk_best.csv；
- budget_matched_performance_risk_all_policies.csv；
- budget_30_performance_risk_best.csv；
- budget_50_performance_risk_best.csv。

如存在，可以删除。

## 7. 报告优先级

后续写报告或 README 时，引用优先级为：

1. `v082b_final_summary.md`
2. `v082b_key_findings.md`
3. `budget_30_performance_risk_best.csv`
4. `budget_50_performance_risk_best.csv`
5. `all_budget_performance_risk_best.csv`
6. `controlled_performance_risk_summary.csv`
7. 其他 diagnostic / audit-only 表
