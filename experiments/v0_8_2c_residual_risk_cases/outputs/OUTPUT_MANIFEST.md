# v0.8.2c 输出文件说明

## 1. 主结论文件

### v082c_final_summary.md

v0.8.2c 最终阶段总结。
用于后续 README、汇报和论文草稿中的病例级 residual risk audit 说明。

### v082c_key_findings.md

v0.8.2c 关键发现。
记录三个 fixed operating points 的病例级结果、fixed-risk overlap 和 hard case 发现。

## 2. 主证据表

### case_export_summary.csv

病例级汇总表。
汇总三个 operating points 的：

- selected_n；
- base error；
- final error；
- expert corrected；
- expert induced error；
- self-risk recall；
- fixed-risk-pool recall。

这是 v0.8.2c 的第一主表。

### protocol_overlap_summary_by_fixed_event.csv

固定风险池 overlap 汇总表。
回答三个 operating points 在同一批 fixed-risk cases 上：

- 谁抓到了；
- 谁漏掉了；
- 有多少病例被 0 / 1 / 2 / 3 个 protocols 选中；
- 是否存在 protocol 独有贡献。

这是 v0.8.2c 的第二主表。

### hard_case_review_table.csv

hard case 审阅表。
汇总：

- 所有 operating points 都漏掉的 residual hard case；
- 只被某一个 operating point 独有选中的病例；
- 三个 protocol 的 scout_pred / expert_pred / final_pred / selected_for_expert / routing_score。

这是 v0.8.2c 的第三主表。

## 3. Raw audit 表

### operating_point_case_table.csv

三个 operating points 的逐病例完整表。
包含每个病例在每个 protocol 下的预测、routing score、是否选中、expert 是否修正、是否 residual risk。

这是病例级审计总表，保留用于追溯，不建议直接作为主报告表。

### selected_risk_cases.csv

被送入 expert 通道的风险或错误病例。
用于追踪系统认为哪些病例值得 expert 复核。

### residual_risk_cases.csv

未进入 expert 通道或最终仍错误/危险的病例。
用于 residual risk audit。

### protocol_overlap_cases.csv

逐病例 protocol overlap 表。
用于生成 overlap summary，不建议直接展示。

### fixed_risk_cases_missed_by_all_protocols.csv

所有 operating points 都未选中的 fixed-risk cases。
当前关键发现：`bfdee9be1f1d` 是三类 fixed risk pool 共同 hard case。

### fixed_risk_cases_uniquely_selected.csv

只被某一个 operating point 独有选中的 fixed-risk cases。
当前关键发现：`safety_swin_50` 独有选中 2 个 severe/PDR fixed-risk cases。

## 4. 报告优先级

后续报告优先引用：

1. `v082c_final_summary.md`
2. `v082c_key_findings.md`
3. `case_export_summary.csv`
4. `protocol_overlap_summary_by_fixed_event.csv`
5. `hard_case_review_table.csv`

其余 CSV 只作为 raw audit / traceability，不再继续扩展分析。

## 5. 阶段边界

v0.8.2c 到此为止，不继续扩表。

后续方向应进入：

- v0.8.2d：prediction record schema + adapter template；
- 或 v0.8.3：外部验证协议冻结；
- 后续如需展示，再基于 `hard_case_review_table.csv` 做少量病例可视化。
