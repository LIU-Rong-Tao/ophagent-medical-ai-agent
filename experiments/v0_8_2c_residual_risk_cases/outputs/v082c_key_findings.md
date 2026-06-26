# v0.8.2c 关键发现：Residual Risk Case Export

## 1. 阶段定位

v0.8.2c 的目标不是继续搜索新协议，而是把 v0.8.2b 中确定的 operating points 落到病例级证据。

本阶段固定三个 operating points：

1. `efficiency_multiscout_30`
   - 协议：`convnext_swin_to_retfound`
   - 类型：multi-scout ensemble-base routing
   - budget：30%
   - policy：`disagreement_then_uncertainty`

2. `safety_convnext_50`
   - 协议：`convnext_to_retfound`
   - 类型：single-scout routing
   - budget：50%
   - policy：`low_confidence`

3. `safety_swin_50`
   - 协议：`swin_to_retfound`
   - 类型：single-scout routing
   - budget：50%
   - policy：`low_confidence`

本阶段不重新选择协议，不做新的 full combination search。

## 2. 病例级导出文件

主要输出包括：

- `operating_point_case_table.csv`：三个 operating points 的逐病例完整表；
- `selected_risk_cases.csv`：被送入 expert 通道的风险/错误病例；
- `residual_risk_cases.csv`：未进入 expert 通道或最终仍错误/危险的病例；
- `protocol_overlap_cases.csv`：三个 operating points 在病例级的选中/漏掉重叠表；
- `case_export_summary.csv`：病例级汇总；
- `protocol_overlap_summary_by_fixed_event.csv`：固定风险池的协议重叠汇总；
- `fixed_risk_cases_missed_by_all_protocols.csv`：所有协议都漏掉的 fixed-risk hard cases；
- `fixed_risk_cases_uniquely_selected.csv`：只被某一个 operating point 独有选中的 fixed-risk cases。

## 3. Operating point 病例级汇总

### 3.1 Efficiency point：multi-scout 30%

`efficiency_multiscout_30`：

- selected_n：330；
- base error：184；
- final error：161；
- expert corrected：48；
- expert induced error：25；
- fixed large undergrading：33/45，recall 0.733333；
- fixed referable miss：83/90，recall 0.922222；
- fixed severe/PDR miss：48/75，recall 0.640000。

结论：

> Multi-scout 30% 是效率优先点。它显著降低 expert 调用量，同时在 referable miss 上保持较强覆盖，但对 severe/PDR miss 的 fixed-risk 覆盖不足。

### 3.2 Safety point：ConvNeXt 50%

`safety_convnext_50`：

- selected_n：550；
- base error：205；
- final error：163；
- expert corrected：83；
- expert induced error：41；
- fixed large undergrading：44/45，recall 0.977778；
- fixed referable miss：89/90，recall 0.988889；
- fixed severe/PDR miss：72/75，recall 0.960000。

结论：

> ConvNeXt 50% 是 large undergrading / referable miss 更强的安全复核点。

### 3.3 Safety point：Swin 50%

`safety_swin_50`：

- selected_n：550；
- base error：188；
- final error：163；
- expert corrected：65；
- expert induced error：40；
- fixed large undergrading：44/45，recall 0.977778；
- fixed referable miss：88/90，recall 0.977778；
- fixed severe/PDR miss：74/75，recall 0.986667。

结论：

> Swin 50% 是 severe/PDR miss 覆盖最强的安全复核点。

## 4. Expert corrected 与 induced error

病例级导出显示，expert 通道不是单向收益。

三个 operating points 都存在：

- expert corrected：expert 修正 base/scout 错误；
- expert induced error：base/scout 原本正确，但 expert 覆盖后变错。

这说明 routing 不能只报告“送入 expert 的风险召回率”，还必须报告：

- expert 修正了哪些病例；
- expert 反而改错了哪些病例；
- final prediction 是否仍有 residual danger。

这一点是 v0.8.2c 相比 v0.8.2b 的重要补强。

## 5. Fixed risk pool overlap 发现

固定风险池分母：

- large undergrading union pool：45；
- referable miss union pool：90；
- severe/PDR miss union pool：75。

### 5.1 Large undergrading

- total：45；
- multi-scout 30% selected：33；
- ConvNeXt 50% selected：44；
- Swin 50% selected：44；
- selected by all 3 operating points：33；
- selected by 2 operating points：11；
- missed by all：1。

解释：

> Multi-scout 30% 抓到的 large undergrading 病例全部属于三个 operating points 的高共识风险子集。50% safety 协议主要是在这个基础上扩大覆盖范围。

### 5.2 Referable miss

- total：90；
- multi-scout 30% selected：83；
- ConvNeXt 50% selected：89；
- Swin 50% selected：88；
- selected by all 3 operating points：82；
- selected by 2 operating points：7；
- missed by all：1。

解释：

> Referable miss 上，multi-scout 30% 已经抓到大部分 fixed-risk cases，说明它确实是 referable miss 覆盖较强的效率点。

### 5.3 Severe/PDR miss

- total：75；
- multi-scout 30% selected：48；
- ConvNeXt 50% selected：72；
- Swin 50% selected：74；
- selected by all 3 operating points：48；
- selected by 2 operating points：24；
- selected by 1 operating point：2；
- only selected by Swin 50%：2；
- missed by all：1。

解释：

> Severe/PDR miss 上，Swin 50% 有病例级独有贡献。它独有选中了 2 个 severe/PDR fixed-risk cases，支持“Swin 更偏 severe/PDR miss 风险覆盖”的 operating point 解释。

## 6. 本阶段最重要结论

v0.8.2c 的主要结论不是“某个协议全面最优”，而是：

> 不同 operating points 对 fixed-risk cases 的覆盖结构不同。Multi-scout 30% 抓到的是高共识风险子集；50% safety 协议扩大覆盖范围；Swin 50% 对 severe/PDR miss 有少量独有贡献；所有协议仍各自存在 residual hard cases。

## 7. 下一步

下一步应检查：

1. `fixed_risk_cases_missed_by_all_protocols.csv`
   - 找出所有 operating points 都漏掉的 hard cases；
   - 分析其 true label、base/scout pred、expert pred、routing score 和模型分歧模式。

2. `fixed_risk_cases_uniquely_selected.csv`
   - 查看只被 Swin 50% 独有选中的 severe/PDR cases；
   - 查看是否存在某种可解释的模型偏置。

3. `residual_risk_cases.csv`
   - 检查 residual risk 是否集中在某些 true grade / pred grade / confidence 区间。

本阶段暂不做 UI。病例级 HTML 展示可以放到后续展示层。
