# v0.8.2c 最终总结：Residual Risk Case Export

## 1. 阶段定位

v0.8.2c 不再搜索新协议，也不继续扩展模型池。

本阶段只做一件事：把 v0.8.2b 中确定的三个 operating points 落到病例级证据，分析哪些病例被送入 expert 通道、哪些危险病例仍然残留、不同 operating point 的病例覆盖是否重叠。

固定 operating points：

1. `efficiency_multiscout_30`
   - `convnext_swin_to_retfound`
   - 30% expert budget
   - `disagreement_then_uncertainty`

2. `safety_convnext_50`
   - `convnext_to_retfound`
   - 50% expert budget
   - `low_confidence`

3. `safety_swin_50`
   - `swin_to_retfound`
   - 50% expert budget
   - `low_confidence`

## 2. 主证据文件

v0.8.2c 主结论只依赖三张表：

1. `case_export_summary.csv`
   汇总三个 operating points 的病例级性能、expert 修正、expert induced error 和 fixed-risk 覆盖。

2. `protocol_overlap_summary_by_fixed_event.csv`
   汇总三个 operating points 在 fixed risk pool 上的病例重叠关系。

3. `hard_case_review_table.csv`
   汇总所有协议都漏掉的 hard case，以及只被某一个 operating point 独有抓到的病例。

其余病例级 CSV 作为 raw audit 文件保留，用于追溯，不作为主报告表格继续展开。

## 3. Operating point 病例级结果

### 3.1 Efficiency point：multi-scout 30%

`efficiency_multiscout_30`：

- selected_n：330/1100；
- base error：184；
- final error：161；
- expert corrected：48；
- expert induced error：25；
- fixed large undergrading：33/45；
- fixed referable miss：83/90；
- fixed severe/PDR miss：48/75。

结论：

> Multi-scout 30% 是效率优先点。它以较低 expert budget 接近 dense/reference 性能，并对 referable miss 有较强覆盖，但对 severe/PDR miss 的覆盖不足。

### 3.2 Safety point：ConvNeXt 50%

`safety_convnext_50`：

- selected_n：550/1100；
- base error：205；
- final error：163；
- expert corrected：83；
- expert induced error：41；
- fixed large undergrading：44/45；
- fixed referable miss：89/90；
- fixed severe/PDR miss：72/75。

结论：

> ConvNeXt 50% 更适合作为 large undergrading / referable miss 的安全复核点。

### 3.3 Safety point：Swin 50%

`safety_swin_50`：

- selected_n：550/1100；
- base error：188；
- final error：163；
- expert corrected：65；
- expert induced error：40；
- fixed large undergrading：44/45；
- fixed referable miss：88/90；
- fixed severe/PDR miss：74/75。

结论：

> Swin 50% 更适合作为 severe/PDR miss 的安全复核点。

## 4. Fixed-risk overlap 发现

固定风险池分母：

- large undergrading：45；
- referable miss：90；
- severe/PDR miss：75。

### Large undergrading

- 33 个病例被三个 operating points 同时选中；
- 11 个病例只被两个 50% safety operating points 选中；
- 1 个病例被所有 operating points 漏掉。

### Referable miss

- 82 个病例被三个 operating points 同时选中；
- 7 个病例只被两个 safety operating points 选中；
- 1 个病例被所有 operating points 漏掉。

### Severe/PDR miss

- 48 个病例被三个 operating points 同时选中；
- 24 个病例被两个 operating points 选中；
- 2 个病例只被 `safety_swin_50` 选中；
- 1 个病例被所有 operating points 漏掉。

结论：

> Multi-scout 30% 抓到的是高共识风险子集；50% safety points 主要扩大覆盖范围；Swin 50% 对 severe/PDR miss 存在病例级独有贡献。

## 5. Hard case 发现

### 5.1 All-protocol residual hard case

`bfdee9be1f1d`：

- true_label：4；
- 属于 large undergrading / referable miss / severe-PDR miss 三类 fixed risk pool；
- multi-scout 30%：scout_pred 0，expert_pred 0，final_pred 0，未选中；
- ConvNeXt 50%：scout_pred 0，expert_pred 0，final_pred 0，未选中；
- Swin 50%：scout_pred 0，expert_pred 0，final_pred 0，未选中。

解释：

> 这是一个模型池共同盲区，而不仅是 routing 漏选。即使调用 RETFound expert，该病例也仍会被预测为 0。

### 5.2 Swin 50% 独有 severe/PDR cases

两个 severe/PDR fixed-risk cases 只被 `safety_swin_50` 选中：

- `8c0d05233238`，true_label 3；
- `f576e45d1da2`，true_label 3。

解释：

> 这两个病例支持 Swin 50% 对 severe/PDR miss 有病例级独有贡献。

## 6. 本阶段结论

v0.8.2c 的结论不是“继续产生更多表格”，而是把 v0.8.2b 的指标结论落到病例级：

1. 不同 operating points 覆盖的风险病例结构不同；
2. 30% multi-scout 抓到的是高共识风险子集；
3. 50% safety operating points 扩大风险覆盖；
4. Swin 50% 对 severe/PDR miss 有独有贡献；
5. 存在模型池共同盲区，不能只依赖 router 解决。

## 7. 下一步

下一阶段不应继续扩表。

建议进入：

- v0.8.2d：prediction record schema + adapter template；
- 或 v0.8.3：外部验证协议冻结；
- 后续如需展示，再基于 `hard_case_review_table.csv` 做少量病例可视化。
