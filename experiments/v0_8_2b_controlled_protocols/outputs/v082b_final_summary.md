# v0.8.2b 最终阶段总结：Controlled Protocol Evaluation

## 1. 本阶段做了什么

v0.8.2b 将原本的全组合模型池探索，收束为受控协议评测。

本阶段保留：

- dense baseline；
- single-scout routing；
- multi-scout routing；
- random same-budget baseline；
- oracle same-budget upper bound；
- fixed risk pool coverage；
- 预算匹配性能-风险综合表。

本阶段不再把 full combination search top rows 作为主结论。

## 2. 为什么要做预算匹配

风险覆盖必须在相同 expert budget 下比较。

因此，本阶段额外生成：

- `budget_30_performance_risk_best.csv`
- `budget_50_performance_risk_best.csv`
- `all_budget_performance_risk_best.csv`

这样可以分别讨论：

- 30% expert budget 下的效率优先协议；
- 50% expert budget 下的安全优先协议；
- 0.2 / 0.3 / 0.4 / 0.5 的完整 budget curve。

## 3. 30% expert budget 结论

30% budget 下，`convnext_swin_to_retfound` 是平均性能最强协议。

| Protocol | Policy | Cost ms/image | Acc | n_error | Large recall | Referable recall | Severe/PDR recall |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt+Swin -> RETFound | disagreement_then_uncertainty | 2.153022 | 0.853636 | 161 | 0.733333 | 0.922222 | 0.640000 |
| Swin -> RETFound | high_entropy | 1.665126 | 0.850000 | 165 | 0.644444 | 0.733333 | 0.640000 |
| ConvNeXt -> RETFound | high_entropy | 1.531836 | 0.846364 | 169 | 0.800000 | 0.755556 | 0.746667 |

30% budget 下的核心判断：

- multi-scout 在 Acc 和 referable miss 覆盖上最强；
- ConvNeXt single-scout 在 large undergrading 和 severe/PDR miss 覆盖上更强；
- Swin single-scout 在 30% budget 下不是最优点。

因此，30% budget 的结论不是“multi-scout 全面最优”，而是：

> Multi-scout 30% 是平均性能与 referable miss 覆盖最强的效率优先 operating point；ConvNeXt 30% 是更偏重症低估风险的信号点。

## 4. 50% expert budget 结论

50% budget 下，single-scout routing 更适合作 safety-oriented operating point。

| Protocol | Policy | Cost ms/image | Acc | n_error | Large recall | Referable recall | Severe/PDR recall |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt -> RETFound | low_confidence | 2.234181 | 0.851818 | 163 | 0.977778 | 0.988889 | 0.960000 |
| Swin -> RETFound | low_confidence | 2.367472 | 0.851818 | 163 | 0.977778 | 0.977778 | 0.986667 |
| ConvNeXt+Swin -> RETFound | mean_uncertainty | 2.856536 | 0.850000 | 165 | 0.977778 | 0.988889 | 0.973333 |

50% budget 下的核心判断：

- ConvNeXt 与 Swin single-scout Acc 并列最高；
- ConvNeXt 成本更低，large / referable 风险覆盖强；
- Swin severe/PDR miss 覆盖最高；
- Multi-scout 50% 风险覆盖均衡，但 Acc 不超过 single-scout，成本更高。

因此，50% budget 的结论是：

> 安全复核优先时，ConvNeXt 50% 和 Swin 50% 是更清晰的 operating points；前者偏 large/referable，后者偏 severe/PDR miss。

## 5. 与 dense expert reference 的关系

Dense ConvNeXt+RETFound reference：

- Acc：0.854545；
- Macro-F1：0.718665；
- QWK：0.910170；
- 错误数：160；
- cost：4.003712 ms/image。

30% multi-scout：

- Acc：0.853636；
- 错误数：161；
- cost：2.153022 ms/image。

因此，multi-scout 30% 在平均性能上接近 dense expert reference，但成本明显更低。

但在 fixed risk pool 中，multi-scout 30% 对 severe/PDR miss 覆盖不足，因此不能作为安全覆盖最优点。

## 6. 本阶段最重要的研究发现

v0.8.2b 的关键发现不是某个单一组合最优，而是：

> 不同 budget 下，最优 routing protocol 会变化；不同 scout/routing signal 对不同医学风险事件有偏置。模型编排应该报告 operating points，而不是只报告一个 top-1 协议。

具体来说：

1. 30% budget：ConvNeXt+Swin multi-scout 更适合效率优先；
2. 50% budget：ConvNeXt/Swin single-scout 更适合安全复核优先；
3. ConvNeXt 信号更偏重症低估风险；
4. Multi-scout disagreement 更有利于平均性能和 referable miss 覆盖；
5. Swin 在高预算下对 severe/PDR miss 更敏感。

## 7. 当前边界

本阶段不声称：

- 找到了全局最优模型组合；
- 每个 backbone 都已经达到架构性能上限；
- ViT-B 或 GreenScout 应进入主协议；
- Multi-scout 在所有预算下都优于 single-scout。

本阶段只声称：

> 在固定模型池和预定义 controlled protocols 下，ConvNeXt/Swin 与 RETFound official-protocol 的 scout-to-expert 编排可以形成可解释的性能-成本-风险折中。

## 8. 下一阶段

下一阶段建议：

1. 清理输出文件说明；
2. 导出 residual risk cases；
3. 冻结 external validation 协议；
4. 在 MESSIDOR2 / IDRiD 上只复用预定义协议，不重新调协议。
