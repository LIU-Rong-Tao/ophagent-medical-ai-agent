# v0.8.2b 关键发现：预算匹配的受控协议评测

## 1. 阶段定位

v0.8.2b 的目标是把 v0.8.1/v0.8.2 的全组合模型池探索，收束为预定义 controlled protocols。

主结论不再来自 full combination search top rows，而来自预算匹配后的综合表：

- `controlled_performance_risk_summary.csv`：全量综合表；
- `all_budget_performance_risk_best.csv`：0.2 / 0.3 / 0.4 / 0.5 全预算最佳 policy；
- `budget_30_performance_risk_best.csv`：30% expert budget 对照；
- `budget_50_performance_risk_best.csv`：50% expert budget 对照。

## 2. 30% budget：效率优先点

30% expert budget 下，最佳平均性能协议是：

- 协议：`convnext_swin_to_retfound`
- 类型：multi-scout ensemble-base routing
- scout/base：`convnext_tiny + swin_tiny`
- expert：`retfound_mae_cfp_official_protocol`
- policy：`disagreement_then_uncertainty`
- Acc：0.853636
- Macro-F1：0.710852
- QWK：0.913620
- 错误数：161
- forward cost：2.153022 ms/image

同预算下对照：

- `swin_to_retfound`：Acc 0.850000，错误数 165，cost 1.665126 ms/image；
- `convnext_to_retfound`：Acc 0.846364，错误数 169，cost 1.531836 ms/image。

因此，在 30% expert budget 下，multi-scout routing 是当前平均性能最强的效率优先协议。

## 3. 30% budget：固定风险池覆盖

30% budget 下，multi-scout 并非所有风险事件都最强。

`convnext_swin_to_retfound`：

- large undergrading：33/45，recall 0.733333；
- referable DR miss：83/90，recall 0.922222；
- severe/PDR miss：48/75，recall 0.640000。

`convnext_to_retfound`：

- large undergrading：36/45，recall 0.800000；
- referable DR miss：68/90，recall 0.755556；
- severe/PDR miss：56/75，recall 0.746667。

`Swin_to_retfound`：

- large undergrading：29/45，recall 0.644444；
- referable DR miss：66/90，recall 0.733333；
- severe/PDR miss：48/75，recall 0.640000。

30% budget 的关键发现是：

> Multi-scout routing 在整体准确率和 referable miss 覆盖上最强；ConvNeXt single-scout 对 large undergrading 和 severe/PDR miss 更敏感。

## 4. 50% budget：安全优先点

50% expert budget 下，single-scout routing 更适合作为 safety-oriented operating point。

`convnext_to_retfound`：

- Acc：0.851818；
- 错误数：163；
- cost：2.234181 ms/image；
- large undergrading：44/45，recall 0.977778；
- referable DR miss：89/90，recall 0.988889；
- severe/PDR miss：72/75，recall 0.960000。

`swin_to_retfound`：

- Acc：0.851818；
- 错误数：163；
- cost：2.367472 ms/image；
- large undergrading：44/45，recall 0.977778；
- referable DR miss：88/90，recall 0.977778；
- severe/PDR miss：74/75，recall 0.986667。

`convnext_swin_to_retfound`：

- Acc：0.850000；
- 错误数：165；
- cost：2.856536 ms/image；
- large undergrading：44/45，recall 0.977778；
- referable DR miss：89/90，recall 0.988889；
- severe/PDR miss：73/75，recall 0.973333。

50% budget 的关键发现是：

> ConvNeXt 和 Swin single-scout 是更清晰的安全复核点。ConvNeXt 更偏 large undergrading / referable miss；Swin 更偏 severe/PDR miss。

## 5. 总体结论

v0.8.2b 不应声称存在一个协议在所有预算和所有风险事件上全面最优。

更准确的结论是：

1. 30% expert budget 下，ConvNeXt+Swin multi-scout routing 是平均性能与 referable miss 覆盖最强的 efficiency-oriented protocol。
2. 50% expert budget 下，single-scout routing 更适合作 safety-oriented protocol。
3. 不同 scout/routing signal 对不同医学风险事件有偏置，因此模型编排应报告 operating points，而不是只报告单一 top-1 组合。

## 6. ViT-B 与 GreenScout 的定位

ViT-B 已完成 onboarding 和 unified evaluator 接入，但没有改变当前受控协议 frontier。

ViT-B 当前定位为：

- screening baseline；
- supplementary ablation；
- 不作为主 scout；
- 不作为主 expert。

GreenScout 当前定位为：

- historical scout baseline；
- ablation 对照；
- 不再作为主线 scout 叙事中心。

## 7. 下一阶段建议

下一阶段建议：

- 清理报告；
- 导出 residual risk cases；
- 冻结外部验证协议；
- 在 MESSIDOR2 / IDRiD 上只应用预定义协议，不重新选择协议。
