# GreenScout v0.8.0 Protocol-Control 关键结论

## 1. 实验设置

本轮是 GreenScout v0.8.0 的 official-protocol control 版本。

模型池包括：

- RETFound-Green linear probe
- ConvNeXt-Tiny expert
- RETFound-MAE official-protocol expert

其中 RETFound-MAE official-protocol 使用 RETFound 官方 `main_finetune.py` 训练流程，并使用本地路径加载官方 `RETFound_mae_natureCFP.pth` 权重。

需要强调：

- 这里的 RETFound-MAE official-protocol 不是旧的 OphAgent official-like RETFound。
- 旧 official-like 只能作为第一轮 feasibility / protocol-sensitivity 参考。
- 当前主口径应使用 official-protocol RETFound-MAE control。

## 2. 修正后的模型级结果

| 方法 | Accuracy | Macro-F1 | QWK | 错误数 |
|---|---:|---:|---:|---:|
| Oracle expert selection | 0.9109 | 0.8273 | 0.9386 | 98 |
| Experts-only average | 0.8545 | 0.7187 | 0.9102 | 160 |
| RETFound-MAE official-protocol | 0.8482 | 0.7054 | 0.9129 | 167 |
| All-three average | 0.8455 | 0.7112 | 0.9063 | 170 |
| ConvNeXt-Tiny | 0.8136 | 0.6496 | 0.8615 | 205 |
| RETFound-Green linear probe | 0.7791 | 0.6394 | 0.8653 | 243 |

关键修正：

- `Oracle expert selection` 是事后上界，不可部署。
- `All-three average` 不等于 oracle。
- Green 直接参与三模型概率平均后，性能低于 experts-only average。
- Green 的价值不是作为强分类器参与平均，而是作为 scout / router 提供不确定性排序信号。

因此，不能写成：

> Green + ConvNeXt + RETFound 三模型平均达到 0.9109。

正确写法是：

> Oracle 上界达到 0.9109；真实三模型平均为 0.8455；专家双模型平均为 0.8545。

## 3. Sparse invocation 结果

部署可用的核心问题不是“事后选哪个模型对”，而是：

> 只根据 Green 的输出，能不能挑出最值得调用专家模型的样本？

在 50% expert-call budget 下，Green uncertainty routing + ConvNeXt/RETFound experts 的结果为：

- Accuracy：0.8527
- Macro-F1：约 0.720
- QWK：约 0.910
- 错误数：162

对比：

- RETFound-MAE official-protocol 单模型：0.8482
- Experts-only dense average：0.8545
- Random defer 50% accuracy mean：0.8170
- Random defer 50% accuracy p97.5：0.8273

说明：

> GreenScout 在只调用 50% 专家预算时，接近全量专家平均性能，并明显优于随机专家调用。

这里的“预算”目前是 expert forward-call equivalent cost，不是端到端 wall-clock cost。

## 4. Risk enrichment 结果

Green 本身一共有 243 个错误。

在 50% 预算下，low-confidence / low-margin routing 捕获：

- 225 / 243 个 Green error
- Recall：92.6%
- 相对随机富集：1.85×

这说明 Green 的不确定性排序不是随机噪声，而是能有效筛出 Green 容易出错的样本。

## 5. 高风险低估样本捕获

### 5.1 Severe/PDR miss

定义：

> 真实标签为 Severe DR 或 Proliferative DR，但 Green 预测为 Moderate DR 或更轻。

Green 的 severe/PDR miss 总数为 45。

在 50% 预算下，uncertainty routing 捕获：

- 44 / 45 个 severe/PDR miss
- Recall：97.8%
- 相对随机富集：1.96×

### 5.2 Large undergrading

定义：

> 真实标签为 Proliferative DR，但 Green 预测为 Moderate DR 或更轻。

Green 的 large undergrading 总数为 28。

在 50% 预算下，uncertainty routing 捕获：

- 27 / 28 个 large undergrading
- Recall：96.4%
- 相对随机富集：1.93×

这部分是临床叙事里最重要的证据：

> GreenScout 能把多数高风险漏诊 / 严重低估样本集中到优先专家复核池。

## 6. 专家实际纠错能力

`experts_correct_green_error` 表示：

> Green 预测错误，但 ConvNeXt + RETFound 专家平均能够预测正确。

这类样本总数为 120。

在 50% 预算下，low-confidence / low-margin routing 捕获：

- 114 / 120 个专家可纠错 Green error
- Recall：95.0%
- 相对随机富集：1.90×

这说明 sparse routing 不只是抓到了“Green 错误”，还抓到了“专家确实有机会纠正”的错误。

## 7. 当前主结论

GreenScout 不应被表述为一个强分类器 ensemble 方法。

更准确的表述是：

> GreenScout 是一个低成本 scout / router。Green linear probe 本身性能有限，直接参与平均甚至会拖低专家 ensemble；但 Green 的不确定性信号可以有效识别需要专家模型复核的样本。在 official-protocol RETFound-MAE control 下，50% 专家调用预算即可捕获 92.6% 的 Green 错误、95.0% 的专家可纠错错误、97.8% 的 severe/PDR miss，并达到接近全量专家平均的分类性能。

## 8. 当前边界

当前结果仍然是 prediction-level / expert-call equivalent 分析。

尚未完成：

- 端到端推理 wall-clock 时间 benchmark
- 实际 GPU 显存峰值 benchmark
- 不同 batch size 下的吞吐量 benchmark
- Green-only / expert-only / sparse online inference 的真实成本比较

因此当前不能写：

> GreenScout 已经证明真实推理时间下降 50%。

当前只能写：

> GreenScout 在 expert-call equivalent budget 下显示出有效的风险富集和专家调用节省潜力；真实端到端成本需要后续 benchmark 验证。
