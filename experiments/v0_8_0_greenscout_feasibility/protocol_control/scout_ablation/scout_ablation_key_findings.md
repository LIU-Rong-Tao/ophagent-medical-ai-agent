# v0.8.0d Scout Ablation 关键结论

## 1. 实验设置

本轮实验用于比较不同 scout 选择对 sparse expert invocation 的影响。

当前模型池包括：

- Scout 候选：
  - RETFound-Green linear probe
  - ConvNeXt-Tiny
- Expert 候选：
  - ConvNeXt-Tiny
  - RETFound-MAE official-protocol

对比组别包括：

| 组别 | Scout | Expert |
|---|---|---|
| A | RETFound-Green | ConvNeXt + RETFound-MAE official-protocol average |
| B | RETFound-Green | ConvNeXt only |
| C | RETFound-Green | RETFound-MAE official-protocol only |
| D | ConvNeXt-Tiny | RETFound-MAE official-protocol only |

本轮 budget 表示进入专家模型通道的样本比例，即 expert-call equivalent budget。它不等同于真实端到端 wall-clock 时间、吞吐量或显存收益。

## 2. 三层结论阅读顺序

本轮结果不应解读为“证明某个 scout 绝对最优”。更合适的阅读顺序是：

### 2.1 主效应：sparse routing 是否有效

核心问题是：

> 在固定 expert-call budget 下，基于 scout 不确定性的专家调用是否比随机调用更有效，并能否接近全量专家集成？

因此主指标优先看：

- Accuracy / Macro-F1 / QWK
- 是否超过 random p97.5
- 净错误减少
- 距离 oracle same-budget 上界的差距

当前所有 ablation 组别在 30% / 40% / 50% budget 下均超过对应 random p97.5，说明专家调用不是随机生效，而是 scout uncertainty ranking 提供了有效排序信号。

### 2.2 Scout ablation：不同 scout 带来的折中

本轮比较的是不同 scout 选择带来的性能-风险折中，而不是给出单一最优 scout。

当前结果显示：

- ConvNeXt-Tiny scout 在 30%–40% budget 下 Accuracy 更高；
- RETFound-Green scout + ConvNeXt/RETFound 双专家 average 在 50% budget 下 Accuracy 最高、错误数最低、净错误减少最大；
- 因此 scout 选择会影响 sparse routing 的表现，后续需要统一 scout ranking 和 actual cost benchmark，而不是固定 Green 或 ConvNeXt 为唯一选择。

### 2.3 错误机制：救回、引入与净收益

专家通道是否有价值，不能只看 expert-correctable capture recall。

更关键的是：

> 净错误减少 = 专家救回的 scout 错误 - 专家新引入的错误。

50% budget 下：

| 组别 | 救回 scout 错误 | 专家新引入错误 | 净错误减少 |
|---|---:|---:|---:|
| A: Green → ConvNeXt+RETFound | 114 | 33 | 81 |
| D: ConvNeXt → RETFound | 83 | 41 | 42 |
| C: Green → RETFound | 118 | 42 | 76 |
| B: Green → ConvNeXt | 105 | 59 | 46 |

这说明 `83 / 83` 这类 expert-correctable capture recall 不能单独作为主结论。ConvNeXt scout 的可纠错空间更小，且专家新引入错误更多；Green scout 本身较弱，但在双专家接管后产生了更大的净错误减少。


## 3. 单模型与静态集成基线

| 方法 | Accuracy | Macro-F1 | QWK | 错误数 |
|---|---:|---:|---:|---:|
| RETFound-Green linear probe | 0.7791 | 0.6394 | 0.8653 | 243 |
| ConvNeXt-Tiny | 0.8136 | 0.6496 | 0.8615 | 205 |
| RETFound-MAE official-protocol | 0.8482 | 0.7054 | 0.9129 | 167 |
| Experts-only average | 0.8545 | 0.7187 | 0.9102 | 160 |
| All-three average | 0.8455 | 0.7112 | 0.9063 | 170 |

静态结果说明：

- RETFound-MAE official-protocol 是当前最强单模型。
- Experts-only average 仍是最强静态集成参照。
- Green 直接加入三模型平均后，低于 experts-only average。
- Green 的主要价值仍应定位为 scout / router，而不是普通 ensemble 成员。

## 4. Scout ablation 主要结果

### 3.1 30% expert-call budget

| 组别 | 最优策略 | Accuracy | Macro-F1 | QWK | 错误数 | 是否超过 random p97.5 |
|---|---|---:|---:|---:|---:|---|
| D: ConvNeXt → RETFound | high_entropy | 0.8464 | 0.7024 | 0.9080 | 169 | True |
| C: Green → RETFound | high_entropy | 0.8418 | 0.7054 | 0.9114 | 174 | True |
| A: Green → ConvNeXt+RETFound | high_entropy | 0.8409 | 0.7059 | 0.9056 | 175 | True |
| B: Green → ConvNeXt | high_entropy | 0.8191 | 0.6744 | 0.8754 | 199 | True |

在 30% budget 下，ConvNeXt scout → RETFound expert 的 Accuracy 最高。

### 3.2 40% expert-call budget

| 组别 | 最优策略 | Accuracy | Macro-F1 | QWK | 错误数 | 是否超过 random p97.5 |
|---|---|---:|---:|---:|---:|---|
| D: ConvNeXt → RETFound | high_entropy | 0.8500 | 0.7124 | 0.9119 | 165 | True |
| A: Green → ConvNeXt+RETFound | high_entropy | 0.8464 | 0.7109 | 0.9075 | 169 | True |
| C: Green → RETFound | low_margin | 0.8418 | 0.7005 | 0.9117 | 174 | True |
| B: Green → ConvNeXt | low_margin | 0.8191 | 0.6705 | 0.8723 | 199 | True |

在 40% budget 下，ConvNeXt scout → RETFound expert 仍保持最高 Accuracy。

### 3.3 50% expert-call budget

| 组别 | 最优策略 | Accuracy | Macro-F1 | QWK | 错误数 | 是否超过 random p97.5 |
|---|---|---:|---:|---:|---:|---|
| A: Green → ConvNeXt+RETFound | low_confidence | 0.8527 | 0.7200 | 0.9102 | 162 | True |
| D: ConvNeXt → RETFound | low_confidence | 0.8518 | 0.7120 | 0.9140 | 163 | True |
| C: Green → RETFound | low_confidence | 0.8482 | 0.7116 | 0.9142 | 167 | True |
| B: Green → ConvNeXt | low_margin | 0.8209 | 0.6716 | 0.8686 | 197 | True |

在 50% budget 下，Green scout + 双专家 average 达到最高 Accuracy，并取得最少错误数。

## 5. 专家实际纠错与新引入错误

50% budget 下：

| 组别 | 专家可纠错错误总数 | 捕获专家可纠错错误 | 救回 scout 错误 | 专家新引入错误 | 净错误减少 |
|---|---:|---:|---:|---:|---:|
| A: Green → ConvNeXt+RETFound | 120 | 114 / 120 | 114 | 33 | 81 |
| D: ConvNeXt → RETFound | 83 | 83 / 83 | 83 | 41 | 42 |
| C: Green → RETFound | 125 | 118 / 125 | 118 | 42 | 76 |
| B: Green → ConvNeXt | 110 | 105 / 110 | 105 | 59 | 46 |

这部分说明：

- ConvNeXt scout → RETFound 在 50% budget 下捕获了全部 expert-correctable scout errors。
- 但 ConvNeXt scout 的 expert-correctable error 总数较少，最终净错误减少为 42。
- Green scout + 双专家设置虽然 expert-correctable capture recall 为 95.0%，但可纠错空间更大，最终净错误减少达到 81。
- sparse routing 的瓶颈不只是“能否捕获错误”，还包括专家接管是否会把 scout 原本正确的样本改错。

## 6. DR-specific 高风险低估事件

需要注意：DR-specific 事件是相对于当前 scout 的预测定义的，因此不同 scout 的事件总数不同。这里比较的是不同 scout 自身产生的高风险低估错误规模，而不是同一个固定事件集合。

50% budget 下：

| 组别 | Severe/PDR miss 总数 | 捕获 | Large undergrading 总数 | 捕获 |
|---|---:|---:|---:|---:|
| A: Green → ConvNeXt+RETFound | 45 | 44 / 45 | 28 | 27 / 28 |
| C: Green → RETFound | 45 | 44 / 45 | 28 | 27 / 28 |
| B: Green → ConvNeXt | 45 | 44 / 45 | 28 | 27 / 28 |
| D: ConvNeXt → RETFound | 65 | 62 / 65 | 41 | 40 / 41 |

这部分不能简单解释为某个 scout 在风险事件上全面更优。

ConvNeXt-Tiny 总体分类能力强于 RETFound-Green，但它在当前 APTOS test 的错误结构中产生了更多 DR-specific 高风险低估事件：severe/PDR miss 为 65，large undergrading 为 41。RETFound-Green 对应事件总数分别为 45 和 28。

因此，这里的主要结论是：

> 不同 scout 不仅影响总体 Accuracy，也会改变错误结构和后续专家通道需要处理的风险样本类型。DR-specific 事件捕获率需要结合事件总数一起解释，不能只看 recall。


## 7. 当前主结论

本轮 ablation 不支持“某一个 scout 全面最优”的简单结论。

更准确的结论包括四点：

1. ConvNeXt-Tiny 作为 scout 在 30%–40% expert-call budget 下 Accuracy 更高，说明更强分类能力的 scout 可以改善低预算路由表现。
2. RETFound-Green scout + ConvNeXt/RETFound 双专家 average 在 50% budget 下取得最高 Accuracy、最低错误数和最大净错误减少。
3. ConvNeXt scout 的 expert-correctable capture recall 较高，但 denominator 更小，并且专家接管引入的新错误更多。因此不能只用 `83 / 83` 判断其系统收益。
4. 当前没有单一 scout 全面最优。后续应将问题升级为 scout-to-expert model orchestration framework，通过统一 scout ranking 和真实成本 benchmark 决定实际组合。

从专家纠错机制看，50% budget 下最有解释力的对照是：

| 组别 | 救回 scout 错误 | 专家新引入错误 | 净错误减少 |
|---|---:|---:|---:|
| A: Green → ConvNeXt+RETFound | 114 | 33 | 81 |
| D: ConvNeXt → RETFound | 83 | 41 | 42 |
| C: Green → RETFound | 118 | 42 | 76 |
| B: Green → ConvNeXt | 105 | 59 | 46 |

这说明：

> ConvNeXt scout 起点更强，但留给 RETFound expert 修正的空间更小；RETFound-Green scout 本身较弱，但能暴露更多专家可纠错错误，在双专家接管后形成更大的净错误减少。

因此，当前结果不应表述为 Green 是唯一最佳 scout，也不应表述为 ConvNeXt 是唯一最佳 scout。更稳的表述是：

> Scout 选择本身就是模型中转台需要评估的问题。当前结果支持将 GreenScout 从单一 Green scout 叙事升级为 scout-to-expert model orchestration：不同 scout、不同 expert 和不同 budget 会形成不同的性能-风险折中。

需要进一步注意：本轮实验还不能证明 Green scout 一定具有真实部署成本优势。ConvNeXt scout 提供的是一个更强分类能力的 scout 对照；其真实成本是否高于 Green scout，需要后续 v0.8.0e actual inference cost benchmark 验证。


## 8. 当前边界

当前结果仍然属于 prediction-level sparse routing simulation。

尚未完成：

- 真实端到端推理 wall-clock benchmark
- 不同 batch size 下的吞吐量比较
- peak CUDA memory benchmark
- Green scout 与 ConvNeXt scout 的真实成本差异复核
- 外部数据集固定协议验证

因此当前不能写：

> ConvNeXt scout 或 Green scout 已经证明真实系统成本下降。

当前只能写：

> 在 expert-call equivalent budget 下，不同 scout 选择会影响 sparse routing 的性能-风险折中；真实成本优势需要 v0.8.0e actual inference cost benchmark 验证。
