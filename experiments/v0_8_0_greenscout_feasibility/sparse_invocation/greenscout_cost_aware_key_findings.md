# v0.8.0b Cost-aware sparse routing summary

## 1. 核心问题

本节回答的问题是：

在所有图像先运行 RETFound-Green 的前提下，只对部分图像调用专家模型，是否能够以低于 dense ensemble 的专家调用成本，达到接近或超过 dense ensemble 的性能？

这里的 dense ensemble 指三模型全部运行并做概率平均。它是可部署但高成本的参照。

本节不讨论复杂 Router 训练，也不声称临床部署已经成立，只做成本感知 sparse expert invocation 的离线分析。

## 2. 成本定义

本节中的 cost-aware sparse routing 主要衡量专家 forward-call equivalent cost，而不是静态 GPU 显存成本。

当前 cost-aware 指标的含义是：

* 减少专家 forward 调用次数；
* 减少专家模型计算量；
* 减少专家 forward 产生的 activation 开销；
* 降低平均推理延迟和能耗；
* 提高专家服务吞吐。

因此，relative cost vs dense 表示相对于 dense ensemble 的专家前向调用等价成本，而不是相对于 dense ensemble 的显存占用比例。

例如：

* Dense average ensemble：每张图像都调用 ConvNeXt-Tiny 和 RETFound-MAE，专家 forward-call equivalent cost 记为 100%；
* Green → both experts average，50% defer：只有 50% 图像调用两个专家，专家 forward-call equivalent cost 约为 dense 的 50%；
* Green → ConvNeXt，50% defer：只有 50% 图像调用一个专家，专家 forward-call equivalent cost 约为 dense 的 25%。

如果所有模型都常驻同一张 GPU，sparse invocation 主要节省平均计算成本，而不是静态模型显存。

如果采用 Green 常驻、专家按需加载或专家服务独立部署的系统架构，sparse invocation 还可能进一步减少专家服务负载和硬件需求。但这需要额外系统 benchmark 验证，不能由当前离线实验直接推出。

因此，当前报告应使用如下表述：

Green-first sparse invocation 在 APTOS test 上降低了专家 forward-call equivalent cost，并在部分预算下达到或超过 dense average ensemble 性能。

不应表述为：

Green-first sparse invocation 直接节省了 40%–50% GPU 显存。

## 3. 基线参照

| 方法                                      | Accuracy | 说明           |
| --------------------------------------- | -------: | ------------ |
| RETFound-Green linear probe             |   0.7791 | 低成本 scout    |
| ConvNeXt-Tiny                           |   0.8136 | 最佳单模型        |
| RETFound-MAE existing reproduced expert |   0.8036 | 单专家          |
| Dense average ensemble                  |   0.8309 | 三模型全部运行，概率平均 |
| Oracle any-expert upper bound           |   0.9191 | 后验理论上限，不可部署  |

Dense average ensemble 是现实可部署的高成本参照。

Oracle any-expert upper bound 只用于说明模型池存在互补空间，不作为现实目标。

## 4. 主要结论

当前结果支持 Green-first sparse invocation 存在正向成本-性能折中。

Green 本身不是最强分类器，但它的 uncertainty signal 能帮助挑出更值得调用专家的样本。

整体结论可以分成三档：

1. 轻量模式：只调用 ConvNeXt，成本低，但性能上限有限。
2. 性能模式：调用 both experts average，40%–50% dense expert cost 下超过 dense average ensemble。
3. 成本效率模式：30% both experts average 已基本接近 dense average ensemble，单位专家调用收益较高。

## 5. 最强性能点

当前最高性能来自：

| Policy         | Expert setting | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Above random 97.5% |
| -------------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------ |
| low_margin     | both_average   |              50% |                   0.50 |   0.8427 |         +0.0118 | True               |
| low_confidence | both_average   |              50% |                   0.50 |   0.8418 |         +0.0109 | True               |
| high_entropy   | both_average   |              50% |                   0.50 |   0.8409 |         +0.0100 | True               |

解释：

在只使用 dense ensemble 50% 专家调用成本的情况下，Green low-margin defer + both experts average 达到 Acc 0.8427，高于 dense average ensemble 的 Acc 0.8309。

这说明 Green uncertainty signal 不是随机噪声，而是能够挑出更值得调用专家的样本。

## 6. 成本更低的超过 dense 点

40% expert call 下，Green uncertainty defer 仍然超过 dense average ensemble：

| Policy         | Expert setting | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Above random 97.5% |
| -------------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------ |
| low_confidence | both_average   |              40% |                   0.40 |   0.8364 |         +0.0055 | True               |
| low_margin     | both_average   |              40% |                   0.40 |   0.8355 |         +0.0045 | True               |
| high_entropy   | both_average   |              40% |                   0.40 |   0.8345 |         +0.0036 | True               |

解释：

40% dense expert cost 已经可以超过 dense average ensemble，说明 Green-first sparse invocation 不只是提高性能，也确实带来成本节省空间。

## 7. 成本效率最高区域

30% expert call 下，Green uncertainty defer 已基本接近 dense average ensemble：

| Policy         | Expert setting | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Gain per 100 expert calls |
| -------------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------------: |
| high_entropy   | both_average   |              30% |                   0.30 |   0.8300 |         -0.0009 |                  0.007713 |
| low_confidence | both_average   |              30% |                   0.30 |   0.8255 |         -0.0055 |                  0.007025 |
| low_margin     | both_average   |              30% |                   0.30 |   0.8227 |         -0.0082 |                  0.006612 |

解释：

30% dense expert cost 下，high_entropy defer 基本接近 dense average ensemble。虽然略低于 dense average ensemble，但单位专家调用收益高，适合作为成本效率候选点。

## 8. 单专家轻量模式

单专家 ConvNeXt defer 是更轻量的可部署候选：

| Expert   | Policy         | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Above random 97.5% |
| -------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------ |
| ConvNeXt | low_confidence |              50% |                   0.25 |   0.8209 |         -0.0100 | True               |
| ConvNeXt | high_entropy   |              30% |                   0.15 |   0.8191 |         -0.0118 | True               |
| ConvNeXt | low_margin     |              40% |                   0.20 |   0.8191 |         -0.0118 | True               |

解释：

ConvNeXt 单专家模式成本低，50% expert call 只相当于 dense expert cost 的 25%。

但它的性能上限低于 dense average ensemble，因此更适合作为轻量模式，而不是性能最强方案。

## 9. 与 random defer 的关系

所有主要 Green uncertainty defer 策略均超过对应 random defer 的 97.5% 上界。

这说明当前结果不是随机调用专家造成的偶然提升，而是 Green uncertainty signal 确实提供了有效排序信息。

以 both_average、50% expert call 为例：

| 方法                             | Accuracy |
| ------------------------------ | -------: |
| Random defer mean              |   0.8093 |
| Random defer 97.5% upper bound |   0.8209 |
| Green low-margin defer         |   0.8427 |
| Green low-confidence defer     |   0.8418 |
| Green high-entropy defer       |   0.8409 |

## 10. 当前判断

当前结论比初版更积极：

* Green-only 不是最强模型；
* 但 Green 的 uncertainty signal 能识别更值得专家介入的样本；
* random defer 不能有效释放互补性；
* Green uncertainty defer 明显优于 random defer；
* 40%–50% dense expert cost 下，Green-first sparse invocation 可以超过 dense average ensemble；
* 30% dense expert cost 下，Green-first sparse invocation 已基本接近 dense average ensemble；
* 单专家 ConvNeXt defer 更轻量，但性能上限低于 both experts average；
* 因此，v0.8.0b 支持继续做 cost-aware sparse routing analysis。

## 11. 边界

当前结果仍不能解释为“复杂 Router 已经成功”。

必须保留以下边界：

1. 当前最强策略依赖 both experts average，即被 defer 的样本同时调用 ConvNeXt 和 RETFound-MAE。
2. both_average 的准确率最高，但成本高于只调用单个专家。
3. Oracle any-expert upper bound 是后验理论上限，不可部署。
4. 当前结果只在 APTOS test 上成立，尚未验证外部数据集。
5. 当前 RETFound-MAE 是 existing reproduced checkpoint，尚未完成 official-protocol control baseline。
6. 当前 routing signal 仍然是简单 uncertainty signal，不是 learned router。
7. 当前 cost-aware 指标衡量的是专家 forward-call equivalent cost，不是静态 GPU 显存成本。
8. 在线部署中的模型常驻、按需加载、CPU offload、多服务部署等属于后续工程优化问题，不作为 v0.8.0b 的主实验目标。

## 12. Go/No-Go 更新

当前判断：

* Go for sparse routing feasibility.
* Go for lightweight routing signal analysis.
* Go for cost-aware sparse routing summary.
* No-Go for claiming routing success.
* No-Go for immediate complex Router training.

下一步应优先做：

1. 同步更新 `go_no_go_summary.md`；
2. 补充 official-protocol RETFound-MAE control baseline；
3. 在外部数据集上验证 Green uncertainty defer 是否仍优于 random defer；
4. 再决定是否进入 learned lightweight router。
