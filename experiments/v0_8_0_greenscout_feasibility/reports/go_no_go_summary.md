# v0.8.0 GreenScout Routing Feasibility Audit：Go/No-Go Summary

## 1. 当前阶段边界

本阶段验证的问题是：

低成本 RETFound-Green scout 与现有专家模型之间，是否存在值得继续探索的成本感知稀疏专家调用空间。

当前不训练复杂 Router，不做临床分流，不定义临床阈值，不声称执行前路由方法已经成立。

当前所有 sparse invocation 结果均为 APTOS test 上的离线分析，用于验证 feasibility，不等同于线上部署 benchmark。

## 2. 当前 Go/No-Go 结论

当前判断：

* Go for sparse routing feasibility.
* Go for lightweight routing signal analysis.
* Go for cost-aware sparse routing summary.
* No-Go for claiming routing success.
* No-Go for immediate complex Router training.

中文表述：

* 继续推进 Green-first sparse invocation 可行性分析；
* 继续分析轻量 routing signal；
* 继续做成本感知专家调用核算；
* 暂不声称路由方法已经成功；
* 暂不进入复杂 learned Router 训练。

## 3. Green 工程可运行性

RETFound-Green 已在当前 ophagent 环境中跑通。

* checkpoint：83.35 MB
* 架构：vit_small_patch14_reg4_dinov2
* 输入尺寸：392 × 392
* 输出：384 维 embedding
* demo_samples：15 / 15 成功
* APTOS embedding：

  * train：2048 张
  * val：514 张
  * test：1100 张
* APTOS test embedding median latency：约 1.87 ms / image
* 全量 embedding 峰值显存：约 592 MB

结论：

Green 作为低成本 embedding scout 的工程可运行性通过。

注意：

当前 Green 的资源占用是在 Green embedding 导出脚本下实测得到。后续如需严格比较 Green / ConvNeXt / RETFound-MAE 的部署成本，应使用同一 benchmark 脚本、同一 batch size、同一设备、同一计时口径重新对齐。

## 4. Green linear probe 结果

冻结 RETFound-Green embedding 后，使用 Logistic Regression 训练五分类线性头。

APTOS test：

* Accuracy：0.7791
* Macro-F1：0.6394
* QWK：0.8653

该结果说明 Green embedding 具备基本五分类可用性，但 Green 单模型不是当前最佳专家模型。

## 5. 三模型互补性结果

参与模型：

1. RETFound-Green linear probe
2. ConvNeXt-Tiny
3. RETFound-MAE existing reproduced expert

APTOS test，1100 张图像：

| 方法                                      | Accuracy | Macro-F1 |    QWK | Error |
| --------------------------------------- | -------: | -------: | -----: | ----: |
| Oracle any-expert upper bound           |   0.9191 |      不主报 |    不主报 |    89 |
| Dense average ensemble                  |   0.8309 |   0.6935 | 0.8875 |   186 |
| ConvNeXt-Tiny                           |   0.8136 |   0.6496 | 0.8615 |   205 |
| RETFound-MAE existing reproduced expert |   0.8036 |   0.5834 | 0.8659 |   216 |
| RETFound-Green linear probe             |   0.7791 |   0.6394 | 0.8653 |   243 |

重要修正：

此前结果中曾将 average ensemble 写成 0.9191。重新核查后确认，0.9191 对应的是 oracle any-expert upper bound，即三个模型中只要有一个模型预测正确，就视为 oracle 可选对。

真实的 dense average ensemble 是对三个模型的五分类概率取平均后再 argmax，其 Accuracy 为 0.8309，Macro-F1 为 0.6935，QWK 为 0.8875。

Oracle any-expert upper bound 的 Accuracy 为 0.9191，Error 为 89。该指标使用测试集真实标签信息，是后验理论上限，不是现实可部署方法。

Oracle any-expert 的 Macro-F1 / QWK 不作为主结论报告，因为在三模型全部预测错误的 89 张图像上，oracle 仍无法选到正确标签；若对这些样本采用不同 fallback 策略，Macro-F1 / QWK 会发生变化。因此后续主要使用 Oracle Accuracy / Error 描述理论上限。

关键数字：

* 最佳单模型：ConvNeXt-Tiny，Accuracy 0.8136
* Dense average ensemble：Accuracy 0.8309
* Oracle any-expert upper bound：Accuracy 0.9191
* Dense average ensemble 相对最佳单模型 Accuracy 增益：+0.0173
* Oracle any-expert 相对最佳单模型 Accuracy 增益：+0.1055
* Oracle any-expert 相对 dense average ensemble Accuracy 增益：+0.0882
* 三模型全部错误：89 张
* 三模型全部正确：729 张
* 至少一个模型正确且至少一个模型错误：282 张

独有纠错：

* RETFound-MAE existing reproduced expert：44 张
* RETFound-Green linear probe：36 张
* ConvNeXt-Tiny：35 张

解释：

dense average ensemble 的提升有限，说明简单概率平均只能利用一小部分模型互补性。

oracle any-expert upper bound 很高，说明模型池中确实存在明显互补空间。

因此当前结论不是“average ensemble 已经吃满 oracle”，而是：

模型池存在较大 oracle 空间，但简单 dense average ensemble 只利用了其中一小部分。后续重点应转向 routing signal 是否能识别“Green 错、专家对”的样本。

## 6. Sparse invocation 结果

v0.8.0b 进一步分析了 Green-first sparse expert invocation。

核心问题：

在所有图像先运行 Green 的前提下，只对一部分样本调用专家模型，能否以低于 dense ensemble 的专家调用成本，达到接近或超过 dense ensemble 的性能？

### 6.1 Random defer baseline

random defer 是现实可部署但无信息的下限，用来判断 Green uncertainty defer 是否真的有用。

以 50% expert call 为例：

| Expert 设置                    | Random defer Acc mean |        95% 区间 | 说明                          |
| ---------------------------- | --------------------: | ------------: | --------------------------- |
| Green → ConvNeXt             |                0.7966 | 0.7845–0.8082 | 低于最佳单模型 ConvNeXt            |
| Green → RETFound-MAE         |                0.7915 | 0.7791–0.8045 | 低于最佳单模型 ConvNeXt            |
| Green → both experts average |                0.8093 | 0.7982–0.8209 | 接近但仍低于 ConvNeXt             |
| Green → dense average        |                0.8052 | 0.7964–0.8136 | 明显低于 dense average ensemble |

random defer 的结果说明：仅仅“随机调用专家”并不能有效释放模型池互补性。

### 6.2 Green uncertainty defer

使用 Green confidence / margin / entropy 作为路由信号时，结果明显强于 random defer。

代表性结果：

| 路由设置                                        | Expert call rate | Accuracy | Macro-F1 |    QWK | 说明                        |
| ------------------------------------------- | ---------------: | -------: | -------: | -----: | ------------------------- |
| Green low-margin → both experts average     |              50% |   0.8427 |   0.7097 | 0.8952 | 当前最佳 uncertainty defer    |
| Green low-confidence → both experts average |              50% |   0.8418 |   0.7084 | 0.8950 | 接近最佳                      |
| Green high-entropy → both experts average   |              50% |   0.8409 |   0.7069 | 0.8948 | 接近最佳                      |
| Green low-confidence → both experts average |              40% |   0.8364 |   0.6981 | 0.8938 | 超过 dense average ensemble |
| Green low-margin → both experts average     |              40% |   0.8355 |   0.6978 | 0.8937 | 超过 dense average ensemble |
| Green high-entropy → both experts average   |              30% |   0.8300 |   0.6928 | 0.8914 | 接近 dense average ensemble |

关键观察：

1. 50% expert call 时，Green uncertainty defer + both experts average 最高达到 Acc 0.8427，高于 dense average ensemble 的 0.8309。
2. 40% expert call 时，low-confidence / low-margin + both experts average 已经超过 dense average ensemble。
3. 30% expert call 时，high-entropy + both experts average 接近 dense average ensemble。
4. uncertainty defer 明显强于 random defer。以 both experts average、50% expert call 为例：

   * random defer mean Acc：0.8093
   * random defer 97.5% 上界：0.8209
   * low-margin uncertainty defer Acc：0.8427

结论：

Green uncertainty signal 并非无效。它已经能在部分预算下识别出更值得调用专家的样本。

## 7. Cost-aware sparse routing 结果

只看 Accuracy 不够，因为 both experts average 的准确率最高，但它对被 defer 的样本同时调用 ConvNeXt 和 RETFound-MAE，成本高于只调用单个专家。

本阶段使用 expert forward-call equivalent cost 作为成本指标。

这里的 cost-aware sparse routing 主要衡量：

* 专家 forward 调用次数；
* 专家模型计算量；
* 专家 forward activation 开销；
* 平均推理延迟和能耗；
* 专家服务负载。

它不直接表示静态 GPU 显存占用。

### 7.1 最强性能点

| Policy         | Expert setting | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Above random 97.5% |
| -------------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------ |
| low_margin     | both_average   |              50% |                   0.50 |   0.8427 |         +0.0118 | True               |
| low_confidence | both_average   |              50% |                   0.50 |   0.8418 |         +0.0109 | True               |
| high_entropy   | both_average   |              50% |                   0.50 |   0.8409 |         +0.0100 | True               |

解释：

在只使用 dense ensemble 50% 专家调用成本的情况下，Green low-margin defer + both experts average 达到 Acc 0.8427，高于 dense average ensemble 的 Acc 0.8309。

### 7.2 成本更低的超过 dense 点

40% expert call 下，Green uncertainty defer 仍然超过 dense average ensemble：

| Policy         | Expert setting | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Above random 97.5% |
| -------------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------ |
| low_confidence | both_average   |              40% |                   0.40 |   0.8364 |         +0.0055 | True               |
| low_margin     | both_average   |              40% |                   0.40 |   0.8355 |         +0.0045 | True               |
| high_entropy   | both_average   |              40% |                   0.40 |   0.8345 |         +0.0036 | True               |

解释：

40% dense expert cost 已经可以超过 dense average ensemble，说明 Green-first sparse invocation 不只是提高性能，也确实带来成本节省空间。

### 7.3 成本效率最高区域

30% expert call 下，Green uncertainty defer 已基本接近 dense average ensemble：

| Policy         | Expert setting | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Gain per 100 expert calls |
| -------------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------------: |
| high_entropy   | both_average   |              30% |                   0.30 |   0.8300 |         -0.0009 |                  0.007713 |
| low_confidence | both_average   |              30% |                   0.30 |   0.8255 |         -0.0055 |                  0.007025 |
| low_margin     | both_average   |              30% |                   0.30 |   0.8227 |         -0.0082 |                  0.006612 |

解释：

30% dense expert cost 下，high_entropy defer 基本接近 dense average ensemble。虽然略低于 dense average ensemble，但单位专家调用收益高，适合作为成本效率候选点。

### 7.4 单专家轻量模式

单专家 ConvNeXt defer 是更轻量的可部署候选：

| Expert   | Policy         | Expert call rate | Relative cost vs dense | Accuracy | Gain over dense | Above random 97.5% |
| -------- | -------------- | ---------------: | ---------------------: | -------: | --------------: | ------------------ |
| ConvNeXt | low_confidence |              50% |                   0.25 |   0.8209 |         -0.0100 | True               |
| ConvNeXt | high_entropy   |              30% |                   0.15 |   0.8191 |         -0.0118 | True               |
| ConvNeXt | low_margin     |              40% |                   0.20 |   0.8191 |         -0.0118 | True               |

解释：

ConvNeXt 单专家模式成本低，50% expert call 只相当于 dense expert cost 的 25%。

但它的性能上限低于 dense average ensemble，因此更适合作为轻量模式，而不是性能最强方案。

## 8. 成本定义与部署边界

本阶段的 cost-aware sparse routing 主要衡量专家 forward-call equivalent cost，而不是静态 GPU 显存成本。

在最简单的低延迟在线部署方式下，RETFound-Green、ConvNeXt-Tiny 和 RETFound-MAE 可能全部常驻 GPU。此时，即使某张图像没有调用专家模型，专家模型的权重仍然占用显存。因此，Green-first sparse invocation 不应被表述为直接节省模型权重显存。

如果所有模型都常驻同一张 GPU，sparse invocation 主要节省平均计算成本，而不是静态模型显存。

如果采用 Green 常驻、专家按需加载或专家服务独立部署的系统架构，sparse invocation 还可能进一步减少专家服务负载和硬件需求。但这需要额外的系统 benchmark 验证，不能由当前离线实验直接推出。

因此，当前报告应使用如下表述：

Green-first sparse invocation 在 APTOS test 上降低了专家 forward-call equivalent cost，并在部分预算下达到或超过 dense average ensemble 性能。

不应表述为：

Green-first sparse invocation 直接节省了 40%–50% GPU 显存。

## 9. 辩证判断

### 9.1 支持继续的证据

1. Green 能稳定加载、推理并导出 384 维 embedding。
2. Green linear probe 在 APTOS test 上达到 Acc 0.7791 / Macro-F1 0.6394 / QWK 0.8653，不是废模型。
3. Green 有 36 张 unique correction，说明它不是完全冗余模型。
4. 三模型之间存在 282 张 mixed correct/wrong cases，说明存在明显互补性。
5. Oracle any-expert upper bound 明显高于最佳单模型，说明专家选择存在理论上限空间。
6. Green uncertainty defer 明显强于 random defer，说明 Green uncertainty signal 有有效排序信息。
7. 在 both experts average 设置下，40%–50% dense expert cost 已经可以超过 dense average ensemble。
8. 30% dense expert cost 下，Green-first sparse invocation 已基本接近 dense average ensemble。

### 9.2 主要风险

1. Oracle any-expert upper bound 是后验理论上限，不代表现实可部署路由能力。
2. 当前最强策略依赖 both experts average，即被 defer 的样本同时调用 ConvNeXt 和 RETFound-MAE。
3. both_average 的准确率最高，但成本高于只调用单个专家。
4. 当前结果只在 APTOS test 上成立，尚未验证外部数据集。
5. 当前 RETFound-MAE 是 existing reproduced checkpoint，尚未完成 official-protocol control baseline。
6. 当前 routing signal 仍然是简单 uncertainty signal，不是 learned router。
7. 当前 cost-aware 指标衡量的是专家 forward-call equivalent cost，不是静态 GPU 显存成本。
8. 在线部署中的模型常驻、按需加载、CPU offload、多服务部署等属于后续工程优化问题，不作为 v0.8.0b 的主实验目标。

## 10. 下一步

下一步应优先做三件事：

1. 补充 official-protocol RETFound-MAE control baseline，检查当前结论是否依赖 existing reproduced RETFound-MAE checkpoint。
2. 在外部数据集上验证 Green uncertainty defer 是否仍优于 random defer。
3. 再决定是否进入 learned lightweight router。

当前最稳妥的下一步不是训练复杂 Router，而是完成 protocol-control 和外部验证准备。
