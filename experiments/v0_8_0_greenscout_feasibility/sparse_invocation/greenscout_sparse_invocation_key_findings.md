# v0.8.0b Green-first sparse expert invocation curve

## 1. 基线结果

APTOS test，1100 张图像。

| 方法                                      | Accuracy | Macro-F1 |    QWK | Error | 说明                   |
| --------------------------------------- | -------: | -------: | -----: | ----: | -------------------- |
| RETFound-Green linear probe             |   0.7791 |   0.6394 | 0.8653 |   243 | 可部署低成本 scout         |
| ConvNeXt-Tiny                           |   0.8136 |   0.6496 | 0.8615 |   205 | 可部署单专家               |
| RETFound-MAE existing reproduced expert |   0.8036 |   0.5834 | 0.8659 |   216 | 可部署单专家               |
| Dense average ensemble                  |   0.8309 |   0.6935 | 0.8875 |   186 | 三模型全部运行，概率平均，可部署但成本高 |
| Oracle any-expert upper bound           |   0.9191 |      不主报 |    不主报 |    89 | 后验理论上限，不可部署          |

重要修正：

此前 complementarity 结果中 average ensemble 被错误写成 0.9191。重新核查后确认，0.9191 对应的是 oracle any-expert upper bound，不是 dense average ensemble。

真实 dense average ensemble 是三模型概率平均后再 argmax，Accuracy 为 0.8309，Macro-F1 为 0.6935，QWK 为 0.8875。

Oracle any-expert upper bound 的 Accuracy 为 0.9191，Error 为 89。该指标只作为模型池互补空间的后验诊断指标，不作为现实可部署方法的性能结果。

Oracle any-expert 的 Macro-F1 / QWK 不作为主结论报告，因为在三模型全部预测错误的 89 张图像上，oracle 仍无法选到正确标签；若对这些样本采用不同 fallback 策略，Macro-F1 / QWK 会发生变化。因此后续报告中主要使用 Oracle Accuracy / Error 来描述理论上限。

## 2. Oracle upper bound

以 Green-only 为起点：

| Expert 设置                    | Expert call budget | Oracle up-to-k Accuracy | 说明                                           |
| ---------------------------- | -----------------: | ----------------------: | -------------------------------------------- |
| Green → ConvNeXt             |                10% |                  0.8791 | 110 张 Green 错、ConvNeXt 对，10% budget 已吃满收益    |
| Green → RETFound-MAE         |                10% |                  0.8791 | 10% budget 已获得主要收益                           |
| Green → RETFound-MAE         |                20% |                  0.8873 | 119 张 Green 错、RETFound-MAE 对，20% budget 吃满收益 |
| Green → both experts average |                10% |                  0.8791 | 10% budget 已获得主要收益                           |
| Green → both experts average |                20% |                  0.8855 | 20% budget 吃满收益                              |
| Green → dense average        |                10% |                  0.8545 | 83 张 Green 错、dense average 对，10% budget 吃满收益 |

这说明低成本稀疏专家调用本身存在明显理论空间。

但该空间是 oracle upper bound，即事后知道哪些图像应该调用专家，不代表当前已有可部署路由信号。

## 3. Random defer baseline

random defer 是现实可部署但无信息的下限，用来判断 Green uncertainty defer 是否真的有用。

代表性结果：

| Expert 设置                    | Expert call rate | Random defer Acc mean |        95% 区间 | 说明                          |
| ---------------------------- | ---------------: | --------------------: | ------------: | --------------------------- |
| Green → ConvNeXt             |              50% |                0.7966 | 0.7845–0.8082 | 低于最佳单模型 ConvNeXt            |
| Green → RETFound-MAE         |              50% |                0.7915 | 0.7791–0.8045 | 低于最佳单模型 ConvNeXt            |
| Green → both experts average |              50% |                0.8093 | 0.7982–0.8209 | 接近但仍低于 ConvNeXt             |
| Green → dense average        |              50% |                0.8052 | 0.7964–0.8136 | 明显低于 dense average ensemble |

random defer 的结果说明：仅仅“随机调用专家”并不能有效释放模型池互补性。

因此，后续判断 routing signal 是否有价值，必须比较 uncertainty defer 是否显著超过 random defer。

## 4. Green uncertainty defer 结果

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

这说明 Green uncertainty signal 并非无效。它已经能在部分预算下识别出更值得调用专家的样本。

## 5. 当前判断

当前结果比初版判断更积极，但仍不能声称复杂 Router 已经成立。

更准确的结论是：

* Green 作为低成本 scout 可运行；
* Green linear probe 具备基本五分类能力；
* 三模型池存在明显 oracle 互补空间；
* dense average ensemble 的实际提升有限；
* random defer 不能有效释放互补性；
* Green confidence / margin / entropy defer 明显优于 random defer；
* 在 both experts average 设置下，40%–50% expert call 已经可以超过 dense average ensemble；
* 但 oracle any-expert 0.9191 仍是不可部署的后验理论上限，不能作为现实目标；
* 是否值得 learned router，取决于后续是否能进一步超过当前 uncertainty defer。

当前 Go/No-Go 结论应调整为：

* Go for sparse routing feasibility.
* Go for lightweight routing signal analysis.
* No-Go for claiming routing success.
* No-Go for immediate complex Router training.

## 6. 方法边界

当前 v0.8.0b 仍然不是临床部署方案。

必须明确以下边界：

1. Oracle any-expert upper bound 使用真实标签，是后验理论上限，不可部署。
2. Oracle up-to-k 只用于判断理论空间，不代表现实路由性能。
3. Dense average ensemble 是可部署但高成本的参照，不是低成本方案。
4. Random defer 是可部署下限，用于判断 uncertainty defer 是否真的有信息量。
5. Green uncertainty defer 是当前最简单的可部署路由信号。
6. both experts average 表示被 defer 的样本同时调用 ConvNeXt 和 RETFound-MAE 并做专家平均，成本高于只调用单个专家。
7. 当前结果只基于 APTOS test，尚未证明外部数据集可复现。
8. 当前 RETFound-MAE 来自 existing reproduced checkpoint，尚未完成 official-protocol control baseline。

## 7. 下一步

下一步应做三件事。

第一，整理 sparse invocation 的正式结果表：

* baseline methods
* random defer summary
* uncertainty defer curve
* oracle up-to-k curve
* best policy by budget
* gap to best single
* gap to dense average ensemble
* gain over random defer

第二，补充成本感知指标。

当前 both experts average 在 40%–50% expert call 下效果最好，但它对被 defer 样本同时调用两个专家，真实成本高于只调用 ConvNeXt 或只调用 RETFound-MAE。

因此下一步不能只看 Accuracy，还要计算：

* expert_call_rate
* expert_call_count
* single_expert_call_equivalent
* relative_cost_vs_dense_ensemble
* Accuracy per expert call
* gain over random per expert call

第三，决定是否进入 learned routing signal。

只有当 lightweight learned signal 能稳定超过：

1. random defer；
2. confidence / margin / entropy；
3. best single；
4. dense average ensemble 附近的现实参照；

才值得进入 learned router。

当前最稳妥的下一步不是训练复杂 Router，而是做 cost-aware sparse routing analysis，把“性能收益”和“专家调用成本”放到同一张表里。
