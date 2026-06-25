# RETFound-MAE 协议控制基线计划

## 1. 目的

当前 GreenScout 第一轮可行性分析使用的是已有 RETFound-MAE 复现实验 checkpoint。

该 checkpoint 可以作为第一轮可行性证据，但不应直接表述为严格官方协议复现。

为了避免 GreenScout 结论依赖单一专家训练配置，v0.8.0b 将补充 official-protocol RETFound-MAE control baseline，并在该基线上重复互补性分析与稀疏专家调用曲线。

## 2. 当前已有 RETFound-MAE 专家模型

当前使用的已有专家模型路径：

experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/checkpoints/retfound_mae_cfp_best.pth

当前配置文件路径：

experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/configs/config.json

定位：

existing reproduced RETFound-MAE expert

说明：

该模型是当前 GreenScout 第一轮分析中的专家模型之一，但不能直接等同于严格 official-protocol RETFound-MAE reproduction。

## 3. 新增 official-protocol control baseline

新增模型定位：

official-protocol RETFound-MAE control baseline

该 baseline 的作用不是推翻当前结果，而是作为协议控制实验，用于回答以下问题：

1. 当前 GreenScout 互补性是否依赖某个 RETFound-MAE 训练配置。
2. 如果 RETFound-MAE 按官方协议重新训练，Green 的 unique correction 是否仍然存在。
3. Green-first sparse expert invocation curve 是否对专家训练协议敏感。
4. 稀疏调用收益是否随着专家模型强度变化而变化。

## 4. 需要明确记录的官方协议信息

训练 official-protocol RETFound-MAE 前，需要记录：

- protocol source：采用哪个官方仓库、哪个 README、哪个脚本
- model / architecture
- pretrained checkpoint
- input size
- batch size
- epochs
- base learning rate
- layer decay
- weight decay
- drop path
- warmup schedule
- label smoothing
- global pooling
- checkpoint selection rule
- random seed
- train / val / test split
- 输出 checkpoint 路径
- 输出 prediction CSV 路径

如果旧 RETFound_MAE 仓库与当前统一 RETFound 仓库的 fine-tuning recipe 存在差异，需要显式说明采用哪一套 protocol，不混写为唯一官方最优设置。

## 5. 对比模型池

### Pool A：当前已有专家池

- RETFound-Green linear probe
- ConvNeXt-Tiny existing expert
- RETFound-MAE existing reproduced expert

### Pool B：官方协议控制专家池

- RETFound-Green linear probe
- ConvNeXt-Tiny existing expert
- RETFound-MAE official-protocol control

## 6. 两个模型池都需要重复的分析

Pool A 和 Pool B 都需要计算：

1. 单模型 Acc / Macro-F1 / QWK
2. average ensemble
3. oracle expert selection
4. pairwise error overlap
5. unique correction
6. Green-first sparse expert invocation curve
7. random defer baseline
8. oracle defer upper bound

## 7. 判断逻辑

如果 Pool A 和 Pool B 得到一致趋势：

GreenScout 的成本-性能折中结论对 RETFound-MAE 训练协议不敏感。

如果 Pool A 和 Pool B 差异明显：

GreenScout 的收益依赖专家模型强度和专家池构成。

如果 official-protocol RETFound-MAE 显著减少 Green unique correction，并使 sparse invocation gain 消失：

GreenScout 应降级，只保留低成本 embedding scout 的工程可行性，不继续训练复杂 Router。

如果 official-protocol RETFound-MAE 下 Green 仍有 unique correction，且 sparse invocation curve 仍优于 random defer：

GreenScout 方向得到更强支持，可以继续考虑 learned router。

## 8. 当前阶段结论表述

当前结果应表述为：

基于 existing reproduced RETFound-MAE expert 的第一轮 GreenScout feasibility evidence。

不能表述为：

基于严格官方协议 RETFound-MAE 的最终对照结论。

最终 GreenScout 结论需要在 official-protocol RETFound-MAE control baseline 上复核。

## 9. Protocol-control 判定阈值

official-protocol RETFound-MAE control baseline 的目的不是要求结果与 existing reproduced checkpoint 完全一致，而是判断 Green-first sparse invocation 的核心结论是否依赖某一个 RETFound-MAE 复现实验 checkpoint。

所有判断必须在同一模型池内完成。

Pool A：

* RETFound-Green linear probe
* ConvNeXt-Tiny existing expert
* RETFound-MAE existing reproduced expert

Pool B：

* RETFound-Green linear probe
* ConvNeXt-Tiny existing expert
* RETFound-MAE official-protocol control expert

Pool B 需要重新计算：

1. 单模型 Acc / Macro-F1 / QWK；
2. Dense average ensemble；
3. Oracle any-expert upper bound；
4. Random defer baseline；
5. Green confidence / margin / entropy defer；
6. Cost-aware sparse routing summary；
7. Single-vs-both expert tradeoff。

### 9.1 Strong pass

满足以下条件时，认为 GreenScout sparse invocation 结论对 RETFound-MAE protocol 较稳健：

1. 在 Pool B 中，Green uncertainty defer 至少有一个 30%–50% expert-call 设置超过对应 random defer 的 97.5% 上界。
2. 最优 Green uncertainty defer 相对 random defer mean 的 Accuracy 增益不低于 +0.02。
3. 至少一个 40%–50% cost setting 达到或超过 Pool B 的 dense average ensemble。
4. 或者至少一个 30% cost setting 接近 Pool B dense average ensemble，差距不超过 0.005。
5. 结果仍然支持“Green uncertainty signal 能挑出更值得专家介入的样本”。

此时结论可以写为：

Green-first sparse invocation shows a robust positive cost-performance tradeoff under both existing reproduced and official-protocol RETFound-MAE expert settings.

### 9.2 Moderate pass

满足以下条件时，认为结论部分稳健，但需要降级表述：

1. Green uncertainty defer 明显超过 random defer 的 97.5% 上界；
2. 最优 Green uncertainty defer 相对 random defer mean 的 Accuracy 增益不低于 +0.01；
3. 但 40%–50% cost setting 不能达到 Pool B dense average ensemble；
4. 或者只能超过 best single，不能超过 dense average ensemble。

此时结论应降级为：

Green uncertainty signal remains informative under official-protocol control, but the claim of best single，不能超过 dense average ensemble。

此时结论应降 exceeding dense average ensemble is not robust.

### 9.3 Weak pass / downgrade

出现以下情况时，结论降级：

1. Green uncertainty defer 仍然高于 random defer mean，但不能稳定超过 random defer 97.5% 上界；
2. 最优 Green uncertainty defer 相对 random defer mean 的 Accuracy 增益小于 +0.01；
3. sparse invocation 只能带来轻微提升，不能稳定超过 best single 或 dense average ensemble。

此时结论应写为：

Green-first sparse invocation shows limited routing signal under official-protocol control. The current result should be treated as preliminary feasibility evidence rather than a robust routing result.

### 9.4 No-Go

出现以下情况时，不进入 learned Router：

1. Green uncertainty defer 与 random defer 基本相当；
2. Green uncertainty defer 不能超过 random defer 97.5% 上界；
3. 30%–50% expert-call setting 不能超过 best single；
4. oracle upper bound 仍然较高，但现实可实现 routing signal 无法挖出该空间。

此时结论应写为：

The model pool may contain oracle complementarity, but Green uncertainty is insufficient as a deployable routing signal. Learned Router training is not justified at this stage.

### 9.5 重要边界

Pool B 的阈值不能直接套用 Pool A 的绝对数字。

例如，Pool A 中 dense average ensemble Acc 为 0.8309，best sparse result 为 0.8427。但 Pool B 更换 official-protocol RETFound-MAE 后，dense average ensemble、random defer、oracle upper bound 都可能变化。

因此，所有判断必须使用 Pool B 内部重新计算的相对指标：

* gain over Green-only；
* gain over best single；
* gain over dense average ensemble；
* gain over random defer mean；
* whether above random defer 97.5% upper bound；
* relative cost vs dense；
* gain per expert forward-call equivalent cost。

不能用 Pool A 的 0.8427 作为 Pool B 的硬性通过阈值。

## 10. Protocol source lock

本控制实验采用“协议来源锁定”原则。official-protocol RETFound-MAE control baseline 不表述为唯一官方最优设置，而表述为基于公开官方 fine-tuning recipe 的协议控制复现。

当前锁定的主要 protocol source 为：

* Repository：rmaphoh/RETFound
* Fine-tuning entry：train.sh
* Adaptation mode：finetune
* Model family：RETFound-MAE
* MODEL：RETFound_mae
* MODEL_ARCH：retfound_mae
* FINETUNE：RETFound_mae_natureCFP
* input_size：224
* epochs：50
* batch_size：24
* global_pool：enabled
* world_size：1
* downstream task：APTOS2019 5-class DR classification
* split：保持当前 APTOS train / val / test 划分不变

本实验不声称该 protocol 是 RETFound-MAE 的唯一官方最优训练设置。若旧 RETFound_MAE / Nature repo 与当前统一 rmaphoh/RETFound repo 的 fine-tuning recipe 存在差异，应在报告中显式说明采用哪一个 protocol source。

本实验的目的不是追求最高单模型性能，而是检查 v0.8.0b Green-first sparse invocation 的结论是否依赖 existing reproduced RETFound-MAE checkpoint。

因此，official-protocol control baseline 需要重新生成：

1. RETFound-MAE official-protocol checkpoint；
2. RETFound-MAE official-protocol APTOS test prediction CSV；
3. Pool B complementarity metrics；
4. Pool B random defer baseline；
5. Pool B uncertainty defer curve；
6. Pool B cost-aware sparse routing summary；
7. Pool B single-vs-both expert tradeoff。

所有 Pool B 判断必须使用 Pool B 内部重新计算的相对指标，不能直接套用 Pool A 的绝对数值。


