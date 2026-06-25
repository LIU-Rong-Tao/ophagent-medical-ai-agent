# v0.8.0 Protocol-Control Summary: Scout-to-Expert Model Orchestration

## 1. 阶段定位

本阶段从单一 GreenScout feasibility 进一步推进到 protocol-control setting 下的 scout-to-expert model orchestration 分析。

核心问题不再是证明某一个 scout 一定最优，而是回答：

- sparse expert invocation 是否优于随机专家调用；
- 不同 scout / expert / budget 组合形成怎样的 accuracy-cost tradeoff；
- expert-call budget 是否能转化为实际 forward-cost 节省；
- GreenScout 叙事是否应升级为更一般的模型中转台 / model orchestration framework。

## 2. 实验组成

### v0.8.0d: Scout Ablation

输入为 APTOS2019 test split 上的三模型 standardized predictions：

- `retfound_green_linear_probe`
- `convnext_tiny`
- `retfound_mae_cfp_official_protocol`

比较四类 sparse routing 设置：

- A: Green scout -> ConvNeXt + RETFound official-protocol average
- B: Green scout -> ConvNeXt only
- C: Green scout -> RETFound official-protocol only
- D: ConvNeXt scout -> RETFound official-protocol only

主要评估：

- accuracy / macro-F1 / QWK
- above random p97.5
- oracle same-budget upper bound
- rescued scout errors
- expert-induced errors
- net error reduction
- DR-specific high-risk undergrading event capture

### v0.8.0e: Actual Forward-Cost Benchmark

在同一 APTOS2019 test split 上重新测量三条 online inference 链路的 forward-only cost：

- Green online: RETFound-Green encoder forward + sklearn linear-probe predict_proba
- ConvNeXt online: ConvNeXt-Tiny classifier forward
- RETFound online: RETFound-MAE official-protocol classifier forward

并基于 v0.8.0d 的 selected_n 估算 sparse system cost。

当前成本结果是 single-GPU forward-only benchmark and sparse system estimate，不等同于完整服务压测。

## 3. 主要效果结论

v0.8.0d 结果显示，sparse routing 并非随机有效，而是具有明确 routing signal：

- 所有主要 scout ablation 设置在 30% / 40% / 50% expert-call budget 下均超过 random p97.5。
- 30%–40% budget 下，ConvNeXt scout -> RETFound official-protocol 的 accuracy 表现更强。
- 50% budget 下，Green scout -> ConvNeXt + RETFound average 取得最高 accuracy 和最大净错误减少。
- 但没有单一 scout 在所有指标、所有 budget 下全面最优。

50% budget 的关键对照：

| setting | accuracy | n_error | rescued_scout_errors | expert_induced_errors | net_error_reduction |
|---|---:|---:|---:|---:|---:|
| A: Green -> ConvNeXt+RETFound | 0.8527 | 162 | 114 | 33 | 81 |
| C: Green -> RETFound | 0.8482 | 167 | 118 | 42 | 76 |
| D: ConvNeXt -> RETFound | 0.8518 | 163 | 83 | 41 | 42 |

解释上，Green scout 暴露出更多 expert-correctable errors，因此在双专家接管时获得较大净错误减少；ConvNeXt scout 自身更强，但留给 RETFound expert 修正的空间更小。

## 4. 主要成本结论

v0.8.0e 结果显示，Green checkpoint 更小，但 forward latency 并不优于 ConvNeXt-Tiny。

单模型 forward-only cost：

| model | estimated ms/image | images/s | PyTorch allocated peak memory MB | checkpoint MB |
|---|---:|---:|---:|---:|
| RETFound-Green linear probe | 1.893 | 528.2 | 592.0 | 83.3 |
| ConvNeXt-Tiny | 0.478 | 2090.7 | 538.9 | 106.2 |
| RETFound-MAE official-protocol | 3.512 | 284.8 | 1479.0 | 1157.1 |

50% budget 的关键 system-cost estimate：

| setting | online no-cache ms/image | cached scout ms/image |
|---|---:|---:|
| A: Green -> ConvNeXt+RETFound | 3.888 | 1.995 |
| C: Green -> RETFound | 3.649 | 1.756 |
| D: ConvNeXt -> RETFound | 2.234 | 1.756 |
| experts-only dense | 3.990 | - |
| all-three dense | 5.883 | - |

因此，50% expert-call budget 确实可以转化为 forward-cost 节省，但节省幅度强烈依赖 scout 本身成本与 expert 组合。

## 5. 综合判断

当前结果不支持继续把主线表述为“GreenScout 是低成本最优 scout”。

更稳妥的结论是：

1. GreenScout 是一个可用的 scout 候选，checkpoint 小，但当前 forward-only latency 不优于 ConvNeXt-Tiny。
2. ConvNeXt-Tiny 在当前环境下是更快的 scout，并且 `D_convnext_scout_to_retfound_only` 在 50% budget 下形成更好的 cost-performance tradeoff。
3. `A_green_scout_to_convnext_retfound_avg` 在 accuracy 上略强，但 online no-cache 成本几乎接近 experts-only dense，因此更适合作为高 accuracy 方案，而不是明显低成本方案。
4. 当前阶段真正成立的主线是 scout-to-expert model orchestration：不同 scout / expert / budget 组合需要在 accuracy、risk capture、expert-induced errors 和 forward cost 之间联合评估。

## 6. 当前边界

- sparse system cost 基于 measured model forward cost 与 selected_n 估算，不是实际部署服务压测。
- forward-only benchmark 不包含 PIL 解码、Resize、Normalize、DataLoader workers、磁盘 I/O、请求排队、模型动态加载和并发调度。
- 当前只在 APTOS2019 DR 五分类任务上验证，DR-specific undergrading 事件不能直接泛化到其他眼病任务。
- 当前不应发布正式 release；更适合作为 v0.8.0 protocol-control 阶段性实验节点。
