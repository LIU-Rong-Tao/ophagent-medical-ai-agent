# v0.8.2b 受控协议评测说明

## 1. 为什么要做 v0.8.2b

v0.8.2 已经完成了模型接入与自动化成本评测，新增了 Swin-Tiny 和 ViT-B ImageNet 作为通用模型筛查候选。

五模型探索结果有价值，但它更适合定位为“模型池探索结果”，不能直接作为最终主实验结论。原因是当前统一评测器会枚举大量组合：

- 多个 scout 候选；
- 多个 expert 子集；
- 多种静态 ensemble；
- 多种 routing policy；
- 多个 expert 调用预算；
- 多种风险事件输出。

这种全组合搜索可以帮助发现候选方向，但如果直接把 top rows 当作主结论，会有明显问题：

1. 实验空间过大，解释成本高；
2. 容易变成在 test set 上挑最优协议；
3. scout/expert 角色边界不清；
4. 多模型组合越来越多，但不一定更好；
5. 主结论会被 exploratory search 淹没。

因此，v0.8.2b 的目标是把当前 evaluator 从“全组合搜索器”收束为“受控协议评测器”。

## 2. 当前五模型探索结果的阶段性判断

当前模型池包括：

- `retfound_green_linear_probe`
- `convnext_tiny`
- `swin_tiny`
- `vit_b_imagenet`
- `retfound_mae_cfp_official_protocol`

五模型探索的主要结论是：

1. `vit_b_imagenet` 接入成功，可以作为普通 ViT screening baseline；
2. `vit_b_imagenet` 没有实质改变当前 cost-performance frontier；
3. 当前最值得保留的 scout 候选仍然是 `convnext_tiny` 和 `swin_tiny`；
4. 当前最重要的 expert 候选仍然是 `retfound_mae_cfp_official_protocol`；
5. GreenScout 和 ViT-B 更适合作为筛查/消融候选，而不是当前主协议核心。

因此，当前不能说“找到了全局最优组合”。更稳的说法是：

> 五模型筛查显示，ViT-B 可接入但未改变当前最优模型编排边界；当前最有前景的是 ConvNeXt/Swin 作为 scout、RETFound official-protocol 作为 expert 的稀疏调用协议。

## 3. Screening、Recipe-control 与 Fixed routing 的边界

本项目不追求为每个 backbone 找到最优训练超参数。

不同架构确实适配不同训练 recipe。ViT、Swin、ConvNeXt、RETFound 可能需要不同的学习率、warmup、weight decay、augmentation、layer-wise learning rate decay 和 scheduler。硬用一套训练参数不能代表每个架构的性能上限。

因此，当前实验分为三层：

### 3.1 Screening baseline

第一层是筛查基线。

目标是用已有可复现 checkpoint 和统一评估口径，快速判断某个模型是否值得进入后续模型池。

这一层可以回答：

- 当前 checkpoint 能不能接入统一 evaluator；
- 当前模型在同一数据集、同一指标、同一成本测量下表现如何；
- 它是否具备成为 scout 或 expert 候选的基本价值。

这一层不能回答：

- 这个架构是否已经达到最优；
- 这个模型是否理论上不如另一个模型；
- 这个模型是否应该被永久淘汰。

### 3.2 Recipe-control for finalists

第二层是候选模型的训练协议控制。

只有当某个模型满足以下条件之一时，才考虑补官方/论文/推荐 recipe：

- 进入 cost-performance frontier；
- 进入 scout 候选前列；
- 进入 expert 候选前列；
- 它的结果会影响最终主结论；
- 它表现异常，但理论上应该是强候选。

例如：

- `retfound_mae_cfp_official_protocol` 是关键 expert，因此已经补 official-protocol control；
- `swin_tiny` 如果持续成为主 scout 候选，后续可以考虑补推荐 recipe；
- `vit_l_official_like` 当前表现低于 ViT-B，暂时只记录为 recipe mismatch signal，不展开成主任务。

### 3.3 Fixed model pool routing

第三层才是本项目的核心。

一旦模型池确定，就冻结 checkpoint 和 prediction CSV。之后主问题变成：

> 在固定模型池中，scout-to-expert routing 能否在有限 expert 调用预算下，实现更好的成本、性能和医学风险覆盖折中？

这一层研究的是模型编排，不是 backbone 调参。

## 4. v0.8.2b 的主任务

v0.8.2b 的主任务是建立受控协议评测。

主表只保留预定义协议，不再展示全组合搜索 top rows 作为主结论。

保留的协议族包括：

### 4.1 Dense baseline

用于提供基础性能和成本参照：

- `convnext_tiny`
- `swin_tiny`
- `vit_b_imagenet`
- `retfound_mae_cfp_official_protocol`
- `convnext_tiny + retfound_mae_cfp_official_protocol`

### 4.2 Single-scout routing

用于回答单个 scout 是否有效：

- `convnext_tiny -> retfound_mae_cfp_official_protocol`
- `swin_tiny -> retfound_mae_cfp_official_protocol`
- `retfound_green_linear_probe -> retfound_mae_cfp_official_protocol`
- `vit_b_imagenet -> retfound_mae_cfp_official_protocol`

其中 GreenScout 和 ViT-B 主要作为筛查/消融对照。

### 4.3 Multi-scout routing

用于回答 scout 是否必须只有一个。

新增多 scout 协议：

- `convnext_tiny + swin_tiny -> retfound_mae_cfp_official_protocol`
- `convnext_tiny + swin_tiny + vit_b_imagenet -> retfound_mae_cfp_official_protocol`

多 scout routing 初版只评估少量预定义信号：

- 平均不确定性；
- 最大不确定性；
- scout 分歧优先，再按不确定性排序。

### 4.4 Bounds

必须保留以下边界：

- random same-budget expert invocation；
- oracle same-budget upper bound；
- dense expert reference。

这些对照用于判断 routing 是否真的优于随机调用，以及距离理论上限还有多远。

## 5. 报告规则

v0.8.2b 之后，报告遵循以下规则：

1. 全组合搜索只作为 exploratory screening；
2. candidate ranking score 只作为筛查信号，不作为最终证据；
3. 主结论只来自 controlled protocols；
4. 不声称任何 screening checkpoint 代表架构性能上限；
5. 不在外部数据集上重新选择 scout/expert 协议；
6. 不把项目扩展成大规模 backbone 超参搜索。

## 6. 当前 Go / No-Go 判断

### Go

继续做 controlled protocol evaluation。

重点评估：

- ConvNeXt/Swin 作为 scout；
- RETFound official-protocol 作为 expert；
- single-scout 与 multi-scout routing；
- random / oracle / dense expert 对照；
- DR-specific 风险事件覆盖。

### No-Go

不要继续扩大无约束全组合搜索。

不要把五模型全组合 top rows 当最终主结论。

不要因为 ViT-L 当前表现异常，就把下一阶段变成 ViT-L 调参研究。

## 7. 当前一句话结论

ViT-B 扩展验证通过，但没有改变当前最优模型编排边界；当前最有前景的是 ConvNeXt/Swin 作为 scout、RETFound official-protocol 作为 expert 的稀疏调用协议。下一步需要把全组合探索收敛为预定义 controlled protocols，并加入 multi-scout routing。
