# OphAgent v0.6.6：无真实标签预审风险排序

## 1. 版本目标

v0.6.6 的目标是将 OphAgent 从 v0.6.5 的“事后错误审计展示”推进到“无真实标签预审风险排序”。

核心问题：

> 在不使用真实标签的前提下，仅根据模型输出概率、置信度、Top-1/Top-2 margin、entropy 和类别严重程度关系，能否生成一个有价值的人工复核优先级队列？

该版本重点不是直接提高分类准确率，而是验证一个更贴近真实接入流程的问题：

> 模型输出之后，哪些样本应该优先进入人工复核？

## 2. 方法设计

v0.6.6 将流程拆成两个阶段。

### 2.1 预审排序阶段

预审排序阶段不使用真实标签。

允许使用的信号包括：

- `pred_label`
- `confidence`
- `top2_label`
- `top2_confidence`
- `margin`
- `entropy`
- `prob_No DR`
- `prob_Mild DR`
- `prob_Moderate DR`
- `prob_Severe DR`
- `prob_Proliferative DR`

禁止用于排序的后验字段包括：

- `true_label`
- `gt_label`
- `correct`
- `is_correct`
- `error_type`
- `severe_underestimate`

输出字段包括：

- `pre_review_risk_score`
- `pre_review_risk_level`
- `risk_reasons`
- `review_priority_rank`

### 2.2 后验验证阶段

后验验证阶段才使用真实标签。

真实标签不参与风险排序，只用于评估排序是否真的更容易发现：

- 真实错误样本
- 重症低估样本
- 边界不稳定样本

评估指标包括：

- Top-K error rate
- Top-K enrichment ratio
- Severe underestimation recall@K
- Risk group error rate

## 3. 核心实验结果

实验数据为 APTOS2019 test split，共 1100 张样本。实验覆盖 6 个 backbone / 配置：

- ConvNeXt-Tiny
- Swin-Tiny
- ViT-B ImageNet
- ViT-B official-like
- ViT-L official-like
- RETFound official-like

汇总结果位于：

- `experiments/summary/v0_6_6/full_test_backbones/backbone_pre_review_ranking_summary.csv`
- `experiments/summary/v0_6_6/full_test_backbones/backbone_pre_review_ranking_summary.md`

## 4. 主要发现

### 4.1 Top-K 风险排序在多个 backbone 上均优于随机抽样

6 个 backbone 上，Top 10%、Top 20%、Top 30% 风险队列的 enrichment ratio 均大于 1。

这说明预审风险排序能够把真实错误样本更集中地排到前面，而不是随机抽样。

### 4.2 低风险组错误率明显更低

在所有 backbone 上，low risk 组错误率均明显低于整体错误率。

例如：

| Backbone | Overall Error Rate | Low Risk Error Rate |
|---|---:|---:|
| ConvNeXt-Tiny | 0.1864 | 0.1207 |
| Swin-Tiny | 0.1709 | 0.1424 |
| ViT-B ImageNet | 0.1818 | 0.1440 |
| ViT-B official-like | 0.2009 | 0.0612 |
| ViT-L official-like | 0.1991 | 0.0822 |
| RETFound official-like | 0.1964 | 0.0904 |

这说明当前规则具备一定“低风险排除”能力，可用于降低常规样本的复核优先级。

### 4.3 对重症低估具有较强覆盖能力

以 ConvNeXt-Tiny 为例：

- 仅复核 Top 10% 风险样本，可覆盖 50.77% 的重症低估案例。
- 复核 Top 20% 风险样本，可覆盖 66.15% 的重症低估案例。
- 复核 Top 30% 风险样本，可覆盖 81.54% 的重症低估案例。

这说明预审风险排序对“重症被低估”这类高价值风险具有一定发现能力。

## 5. 需要客观看待的限制

当前结果不能表述为“高风险组永远错误率最高”。

部分 backbone 中，medium risk 组错误率高于 high risk 组，例如 RETFound、ViT-B ImageNet 和 ViT-L official-like。说明当前风险规则仍是第一版启发式规则，风险分数和风险等级阈值还有优化空间。

更稳妥的结论是：

> high / medium 复核候选整体错误率明显高于 low risk 组，且 Top-K 风险排序在所有 backbone 上均优于随机抽样。

## 6. 对接价值

v0.6.6 证明了 OphAgent 的风险排序不依赖真实标签，不是事后看答案挑错。

在真实系统中，只要已有模型输出概率或 logits，即可生成：

- 预审风险排序表
- 高风险复核候选队列
- 风险原因说明
- 后验验证报告

如果接入医院主线流程，该模块可以作为模型输出后的风险分层组件，用于辅助技术老师和医生优先查看更值得复核的样本。

## 7. 后续方向

后续可继续优化：

1. 引入模型校准指标，例如 ECE、Brier score。
2. 引入 TTA uncertainty，观察预测稳定性。
3. 引入多模型 disagreement，发现 backbone 间不一致样本。
4. 引入图像质量评分，发现低质量高置信样本。
5. 根据医生反馈修正规则权重和风险等级阈值。
