# v0.5.1 多指标 Benchmark 更新

## Overview

v0.5.1 在原始 v0.5 benchmark 基础上，
从单一 Macro-F1 指标扩展为更完整的多指标评测框架。

当前目标不仅是比较不同 backbone 的分类性能，
还希望进一步分析：

- 类别不均衡下的表现差异
- DR grading 的 ordinal consistency
- 不同 representation 的 uncertainty characteristics
- 不同 backbone 的错误模式差异

当前评测 backbone：

- ConvNeXt-Tiny
- Swin-Tiny
- RETFound-MAE-CFP

数据集：

- APTOS2019 diabetic retinopathy grading benchmark

---

## 当前评测指标

当前版本加入：

- Accuracy
- Macro-F1
- Weighted-F1
- Quadratic Weighted Kappa (QWK)
- Per-class F1
- Prediction entropy
- Top1-top2 margin
- Confusion matrix

相比仅使用 Macro-F1，
当前版本能够更完整分析：

- class imbalance
- ordinal grading consistency
- prediction confidence
- uncertainty behavior

---

## Main Benchmark Results

| Backbone | Accuracy | Macro-F1 | Weighted-F1 | QWK |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.814 | 0.650 | 0.809 | 0.862 |
| Swin-Tiny | 0.829 | 0.657 | 0.820 | 0.898 |
| RETFound-MAE-CFP | 0.790 | 0.552 | 0.769 | 0.834 |

---

## 当前观察

### Swin-Tiny 当前表现最稳定

Swin-Tiny 在：

- Accuracy
- Macro-F1
- Weighted-F1
- QWK

等指标上均取得当前最佳结果。

同时：

- prediction entropy 最低
- top1-top2 margin 最高

说明其预测结果整体更加稳定，
模型输出也更加 confident。

---

### RETFound 展现出不同的 representation behavior

RETFound-MAE-CFP 当前并未超过 lightweight ConvNeXt/Swin baseline 的整体分类性能。

但其表现出一些不同于普通 backbone 的行为特征：

- Moderate DR recall 较高
- prediction entropy 更高
- confidence margin 更低
- Severe DR 类别识别较弱

说明 ophthalmic foundation representation
可能具有不同于传统 ImageNet representation 的 uncertainty behavior 与 grading characteristics。

当前结果也表明：

foundation representation
并不一定直接带来更高的 raw classification metrics。

---

## Per-class Findings

### ConvNeXt-Tiny

- 整体表现较均衡
- Severe DR F1 高于 Swin

### Swin-Tiny

- Moderate DR 表现最强
- 当前 benchmark 整体指标最佳
- Severe DR recall 较低

### RETFound-MAE-CFP

- Moderate DR recall 较高
- uncertainty characteristics 更明显
- Severe DR performance drop 较明显

---

## 当前限制

当前 benchmark 仍主要关注 classification behavior。

尚未涉及：

- calibration reliability
- explanation faithfulness
- CAM stability
- uncertainty-aware referral behavior

这些方向将在后续版本中继续推进。

---

## Next Step

下一阶段计划：

```text
v0.6.0 = Explainability Consistency Benchmark
```

后续重点包括：

- CAM consistency under perturbations
- explanation stability
- hardcase reliability
- uncertainty-aware evaluation