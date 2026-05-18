# v0.5.0 Backbone 对比（APTOS2019）

## 实验设置

- Dataset: APTOS2019
- Task: Diabetic Retinopathy Classification
- Image Size: 224
- Epochs: 10
- Batch Size:
  - ConvNeXt / Swin / ViT Foundation-style: 32
- Seed: 42

---

# Backbone Benchmark Comparison

| Backbone | 表征类型 | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Best Val Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | CNN-based Representation | 0.8136 | 0.7079 | 0.6555 | 0.6496 | 0.8093 | 0.8366 |
| Swin-Tiny | Hierarchical Transformer | 0.8291 | 0.7041 | 0.6342 | 0.6567 | 0.8202 | 0.8346 |
| ViT-B/16 ImageNet Baseline | Foundation-style Global Representation | 0.7899 | 0.6300 | 0.5400 | 0.5500 | 0.7700 | 0.7899 |

---

# 初步观察

## 1. ConvNeXt-Tiny

特点：

- 整体性能稳定
- Weighted F1 较高
- 对多数类别具有较强鲁棒性
- CNN 局部纹理表征能力较强

问题：

- 对少数类（Severe DR / Proliferative DR）仍存在明显召回不足

---

## 2. Swin-Tiny

特点：

- 当前整体 Accuracy 最优
- Macro F1 略高于 ConvNeXt
- Transformer 分层结构能够兼顾局部与全局信息

问题：

- 对类别不平衡仍较敏感
- 少数类 recall 仍然有限

---

## 3. ViT-B/16（Foundation-style Interface）

说明：

当前阶段使用：

- ViT-B/16
- ImageNet pretrained weights

用于验证：

- foundation backbone integration
- unified benchmark pipeline
- explainability compatibility

当前为 ViT-B/16 ImageNet 预训练基线；真正 retinal foundation pretrained checkpoint 将作为独立 backbone 接入。

特点：

- Moderate DR recall 较高
- 全局 token representation 更明显
- 已成功接入统一 train / eval / explain pipeline

问题：

- Severe DR recall 明显偏低
- Accuracy 暂低于 ConvNeXt / Swin
- attention 可能更 diffuse（待后续 CAM consistency 分析）

---

# 下一阶段研究方向

后续重点不再是：

- 单纯提升 accuracy
- 堆叠更多 backbone

而是研究：

## Foundation Representation Analysis

包括：

- CAM consistency
- failure behavior
- localization stability
- attention drift
- OOD robustness

重点问题：

> retinal foundation representation 是否不仅提升分类性能，
> 还能够提升模型解释一致性与可信性？