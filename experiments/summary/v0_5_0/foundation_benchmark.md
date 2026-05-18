# OphAgent v0.5 Foundation Representation Benchmark

## 项目阶段

v0.5 开始，OphAgent 从：

```text
眼科医学 AI workflow demo
```

逐步演进为：

```text
Foundation Representation Benchmark
```

当前研究重点不再只是分类准确率，而是：

```text
不同视觉表征（representation）
在眼科医学任务中的行为差异
```

包括：

- 分类性能
- 类别均衡能力
- explainability consistency
- attention behavior
- failure pattern

---

# Benchmark 结构

当前 benchmark 分为两层：

## Level 1：Unified Lightweight Baseline

用于：

```text
验证 workflow / benchmark infrastructure
```

统一 protocol：

- Dataset: APTOS2019
- Image Size: 224
- Batch Size: 32
- Epochs: 10
- Learning Rate: 1e-4
- Seed: 42

## 实验结果

| Backbone | 类型 | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| ConvNeXt-Tiny | CNN baseline | 0.8136 | 0.6496 | 0.8093 |
| Swin-Tiny | Hierarchical Transformer | 0.8291 | 0.6567 | 0.8202 |
| ViT-B/16 | Vanilla ViT | 0.7899 | 0.5500 | 0.7700 |

---

## Level 2：Foundation-style Controlled Benchmark

用于：

```text
控制变量后比较：
普通视觉预训练
vs
Retinal Foundation Pretraining
```

统一 protocol：

- Dataset: APTOS2019
- Image Size: 224
- Batch Size: 8
- Epochs: 50
- BLR Scaling: 0.005
- Warmup Epochs: 10
- Cosine LR Schedule
- Weight Decay: 0.05
- Drop Path: 0.2
- Label Smoothing: 0.1
- Seed: 42

## 实验结果

| Backbone | Pretraining | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Best Val Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | ImageNet | 0.7918 | 0.6409 | 0.5653 | 0.5800 | 0.7773 | 0.7899 |
| RETFound-MAE-CFP | Retinal Foundation | 0.7955 | 0.6412 | 0.5954 | 0.6095 | 0.7874 | 0.8288 |

---

# 当前观察

在：

```text
相同 architecture
相同 training protocol
相同 finetuning strategy
```

条件下：

```text
RETFound-MAE-CFP
相比
ImageNet-pretrained ViT-B/16
```

获得了更高的：

```text
Macro F1
```

说明：

```text
retinal foundation representation
对于类别均衡性能具有正向作用
```

尤其：

```text
Severe DR
PDR
```

类别存在提升趋势。

---

# 当前定位

v0.5 当前属于：

```text
single-seed pilot benchmark
foundation representation study
```

不是严格：

```text
size-controlled benchmark
```

因此当前结论：

```text
不解释为“架构绝对优劣”
```

而是：

```text
representation behavior comparison
```

---

# 下一阶段

v0.6 将进入：

```text
Explainability Consistency Analysis
```

重点包括：

- same-image CAM comparison
- focus drift
- failure behavior
- representation consistency
- grounded attention behavior

---

# 图表

![v0.5 Foundation Benchmark](../../../docs/assets/v0_5_foundation_benchmark_macro_f1.png)