# OphAgent

## 面向眼科基础模型的可信评测工作流

![OphAgent 项目总览](docs/assets/ophagent_v0_5_overview.png)

OphAgent 是一个面向眼科医学影像的 AI workflow 项目，当前正在从早期的：

```text
workflow demo
```

逐步演进为：

```text
ophthalmic foundation representation benchmark
```

项目当前重点关注：

- 眼科影像分类训练与评测流程
- 多种视觉 backbone 的 benchmark 对比
- RETFound 等眼科基础模型表示能力分析
- Grad-CAM 可解释性展示
- 后续可信评测方向探索

> 本项目目前以 APTOS2019 眼底图像分类任务为主要实验场景，重点分析不同视觉表示在糖尿病视网膜病变分级任务中的表现差异。

---

## Research Focus

本项目关注的问题不是单纯追求最高分类准确率，而是希望进一步分析：

```text
不同 ophthalmic foundation representations 在性能、解释性和 hardcase 行为上的差异。
```

当前研究关注点包括：

- 不同 backbone 在类别不均衡 DR grading 任务中的表现差异
- RETFound 等眼科基础模型与通用 ImageNet backbone 的对比
- Grad-CAM 热力图是否能辅助理解模型决策
- 错分样本、低置信样本和 hardcase 的表现特征
- 后续向 calibration、explainability consistency 和 uncertainty-aware triage 扩展

---

## Workflow

当前 workflow 包括：

```text
APTOS2019 / Fundus Image
        ↓
Training Engine
        ↓
Backbone Benchmark
        ↓
Evaluation Schema
        ↓
Grad-CAM Explainability
        ↓
Representation Comparison
        ↓
Grounded Reasoning Exploration
```

---

## v0.5 Foundation Benchmark

![OphAgent v0.5 Benchmark 总览](docs/assets/ophagent_v0_5_benchmark_overview.png)

### Benchmark Results

| Backbone | Setting | Macro-F1 | Notes |
|---|---|---:|---|
| ConvNeXt-Tiny | lightweight baseline | 0.6496 | 通用视觉 CNN baseline |
| Swin-Tiny | lightweight baseline | 0.6567 | 通用视觉 Transformer baseline |
| ViT-B/16 | official-like | 0.5800 | ImageNet-pretrained ViT baseline |
| RETFound-MAE-CFP | official-like | 0.6095 | 眼科基础模型表示 |

### 当前观察

在当前实验设置下：

```text
RETFound-MAE-CFP 相比 ImageNet-pretrained ViT-B/16 获得了更高的 Macro-F1，
但尚未超过 lightweight ConvNeXt / Swin baseline。
```

这说明：

- 眼科基础模型并不一定直接带来最高的分类分数；
- Macro-F1 受到 APTOS2019 类别不均衡影响较大；
- 后续需要结合 Weighted-F1、QWK、calibration、hardcase analysis 等指标进行更完整评估；
- RETFound 的价值不应只从 raw accuracy 判断，也需要进一步分析其解释稳定性和不确定性行为。

---

## Explainability

![Grad-CAM 示例](docs/assets/gradcam_good.png)

当前项目已集成 Grad-CAM 可解释性流程，用于观察不同 backbone 在眼底图像分类任务中的关注区域差异。

当前 explainability 模块包括：

- Grad-CAM
- backbone-specific Grad-CAM wrapper
- heatmap overlay 导出
- 可解释性结果展示

当前 explainability 模块主要用于：

```text
从可视化角度观察不同 backbone 的关注区域差异。
```

后续可进一步扩展：

- HiResCAM / EigenCAM / LayerCAM 等 CAM variants
- explainability consistency evaluation
- 同一图像在不同扰动、增强或 checkpoint 下的 CAM 稳定性分析

---

## Current Progress

### 已完成

- APTOS2019 眼底图像分类训练流程
- ConvNeXt / Swin / ViT / RETFound 多 backbone benchmark
- RETFound-MAE-CFP 权重接入与实验适配
- Macro-F1 based benchmark summary
- Grad-CAM 可解释性展示
- 实验结果、配置与 artifact 管理
- v0.5 benchmark 文档整理

### 进行中

- 更完整的 performance evaluation
- Weighted-F1 / QWK / confusion matrix 补充
- calibration 相关指标设计
- hardcase 样本分析
- explainability consistency 方案设计

### 后续方向

- Explainability Consistency Analysis
- Calibration Evaluation
- Uncertainty-aware Hardcase Triage
- Structured Clinical Findings
- Grounded Ophthalmology Reasoning

---

## Documentation

| 页面 | 内容 |
|---|---|
| [v0.5 Foundation Benchmark](experiments/summary/v0_5_0/foundation_benchmark.md) | v0.5 benchmark 设计、结果与当前观察 |
| [v0.4.2 README Archive](docs/v0_4_2_readme_archive.md) | v0.5 重构前的旧版 README 归档 |
| [Experiment Summaries](experiments/summary/) | 各阶段实验汇总 |
| [Development Notes](notes/) | 版本开发记录 |
| [Grad-CAM Assets](docs/assets/) | 可解释性与展示图片 |
| [Changelog](CHANGELOG.md) | 版本更新记录 |

---

## Repository Structure

```text
ophagent-medical-ai-agent/
├── app/                    # Demo 与展示入口
├── configs/                # 训练与模型配置
├── datasets/               # 数据集相关脚本
├── docs/                   # 项目文档
│   ├── assets/             # README 与文档图片资源
│   ├── v0_5_foundation_benchmark.md
│   └── v0_4_2_readme_archive.md
├── experiments/            # 实验输出与 summary
├── explain/                # Grad-CAM 可解释性模块
├── metrics/                # 评测指标与结果处理
├── models/                 # backbone 与模型定义
├── notes/                  # 开发记录
├── scripts/                # 工具脚本
└── train/                  # 训练入口与训练逻辑
```

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| v0.5 | Foundation Representation Benchmark | Completed |
| v0.5.x | Performance Metrics & Hardcase Analysis | Ongoing |
| v0.6 | Explainability Consistency | Planned |
| Future | Calibration & Uncertainty-aware Triage | Planned |

---

## 使用声明

本项目仅用于科研、工程实践与项目展示，不用于临床诊断、治疗建议或真实医疗决策。
