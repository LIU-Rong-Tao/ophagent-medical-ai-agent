# OphAgent

## Grounded & Explainable Ophthalmology AI Workflow

![teaser](docs/assets/ophagent_v0_5_overview.png)

OphAgent 是一个面向眼科医学影像的 AI workflow 项目，当前正在从：

```text
workflow demo
```

逐步演进为：

```text
foundation representation benchmark
```

当前重点：

- benchmark infrastructure
- foundation representation comparison
- explainability consistency
- grounded ophthalmology reasoning

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
Grounded Reasoning
```
---

## v0.5 Foundation Benchmark

![benchmark](docs/assets/ophagent_v0_5_benchmark_overview.png)

### Core Results

| Backbone | Setting | Macro F1 |
|---|---|---:|
| ConvNeXt-Tiny | lightweight baseline | 0.6496 |
| Swin-Tiny | lightweight baseline | 0.6567 |
| ViT-B/16 | official-like | 0.5800 |
| RETFound-MAE-CFP | official-like | 0.6095 |

### Key Observation

```text
RETFound-MAE-CFP 在相同 foundation-style training protocol 下，
相比 ImageNet-pretrained ViT-B/16 获得了更高的 Macro F1。
```

---

## Documentation

| 页面 | 内容 |
|---|---|
| [v0.5 Foundation Benchmark](docs/v0_5_foundation_benchmark.md) | v0.5 benchmark 设计、结果与当前观察 |
| [v0.4.2 README Archive](docs/v0_4_2_readme_archive.md) | v0.5 重构前的旧版 README 归档 |
| [Experiment Summaries](experiments/summary/) | 各阶段实验汇总 |
| [Development Notes](notes/) | 版本开发记录 |
| [Grad-CAM Assets](docs/assets/) | 可解释性与展示图片 |
| [Changelog](CHANGELOG.md) | 版本更新记录 |

---

## Roadmap

| Version | Focus |
|---|---|
| v0.5 | Foundation Representation Benchmark |
| v0.6 | Explainability Consistency |
| v0.7 | Structured Clinical Findings |
| v0.8 | Clinical Reasoning Prototype |
| v1.0 | Grounded Ophthalmology Agent |

---

## Disclaimer

This project is for research and engineering demonstration only. It is not intended for clinical diagnosis.