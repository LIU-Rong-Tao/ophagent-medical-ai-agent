````markdown
# OphAgent

轻量级眼科医学 AI Workflow 与 Benchmark Infrastructure。

当前项目聚焦于：

- DR（糖尿病视网膜病变）分类
- Explainability（Grad-CAM / HiResCAM）
- Structured Findings
- Rule-based / OpenAI Reasoning
- Lightweight Agent Runner Workflow
- Benchmark Infrastructure

本项目主要用于：

- 医学 AI Workflow 展示
- Explainability 分析
- Research Engineering Demo
- Ophthalmology AI 原型验证

---

# 当前定位

OphAgent 当前不是：

```text
单纯 DR 分类 baseline
```

而是：

```text
Ophthalmology Medical AI Workflow + Benchmark Infrastructure
```

重点包括：

```text
classification
explainability
structured findings
reasoning
workflow integration
benchmark infrastructure
```

---

# Workflow

```text
image
  ↓
run_agent(...)
  ↓
classification
  ↓
top-k prediction
  ↓
structured findings
  ↓
rule-based / OpenAI reasoning
```

---

# 当前功能

## 1. DR Classification

支持：

- No DR
- Mild DR
- Moderate DR
- Severe DR
- Proliferative DR

当前 baseline backbones：

```text
ConvNeXt-Tiny
Swin-Tiny
```

---

## 2. Structured Findings

当前 findings 模块：

```text
classification
    ↓
structured findings
```

说明：

- 当前不是 lesion detector
- findings 基于分类先验与 explainability context
- 用于 workflow explanation
- 不作为临床病灶检测结果

---

## 3. Rule-based / OpenAI Reasoning

支持：

```text
rule_based
openai
```

当前 OpenAI report：

- 可选生成
- 默认不调用 API
- Streamlit 中按钮触发
- 避免重复 rerun

---

## 4. Explainability Gallery

当前包含：

```text
good_cases/
failure_cases/
interesting_cases/
```

用于：

- Grad-CAM showcase
- failure analysis
- explainability discussion

---

## 5. Benchmark Infrastructure

当前支持：

```text
统一 experiment schema
统一 evaluation schema
summary builder
backbone comparison
benchmark summary
```

当前 benchmark schema：

训练产物：

```text
logs/
checkpoints/
configs/
figures/
```

评估产物：

```text
evaluation/test/
  metrics.json
  classification_report.txt
  test_predictions.csv
```

---

# 项目结构

```text
agent/
  __init__.py
  schema.py
  runner.py
  providers.py

app/
  demo.py

configs/

findings/

reasoning/

explain/

docs/
  gradcam_gallery/

experiments/
  summary/
```

---

# Demo 启动

## 环境

建议：

```text
Python 3.10+
PyTorch
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## Checkpoints

当前仓库不包含模型权重文件。

请将 checkpoint 放置于：

```text
experiments/.../checkpoints/
```

例如：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/
```

---

## 启动 Streamlit Demo

```bash
streamlit run app/demo.py
```

---

# Demo 页面

当前 Demo 包含：

## Classification & VL Reasoning Report

支持：

- 上传眼底图像
- 使用 demo samples
- classification
- Top-k prediction
- structured findings
- rule-based report
- optional OpenAI report

---

## Grad-CAM / HiResCAM Gallery

展示：

- good cases
- failure cases
- interesting cases

用于 explainability showcase。

---

## Model / Evaluation Info

展示：

- unified evaluation metrics
- benchmark summary
- training summary
- backbone comparison

---

# 当前版本

```text
v0.4.2
```

---

# v0.4.x 当前阶段

当前阶段重点：

```text
benchmark infrastructure
experiment packaging
multi-backbone baseline
unified evaluation schema
```

当前已经完成：

```text
v0.4.0
  - experiment summary builder

v0.4.1
  - Swin-Tiny baseline
  - unified evaluation schema
  - backbone comparison summary

v0.4.2
  - benchmark infrastructure cleanup
  - experiment metadata consistency
  - summary builder improvements
```

---

# 当前 Benchmark

统一 benchmark protocol：

- APTOS2019
- 224×224
- batch size = 32
- lr = 1e-4
- seed = 42
- 10 epochs
- test split evaluation

当前单 seed benchmark：

| Backbone | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.8136 | 0.6496 | 0.8093 |
| Swin-Tiny | 0.8291 | 0.6567 | 0.8202 |

当前结果中：

```text
Swin-Tiny 略高于 ConvNeXt-Tiny
```

但当前仍属于：

```text
single-seed comparison
```

尚未形成完整 benchmark。

---

# Reproducibility

当前 benchmark：

- single-seed evaluation
- fixed training protocol
- unified evaluation schema

后续计划加入：

- multi-seed benchmark
- leaderboard aggregation
- formal benchmark protocol

---

# 当前设计目标

当前重点不是：

```text
继续堆叠 demo 页面
```

而是：

```text
构建稳定 benchmark workflow
```

当前目标：

- unified experiment schema
- reusable evaluation pipeline
- benchmark summary generation
- multi-backbone infrastructure
- stable workflow demo

---

# 当前阶段不包含

当前暂不包含：

- OCT / 3D modality
- Qwen-VL pipeline
- Real-time CAM generation
- Clinical deployment
- Lesion detection

这些内容将在后续阶段逐步扩展。

---

# 后续计划

## v0.4.3

```text
multi-experiment aggregation
benchmark table generation
leaderboard summary
```

---

## v0.5.0

```text
formal multi-backbone benchmark
ConvNeXt
Swin
ViT / EVA
RETFound-style backbone
```

---

## v0.6.0

```text
multimodal reasoning
professional medical report generation
Qwen / OpenAI providers
```

---

## v0.7.0

```text
OCT / 3D ophthalmology modality
```

---

# Disclaimer

本项目仅用于：

```text
research
education
engineering demo
```

不用于：

```text
clinical diagnosis
medical decision making
```
````
