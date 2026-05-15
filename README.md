# OphAgent

轻量级眼科医学 AI Workflow Demo。

当前项目聚焦于：

- DR（糖尿病视网膜病变）分类
- Explainability（Grad-CAM / HiResCAM）
- Structured Findings
- Rule-based / OpenAI Reasoning
- Lightweight Agent Runner Workflow

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
Ophthalmology Medical AI Workflow Demo
```

重点包括：

```text
classification
explainability
structured findings
reasoning
workflow integration
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

当前默认 backbone：

```text
ConvNeXt-Tiny
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
```

---

# Demo 启动

## 环境

建议：

```text
Python 3.10+
CUDA + PyTorch
```

安装依赖：

```bash
pip install -r requirements.txt
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

- checkpoint metadata
- evaluation metrics
- training summary

---

# 当前版本

```text
v0.3.0
```

---

# v0.3.0 更新内容

```text
- introduce lightweight agent runner workflow
- unified run_agent(...) interface
- integrate structured findings
- integrate reasoning providers
- simplify Streamlit workflow
- optional OpenAI reasoning report
- improve project architecture
```

---

# 当前设计目标

v0.3 重点不是：

```text
增加更多页面
```

也不是：

```text
继续堆 demo
```

而是：

```text
workflow 开始稳定
```

当前目标：

- unified workflow
- reusable run_agent(...)
- stable Streamlit demo
- reusable reasoning pipeline

---

# 后续计划

## v0.4+

计划方向：

```text
benchmark
multiple backbones
VLM integration
Qwen / multimodal reasoning
online CAM generation
real clinical case analysis
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