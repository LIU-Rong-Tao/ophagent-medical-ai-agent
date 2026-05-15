# OphAgent

> 眼科医学影像 AI Demo  
> DR Classification · Explainability · Lightweight VL Reasoning

---

## 项目简介

OphAgent 是一个面向眼科医学影像的研究型 AI 项目，
当前聚焦于：

- 糖尿病视网膜病变（DR）分类
- Explainability（Grad-CAM / HiResCAM）
- 结构化 Findings 生成
- Lightweight VL Reasoning
- 可复现实验流程

当前 Demo 流程：

```text
输入眼底图像
↓
DR 分类预测
↓
Grad-CAM / HiResCAM
↓
结构化 Findings
↓
Rule-based / OpenAI 中文报告
```

当前 baseline 使用：

- ConvNeXt-Tiny
- APTOS2019 DR 数据集

> 本项目仅用于科研与工程展示，  
> 不用于临床诊断。

---

## 功能特性

- ConvNeXt-Tiny DR 分类 baseline
- Grad-CAM / HiResCAM Explainability
- Explainability Failure-case Gallery
- Streamlit 交互式 Demo
- Rule-based 医学报告生成
- OpenAI Report Provider（支持 fallback）
- YAML 配置化实验管理
- 可复现实验目录结构

---

## Demo

统一 Streamlit 入口：

```bash
streamlit run app/demo.py
```

当前 Demo 包含：

- Classification & VL Reasoning Report
- Grad-CAM / HiResCAM Gallery
- Model / Evaluation Info

---

## Explainability Showcase

当前项目包含：

- Grad-CAM
- HiResCAM
- Failure-case Analysis
- Interesting-case Visualization

Explainability Gallery 位于：

```text
docs/gradcam_gallery/
```

包含：

```text
good_cases/
failure_cases/
interesting_cases/
```

用于展示：

- 热力图与病变区域对应情况
- Explainability failure cases
- 模型关注区域分析
- 不同 DR 等级下的 CAM 表现

---

## 示例流程

```text
眼底图像
→ 分类预测
→ Top-3 Confidence
→ Explainability Visualization
→ Structured Findings
→ 中文推理报告
```

---

## 数据集

当前 baseline 使用：

- APTOS2019 Blindness Detection

`demo_samples/` 为从 APTOS2019 test split 中选取的代表性样例，
用于定性展示与 explainability showcase。

---

## 项目结构

```text
app/                Streamlit Demo
configs/            YAML configs
demo_samples/       Demo 输入样例
docs/               Explainability Gallery / Assets
explain/            Grad-CAM / HiResCAM
findings/           Structured Findings
models/             Classification Models
reasoning/          Report Providers / Reasoning
scripts/            Utility Scripts
```

---

## 当前版本

当前版本：`v0.2.2`

近期更新：

- 统一 Streamlit Demo 入口
- Lightweight VL Reasoning 工作流
- Rule/OpenAI provider abstraction
- Structured Findings generation
- Explainability Gallery 重构

---

## Quick Start

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Demo：

```bash
streamlit run app/demo.py
```

