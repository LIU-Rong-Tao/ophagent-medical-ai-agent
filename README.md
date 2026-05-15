````markdown
# OphAgent

> 眼科医学影像 AI Demo  
> DR Classification · Explainability · Lightweight VL Reasoning

---

## 项目简介

OphAgent 是一个面向眼科医学影像的研究型 AI 项目，
当前聚焦于：

- 糖尿病视网膜病变（DR）分类
- Explainability（Grad-CAM / HiResCAM）
- 结构化 findings 生成
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
- Grad-CAM / HiResCAM 可解释性可视化
- Explainability failure-case gallery
- Streamlit 交互式 Demo
- Rule-based 医学报告生成
- 可选 OpenAI Report Provider（带 fallback）
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

## 当前版本

当前版本：`v0.2.2`

近期更新：

- 统一 Streamlit Demo 入口
- Lightweight VL Reasoning 工作流
- Rule/OpenAI provider abstraction
- Structured findings generation
- Explainability Gallery 重构
````
