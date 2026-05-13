# OphAgent

OphAgent 是一个面向眼科医学影像的 AI 项目。  
当前版本：**v0.1.0 Vision Baseline**

> 注意：当前版本还不是完整的 Agent 系统。  
> v0.1.0 主要目标是构建一个可复现、可评估、可扩展的眼底图像分类 baseline。

---

# 项目目标

OphAgent 的长期目标是构建一个多模态眼科 AI 助手，支持：

- 眼底图像分析
- 疾病分类
- 病灶可解释性分析
- 结构化医学 Findings
- 自动报告生成
- Agent 化临床交互

当前阶段仅聚焦于视觉 baseline。

---

# 当前版本：v0.1.0 Vision Baseline

当前已实现：

- APTOS2019 ImageFolder 数据加载
- ConvNeXt-Tiny 训练 pipeline
- YAML 配置化实验管理
- 可复现实验目录结构
- 训练日志与训练曲线保存
- 单张图片推理
- Test set 批量评估
- 混淆矩阵可视化

当前未实现：

- Grad-CAM 可解释性分析
- 外部数据集泛化验证
- Structured Findings
- 自动报告生成
- RAG
- Agent Workflow

---

# 数据集结构

当前使用数据集：

```text
/data/LRT/RETFound/Data_split/APTOS2019/
├── train/
├── val/
└── test/