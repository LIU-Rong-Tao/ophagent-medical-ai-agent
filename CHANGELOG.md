# 更新日志

## v0.1.0 - Vision Baseline

### 新增功能

- APTOS2019 ImageFolder 数据加载
- ConvNeXt-Tiny 训练 pipeline
- YAML 配置化实验管理
- 可复现实验目录结构
- 训练日志与训练曲线保存
- 单张图片推理
- Test set 批量评估
- 混淆矩阵可视化

### 实验结果

- Test Accuracy：81.36%
- Macro Precision：70.79%
- Macro Recall：65.55%
- Macro F1：64.96%
- Weighted F1：80.93%

### 当前限制

- 暂未实现 Grad-CAM 可解释性分析
- 暂未实现外部数据集泛化验证
- 暂未实现结构化医学 Findings
- 暂未实现自动报告生成
- 暂未实现 Agent Workflow