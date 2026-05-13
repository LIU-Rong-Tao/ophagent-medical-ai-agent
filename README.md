# OphAgent

OphAgent 是一个面向眼科医学影像的 AI 项目。

当前版本：**v0.1.1 Vision Baseline Polish Release**

> 注意：当前版本还不是完整的 Agent 系统。  
> v0.1.1 主要目标是构建一个可复现、可评估、可运行 Demo 的眼底图像分类 baseline。

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

# 当前版本：v0.1.1 Vision Baseline

当前已实现：

- APTOS2019 ImageFolder 数据加载
- ConvNeXt-Tiny 训练 pipeline
- YAML 配置化实验管理
- 可复现实验目录结构
- 训练日志与训练曲线保存
- 单张图片推理
- Test set 批量评估
- 混淆矩阵可视化
- Streamlit Demo 展示
- checkpoint 缺失时的友好提示
- GitHub Release 权重下载说明

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
```

---

# 类别定义

| Label Folder | Clinical Meaning |
| --- | --- |
| `anodr` | No Diabetic Retinopathy |
| `bmilddr` | Mild Diabetic Retinopathy |
| `cmoderatedr` | Moderate Diabetic Retinopathy |
| `dseveredr` | Severe Diabetic Retinopathy |
| `eproliferativedr` | Proliferative Diabetic Retinopathy |

---

# 安装依赖

建议先创建独立 Python 环境：

```bash
conda create -n ophagent python=3.10 -y
conda activate ophagent
```

---

## PyTorch Installation (CUDA 12.1)

当前项目默认使用 CUDA 12.1 环境训练与推理。

请先安装对应版本的 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你使用 CPU 环境，可以参考 PyTorch 官网安装对应版本。

---

## Install Other Dependencies

安装其余项目依赖：

```bash
pip install -r requirements.txt
```

---

## Verify Installation

可以通过以下命令验证 PyTorch CUDA 是否正常：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出：

```text
True
```

说明 CUDA 环境安装成功。

---

# 预训练权重

模型权重文件不会直接提交到 Git 仓库中。

原因：

- `.pth` 文件体积较大
- GitHub 仓库不适合长期保存模型权重
- 当前 `.gitignore` 已忽略 PyTorch checkpoint 文件

请从 GitHub Release 下载预训练权重：

```text
convnext_tiny_best.pth
```

下载后放到以下路径：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth
```

同时确保类别映射文件存在：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json
```

同时建议下载对应的模型元信息文件：

```text
checkpoint_meta.json

说明：

- checkpoint 用于复现 v0.1.1 Demo 与推理结果
- 同一 checkpoint 与同一输入图像下，推理结果固定可复现
- checkpoint 仅用于科研和工程演示，不能用于临床诊断

---

# 模型训练

```bash
python -m models.classifiers.train_classifier \
  --config configs/vision_baseline.yaml
```

训练完成后，实验文件会保存到：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/
├── checkpoints/
├── configs/
├── figures/
├── logs/
└── evaluation/
```

主要输出包括：

```text
checkpoints/convnext_tiny_best.pth
configs/class_to_idx.json
logs/train_log.csv
figures/loss_curve.png
figures/val_acc_curve.png
env_info.json
```

---

# 单图推理

```bash
python -m models.classifiers.infer_classifier \
  --image /path/to/image.png \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth
```

输出示例：

```text
Prediction: Severe DR
Confidence: 0.81
Raw Class: dseveredr
```

---

# Test Set 批量评估

```bash
python -m models.classifiers.evaluate_classifier \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
  --split test
```

评估结果会保存到：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/
├── metrics.json
├── classification_report.txt
├── confusion_matrix.png
└── test_predictions.csv
```

---

# Streamlit Demo

启动 Demo：

```bash
streamlit run app/demo_v1.py
```

Demo 支持：

- 上传眼底图像
- 选择内置 demo sample
- 显示输入图像
- 显示真实标签
- 显示预测标签
- 显示 confidence
- 显示 Top-3 probabilities
- 显示预测是否正确

说明：

`demo_samples/` 中仅包含少量随机抽样测试图片，用于 Streamlit Demo 展示，不代表完整测试集结果。

如果缺少 checkpoint，Demo 会提示先从 GitHub Release 下载模型权重。

---

# v0.1.1 实验结果

实验设置：

| 项目 | 内容 |
| --- | --- |
| Dataset | APTOS2019 |
| Backbone | ConvNeXt-Tiny |
| Input Size | 224 × 224 |
| Seed | 42 |

整体指标：

| 指标 | 数值 |
| --- | ---: |
| Test Accuracy | 81.36% |
| Macro Precision | 70.79% |
| Macro Recall | 65.55% |
| Macro F1 | 64.96% |
| Weighted F1 | 80.93% |

---

# 训练曲线

![Loss Curve](experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/figures/loss_curve.png)

![Validation Accuracy Curve](experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/figures/val_acc_curve.png)

---

# 混淆矩阵

![Confusion Matrix](experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/confusion_matrix.png)

---

# 局限性

v0.1.1 仅是视觉 baseline，仍存在明显局限：

- 当前模型仅在 APTOS2019 数据集上训练与评估
- 尚未进行外部数据集验证
- Severe DR 与 Proliferative DR 等重症类别表现仍需提升
- 当前模型可能漏检严重病变样本
- 当前版本没有 Grad-CAM 或病灶级解释
- 当前版本不具备临床部署能力

本项目当前仅用于科研、学习与工程演示，不能用于真实临床诊断或医疗决策。

---

# 项目结构

```text
ophagent-medical-ai-agent/
├── app/
│   └── demo_v1.py
├── configs/
│   └── vision_baseline.yaml
├── demo_samples/
├── experiments/
│   └── aptos_convnext_tiny/
│       ├── legacy_v0_1_baseline/
│       └── lr1e-4_bs32_seed42/
│           ├── checkpoints/
│           ├── configs/
│           ├── figures/
│           ├── logs/
│           └── evaluation/
├── explain/
│   └── gradcam.py
├── findings/
│   └── finding_schema.py
├── models/
│   └── classifiers/
│       ├── train_classifier.py
│       ├── infer_classifier.py
│       └── evaluate_classifier.py
├── utils/
├── README.md
├── CHANGELOG.md
├── requirements.txt
└── LICENSE
```

---

# Roadmap

## v0.2.0 Explainability

- Grad-CAM 可解释性分析
- 热力图可视化
- 病灶区域定位
- Streamlit Demo 集成 Grad-CAM

## v0.3.0 Evaluation

- 外部数据集泛化验证
- Cross-dataset evaluation
- Domain shift 分析
- AUC / Sensitivity / Specificity / Calibration 指标补充

## v0.4.0 Structured Findings

- 医学结构化结果生成
- 病灶类型、位置、严重程度描述
- 为自动报告生成提供标准化输入

## v0.5.0 Report Generation

- 自动报告生成
- 多模态报告模板
- Findings-to-Report pipeline

## v0.6.0 Agent Workflow

- OphAgent Agent Workflow
- 工具调用
- 多模块协同
- 交互式医学影像分析流程

---

# Disclaimer

本项目仅用于科研、学习与工程演示。

本项目不用于真实临床诊断、治疗建议或医疗决策。