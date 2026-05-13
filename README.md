# OphAgent

OphAgent 是一个面向眼科医学影像的深度学习实验项目，当前聚焦于糖尿病视网膜病变（Diabetic Retinopathy, DR）分类与模型可解释性研究。

当前版本基于 APTOS2019 数据集与 ConvNeXt-Tiny baseline，实现了：

- 可复现训练流程
- 配置化实验管理
- 单图推理与批量评估
- Streamlit Demo
- Grad-CAM Explainability（v0.2）

---

## 当前结果

测试集结果：

- Accuracy：81.36%
- Macro Precision：70.79%
- Macro Recall：65.55%
- Macro F1：64.96%
- Weighted F1：80.93%

---

## 项目结构

```text
ophagent-medical-ai-agent/
├── app/
├── configs/
├── demo_samples/
├── docs/
├── evaluation/
├── experiments/
├── explain/
├── findings/
├── scripts/
├── src/
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

## 环境安装

建议使用 Python 3.10。

安装依赖：

```bash
pip install -r requirements.txt
```

开发依赖：

```bash
pip install -r requirements-dev.txt
```

---

## 下载预训练权重

请从 GitHub Release 下载以下文件：

- convnext_tiny_best.pth
- checkpoint_meta.json

下载后放到：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/
```

目录结构应如下：

```text
experiments/
└── aptos_convnext_tiny/
    └── lr1e-4_bs32_seed42/
        ├── checkpoints/
        │   ├── convnext_tiny_best.pth
        │   └── checkpoint_meta.json
        └── configs/
            └── class_to_idx.json
```

同时确保类别映射文件存在：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json
```

---

## 模型训练

```bash
python train_classifier.py \
  --config configs/vision_baseline.yaml
```

训练结果默认保存到：

```text
experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/
```

包括：

```text
checkpoints/
configs/
evaluation/
figures/
logs/
```

---

## 单图推理

```bash
python infer_classifier.py \
  --image demo_samples/cmoderatedr/b9127e38d9b9.png \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
  --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json
```

---

## 批量评估

```bash
python evaluate_classifier.py \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
  --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json
```

评估结果包括：

```text
metrics.json
classification_report.txt
confusion_matrix.png
test_predictions.csv
```

---

## Streamlit Demo

启动 Demo：

```bash
streamlit run app/demo_v1.py
```

当前支持：

- 上传眼底图像
- 内置 demo_samples
- 显示预测类别与 confidence
- Top-3 分类结果
- 显示模型版本与测试集指标

注意：

Softmax confidence 不等同于医学诊断可信度。

---

## Grad-CAM 可解释性（v0.2）

OphAgent v0.2 新增了基于 CAM 的眼底分类可解释性支持，用于观察模型在糖尿病视网膜病变分类中的关注区域。

当前支持：

- GradCAM
- HiResCAM
- EigenCAM
- LayerCAM

默认配置：

- Method：HiResCAM
- Target Layer：stage3
- Smoothing：关闭

该默认配置基于多种 CAM 方法、不同 target layer 以及 smoothing 组合的定性对比结果选择。

示例命令：

```bash
python -m explain.gradcam \
  --image demo_samples/cmoderatedr/b9127e38d9b9.png \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
  --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json \
  --method hirescam \
  --target-layer stage3 \
  --output experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/explain/single_test/
```

生成结果包括：

```text
original.png
heatmap.png
overlay.png
```

项目同时提供了定性 Explainability Gallery：

```text
docs/gradcam_gallery/
├── good_cases/
├── failure_cases/
└── interesting_cases/
```

注意：

当前 CAM 可视化仅用于模型行为分析与定性解释，不等同于医学病灶分割或临床诊断结果。

---

## 当前限制

当前项目仍存在以下限制：

- 仅使用 ConvNeXt-Tiny baseline
- 尚未进行多模型对比
- DR 类别存在类别不平衡问题
- Explainability 仍属于 preliminary stage
- CAM 可视化可能受到边缘、亮度与图像质量影响

---

## Roadmap

计划中的方向包括：

- 更强 backbone（Swin / ViT）
- 多模态 Ophthalmology Agent
- 多任务学习
- 更稳定的 Explainability
- 更完整的 Streamlit Demo
- Retina report generation
- Medical VLM integration

---

## License

MIT License