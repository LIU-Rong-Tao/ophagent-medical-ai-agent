# v0.6.0 Evidence-Bottleneck Case Report Prototype

## 本版本定位

v0.6.0 将 OphAgent 从 v0.5 阶段的 benchmark / CAM explainability workflow，进一步扩展为一个可展示、可追踪、可审核的病例报告原型。

本版本的核心不是训练医学报告生成模型，而是建立一个 evidence-bottleneck case report pipeline：

```text
Fundus Image
→ Prediction
→ CAM-based Weak Evidence
→ Structured Findings
→ Claim-level Validation
→ Report Draft
→ Case Artifact
```

该 pipeline 用于在当前缺少图像-报告配对数据、病灶级标注和完整临床上下文的条件下，生成一个 research/demo 级别的 case report artifact。

---

## 新增内容

### 1. Evidence-Bottleneck Case Report Pipeline

新增脚本：

```text
scripts/run_case_report.py
```

该脚本支持从单张眼底图像生成完整 case artifact，包括：

- 模型预测结果
- CAM 弱视觉证据
- 结构化 findings
- claim-level validation
- Markdown 报告草稿
- HTML 报告草稿
- metadata 记录

---

### 2. Case Findings Schema

新增 schema 文档：

```text
docs/schema/case_findings_schema_v0_6.md
```

该 schema 规范了：

- `findings.json`
- `validation.json`
- prediction / evidence / findings / report_claims / quality_control 的字段关系

其中核心设计是：

```text
report.md / report.html 只能渲染 findings.json 中已有的 report_claims。
每条 report claim 必须通过 supported_by 指向 prediction / evidence / finding。
```

---

### 3. Claim-level Validation

新增 `validation.json` 输出，用于检查 case artifact 是否满足 evidence-bottleneck 约束。

当前 validation 检查内容包括：

- schema validity
- required files
- required disclaimer
- human-review-required statement
- CAM weak-evidence wording
- clinical diagnosis claim
- unsupported claim count
- evidence coverage rate
- image quality overclaim
- report reproducibility

---

### 4. README Landing Page 更新

根目录 `README.md` 已从 v0.5 阶段的 benchmark-oriented 首页，调整为更适合展示的 landing page。

旧版 v0.5.3 README 已归档到：

```text
docs/v0_5_3_readme_archive.md
```

---

## 示例 Case Artifact

当前示例 case 位于：

```text
experiments/case_reports/d9bbdc33db83/
```

包含文件：

```text
experiments/case_reports/d9bbdc33db83/
├── input.png
├── prediction.json
├── findings.json
├── validation.json
├── report.md
├── report.html
├── metadata.json
└── cam/
    ├── original.png
    ├── heatmap.png
    └── overlay.png
```

对应输入图像：

```text
demo_samples/cmoderatedr/d9bbdc33db83.png
```

默认模型配置：

```text
Backbone: ConvNeXt-Tiny
Config: configs/vision_baseline.yaml
Checkpoint: experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth
CAM: gradcam_stage3_eigen
```

---

## 示例运行命令

```bash
python scripts/run_case_report.py \
  --image demo_samples/cmoderatedr/d9bbdc33db83.png \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
  --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json \
  --output experiments/case_reports/d9bbdc33db83
```

---

## 示例预测结果

当前示例 case 的模型输出为：

| Field | Value |
|---|---|
| Predicted class | Moderate DR |
| Raw class | cmoderatedr |
| Confidence | 0.6026 |
| Top-2 class | Severe DR |
| Top-2 confidence | 0.3330 |

该结果仅表示模型分类倾向，不代表临床诊断。

---

## Validation 结果

当前示例 case 的 `validation.json` 关键结果如下：

```json
{
  "schema_valid": true,
  "required_files_present": true,
  "required_disclaimer_present": true,
  "human_review_required": true,
  "cam_described_as_weak_evidence": true,
  "clinical_diagnosis_claim_present": false,
  "unsupported_claim_count": 0,
  "evidence_coverage_rate": 1.0,
  "image_quality_overclaimed": false,
  "report_reproducible": true
}
```

该结果说明：

- case artifact 文件完整
- report 中包含必要 disclaimer
- report 中声明需要 human review
- CAM 被描述为 weak evidence
- report 中未检测到临床诊断声明
- 所有 report claims 都有 supported_by 支撑
- report 可由 structured findings 复现

---

## 当前限制

v0.6.0 仍然是 research/demo prototype，不是临床报告生成系统。

当前限制包括：

- 未训练端到端医学报告生成模型
- 未接入 LLM report generator
- 未接入 RAG 知识库
- 未做 LoRA / SFT
- 未使用图像-医生报告配对数据
- 未使用 lesion-level annotation
- 未实现自动图像质量评估
- CAM 只是 weak model attention evidence，不是病灶标注
- `validation.json` 只检查 artifact schema、安全声明和 claim 支撑关系，不评估医学正确性

---

## 与 v0.5.3 的关系

v0.5.3 主要完成 unified CAM adapter foundation，使不同 backbone 可以通过统一接口生成 CAM 可解释性结果。

v0.6.0 在此基础上，将 CAM 结果作为 weak visual evidence 接入 case report pipeline，并通过 `findings.json` 和 `validation.json` 建立 claim-level 可追踪机制。

因此，v0.6.0 不是替代 v0.5 benchmark，而是在 v0.5 benchmark / explainability 基础上的 case-level workflow 扩展。

---

## 下一步

### v0.6.1：Guarded LLM Report Drafting

计划方向：

- 增加 LLM report renderer
- 增加 prompt builder
- 支持 schema-constrained generation
- 增加 unsupported claim checker
- 增加 template fallback
- 保持 report 只能基于 structured evidence 生成

### v0.6.2：Demo and Deployment

计划方向：

- 增加 Gradio / Streamlit demo
- 增加 Dockerfile
- 支持 one-command local demo
- 增强 README 展示图和 case artifact 可视化

### v0.7.0：Knowledge-grounded Report Drafting

计划方向：

- 构建小型眼科知识库
- 接入 DR grading / terminology / report template retrieval
- 支持 RAG-grounded report drafting
- 输出 citation / source trace

---

## 使用声明

本版本输出仅用于科研、工程实践和项目展示。

`report.md` 与 `report.html` 是 AI-generated research/demo draft，不是临床诊断报告，也不是治疗建议。

CAM 是 weak model attention evidence，不是 lesion annotation。

所有输出均需要人工审核。
