# OphAgent

## 眼科基础模型评测与证据瓶颈病例报告原型

OphAgent 是一个面向眼科医学影像的 AI workflow 项目，当前从 **眼底图像分类 benchmark** 与 **CAM 可解释性分析**，进一步升级为一个可审计、可追踪、可扩展的 **病例报告原型**。

```text
眼科基础模型评测
+
证据瓶颈病例报告原型
```

- 卡片式 HTML 报告：`experiments/case_reports/d9bbdc33db83/report.html`，GitHub 会显示源码，需下载或本地打开查看

v0.6.0 的核心流程：

```text
眼底图像
→ 模型预测
→ CAM 弱视觉证据
→ 结构化发现
→ 声明级验证
→ 报告草稿
→ 病例级产物
```

v0.6.0 不训练临床报告生成模型，也不声称实现眼科报告生成 SOTA。当前目标是在缺少**图像-报告配对数据、病灶级标注、完整临床上下文**的条件下，构建一个可追踪、可审核、未来可升级的 evidence-bottleneck workflow baseline。

> 本项目仅用于科研、工程实践与项目展示，不用于临床诊断、治疗建议或真实医疗决策。

---

## TL;DR

- **分类评测**：ConvNeXt / Swin / ViT / RETFound 在 APTOS2019 DR grading 上对比
- **可解释性**：统一 CAM adapter，支持 Grad-CAM / HiResCAM / EigenCAM / LayerCAM
- **v0.6 新增**：Evidence-Bottleneck Case Report Prototype，即“证据瓶颈病例报告原型”
  - 每条报告 claim 都必须追溯到 prediction / evidence / finding
  - `validation.json` 自动检查无依据结论和安全声明
  - 输出 `findings.json`、`validation.json`、`report.md`、`report.html`

---

## 病例报告产物展示

v0.6.0 可以对单张眼底图像生成完整 case artifact：

![v0.6 Case Report Showcase](docs/assets/v0_6_case_report_showcase.png)

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

核心验证结果示例：

```json
{
  "schema_valid": true,
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

---

## 为什么不是普通模板报告

v0.6.0 的重点不是“生成一段文本”，而是报告前面的 **evidence control layer（证据控制层）**。

- 每条 report claim 都有 `claim_id`
- 每条 claim 必须通过 `supported_by` 指向 prediction / evidence / finding
- `validation.json` 检查 unsupported claims，即无证据支撑的结论
- CAM 被明确标注为 weak evidence，不允许写成 lesion annotation
- 当前 evidence provider 后续可以替换为 mask / bounding box / lesion detector
- 当前 template renderer 后续可以替换为 LLM / RAG / LoRA report renderer

也就是说，v0.6.0 不是把分类结果硬拼成报告，而是先把模型输出压缩成可验证的结构化证据，再渲染为报告草稿。

---

## Quick Start

```bash
python scripts/run_case_report.py \
  --image demo_samples/cmoderatedr/d9bbdc33db83.png \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
  --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json \
  --output experiments/case_reports/d9bbdc33db83
```

运行后会生成：

```text
prediction.json
findings.json
validation.json
report.md
report.html
metadata.json
cam/overlay.png
```

---

## Benchmark Results

### v0.5 代表性结果

| Backbone | Setting | Accuracy | Macro-F1 | Weighted-F1 | QWK |
|---|---|---:|---:|---:|---:|
| Swin-Tiny | lightweight baseline | 0.829 | 0.657 | 0.820 | 0.898 |
| ConvNeXt-Tiny | lightweight baseline | 0.814 | 0.650 | 0.809 | 0.862 |
| ViT-B/16 | lightweight baseline | 0.818 | 0.646 | 0.814 | 0.876 |
| RETFound-MAE-CFP | official-like | 0.804 | 0.583 | 0.789 | 0.866 |

当前结果基于 single-run / seed=42，更适合作为 representation behavior analysis 与 benchmark infrastructure validation，不作为统计显著性结论。

---

## Explainability

v0.5.3 引入统一 CAM adapter，使不同 backbone 可以通过统一接口生成 CAM 可解释性结果。

当前支持：

- ConvNeXt-Tiny
- Swin-Tiny
- ViT-B/16
- ViT-L/16
- RETFound-MAE-CFP

当前 CAM 结果仅作为模型注意力可视化和 qualitative sanity check，不作为病灶标注或临床定位依据。

---

## Documentation

| 页面 | 内容 |
|---|---|
| [v0.6.0 Report Generation Design](notes/v0.6.0_report_generation_design.md) | v0.6.0 证据瓶颈病例报告原型设计 |
| [v0.6.0 Case Findings Schema](docs/schema/case_findings_schema_v0_6.md) | `findings.json` 与 `validation.json` 的字段规范 |
| [Example Case Report](experiments/case_reports/d9bbdc33db83/report.md) | 示例病例报告草稿 |
| [Example Validation](experiments/case_reports/d9bbdc33db83/validation.json) | 声明级级别验证结果 |
| [v0.5.3 README Archive](docs/v0_5_3_readme_archive.md) | 旧版 benchmark-oriented README 归档 |
| [Changelog](CHANGELOG.md) | 版本更新记录 |

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| v0.5.x | Benchmark + CAM adapter | Completed |
| v0.6.0 | Evidence-bottleneck case report prototype | Current |
| v0.6.1 | Guarded LLM report drafting | Planned |
| v0.6.2 | Demo and deployment | Planned |
| v0.7.0 | Knowledge grounding / RAG | Planned |
| v0.8.0 | LoRA report verbalization adapter | Planned |

---

## Disclaimer

本项目仅用于科研、工程实践与项目展示。

生成的病例报告是 AI-generated research/demo draft，不是临床诊断报告，也不是治疗建议。

CAM 是 weak model attention evidence，不是病灶标注。

v0.6.0 尚未实现自动图像质量评估。

所有输出都需要人工审核。
