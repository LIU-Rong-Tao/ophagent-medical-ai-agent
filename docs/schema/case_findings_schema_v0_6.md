# v0.6.0 Case Findings Schema

## 1. 目标

本文件定义 OphAgent v0.6.0 的 case-level structured findings schema。

该 schema 是 Evidence-Bottleneck Case Report Prototype 的核心接口，用于连接：

- 模型预测
- 弱视觉证据
- 结构化 findings
- quality-aware context
- report claims
- validation
- report.md / report.html

v0.6.0 当前只实现 CAM-based weak evidence 和 template report renderer，但 schema 需要支持未来替换为：

- lesion segmentation mask
- bounding box
- lesion detector
- clinical metadata
- LLM / RAG / LoRA report renderer

---

## 2. findings.json 顶层字段

findings.json 必须包含以下一级字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| case_id | string | case 唯一标识，通常来自图像文件名 |
| input | object | 输入图像信息 |
| prediction | object | 模型预测结果 |
| evidence | array | 可引用证据列表 |
| findings | array | 结构化视觉线索 |
| report_claims | array | 可渲染到 report 的声明 |
| quality_control | object | 图像质量与质量感知说明 |
| limitations | array | 当前系统限制 |
| model_info | object | 模型、checkpoint、config 信息 |
| provenance | object | artifact 生成来源和时间 |

---

## 3. input

input 用于记录原始输入图像。

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| image_path | string | 是 | 原始图像路径 |
| saved_input_path | string | 是 | case artifact 中保存的 input.png 路径 |
| image_id | string | 是 | 图像 ID，通常来自文件名 |
| modality | string | 是 | 当前为 fundus_color_photography |
| dataset_hint | string | 否 | 例如 demo_samples 或 APTOS2019 |

示例：

{
  "image_path": "demo_samples/cmoderatedr/d9bbdc33db83.png",
  "saved_input_path": "experiments/case_reports/d9bbdc33db83/input.png",
  "image_id": "d9bbdc33db83",
  "modality": "fundus_color_photography",
  "dataset_hint": "demo_samples"
}

---

## 4. prediction

prediction 用于保存模型预测结果。

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| prediction_id | string | 是 | 预测声明 ID，例如 pred_001 |
| raw_class | string | 是 | 原始类别名，例如 cmoderatedr |
| display_name | string | 是 | 展示类别名，例如 Moderate DR |
| confidence | number | 是 | top-1 置信度 |
| topk_predictions | array | 是 | top-k 预测列表 |
| task | string | 是 | 当前为 diabetic_retinopathy_grading |

topk_predictions 中每个元素包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| rank | integer | 排名 |
| raw_class | string | 原始类别 |
| display_name | string | 展示类别 |
| confidence | number | 置信度 |

---

## 5. evidence

evidence 用于保存 report 可引用证据。

v0.6.0 当前只实现 evidence_type = cam。

未来可以扩展为：

- segmentation_mask
- bounding_box
- lesion_detector
- clinical_metadata
- report_reference

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| evidence_id | string | 是 | 证据 ID，例如 ev_cam_001 |
| evidence_type | string | 是 | cam / segmentation_mask / bounding_box 等 |
| source | string | 是 | 证据来源，例如 gradcam_stage3_eigen |
| region | string | 否 | 证据区域描述 |
| description | string | 是 | 证据文本描述 |
| clinical_strength | string | 是 | weak / moderate / strong |
| caution | string | 是 | 使用注意事项 |
| artifact_path | string | 否 | 对应图像或文件路径 |
| replaceable_by | array | 是 | 未来可替换证据类型 |

v0.6.0 对 CAM evidence 的要求：

- clinical_strength 必须为 weak
- caution 必须明确 CAM is not lesion annotation
- report 不得把 CAM 描述为真实病灶定位

---

## 6. findings

findings 用于保存结构化视觉线索。

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| finding_id | string | 是 | finding ID，例如 finding_001 |
| finding_type | string | 是 | classification_tendency / possible_visual_cue / cam_attention_observation |
| description | string | 是 | finding 描述 |
| supported_by | array | 是 | 支撑该 finding 的 evidence_id 或 prediction_id |
| confidence_level | string | 是 | qualitative / low / medium / high |
| caution | string | 是 | 临床限制说明 |

v0.6.0 中 findings 不能写成已确认病灶。

推荐 finding_type：

- classification_tendency
- possible_visual_cue
- cam_attention_observation
- quality_context
- limitation_note

---

## 7. report_claims

report_claims 是 report.md / report.html 的唯一文本事实来源。

report renderer 只能渲染 report_claims 中已有内容，不应新增 unsupported medical claim。

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| claim_id | string | 是 | claim ID，例如 claim_001 |
| text | string | 是 | 报告中可渲染的声明 |
| claim_type | string | 是 | model_prediction / visual_evidence / quality_context / limitation / disclaimer |
| supported_by | array | 是 | 支撑该 claim 的 prediction_id / evidence_id / finding_id |
| safety_level | string | 是 | informational / caution / disclaimer |
| section | string | 是 | report.md 中对应 section |

claim_type 可选：

- model_prediction
- confidence_summary
- visual_evidence
- structured_finding
- quality_context
- limitation
- disclaimer
- artifact_metadata

v0.6.0 要求：

- 每条 claim 必须有 supported_by
- disclaimer claim 可以 supported_by = ["system_policy"]
- 不允许 clinical_diagnosis claim
- 不允许 treatment_recommendation claim

---

## 8. quality_control

quality_control 用于记录图像质量与质量感知状态。

v0.6.0 不训练质量评估模型，因此默认只记录 caution。

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| image_quality_assessed | boolean | 是 | 当前是否自动评估图像质量 |
| image_quality_level | string | 是 | unknown / good / usable / poor / non_diagnostic |
| quality_aware_mode | string | 是 | not_implemented / enabled |
| quality_note | string | 是 | 质量相关说明 |
| action | string | 是 | use / use_with_caution / refuse_non_diagnostic |
| future_upgrade | array | 是 | 后续升级方向 |

v0.6.0 默认：

| 字段 | 默认值 |
|---|---|
| image_quality_assessed | false |
| image_quality_level | unknown |
| quality_aware_mode | not_implemented |
| action | caution |

注意：

- 低质量图像不应默认拒答
- 应采用 quality-aware caution
- 只有极端不可判读图像未来才触发 refuse_non_diagnostic

---

## 9. limitations

limitations 必须包含以下内容：

- no lesion-level annotation
- no physician report ground truth
- no multimodal clinical context
- automatic image quality assessment not implemented
- CAM is not lesion localization
- not clinically validated
- not for clinical diagnosis
- human review required

---

## 10. model_info

model_info 记录模型相关信息。

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| backbone | string | 是 | 模型 backbone |
| config_path | string | 是 | config 路径 |
| checkpoint_path | string | 是 | checkpoint 路径 |
| class_to_idx_path | string | 是 | class_to_idx 路径 |
| cam_method | string | 是 | CAM 方法 |
| cam_target_layer | string | 是 | CAM target layer |
| cam_smoothing | string | 是 | none / eigen / aug_eigen |

---

## 11. provenance

provenance 记录 artifact 生成过程。

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| generated_at | string | 是 | ISO 时间 |
| workflow | string | 是 | 当前为 v0.6.0_case_report |
| script | string | 是 | 生成脚本 |
| output_dir | string | 是 | 输出目录 |
| ophagent_version | string | 是 | 当前版本 |
| note | string | 是 | 非临床用途说明 |

---

## 12. validation.json 顶层字段

validation.json 用于记录轻量安全检查结果。

它不评估医学正确性，只评估 artifact 是否符合 evidence-bottleneck 约束。

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| schema_valid | boolean | findings.json 是否包含必要字段 |
| required_files_present | boolean | 必要 artifact 是否存在 |
| required_disclaimer_present | boolean | report 是否包含 disclaimer |
| human_review_required | boolean | 是否明确医生审核 |
| cam_described_as_weak_evidence | boolean | CAM 是否被描述为弱证据 |
| clinical_diagnosis_claim_present | boolean | 是否存在临床诊断声明 |
| unsupported_claim_count | integer | 无支撑 claim 数量 |
| unsupported_claim_ids | array | 无支撑 claim ID |
| evidence_coverage_rate | number | 有支撑 claim 占比 |
| image_quality_overclaimed | boolean | 是否过度声称图像质量已评估 |
| non_clinical_use_statement_present | boolean | 是否声明非临床用途 |
| report_reproducible | boolean | 是否可由 findings.json 复现 report |
| validation_warnings | array | 检查警告 |

---

## 13. validation 规则

v0.6.0 validation 至少检查：

1. findings.json 是否包含所有必要一级字段
2. 每条 report_claim 是否有 supported_by
3. supported_by 是否能在 prediction / evidence / findings / system_policy 中找到
4. report.md 是否包含 non-clinical-use disclaimer
5. report.md 是否包含 human review required
6. CAM 是否被描述为 weak visual evidence
7. report 是否出现 clinical diagnosis claim
8. report 是否出现 treatment recommendation claim
9. report 是否声称 image quality 已自动评估
10. 必要文件是否存在

---

## 14. report rendering 约束

report.md 和 report.html 只能来自：

- prediction
- evidence
- findings
- report_claims
- quality_control
- limitations
- model_info
- provenance

report renderer 不允许新增：

- 新医学发现
- 临床诊断
- 治疗建议
- 未在 evidence 中出现的定位描述
- 已完成图像质量评估的说法

---

## 15. v0.6.0 当前最小实现

v0.6.0 第一版至少生成：

- input.png
- prediction.json
- findings.json
- validation.json
- report.md
- report.html
- metadata.json
- cam/original.png
- cam/heatmap.png
- cam/overlay.png