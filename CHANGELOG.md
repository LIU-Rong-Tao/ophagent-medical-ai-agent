# 更新日志

---

## v0.6.2 — Safety Regression and Audit Metadata

### 新增

- 新增 `tests/test_llm_report_safety_checker.py`，使用 Python 标准库 `unittest` 覆盖 `RuleBasedSafetyChecker` 的核心行为，不额外引入 pytest 依赖
- 新增安全草稿、临床诊断越权、CAM / heatmap 夸大、图像质量夸大、临床用途夸大、缺少非临床用途声明、缺少人工审核声明等 7 类回归测试
- 为 deterministic template provider 和 mock LLM provider 增加 `provider_version` 元数据
- 为 `safety_report.json` 增加 `audit_metadata`，包括 `generated_at`、`prompt_hash`、`provider_version`、`checker_version`、`safety_policy_version` 和 deterministic provider metadata
- 新增 `docs/safety/llm_report_safety_rule_boundaries.md`，说明 `RuleBasedSafetyChecker` 的覆盖范围、已知 false negative / false positive 风险和后续扩展方向
- 新增 `notes/v0.6.2_safety_regression_audit_plan.md`，记录 v0.6.2 的 safety regression 与 audit metadata 计划

### 变更

- README Roadmap 调整为先完成 safety regression tests 与 audit metadata，再推进 controlled real LLM provider integration

### 说明

- v0.6.2 仍不接入真实 LLM API
- `RuleBasedSafetyChecker` 仍是确定性的 rule-based safety guard，不是完整 hallucination detector，也不是 clinical safety verifier
- 真实 LLM provider、LLM-as-judge、医学事实校验和图像质量评估保留到后续版本

---

## v0.6.1 — Guarded LLM Report Drafting with Explicit Safety Trace

### 新增

- 新增 `reasoning/llm_report/` 模块，用于受控 LLM 报告草稿生成实验
- 新增 `prompt_builder.py`，将 `prediction.json`、`findings.json`、`validation.json` 和 `metadata.json` 转换为受约束 prompt
- 新增 `provider.py`，提供确定性的 `TemplateProvider` 与 `MockLLMProvider`
- 新增 `safety_checker.py`，实现规则型生成后安全检查
- 新增 `renderer.py`，串联 prompt builder、provider 和 safety checker，生成 guarded report artifacts
- 新增 `safety_report.json`，记录 LLM draft 的安全检查结果、flagged claims、fallback 决策和审计轨迹
- 新增 `reports/` 中间产物语义：
  - `llm_raw.md`：Provider 原始输出
  - `llm_checked.md`：通过安全检查后的草稿
  - `llm_guarded.html`：简化版 guarded report HTML
  - `template.md` / `template.html`：确定性模板 fallback
- 新增 `scripts/run_case_report.py` 参数：
  - `--report-provider template/mock_llm`
  - `--mock-llm-mode safe/unsafe_diagnosis/unsafe_cam/unsafe_mixed`
- 新增 v0.6.1 轻量展示产物：
  - `experiments/summary/v0_6_1/`

### 变更

- 默认 `--report-provider template`，保持 v0.6.0 确定性模板报告路径不变
- 当显式使用 `--report-provider mock_llm` 时，系统会在 v0.6.0 artifact 生成后追加 guarded report rendering
- `metadata.json` 新增 `report_provider` 与 `guarded_report` 字段，用于记录报告生成模式和 fallback 状态

### 验证

- safe mock case：
  - `overall_pass = true`
  - `fallback_triggered = false`
  - LLM draft 被保留为 `llm_checked.md`
- unsafe CAM mock case：
  - `overall_pass = false`
  - `fallback_triggered = true`
  - CAM / heatmap overclaim 被拦截
  - 最终报告回退到 deterministic template report

### 说明

- v0.6.1 不接入真实 LLM API
- v0.6.1 不实现 LLM-as-judge
- v0.6.1 不声称实现临床级报告生成
- 当前 MockLLMProvider 用于验证 guarded generation 控制链路，不代表真实模型能力
- 当前 HTML 仅作为辅助展示，v0.6.1 的核心展示重点是 safety trace，而不是 UI 美观

---

## v0.6.0 — Evidence-Bottleneck Case Report Prototype

### 新增

- 增加 evidence-bottleneck case report pipeline，用于从单张眼底图像生成可追踪的病例级 artifact
- 新增 `scripts/run_case_report.py`，支持一条命令生成：
  - `prediction.json`
  - `findings.json`
  - `validation.json`
  - `report.md`
  - `report.html`
  - `metadata.json`
  - CAM `original.png` / `heatmap.png` / `overlay.png`
- 新增 v0.6.0 case findings schema，用于规范 `findings.json` 与 `validation.json`
- 新增 claim-level traceability，每条 report claim 通过 `supported_by` 指向 prediction / evidence / finding
- 新增 lightweight validation 输出，用于检查：
  - schema validity
  - required disclaimer
  - human-review-required statement
  - CAM weak-evidence wording
  - unsupported claim count
  - evidence coverage rate
  - image-quality overclaim
  - report reproducibility
- 新增 v0.6.0 example case artifact：
  - `experiments/case_reports/d9bbdc33db83/`

### 变更

- README 从 benchmark-oriented 首页调整为展示型 landing page
- 旧版 v0.5.3 README 归档到 `docs/v0_5_3_readme_archive.md`
- 根目录 README 重点展示：
  - v0.5 benchmark 代表性结果
  - v0.5.3 CAM adapter
  - v0.6.0 evidence-bottleneck case report prototype
  - `validation.json` 安全与可追踪检查结果
- v0.6.0 报告生成路线从“自由文本生成”明确收敛为：
  - prediction
  - weak visual evidence
  - structured findings
  - claim-level validation
  - report draft

### 说明

- v0.6.0 不训练端到端医学报告生成模型
- v0.6.0 不声称实现 ophthalmic report generation SOTA
- 当前 report 是 AI-generated research/demo draft，不是临床诊断报告
- CAM 仅作为 weak model attention evidence，不是 lesion annotation
- 当前尚未实现自动图像质量评估，仅保留 quality-aware caution
- `validation.json` 只检查 artifact 的 schema、安全声明、claim 支撑关系和可追踪性，不评估医学正确性

---

## v0.5.3 — CAM Adapter Foundation

### 新增

- 增加 unified CAM adapter，支持 ConvNeXt / Swin / ViT-B / ViT-L / RETFound
- 增加 Transformer backbone 的 relative block depth target selection：`early` / `middle` / `late`
- 增加 CAM grid generation，用于 `method × target layer/depth × smoothing` 的 qualitative sanity check
- 增加 representative fundus image 的 selected CAM comparison

### 变更

- `explain/gradcam.py` 改为通过 backbone adapter 获取 target layer 与 reshape transform
- `scripts/run_gradcam_grid.py` 支持 CNN stage 与 Transformer block depth 两类 target layer schema
- CAM visualization selection 从“热力图好看”调整为“眼底病灶证据对齐优先”

### 说明

- v0.5.3 的 CAM 结果仅用于 qualitative visualization sanity check
- 当前 selected CAM setting 不作为医学病灶定位、临床诊断或 explanation faithfulness 结论
- 正式 CAM consistency evaluation 延后到 v0.6

## v0.5.2 — Benchmark Consistency Repair

### 修复
- 修复历史 benchmark artifact inconsistency 问题
- 修复 legacy experiment naming 与 checkpoint mismatch 导致的实验污染
- 删除受污染的 `aptos_vit_base_patch16` 历史实验目录
- 修正 RETFound 对比中 initialization-only controlled benchmark 的表述风险

### 新增
- clean ViT-B/16 ImageNet lightweight baseline
- ViT-L/16 official-like baseline
- RETFound-MAE-CFP official-like setting
- backbone-scale-aligned official-like comparison

### 变更
- 统一 benchmark experiment namespace
- 统一 checkpoint naming schema：`{backbone}_best.pth`
- 重构 benchmark experiment 与 official-like config 结构

## v0.5.1 - Multi-metric Benchmark Evaluation

### 新增

- multi-metric benchmark evaluation
- QWK evaluation
- per-class F1 analysis
- prediction entropy analysis
- top1-top2 margin analysis
- confusion matrix generation

### 新增文件

```text
scripts/build_benchmark_table.py

experiments/summary/v0_5_1/benchmark_metrics.csv
experiments/summary/v0_5_1/per_class_f1.csv
experiments/summary/v0_5_1/confusion_matrices/
experiments/summary/v0_5_1/metrics_update.md
```

### Benchmark

| Backbone | Accuracy | Macro-F1 | Weighted-F1 | QWK |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.814 | 0.650 | 0.809 | 0.862 |
| Swin-Tiny | 0.829 | 0.657 | 0.820 | 0.898 |
| ViT-B/16 | 0.818 | 0.646 | 0.814 | 0.876 |
| RETFound-MAE-CFP | 0.790 | 0.552 | 0.769 | 0.834 |

### 当前观察

- Swin-Tiny 在当前 benchmark 中表现最稳定
- ViT-B/16 lightweight baseline 已接近 ConvNeXt-Tiny 与 Swin-Tiny
- ConvNeXt-Tiny 在 Severe DR 类别上表现相对更稳定
- RETFound-MAE-CFP 展现出不同的 uncertainty characteristics 与 class-wise behavior

### 当前限制

- 当前 benchmark 仍基于 single-seed evaluation
- 当前 benchmark 同时包含 lightweight baseline 与 official-like foundation setting
- 不同 backbone 的 training protocol 并不完全一致
- 当前结果更适合作为 representation behavior observation
- 尚未形成严格 controlled benchmark leaderboard

---

## v0.4.2 - Benchmark Infrastructure Cleanup

### 新增

- Swin-Tiny checkpoint metadata
- experiment version metadata
- benchmark artifact consistency improvements

### 修复

- 修复 Streamlit demo 版本号不一致问题
- 修复 unified evaluation metrics 路径
- 修复 summary builder 中 `Version: None`
- 修复 benchmark artifact relative path consistency

### 改进

- experiment-root relative artifact path
- summary builder version support
- benchmark release consistency
- benchmark portability 与 reproducibility

---

## v0.4.1 - Second Backbone Baseline

### 新增

- Swin-Tiny baseline
- unified evaluation schema
- backbone comparison summary

### Benchmark

| Backbone | Test Accuracy | Macro F1 |
|---|---:|---:|
| ConvNeXt-Tiny | 0.8136 | 0.6496 |
| Swin-Tiny | 0.8291 | 0.6567 |

### 当前限制

- 当前 benchmark 仍为 single-seed evaluation
- 尚未形成 formal benchmark leaderboard

---

## v0.4.0 - Experiment Summary Builder

### 新增

- `build_experiment_summary.py`
- unified experiment summary artifacts
- training/evaluation aggregation
- benchmark summary generation

### 输出

```text
summary.csv
summary.md
class_mapping.csv
training_curve_summary.csv
```

## v0.3.0 - Lightweight Agent Runner

### 新增

- reusable `run_agent(...)`
- provider abstraction
- structured findings integration
- optional OpenAI reasoning
- unified workflow pipeline

### Workflow

```text
image
  ↓
run_agent(...)
  ↓
classification
  ↓
structured findings
  ↓
reasoning
```

---

## v0.2.x - Workflow Demo

### 新增

- unified Streamlit demo
- Grad-CAM / HiResCAM gallery
- structured findings
- lightweight VL reasoning
- workflow integration