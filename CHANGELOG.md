# 更新日志

---

## v0.6.6 - 预审风险排序内部验证

- 完成无真实标签预审风险排序方向的内部实验验证。
- 完整实验脚本、规则实现、逐 backbone 结果和复现链路已合并回明线。
- 当前结果属于研究验证，不作为正式临床评估结论；后续根据正式协作接口适配交付。


## v0.6.5 — Integrated Showcase + Demo Risk Case Table

### 新增

- 新增 `experiments/summary/v0_6_5/`，作为医院线下展示前的集成展示入口。
- 新增 `integrated_showcase.html`，将展示顺序调整为：
  - 项目定位
  - 核心价值：普通模型输出 → OphAgent 审计后的复核优先级
  - 15-case demo risk table
  - 典型高风险样本深度分析
  - 系统工作流
  - 安全审计示例
  - 线下对接第一步
  - 当前系统不声称
- 新增 `scripts/build_demo_risk_case_table.py`，用于从 `demo_samples` 构建 15 张样本的风险样本表。
- 新增风险样本表产物：
  - `demo_risk_case_table.csv`
  - `demo_risk_case_table.md`
  - `demo_risk_case_table_cn.md`
  - `demo_risk_case_summary.json`
- 新增 `high_priority_case_analysis.md`，对一个 high-priority 样本进行中文深度分析。
- 新增 `pipeline_overview.md` 和 `raw_vs_guarded_example.md`，用于说明模型输出审计和安全模板切换逻辑。

### 结果

15 张 demo samples 风险样本表：

- total_cases: 15
- correct_count: 11
- incorrect_count: 4
- accuracy_on_demo_samples: 0.7333
- high_priority_human_review: 3
- human_review_recommended: 3
- routine_review: 9

主要风险类型：

- severe_underestimate: 3
- adjacent_grade_confusion: 3
- low_margin_uncertain: 2
- low_conf_correct: 2

### 变更

- 将 v0.6.5 从单纯 artifact 汇总调整为“模型输出审计与失败样本发现”的医院展示版本。
- 根 README 的 Documentation 区域收敛为少量核心入口，避免 v0.6.x 小版本链接过散。
- CAM / heatmap 不再作为核心展示证据，只保留为弱参考说明；当前展示重点转向 risk table 和人工复核优先级。
- 医院线下对接部分不再直接提出数据需求，而是强调先了解项目现状、账号权限、现有系统和团队分工。

### 说明

- v0.6.5 不新增模型训练。
- v0.6.5 不新增 safety rule。
- v0.6.5 不接入 RAG、LoRA 或 SFT。
- demo risk table 使用 `demo_samples` 的目录名作为 weak label，仅用于展示模型输出审计流程。
- 该版本不证明临床安全性、医学事实正确性或临床可用性。

---

## v0.6.4 — Real LLM Safety Probe on Small Case Set

### 新增

- 新增 `scripts/run_real_llm_safety_probe.py`，用于在一批 case artifacts 上运行 guarded report safety probe
- 支持从 CSV manifest 读取 `case_id,case_dir`
- 支持 `real_llm`、`mock_llm`、`template` provider
- 新增 probe summary 输出：
  - `safety_probe_results.json`
  - `safety_probe_table.md`
- 新增 `--save-samples`，可保存脱敏后的 per-case `llm_raw.md` 与 `safety_report.json`
- 新增 secret redaction，避免 API key / Bearer token 被写入 probe 结果
- 新增 v0.6.4 summary artifacts：
  - 5-case real LLM constrained prompt probe
  - 5-case unsafe mock positive control
  - unsafe mock raw draft example
  - unsafe mock safety report example

### 结果

Real LLM constrained prompt probe：

- provider: real_llm
- model: gpt-4o-mini
- total_cases: 5
- success_count: 5
- api_failure_count: 0
- safety_pass_count: 5
- fallback_count: 0
- fallback_rate: 0.0

Unsafe mock positive control：

- provider: mock_llm
- mock_llm_mode: unsafe_mixed
- total_cases: 5
- success_count: 5
- safety_pass_count: 0
- fallback_count: 5
- fallback_rate: 1.0

Aggregated flagged claim types：

- clinical_diagnosis_overclaim: 5
- cam_or_heatmap_overclaim: 5
- unsupported_lesion_localization: 5
- missing_non_clinical_use_statement: 5
- missing_human_review_statement: 5
- image_quality_overclaim: 5
- clinical_use_overclaim: 10

### 说明

- v0.6.4 是 small-scale pilot，不是统计性实验
- real LLM constrained prompt 全部通过，不代表真实 LLM 安全性已被完整验证
- unsafe mock positive control 只证明 obvious unsafe draft 能被规则拦截并触发 fallback
- 当前仍不评估医学事实正确性、病灶定位正确性或临床可用性

---

## v0.6.3 — Controlled Real LLM Provider Integration

### 新增

- 新增 OpenAI-compatible `real_llm` provider，用于受控接入真实 LLM 报告草稿生成
- `scripts/run_case_report.py` 新增 `--report-provider real_llm`
- 新增真实 LLM provider 配置失败路径测试，覆盖缺少 API key 和缺少 model name 的情况
- 新增 OpenAI-compatible response parsing 测试，不依赖真实网络
- 新增 real_llm provider 与 guarded renderer 的离线集成测试，覆盖 safe response 与 unsafe fallback 两条路径
- 新增 `scripts/dev/smoke_real_llm_provider.py`，用于手动验证真实 OpenAI-compatible endpoint
- 新增 v0.6.3 real LLM summary artifacts：
  - `experiments/summary/v0_6_3/llm_raw_real_llm.md`
  - `experiments/summary/v0_6_3/llm_checked_real_llm.md`
  - `experiments/summary/v0_6_3/llm_guarded_real_llm.html`
  - `experiments/summary/v0_6_3/safety_report_real_llm.json`
  - `experiments/summary/v0_6_3/README.md`

### 变更

- real LLM 输出现在进入既有 guarded report workflow：
  - constrained prompt
  - provider draft
  - RuleBasedSafetyChecker
  - checked report 或 template fallback
  - safety_report.json audit metadata
- README Documentation 增加 v0.6.3 real LLM summary artifacts 入口

### 验证

- 离线测试：
  - `python -m unittest discover -s tests -p "test_*.py" -v`
  - 当前测试数：12
- 语法检查：
  - `python -m py_compile scripts/run_case_report.py reasoning/llm_report/*.py tests/*.py`
- 手动 smoke run：
  - provider: real_llm
  - model: gpt-4o-mini
  - real_llm_used: True
  - overall_pass: True
  - fallback_triggered: False
  - API key leakage check: contains_sk_key = False

### 说明

- v0.6.3 接入真实 LLM provider，但仍不声称实现临床报告生成
- 真实 LLM draft 必须经过 RuleBasedSafetyChecker
- unsafe draft 仍会触发 deterministic template fallback
- 当前 HTML 展示仍是辅助产物，不是本版本核心卖点
- 当前版本不包含 LLM-as-judge、医学事实自动校验、图像质量自动判断或 Web demo

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