# CHANGELOG

## v0.8.6 - 交互式眼科模型中转台

### 新增

- 新增交互式 Model Hub 页面，包含模型工程区、研究评测区、病例回放与路由解释。
- 新增全局模型库与任务兼容性判断，模型按当前任务标记为：可直接推理、仅离线回放、可适配当前任务、不可接入。
- 新增 timm ImageFolder 训练适配流程，支持 ConvNeXt、Swin、ViT 等 timm 模型通过 YAML recipe 提交训练任务。
- 新增训练配置注册系统，统一保存 base_recipe.yaml、submitted_config.yaml、effective_config.yaml、validation_report.json 和 run_manifest.yaml。
- 新增四类工程训练模板：快速链路验证、通用全量微调、冻结骨干只训分类头、低学习率保守微调。
- 新增官方锚点档案，记录 ConvNeXt-Tiny、Swin-Tiny、ViT-B/16、ViT-L/16 的官方配置来源和当前可执行边界。
- 新增固定预算 LR×WD 验证集搜索计划，用于科研候选模型的等预算调参规划；当前仅作为规划和预览层。
- 新增训练任务记录与本地曲线展示，支持查看训练/验证损失、验证集指标、测试结果和原始训练日志。
- 新增路由/专家组合实验台，支持单路由、多路由、单专家、多专家、固定专家接管和专家池概率平均融合。
- 新增全局候选扫描，支持在当前模型池内探索不同路由策略、专家调用比例和专家组合下的成本-性能操作点。
- 新增病例回放与研究审计隔离，默认视图不显示真实标签、残余事件和原始 JSON。

### 变更

- 将“基础输出模型”改为“默认输出模型”，更准确地区分默认输出、路由模型和专家接管。
- 全新微调默认从 timm 原始预训练权重初始化，不再静默继承已有眼病 checkpoint。
- 旧 checkpoint 仅在显式选择继续训练或跨疾病迁移研究时使用。
- DR 代理风险事件说明改为行内问号 Tooltip。
- 组合对比表按任务注册协议展示指标，并对工程字段进行中文化。
- 模型元信息进一步区分模型族、具体架构和预训练来源。
- 青光眼 prediction 的 image_key 处理改为稳定的“类别/文件名”形式，避免不同类别下同名图片冲突。

### 同步与清理

- 更新 v0.8.5 / v0.8.5b / v0.8.5c 相关模型注册、任务注册、known-model inventory 和 adapter onboarding 产物。
- 清理部分旧 official-like 实验配置与结果，避免与当前 Model Hub 资产和官方协议档案混淆。
- 将训练运行状态、runtime、work、checkpoint 和日志从版本控制中隔离。

### 已知限制

- 自动训练当前主要支持 timm_imagefolder_v1。
- RETFound、RETFound-Green、DINOv2 等模型仍需要后续专用 trainer/loader adapter。
- 全局候选扫描是当前模型池上的探索性工具，不是最终论文结论。
- 工程 recipe 用于链路验证和统一初筛，不代表每个模型的官方最优训练协议。
- forward-only cost 不包含图像解码、预处理、I/O、服务排队和真实部署开销。


## v0.8.5c - timm adapter 激活

- 激活 ConvNeXt、Swin、ViT-B ImageNet 和青光眼 ConvNeXt 的 timm adapter 链路。
- 生成统一预测、基础指标、前向成本和路由 replay 产物。
- 验证 timm adapter 输出与既有实验结果的一致性。


## v0.8.5b - known-model inventory 与 adapter onboarding

- 新增已知模型清单、模型资产发现、adapter onboarding 和路由 replay 流程。
- 补充模型族、架构、预训练来源等结构化字段。
- 支持对已知模型进行注册、检查和受控产物发布。


## v0.8.5 - 模型注册与路由协议

- 新增模型注册、任务注册和 scout-expert 协议配置。
- 将 DR 和青光眼任务纳入统一任务注册表。
- 建立模型、任务、artifact、路由协议之间的基础映射。


## v0.8.4b - 青光眼 forward-only 成本闭环

- 新增青光眼三分类任务的 forward-only 成本评估。
- 形成 ConvNeXt scout 与 RETFound-DINOv2 expert 的成本-性能对照。
- 明确 forward-only cost 不包含图像解码、预处理、I/O、服务排队和真实部署开销。

## v0.7.4 - Audit Demo case detail and checkpoint discovery

- 新增六个 APTOS backbone 的 checkpoint / artifact 自动发现。
- 新增病例复核详情弹窗，不增加第六页导航。
- 新增复核容量 N、风险 Top N 和随机抽 N 对照，用于模拟科室动态复核工作量。
- 新增病例队列筛选、搜索和每页 12 条分页。
- 新增预审/后验数据隔离，默认临床详情不显示真实标签和后验事件。
- 新增 generic multiclass 边界，非 DR CSV 不启用 VTDR miss、large undergrading、expected grade 等 DR 专属指标。
- 新增 checkpoint discovery、case detail、容量抽样和协议边界相关测试。
- 修复单病例、批量卡片和详情弹窗的展示语义，统一区分“模型预测等级”和“模型输出复核优先级”，避免将 PDR / Severe 等重症预测误解为普通病例。
- 修复上传图像在线 checkpoint 推理链路，上传图像无历史 prediction record 时也可基于当前模型生成五级概率；失败时显示固定阶段错误，不回退教学概率。

## v0.7.3 - Audit Demo clinical UI

- 将 Streamlit demo 升级为五页 OphAgent Audit Demo。
- 新增临床展示 / 研究审计双模式。
- 默认展示临床病例卡与红黄绿复核队列。
- 修复中文显示、数值遮挡、标签重叠和候选卡省略号问题。
- 保留 v0.7.1b / v0.7.2 外部冻结研究证据展示。
- 保留旧版入口 `app/demo_legacy_v0_4_2.py`。

## v0.7.2 - Metric sensitivity audit

- 新增 metric-sensitivity audit，用于检查预审排序结论是否依赖单一评价指标。
- 固定事件目标为 grade-based VTDR miss proxy。
- 比较 AURC、AUGRC、partial_AUGRC_70_90、Top20 event recall 等评价口径。
- 主要结果：
  - AURC：12/12 第一；
  - AUGRC：12/12 第一；
  - partial_AUGRC_70_90：12/12 第一；
  - Top20 event recall：11/12 第一或并列第一。
- 明确该结果表示跨评价口径的一致性，不等同于临床效用证明。

## v0.7.1b - External review ranking protocol completion

- 新增 `scripts/evaluate_v071b_protocol_completion_ci.py`，补全 v0.7.1 外部复核排序的统计验证流程。
- 新增 random gate-only baseline，用于检查 `gated_severe_prob_mass_only` 的增益是否超过单纯预测等级 gate。
- 新增 image-clustered bootstrap CI，同一图像的 6 个 backbone 记录一起重采样。
- 新增 seed sensitivity check，使用 seed=42 / 123 / 2026 / 3407 验证 Monte Carlo 稳定性。
- 主比较固定为 VTDR miss / Top20% / `gated_severe_prob_mass_only` vs `random_gate_only_expected`。
- 主要结果：
  - IDRiD_data：Δ event recall = +0.3385，95% CI [0.2195, 0.4742]。
  - MESSIDOR2：Δ event recall = +0.6268，95% CI [0.5003, 0.7369]。
- 记录 protocol deviation：原 v0.7.0 协议计划保留外部 `learned_logistic` 监督式 baseline，但当前 v0.7.1/v0.7.1b 尚未实现该外部 baseline；本版本 primary gate-only comparison 不受影响。

## v0.7.1 - External DR direct inference and review ranking

- 使用 APTOS-trained frozen checkpoints 直接推理 IDRiD_data / MESSIDOR2 test split。
- 新增外部分类指标、混淆矩阵、逐样本预测表和复核排序评估。
- 输出目录：`experiments/summary/v0_7_1/`。
- 结果显示外部分类迁移存在明显压力；复核排序结果用于 v0.7.1b 的 gate-only 对照和 clustered CI 验证。

## v0.7.0 - External DR protocol freeze and dataset precheck

- 冻结外部 DR 验证前的目标事件、复核预算、排序信号和 checkpoint manifest。
- 新增 IDRiD_data / MESSIDOR2 外部数据预检、类别分布统计和 MD5 重叠审计。
- 未发现 APTOS 与外部 test split 的 MD5 重叠。
- 发现 IDRiD 内部 1 组 train/test MD5 重复且标签冲突，已记录 duplicate exclusion manifest。
- 原协议计划保留外部 `learned_logistic` 监督式 baseline；截至 v0.7.1b，该 baseline 尚未实现，已作为 protocol deviation 记录。
- 输出目录：`experiments/summary/v0_7_0/`。

## v0.6.8b - 稳健性与机制审计

- 同步根目录 `README.md` 和 `CHANGELOG.md` 的项目主页口径。
- 将当前稳定研究节点更新为 v0.6.8b。
- 新增结果目录：`experiments/summary/v0_6_8b/`
- 新增 paired image-key clustered bootstrap。
- 新增 Top20% 捕获重叠分析。
- 新增 Logistic 系数稳定性分析。
- 新增 repeated split sensitivity。
- 修正并冻结正式 bootstrap 评价口径：pooled-backbone training，per-backbone test reporting。
- 结论：`learned_logistic` 有竞争力，但没有稳定超过事件特异性 severity-aware signal。对 `large_undergrading`，`expected_gap_only` 更稳；对 `vision_threatening_dr_miss`，`gated_severe_prob_mass_only` 更稳。

## v0.6.8 - 学习型复核分数

- 新增脚本：`scripts/analyze_v068_learned_deferral_score.py`
- 新增结果目录：`experiments/summary/v0_6_8/`
- 使用 L2 Logistic Regression 构建 supervised learned review score。
- 使用 `decision_function` 作为 learned deferral score，不解释为校准概率。
- 采用 grouped cross-validation，并以 `image_key` 作为分组单位。
- 结论：learned score 有竞争力，但不能替代最强事件特异性规则。

---

## v0.6.7c - 排序信号机制分析

- 新增统一排序方法比较：不确定性基线、`ophagent_combined`、严重程度感知基线。
- 新增多复核预算评估：Top5% 到 Top50%。
- 新增 Top20% overlap analysis 和 residual profile。
- 新增结果目录：`experiments/summary/v0_6_7c/`
- 新增脚本：`scripts/analyze_v067c_ranking_signal_mechanism.py`

核心结果：

- `general_error` 更适合 `margin_only`。
- `large_undergrading` 更适合 `expected_gap_only`。
- `vision_threatening_dr_miss` 更适合 `gated_severe_prob_mass_only`。

结论：`ophagent_combined` 是有效的初版透明审计规则，但不是所有危险错误类型上的最优规则；不同 clinical dangerous events 需要不同 post-hoc risk signals。

---

## v0.6.7b - 严重程度感知信号消融

- 新增 severity-aware baseline ablation。
- 新增脚本：`scripts/analyze_v067_severity_aware_baselines.py`
- 新增结果目录：`experiments/summary/v0_6_7b/`
- 比较 `ophagent_combined`、`expected_gap_only`、`gated_severe_prob_mass_only` 等单信号基线。

核心结果：

- `large_undergrading` 中，`expected_gap_only` 优于 `ophagent_combined`。
- `vision_threatening_dr_miss` 中，`gated_severe_prob_mass_only` 优于 `ophagent_combined`。

结论：v0.6.7b 拆解了 combined 的优势来源，危险低估样本的主要富集能力可以由更简单的严重程度感知信号解释。

---

## v0.6.7 - 临床残余风险审计

- 新增 clinical-risk proxy 分析。
- 新增 review burden 和 release-side residual risk 分析。
- 新增脚本：`scripts/evaluate_clinical_residual_risk.py`
- 新增结果目录：`experiments/summary/v0_6_7/`

跨 6 个 backbone 汇总：

- `general_error`：1249
- `large_undergrading`：263
- `vision_threatening_dr_miss`：391
- `high_confidence_vision_threatening_miss`：120

结论：v0.6.7 将 OphAgent 从普通失败样本发现推进到临床风险代理指标下的残余风险审计。clinical-risk proxy 仍基于 APTOS 五分类标签构造，不等价于真实临床终点。

---

## v0.6.6 - 预审风险排序技术验证

- 完成无真实标签预审风险排序方向的技术验证。
- 完整实验脚本、规则实现、逐 backbone 结果和复现链路已合并到 main。
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
