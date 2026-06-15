# OphAgent

## v0.6.6：预审风险排序技术验证

OphAgent v0.6.6 已完成无真实标签预审风险排序方向的技术验证。该方向用于探索：在真实标签未知时，如何根据模型推理输出生成人工复核优先级队列。

v0.6.6 完整实验脚本、规则实现、逐 backbone 结果和评估表格已合并到 main。当前结果仍属于研究验证，不作为正式临床评估结论；后续如进入正式协作，将根据实际模型输出接口和数据格式进行适配交付。


## 眼科 AI 模型输出审计与失败样本发现原型

OphAgent 是一个面向眼科医学影像的 AI workflow 项目。项目最初从 APTOS2019 糖尿病视网膜病变分级 benchmark、CAM 可解释性分析和病例报告草稿生成出发，在 v0.6.5 收敛为医院线下交流展示版本，并在 v0.6.6 进一步推进到无真实标签预审风险排序：

    普通模型预测结果
    → 模型输出审计
    → 风险样本标记
    → 人工复核优先级
    → 受控报告草稿 / 安全审计辅助

当前项目不声称实现临床诊断系统，不声称实现自动医学报告生成系统，也不声称完成临床安全验证。

更准确的定位是：

    眼科 AI 模型输出审计工具
    +
    失败样本发现 scaffold
    +
    后续真实数据 evidence grounding / VQA evaluation 的前置工程原型

> 本项目仅用于科研、工程实践与项目展示，不用于临床诊断、治疗建议或真实医疗决策。

---

## v0.6.5 Showcase 保留入口

v0.6.5 保留为展示入口；v0.6.6 为当前主要技术验证结果，完整预审排序、强基线对比与选择性复核产物见 `experiments/summary/v0_6_6/`。


v0.6.5 的重点不是把已有 JSON / HTML 文件放到一起，而是展示一个更实际的问题：

    普通分类模型只给出预测结果；
    OphAgent 进一步判断哪些模型输出不应轻信，哪些样本应优先交给医生复核。

### v0.6.5 展示内容

| 模块 | 说明 |
|---|---|
| 项目定位 | OphAgent 是模型输出审计工具，不是临床诊断系统 |
| 核心价值对比 | 普通模型输出 vs OphAgent 审计后的复核优先级 |
| 15-case demo risk table | 将 15 张 demo samples 转化为风险样本表 |
| 高风险样本分析 | 展示 Severe DR 被低估为 Moderate DR 的典型样本 |
| 安全审计示例 | 展示诊断越权、CAM 夸大、临床用途越权如何触发安全模板 |
| 线下对接策略 | 先了解项目现状、账号权限、现有系统和团队分工，再找最小切入点 |
| 不声称内容 | 不证明临床安全性，不替代医生，不把 CAM 当病灶定位 |

---

## 15-case demo risk table

v0.6.5 新增了 15 张 demo samples 的风险样本表。

脚本：

    scripts/build_demo_risk_case_table.py

输出：

    experiments/summary/v0_6_5/demo_risk_case_table.csv
    experiments/summary/v0_6_5/demo_risk_case_table.md
    experiments/summary/v0_6_5/demo_risk_case_table_cn.md
    experiments/summary/v0_6_5/demo_risk_case_summary.json

### 结果摘要

| 指标 | 数值 |
|---|---:|
| total_cases | 15 |
| correct_count | 11 |
| incorrect_count | 4 |
| accuracy_on_demo_samples | 0.7333 |
| high_priority_human_review | 3 |
| human_review_recommended | 3 |
| routine_review | 9 |

### 主要风险类型

| 风险类型 | 数量 | 含义 |
|---|---:|---|
| severe_underestimate | 3 | 重症被低估，应优先复核 |
| adjacent_grade_confusion | 3 | 相邻 DR 等级混淆 |
| low_margin_uncertain | 2 | top-1 与 top-2 差距小，决策边界模糊 |
| low_conf_correct | 2 | 预测正确但置信度偏低 |
| review_not_prioritized | 9 | 当前规则未优先标记，常规抽检即可 |

该表不是正式临床验证结果。它只用于展示 OphAgent 如何把模型预测、置信度、top1-top2 margin、错误类型和严重程度低估转化为人工复核优先级。

---

## 典型高风险样本

v0.6.5 中的一个 high-priority case：

| 字段 | 内容 |
|---|---|
| case_id | 383e72af1955 |
| 真实标签 | Severe DR |
| 模型预测 | Moderate DR |
| 是否正确 | False |
| 置信度 | 0.644 |
| 第二名预测 | Severe DR (0.259) |
| 风险类型 | 相邻等级混淆；重症被低估 |
| 建议动作 | 优先医生复核 |

这个样本的意义不是证明模型“坏”，而是说明：

    普通模型输出只告诉你 Moderate DR；
    OphAgent 审计后会进一步指出这是 Severe DR 被低估，
    因此应优先进入人工复核队列。

---

## 安全审计能力

OphAgent 的报告草稿链路不是为了生成临床报告，而是为了验证模型输出能否在明确证据边界内被组织成可审计文本。

当前安全审计会重点拦截：

| 风险表达 | 示例 | 处理 |
|---|---|---|
| 诊断越权 | The patient is diagnosed with diabetic retinopathy. | 标记风险并回退 |
| CAM / heatmap 夸大 | CAM heatmap localizes retinal lesions. | 标记风险并回退 |
| 临床用途越权 | This report can be used as a clinical reference. | 标记风险并回退 |

相关产物：

    experiments/summary/v0_6_4/
    experiments/summary/v0_6_5/raw_vs_guarded_example.md

注意：RuleBasedSafetyChecker 是 v0.6.x 阶段的快速安全基线，不是完整 hallucination detector。后续 v0.7 应转向真实图像-文本证据匹配和 VQA / grounding evaluation。

---

## Benchmark Results

### APTOS2019 DR grading 代表性结果

| Backbone | Setting | Accuracy | Macro-F1 | Weighted-F1 | QWK |
|---|---|---:|---:|---:|---:|
| Swin-Tiny | lightweight baseline | 0.829 | 0.657 | 0.820 | 0.898 |
| ConvNeXt-Tiny | lightweight baseline | 0.814 | 0.650 | 0.809 | 0.862 |
| ViT-B/16 | lightweight baseline | 0.818 | 0.646 | 0.814 | 0.876 |
| RETFound-MAE-CFP | official-like | 0.804 | 0.583 | 0.789 | 0.866 |

这些结果用于支撑项目中的模型输出审计与失败样本分析，不作为统计显著性结论。

---

## Explainability 说明

项目支持统一 CAM adapter：

| 方法 | 用途 |
|---|---|
| Grad-CAM | 基础模型关注区域可视化 |
| HiResCAM | 更高分辨率关注区域参考 |
| EigenCAM | 梯度无关的主成分关注区域参考 |
| LayerCAM | 分层 attention-like 可视化参考 |

当前 CAM / heatmap 只作为模型关注区域的弱参考，不作为病灶定位、病灶分割或临床诊断依据。

在 v0.6.5 展示中，CAM 不再作为核心证据。当前展示重点转向：

    风险样本表
    +
    复核优先级
    +
    安全审计
    +
    线下协作切入点

---

## Quick Start

### 1. 查看医院展示入口

推荐直接查看 v0.6.5 展示页：

    experiments/summary/v0_6_5/integrated_showcase.html

下载或克隆仓库后，可直接用浏览器打开该 HTML 文件。

配套说明文档：

    experiments/summary/v0_6_5/README.md

### 2. 重新生成 15-case risk table

需要本地已有 ConvNeXt-Tiny checkpoint：

    python scripts/build_demo_risk_case_table.py \
      --samples-root demo_samples \
      --config configs/vision_baseline.yaml \
      --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
      --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json \
      --output-dir experiments/summary/v0_6_5 \
      --max-per-class 3

### 3. 运行单病例 case report pipeline

该路径用于生成传统 evidence-bottleneck case artifact：

    python scripts/run_case_report.py \
      --image demo_samples/cmoderatedr/d9bbdc33db83.png \
      --config configs/vision_baseline.yaml \
      --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
      --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json \
      --output experiments/case_reports/d9bbdc33db83

输出包括：

    prediction.json
    findings.json
    validation.json
    report.md
    report.html
    metadata.json
    cam/overlay.png

---

## Repository Map

| 路径 | 说明 |
|---|---|
| `configs/` | 模型与实验配置 |
| `models/classifiers/` | 分类模型构建、训练、评估 |
| `explain/` | CAM 生成与 adapter |
| `findings/` | findings / validation 相关逻辑 |
| `reasoning/llm_report/` | report provider、safety checker、renderer |
| `scripts/run_case_report.py` | 单病例 evidence-bottleneck pipeline |
| `scripts/run_real_llm_safety_probe.py` | 批量 guarded report safety probe |
| `scripts/build_demo_risk_case_table.py` | v0.6.5 demo risk table 构建脚本 |
| `experiments/summary/v0_6_6/` | 当前主技术结果入口：无真实标签预审风险排序 |
| `experiments/summary/v0_6_5/` | 医院线下展示入口 |
| `experiments/case_reports/` | 单病例 artifact 示例 |

---

## Documentation

| 页面 | 内容 |
|---|---|
| [v0.6.5 Integrated Showcase](experiments/summary/v0_6_5/README.md) | 医院线下展示入口：risk table、高风险样本、模型输出审计、线下对接策略 |
| [v0.6.4 Safety Probe Summary](experiments/summary/v0_6_4/README.md) | 5-case real LLM safety probe 与 unsafe mock positive control |
| [v0.6.3 Real LLM Provider Design](notes/v0.6.3_controlled_real_llm_provider_design.md) | OpenAI-compatible real LLM provider 接入设计 |
| [v0.6 Case Findings Schema](docs/schema/case_findings_schema_v0_6.md) | `findings.json` 与 `validation.json` 字段规范 |
| [Changelog](CHANGELOG.md) | 版本更新记录 |

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| v0.5.x | Benchmark + CAM adapter | Completed |
| v0.6.0-v0.6.4 | Evidence-bottleneck report + guarded LLM + safety probe | Completed |
| v0.6.5 | Integrated showcase + demo risk case table | Completed |
| v0.6.6 | Label-free pre-review risk ranking + post-hoc validation | Completed |
| v0.7.0 | Evidence grounding / VQA evaluation | Planned |
| v0.8.0 | Lesion concept / report verbalization adapter | Planned |

---

## Disclaimer

本项目仅用于科研、工程实践与项目展示。

- 不是临床诊断系统。
- 不是自动医学报告生成系统。
- 不提供治疗建议。
- 不替代眼科医生审核。
- CAM / heatmap 不是病灶标注，也不是病灶定位。
- demo risk table 不是临床验证集结果。
- v0.6.x 的 rule-based safety checker 不是完整 hallucination detector。
- 所有输出都需要人工审核。
