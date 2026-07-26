# OphAgent

> **科研协议边界**：OphAgent 中现有 RETFound 线性探针 recipe 与 trainer 仅用于历史跨仓库
> 集成验证，属于 deprecated / integration-only 资产。正式的冻结特征迁移协议、模型间公平
> 比较和科研产物由 Ophthalmic Foundation Model Benchmark（OphBench）维护；OphAgent 只
> 验证并消费其标准 manifest、预测和指标，再开展模型互补性与路由研究。

OphAgent 是一个面向眼科医学图像模型的交互式模型中转台原型，用于管理模型发现、任务适配、路由模型、专家模型、成本-性能评测和病例级回放。

本项目用于科研、工程实践和项目展示，不用于临床诊断、治疗建议或真实医疗决策。

---

## 当前主线：OphAgent V1 工具化模型中转台

V1 在现有 Model Hub 内统一模型资产、任务资格、冻结概率审计、研究路由模拟和离线病例审阅。病例工作台只使用本地资产和公开 validation 演示病例，不访问公网，不读取冻结 Test，也不提供诊断或处置建议。

统一入口：

    streamlit run app/model_hub_demo.py

离线单机和院内局域网部署方式见
[`docs/MODEL_HUB_V1_OFFLINE_DEPLOYMENT.md`](docs/MODEL_HUB_V1_OFFLINE_DEPLOYMENT.md)，
实际页面验收见
[`docs/v1_acceptance/SCREENSHOT_INDEX.md`](docs/v1_acceptance/SCREENSHOT_INDEX.md)。

---

## 核心能力

### 1. 全局模型库

Model Hub 会展示服务器已发现的模型，并根据当前任务标记其状态：

- 当前任务可直接推理
- 当前任务仅离线回放
- 可适配当前任务
- 不可接入，并显示原因

这避免了将 DR 五分类模型错误地直接用于青光眼三分类任务，同时允许兼容架构作为当前任务的适配骨干。

### 2. 模型适配训练

当前支持基于 `timm_imagefolder_v1` 的 ImageFolder 训练适配流程，可用于 ConvNeXt、Swin、ViT 等 timm 模型。

训练任务采用 YAML recipe 配置，运行后统一保存：

- `base_recipe.yaml`
- `submitted_config.yaml`
- `effective_config.yaml`
- `validation_report.json`
- `run_manifest.yaml`
- prediction、metrics、forward-only cost、registration record

默认全新微调从 timm 原始预训练权重初始化，不再静默继承已有眼病 checkpoint。旧 checkpoint 仅在显式选择继续训练或跨疾病迁移研究时使用。

### 3. 工程训练模板与科研候选档案

当前提供四类工程训练模板：

- 快速链路验证
- 通用全量微调
- 冻结骨干只训分类头
- 低学习率保守微调

这些模板用于链路验证和统一初筛，不代表各模型的官方最优训练协议。

同时，系统提供 ConvNeXt、Swin、ViT 的官方锚点档案，用于记录官方配置来源、当前可执行边界和固定预算 LR×WD 验证集搜索计划。该搜索目前是科研候选实验规划层，不等同于完整官方复现或最终论文冻结协议。

### 4. 路由模型与专家模型组合评测

研究评测区支持：

- 单路由模型
- 多路由模型
- 单专家模型
- 多专家模型
- 固定专家接管
- 专家池概率平均融合
- 不同专家调用预算
- 成本-性能曲线
- Pareto 前沿和推荐操作点

系统区分“默认输出模型”和“路由模型”：未进入专家调用的病例由默认输出模型给出最终输出；进入专家调用后，由所选专家或专家池接管。其他路由模型仅在对应多模型路由机制下参与分歧或平均不确定性计算。

### 5. 病例回放与研究审计隔离

病例回放默认只显示在线推理时可获得的信息，例如图像、模型输出、专家调用状态、最终输出来源和路由解释。

研究审计视图才显示公开测试标签、DR 代理风险事件、是否纠正、残余事件和原始字段，避免把后验评测信息包装成在线临床决策。

---

## 当前支持任务

| 任务 | 数据集 | 标签空间 | 当前用途 |
|---|---|---|---|
| DR 五级分级 | APTOS2019 | ICDR 0-4 | 模型训练、路由评测、风险代理事件分析 |
| 青光眼三分类 | Glaucoma_fundus | normal / early / advanced | 模型训练、路由评测、成本-性能对比 |

---

## 当前支持模型与适配边界

当前自动训练主要支持：

- ConvNeXt
- Swin
- ViT

当前自动训练协议：

- `timm_imagefolder_v1`

仍需后续补充专用 trainer / loader adapter 的模型包括：

- RETFound
- RETFound-Green
- RETFound-DINOv2
- 其他非 timm 或自定义预处理模型

---

## 主要入口文件

| Path | Description |
|---|---|
| `app/model_hub_demo.py` | Model Hub 统一入口 |
| `app/model_hub_engineering.py` | 模型工程区：模型发现、任务适配、训练任务入口 |
| `app/model_hub_research.py` | 研究评测区：路由/专家组合、成本-性能评测 |
| `app/model_hub_clinical.py` | 离线病例审阅与历史路由回放入口 |
| `app/model_hub_review.py` | 离线病例队列、人工审阅、报告与轨迹页面 |
| `app/model_hub_tools.py` | 六项工具契约、资格门禁、错误码与统一 trace |
| `app/training_config.py` | YAML recipe、配置校验、official profile 管理 |
| `app/training_jobs.py` | 后台训练任务状态、运行包和注册记录 |
| `scripts/routing/run_interactive_model_hub.py` | 生成 Model Hub 快照、组合评测和病例 trace |
| `scripts/training/train_timm_classifier.py` | timm ImageFolder 训练器 |
| `scripts/training/run_training_job.py` | 后台训练任务执行入口 |
| `experiments/model_hub/` | Model Hub 资产、recipe、official profile 和运行目录 |
| `experiments/v0_8_6_interactive_model_hub/` | v0.8.6 受控协议配置与发布产物 |

---

## 旧版审计 Demo

早期 v0.7.x 主线关注“模型输出后的风险审计与复核优先级”，核心入口仍保留：

    streamlit run app/demo.py

旧版分类/CAM demo 入口：

    streamlit run app/demo_legacy_v0_4_2.py

对应实验结果和说明主要位于：

- `experiments/summary/v0_7_1b/`
- `experiments/summary/v0_7_2/`
- `notes/v0.7.2_metric_sensitivity_audit.md`
- `notes/v0.7.3_audit_demo_clinical_ui.md`
- `notes/v0.7.4_audit_demo_case_detail_and_checkpoint_discovery.md`

---

## 历史研究节点简表

| Version | Focus |
|---|---|
| v0.8.6 | 交互式眼科模型中转台：模型发现、任务适配、路由/专家组合、训练任务、病例回放 |
| v0.8.5c | timm adapter 激活与 forward-only 成本接入 |
| v0.8.5b | known-model inventory 与 adapter onboarding |
| v0.8.5 | 模型注册与 scout-expert 协议 |
| v0.8.4b | 青光眼 forward-only 成本闭环 |
| v0.7.4 | Audit Demo：六模型产物发现、病例详情弹窗、复核容量模拟 |
| v0.7.2 | metric-sensitivity audit：AURC / AUGRC / partial_AUGRC / Top20 稳健性 |
| v0.7.1b | 外部 DR 复核排序协议补全 |

---

## 当前边界与注意事项

- 本项目所有结果来自公共数据集或回顾性实验，不用于临床诊断、治疗建议或真实医疗决策。
- forward-only cost 仅统计模型前向计算，不包含图像解码、预处理、I/O、服务排队和真实部署开销。
- 工程 recipe 用于链路验证和统一初筛，不代表每个模型的官方最优训练协议。
- 全局候选扫描是探索性工具，不等同于最终论文冻结协议。
- 当前自动训练主要支持 `timm_imagefolder_v1`；RETFound / Green / DINOv2 等模型需要后续专用 adapter。
- 病例回放中的研究审计信息来自公开测试标签和离线评估，不应被解释为在线临床可获得信息。
