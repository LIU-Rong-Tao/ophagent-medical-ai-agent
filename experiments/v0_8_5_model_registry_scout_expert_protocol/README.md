# v0.8.5 Model Registry & Plug-in Scout-Expert Protocol

## 1. 版本目标

v0.8.5 的目标是建立 OphAgent 模型中转台的最小注册协议。

它不是简单把 DR 和青光眼结果拼在一起，而是验证一件更基础的事：

已有标准产物的模型，能否通过注册表被登记为初筛模型（scout）或专家模型（expert），并自动进入统一评估、路由摘要、病例审计和报告归档流程。

本版本只做注册表级接入（registry-level plug-in）/产物级接入（artifact-level plug-in）。

## 2. 三层接入边界

### registry-level / artifact-level plug-in

当前 v0.8.5 支持这一层。

如果一个模型已经有标准产物，例如：

- `prediction_csv`
- `model_baselines.csv`
- `routing_results.csv`
- `cost_summary.csv`
- `case_audit.csv`

就可以在 `model_registry.csv` 与 `route_protocols.csv` 中登记，然后进入统一汇总。

### adapter-level plug-in

这不是 v0.8.5 的范围，计划放到 v0.8.5b。

adapter-level plug-in 指的是：给定 checkpoint、config、data_root 和 class mapping，由系统自动运行模型推理，生成标准 prediction、baseline、forward-only cost 和 adapter manifest。

### training-level plug-in

这也不是 v0.8.5 的范围，计划放到 v0.8.8 或后续训练漏斗。

training-level plug-in 指的是：给定通用模型、数据集和训练 recipe，由系统自动训练或微调模型，再生成标准产物。

## 3. 当前示例任务

### glaucoma_3class

- disease_family：`glaucoma`
- dataset_id：`Glaucoma_fundus`
- risk_mode：`none`
- source：`experiments/v0_8_4b_glaucoma_forward_cost/outputs/glaucoma_convnext_retfound`

青光眼任务当前作为 generic multiclass 示例，不配置 DR 专属风险事件。

### aptos_dr_5class

- disease_family：`diabetic_retinopathy`
- dataset_id：`APTOS2019`
- risk_mode：`dr_risk_events`
- source：`experiments/v0_8_3_controlled_runner/outputs/v082c_dr_replay`

该 DR 源来自 v0.8.3 controlled replay。它包含 `model_baselines.csv`、`routing_results.csv` 和 `case_audit.csv`；DR 风险事件列保留在 `routing_results.csv` 中。该源目录没有独立 `risk_results.csv`，因此 v0.8.5 不会伪造单独风险结果表。

## 4. 如何新增一个 scout/expert 模型

当前版本只支持已有产物接入，新增流程如下：

1. 准备标准 prediction、baseline、cost 或 routing 产物。
2. 在 `task_registry.csv` 中确认任务已经登记。
3. 在 `model_registry.csv` 中登记模型：
   - `artifact_id`
   - `task_id`
   - `model_family`
   - `role_candidates`
   - `baseline_source`
   - `cost_source`
4. 在 `route_protocols.csv` 中指定 scout/expert 组合、路由策略和预算。
5. 在 `cost_registry.csv` 中登记成本来源和成本口径。
6. 运行 protocol。
7. 查看 `summary.html` 和统一输出表。

`role_candidates` 支持：

- `scout`
- `expert`
- `scout|expert`

## 5. 输出文件

运行后输出到：

`experiments/v0_8_5_model_registry_scout_expert_protocol/outputs/`

核心文件：

| 文件 | 作用 |
|---|---|
| `registered_tasks.csv` | 已启用任务注册表 |
| `registered_models.csv` | 已启用模型注册表 |
| `route_protocol_summary.csv` | 每个路由协议的最佳非 oracle 策略摘要 |
| `model_baselines_all.csv` | 多任务模型基线统一表 |
| `routing_results_all.csv` | 多任务路由结果统一表 |
| `risk_results_all.csv` | 多任务风险结果统一表；无风险事件任务可为空 |
| `case_audit_all.csv` | 多任务病例审计统一表 |
| `artifact_manifest.csv` | 输出产物清单 |
| `summary.html` | 最小 HTML 归档报告 |

## 6. 成本口径

当前只使用 forward-only cost（仅前向传播计算成本）。

它不包括：

- 图像读取；
- 图像解码；
- 图像预处理；
- CPU 到 GPU 传输；
- 模型加载；
- 后处理；
- Web 服务排队；
- 临床系统等待时间；
- 真实部署中的并发与调度开销。

因此不能把 forward-only cost 写成真实部署端到端延迟。

## 7. 运行命令

Dry-run：

    python scripts/routing/run_model_registry_scout_expert_protocol.py \
      --config experiments/v0_8_5_model_registry_scout_expert_protocol/configs/protocol.yaml \
      --output-dir experiments/v0_8_5_model_registry_scout_expert_protocol/outputs \
      --dry-run

正式运行：

    python scripts/routing/run_model_registry_scout_expert_protocol.py \
      --config experiments/v0_8_5_model_registry_scout_expert_protocol/configs/protocol.yaml \
      --output-dir experiments/v0_8_5_model_registry_scout_expert_protocol/outputs

通过 controlled runner 运行：

    python scripts/routing/run_controlled_protocol.py \
      --config experiments/v0_8_5_model_registry_scout_expert_protocol/configs/controlled_runner.yaml \
      --resume

## 8. 当前边界

- 不训练新模型；
- 不自动微调；
- 不做 checkpoint adapter；
- 不做 pipeline inference 成本；
- 不做 service end-to-end benchmark；
- 不做复杂交互 UI；
- 不做病例卡片展示；
- 不替代临床判断；
- 不把 DR-specific risk events 套到青光眼。

## 9. 后续路线

- v0.8.5b：adapter-level checkpoint onboarding；
- v0.8.6：Interactive Model Hub Demo / Agent Workflow Showcase；
- v0.8.7：routing policy optimization；
- v0.8.8：training / fine-tuning funnel；
- v0.9：LLM/Agent router + 真实模型接入 + 交互式模型中转台。
