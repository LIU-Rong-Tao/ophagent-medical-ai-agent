# v0.8.5b 已知模型清单盘点与适配器级接入

## 1. 版本目标

v0.8.5b 的目标是把 OphAgent 从“已有产物注册”推进到“已知模型清单盘点与适配器级接入”。

- v0.8.5：已有产物注册。也就是已经生成了 baseline、routing、risk、case audit 的模型，可以登记为 scout/expert 并统一汇总。
- v0.8.5b：已知模型 inventory + adapter onboarding。也就是先盘点项目仓库和服务器上实际存在的 checkpoint、config、class mapping、data root、legacy artifact，再判断哪些模型可以进入 adapter 推理。

本版本不训练、不微调、不做自动微调、不做 UI、不做 Agent、不做服务部署。

## 2. 为什么先做 model inventory

项目里已经有多个阶段的 DR 和青光眼产物，权重文件也可能只存在于服务器而不在 GitHub 仓库中。因此不能靠记忆硬编模型路径。

v0.8.5b 先生成 `model_inventory.csv`，把每个候选模型的实际状态写清楚：

- checkpoint 是否存在；
- config 是否存在；
- class mapping 是否存在；
- data root 是否存在；
- adapter 是否可用；
- 是否只能做 legacy replay；
- 不能 onboard 的原因是什么。

找不到文件不是失败，能把缺失原因写清楚就是本阶段的有效产出。

## 3. adapter 和 routing replay 的区别

adapter 负责 checkpoint → predictions/baseline/cost。

也就是说，adapter 的职责是加载模型、读取图像、执行推理，并生成标准产物：

- `predictions.csv`
- `model_baseline.csv`
- `forward_cost_summary.csv`
- `adapter_manifest.csv`

routing replay 负责 predictions → scout/expert 编排。

也就是说，当 scout 和 expert 的 predictions 都可用时，才复现：

- single scout routing；
- multi scout routing；
- skipped_missing_predictions；
- sanity comparison。

## 4. legacy_replay_only 是什么意思

`legacy_replay_only` 表示该模型已有历史 baseline/routing/prediction 产物，但当前缺少完整 adapter 运行条件，例如缺 checkpoint、data root、class mapping 或 loader。

这种模型可以用于历史结果对照和 registry 汇总，但不能被标记为新的 adapter 推理结果。

## 5. RETFound-DINOv2 的边界

RETFound-DINOv2 已经纳入 inventory，因为它是青光眼 expert 的重要候选。服务器上已确认存在：

`/data/LRT/RETFound/output_dir/retfound_dinov2_Glaucoma_fundus_finetune/checkpoint-best.pth`

但本版本不从零发明 RETFound-DINOv2 loader。若仓库中没有可靠加载代码，它会标记为 `needs_loader_audit`，不作为第一轮必须真实跑通的阻塞项。

## 6. 如何新增一个 adapter job

在 `configs/onboarding_jobs.csv` 中新增一行，至少提供：

- `job_id`
- `task_id`
- `artifact_id`
- `adapter_id`
- `checkpoint_path`
- `data_root` 或后续扩展的输入 manifest
- `class_to_idx_path`
- `num_classes`
- `run_adapter`

只有 `enabled=true` 且 `run_adapter=true` 的 job 才会进入 adapter onboarding。

## 7. 如何接回 v0.8.5 registry

v0.8.5b 生成的标准产物可作为 v0.8.5 registry 的输入：

- `model_baselines_from_adapters.csv`
- `forward_cost_summary_from_adapters.csv`
- `single_scout_routing_results_from_adapters.csv`
- `multi_scout_routing_results_from_adapters.csv`
- `adapter_manifest.csv`

后续可把这些路径登记回 `v0_8_5_model_registry_scout_expert_protocol/configs/*.csv`。

## 8. sanity comparison 不是 strict reproduction

`adapter_vs_legacy_baseline_check.csv` 和 `adapter_vs_legacy_routing_check.csv` 只能叫 sanity comparison / sanity check。

它们不是 strict reproduction。因为 checkpoint、预处理、class mapping、split 或 loader 有任何差异，结果都可能不同。本项目不会为了贴近历史结果而修改数据或后处理。

## 9. 成本口径

所有成本都只能叫 forward-only cost。

forward-only cost 只表示模型前向传播计算成本，不是真实部署端到端延迟。它不包含：

- 图像读取；
- 图像解码；
- 图像预处理；
- CPU-GPU 传输；
- 服务排队；
- 模型加载；
- 后处理；
- 临床系统等待时间。

## 10. 后续路线

- RETFound-DINOv2 loader audit；
- DR task-agnostic 标准重跑；
- v0.8.6 Interactive Model Hub Demo；
- v0.8.8 training / fine-tuning funnel；
- v0.9 LLM/Agent router。
