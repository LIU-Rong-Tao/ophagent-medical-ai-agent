# v0.8.5c：真实 timm 分类模型 adapter 启用

## 目标

v0.8.5b 负责盘点服务器已有模型，并说明缺少哪些运行条件；v0.8.5c 首次真实启用 `timm_classifier_v1`，严格加载已有 checkpoint，在冻结测试集上重新推理并生成标准化 prediction、baseline、forward-only cost 和 adapter manifest。

本版本支持：

- APTOS2019：ConvNeXt-Tiny、Swin-Tiny、ViT-B；
- Glaucoma_fundus：ConvNeXt-Tiny。

RETFound、RETFound-Green 和 RETFound-DINOv2 **本版本不启用**，继续保持 `needs_loader_audit`，不临时拼装未经审计的 loader。

## 输入与 test split

每个 job 使用历史 `test_predictions.csv` 中冻结的 `image_path`、`image_key` 和真实标签生成：

```text
outputs/input_manifests/<job_id>_input_manifest.csv
```

服务器审计已确认 APTOS2019 的 1100 张测试图以及 Glaucoma_fundus 的 465 张测试图均存在。脚本不会用目录遍历重新划分 test split，也不会伪造缺失路径。

## 输出

每个成功 job 生成：

- `predictions.csv`：真实 adapter 推理概率；
- `model_baseline.csv`：Accuracy、Macro-F1、AUROC/AUPR 等通用指标；
- `forward_cost_runs.csv` 与 `forward_cost_summary.csv`：多次前向测量；
- `adapter_manifest.csv`：checkpoint、manifest、结果路径与 SHA256。

DR 的标签结构显式设为 `ordinal`，因此计算 QWK。当前青光眼任务按通用三分类处理，QWK 字段保留但标记为不适用，不把 CFP 三分类包装成完整临床分期。

## 成本口径

`forward-only cost` 只包含模型前向传播计算成本；不包含图像读取、解码、预处理、CPU-GPU 传输、模型加载、后处理、服务排队，**不是真实部署端到端延迟**。

每个模型使用 3 次 warmup、5 次完整测试集重复测量，报告中位数、均值、标准差和变异系数。

## 一致性检查与 replay

新 prediction 与历史 prediction 的比较属于 `sanity check`，用于发现 checkpoint、transform、类别顺序或 split 不一致；它不是 `strict reproduction` 声明，也不会为了贴近历史结果而修改新输出。

当前 replay 使用：

```text
adapter-generated timm scout + legacy-standard RETFound expert
```

因此必须标记为 `mixed_adapter_legacy`。该结果只验证模型中转链路可以运转，不作为新的纯 adapter 科研实验结论。

## 运行

```bash
/data/conda_envs/ophagent/bin/python scripts/routing/run_timm_adapter_activation.py \
  --config experiments/v0_8_5c_timm_adapter_activation/configs/protocol.yaml \
  --output-dir experiments/v0_8_5c_timm_adapter_activation/outputs \
  --dry-run

/data/conda_envs/ophagent/bin/python scripts/routing/run_timm_adapter_activation.py \
  --config experiments/v0_8_5c_timm_adapter_activation/configs/protocol.yaml \
  --output-dir experiments/v0_8_5c_timm_adapter_activation/outputs \
  --stage all
```

## 边界

本版本不训练、不微调、不做 UI、不做 Agent、不启用 RETFound loader，不伪造 prediction 或成本结果。
