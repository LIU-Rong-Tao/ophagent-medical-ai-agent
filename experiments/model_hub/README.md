# Model Hub 统一运行目录

该目录用于收纳模型中转台新生成的受控任务产物，不复制原始数据集或共享预训练权重。

- `registry/training_recipes/*.yaml`：按 `trainer_adapter` 注册的可复用训练模板。
- `runs/training/<task_id>/<artifact_id>/<run_id>/configs/`：`base_recipe.yaml`、`submitted_config.yaml`、`effective_config.yaml` 与 `validation_report.json`。
- `runs/training/<task_id>/<artifact_id>/<run_id>/`：checkpoint、评测指标、预测记录、成本摘要、运行 manifest 与模型注册记录。
- `runtime/training_jobs/`：后台任务状态和日志，仅服务器运行时使用，不提交 Git。
- 数据集真实路径由任务注册表的 `data_root` 字段登记。
- 共享 checkpoint 保留在服务器模型资产目录，运行记录通过 manifest 引用，不重复下载或复制。

新增数据集时，先登记任务、标签空间、类别数和受控 `data_root`，再由模型工程工作区执行预检。正式 UI 不接受未登记的自由路径。

训练器只执行已经通过严格校验的 `effective_config.yaml`。未知字段、类型错误、Adapter 不支持的配置以及跨任务继承疾病 checkpoint 会在提交前被阻断。
