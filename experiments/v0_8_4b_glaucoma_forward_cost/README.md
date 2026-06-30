# v0.8.4b Glaucoma Forward Cost Closure

## 1. 版本目标

本版本用于建立青光眼任务下的 scout-expert routing 最小性能-成本闭环。

核心问题：

- 轻量 scout 模型是否可以作为低成本初筛模型；
- 更强 expert 模型是否能在疑难样本上提升性能；
- 在不同 expert 调用预算下，系统能否同时输出性能指标与 forward-only 计算成本；
- 是否可以将模型基线、路由结果、成本 profile 和病例审计统一到同一套 controlled protocol 输出中。

本版本不是临床部署系统，也不直接定义临床分流规则。

## 2. 方向背景

v0.7.x 主要围绕 DR 风险复核排序展开，包括 dangerous undergrading、vision-threatening DR miss、residual risk、uncertainty ranking 和 learned deferral score。

该方向暴露出的关键问题是：风险事件和阈值依赖公开数据集标签与人为规则。公开数据集标签可能存在噪声，算法阈值也不等同于临床金标准。因此，如果直接将这些阈值包装为临床分流依据，会存在临床意义不稳和责任边界不清的问题。

因此，当前方向从“风险阈值分流”调整为“多眼病模型中转台 / scout-expert 路由”：

- 不直接用非金标准阈值替代临床判断；
- 先评估不同模型在不同眼病任务上的能力边界；
- 将轻量模型作为 scout；
- 将性能更强但成本更高的模型作为 expert；
- 根据不确定性、置信度或后续学习到的路由策略，决定是否调用专家模型。

## 3. 本版本配置

分支：v0.8.4b-glaucoma-forward-cost

提交：78de6fa v0.8.4b add glaucoma forward cost closure

任务：glaucoma_3class

数据：/data/LRT/RETFound/Data_split/Glaucoma_fundus/test

模型角色：

| role | artifact_id | model_family |
|---|---|---|
| scout | convnext_tiny_glaucoma_scout | convnext |
| expert | retfound_dinov2_glaucoma_expert | retfound_dinov2 |

## 4. 关键结果

### 4.1 模型基线

| model | role | Accuracy | Macro-F1 | estimated forward-only cost | throughput |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | scout | 0.8086 | 0.7622 | 0.494492 ms/image | ~2022 images/s |
| RETFound-DINOv2 | expert | 0.8602 | 0.8327 | 4.798021 ms/image | ~208 images/s |

### 4.2 路由结果

路由成本公式：

routing_cost = scout_cost + expert_call_rate × expert_cost

| expert budget | estimated forward-only cost | reduction vs expert-only |
|---:|---:|---:|
| 20% | 1.454096 ms/image | 69.69% |
| 30% | 1.939057 ms/image | 59.59% |
| 50% | 2.888343 ms/image | 39.80% |

以上结果说明，在 forward-only 计算成本口径下，scout-expert routing 可以形成性能与计算成本之间的可解释折中。

## 5. 成本口径边界

本版本的成本是 forward-only cost，仅统计模型在 GPU 上的前向传播计算时间。

不包括：

- 图像读取；
- 图像解码；
- 图像预处理；
- CPU 到 GPU 传输；
- 模型加载；
- 后处理；
- Web 服务排队；
- 临床系统等待时间；
- 真实部署中的并发与调度开销。

因此，本版本不能表述为：

“真实部署端到端延迟降低 69.69%。”

更严谨的表述是：

“在 forward-only 计算成本口径下，20% expert budget 可将平均模型前向计算成本相对 expert-only 降低约 69.69%。”

## 6. 当前输出文件

主要输出目录：

experiments/v0_8_4b_glaucoma_forward_cost/outputs/

关键文件：

| file | meaning |
|---|---|
| glaucoma_model_forward_cost_summary.csv | forward-only cost profile |
| glaucoma_convnext_retfound/model_baselines.csv | scout/expert 模型基线 |
| glaucoma_convnext_retfound/routing_results.csv | 路由性能与成本结果 |
| glaucoma_convnext_retfound/risk_results.csv | 风险事件结果，本任务为空 schema |
| glaucoma_convnext_retfound/case_audit.csv | 病例级路由审计 |
| glaucoma_convnext_retfound/artifact_manifest.csv | 产物清单 |
| glaucoma_convnext_retfound/report.html | HTML 报告 |

## 7. 已完成检查

- benchmark dry-run 通过；
- controlled runner resume 通过；
- 第二次 resume fingerprint unchanged；
- routing_results 已使用新成本 profile；
- risk_results 保持空 schema；
- case_audit 共 1860 行；
- DR-specific leakage columns 为空；
- pytest 通过：41 passed。

## 8. 当前结论

v0.8.4b 完成了青光眼任务下 scout-expert routing 的最小性能-成本闭环。

本版本的意义不是证明系统已经可以临床部署，而是证明：

1. 系统可以从 DR 风险审计扩展到青光眼 generic multiclass 任务；
2. evaluator 已经具备 task-agnostic 评估能力；
3. scout 与 expert 的性能和计算成本可以被放到同一张表中比较；
4. controlled protocol 可以复现成本测量、路由结果和 HTML 报告；
5. 项目方向已经从单一阈值风险分流，推进到多眼病模型中转台 / 专家路由框架雏形。

## 9. 下一步计划

### v0.8.5

统一 DR + 青光眼 task-agnostic evaluator：

- 建立统一 task registry；
- 建立统一 artifact registry；
- 同一套 report 展示 DR 与青光眼结果；
- DR 任务保留 dangerous undergrading / VTDR miss 等风险事件；
- 青光眼任务保持 generic multiclass 评估，不强行套用 DR 风险事件。

### v0.8.5b / v0.8.6

增加 pipeline_inference 成本：

- 图像读取；
- 图像解码；
- 预处理；
- CPU-GPU 传输；
- forward；
- 后处理。

### v0.9

进入 service_e2e 和 UI/Agent：

- HTTP 请求；
- 文件上传；
- 服务排队；
- 模型推理；
- 结果返回；
- 并发测试；
- 可展示的模型中转台原型。
