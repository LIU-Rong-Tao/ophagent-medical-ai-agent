# OphAgent 四数据集 Model Hub 结果总览

> 本文只汇总既有冻结结果，不重新推理、训练或选择路由。所有“危险错误”均为标签等级代理事件，不代表临床真实伤害。

## 1. 结果口径

| 数据集 / 任务 | 模型池 | 选择集 | 冻结结果集 | 当前资格 |
|---|---:|---|---|---|
| APTOS2019 / DR 五级 | 10 | validation 514 例 | retrospective Test 1,100 例 | 离线路由评测完成；`route_eligible=false` |
| DeepDRiD / 冻结迁移 | 6 | 不在 DeepDRiD 选模 | 官方 validation 400 图、100 患者 | 跨数据集冻结迁移；`route_eligible=false` |
| DeepDRiD / 原生适配 | 10 | 官方 train 内部患者级 validation 235 图 | 官方 validation 400 图、100 患者 | 原生任务离线适配完成；`route_eligible=false` |
| Glaucoma_fundus / 青光眼三级 | 5 | 清理后 validation 215 例 | 清理后 Test 464 例 | `split_integrity_limited`；`route_eligible=false` |
| RIM-ONE DL / 青光眼二分类 | 9（5 个冻结迁移 + 4 个原生适配） | 镜像 train 内部 validation 62 例 | 镜像所附 by-hospital Test 174 例 | `third_party_mirror_provenance_limited`；`route_eligible=false` |

指标语义统一如下：

- DR 五级与青光眼三级：**QWK 为任务主指标，Macro-F1 为关键次指标**。
- 青光眼二分类：**Macro-F1 为任务主指标，Balanced Accuracy 为关键次指标**。
- Accuracy 仅作辅助描述，不参与“最佳”判定。
- 不构造加权综合分数。主指标与次指标对应不同模型时分别列出。
- 路由分别报告相对 Scout、相对主指标最佳单模型、标签等级代理事件变化、预算与成本。

代理错误按任务语义分别定义：

- **序数任务（DR 五级、青光眼三级）**：标签等级低估代理事件。DR 为 `y_true >= 3 且 y_pred < 3`；青光眼三级为 `y_true >= 2 且 y_pred < 2`。它们只表示跨越预设等级边界的模型低估，不代表临床后果。
- **二分类任务（青光眼二分类）**：阳性类假阴性代理事件，即 `y_true = 1 且 y_pred = 0`。它只表示模型漏判阳性标签，不得写成危险漏诊。
- `corrected` 表示 Scout 的代理事件被最终输出消除，`introduced` 表示最终输出新增代理事件，`net = corrected - introduced`。三者不能跨任务直接比较严重程度。

DeepDRiD 的两组实验回答不同问题：

- **冻结迁移**：APTOS 模型和路由不在 DeepDRiD 上重训、调参或重校准，检验直接跨数据集迁移。
- **原生适配**：只用 DeepDRiD 官方 train 训练探针并在其内部 validation 选参，官方 validation 只冻结评估一次。

REFUGE 当前仅完成数据准入，不属于上述四数据集既有结果：

- 状态：`admitted_for_labeled_train_validation_only`。
- 来源限制：`third_party_mirror=true`，`official_leaderboard_reproduction=false`。
- 可用范围：train 用于训练及内部选参；带诊断标签的 validation 作为唯一冻结结果集。
- 永久排除：Kaggle test 对应原挑战 onsite Test，诊断标签未公开；Masks 和 gts 是分割标注，不能替代青光眼诊断标签。
- 该准入状态不自动推出 `task_evaluation_status`、`routing_validation_status` 或 `route_eligible`。

## 2. APTOS2019

模型池：ConvNeXt-Tiny、Swin-Tiny、RETFound CFP、RETFound-Green、PRETI、FLAIR、EyeCLIP CFP、KeepFIT CFP、RET-CLIP、RetiZero。

单模型指标最优项：

| 口径 | 指标角色 | 模型 | QWK | Macro-F1 | Accuracy | H100 batch16 成本 |
|---|---|---|---:|---:|---:|---:|
| Validation | 主指标 QWK 最优 | RETFound CFP | 0.9178 | 0.7138 | 0.8541 | 3.421 ms/图 |
| Validation | 次指标 Macro-F1 最优 | RET-CLIP | 0.9166 | 0.7357 | 0.8405 | 1.221 ms/图 |
| 冻结 Test | 主指标 QWK 最优 | RETFound CFP | 0.9131 | 0.7060 | 0.8491 | 3.421 ms/图 |
| 冻结 Test | 次指标 Macro-F1 最优 | FLAIR | 0.9040 | 0.7079 | 0.8155 | 0.872 ms/图 |

冻结 Test 路由：

| 方案 | 预算 | QWK / Macro-F1 | 相对 Scout-only | 相对 QWK 最佳单模型 | corrected / introduced / net | 估算成本 |
|---|---:|---:|---|---|---:|---:|
| FLAIR + RET-CLIP → RETFound CFP | 20% | 0.9211 / 0.7355 | Macro-F1 +2.76 pp；Scout QWK 未单独保存 | QWK +0.80 pp | 0 / 18 / -18 | 2.777 ms/图 |
| FLAIR → Swin-Tiny | 30% | 0.9160 / 0.7375 | QWK +1.20 pp | QWK +0.28 pp | 0 / 24 / -24 | 1.021 ms/图 |
| RETFound CFP + RetiZero → FLAIR | 20% | 0.9196 / 0.7322 | Macro-F1 +2.63 pp；Scout QWK 未单独保存 | QWK +0.64 pp | 18 / 0 / +18 | 8.795 ms/图 |
| RET-CLIP → RETFound CFP | 5% | 0.8993 / 0.6854 | QWK +0.69 pp | QWK -1.39 pp | 1 / 8 / -7 | 1.392 ms/图 |

结论：APTOS 上部分路由提高 QWK，但“主指标提高”和“代理事件净减少”不是同一个目标。性能方案引入了较多代理事件；零新增代理事件方案成本高，必须作为独立研究审计代理方案报告。

## 3. DeepDRiD

### 3.1 APTOS 模型冻结迁移

模型池：ConvNeXt-Tiny、Swin-Tiny、RETFound CFP、RETFound-Green、PRETI、FLAIR。

QWK 与 Macro-F1 的最佳单模型均为 FLAIR：QWK 0.8039、Macro-F1 0.4464（Accuracy 0.5575）。

| 冻结路由 | 预算 | QWK / Macro-F1 | 相对 Scout-only | 相对 QWK 最佳单模型 | corrected / introduced / net | 估算成本 |
|---|---:|---:|---|---|---:|---:|
| RETFound CFP → FLAIR | 10% | 0.7392 / 0.3716 | QWK +2.96 pp | QWK -6.47 pp | 3 / 0 / +3 | 3.509 ms/图 |
| ConvNeXt-Tiny + PRETI → FLAIR | 10% | 0.6470 / 0.3276 | Macro-F1 +2.13 pp；Scout QWK 未单独保存 | QWK -15.69 pp | 5 / 0 / +5 | 1.540 ms/图 |
| ConvNeXt-Tiny → FLAIR | 5% | 0.6572 / 0.3063 | QWK +2.30 pp | QWK -14.67 pp | 5 / 0 / +5 | 0.467 ms/图 |

结论：APTOS 冻结模型存在明显域偏移。路由能捕获部分标签等级代理事件，但不能弥补单模型总体性能下降；冻结迁移适合作为外部稳健性基线，不应替代原生适配。

### 3.2 DeepDRiD 原生适配

模型池为同一 10 个 CFP 模型的冻结编码器探针。

| 结果集 | 指标角色 | 模型 | QWK | Macro-F1 | Accuracy |
|---|---|---|---:|---:|---:|
| 内部 validation | 主指标 QWK 最优 | FLAIR | 0.8243 | 0.6552 | 0.6936 |
| 内部 validation | 次指标 Macro-F1 最优 | FLAIR | 0.8243 | 0.6552 | 0.6936 |
| 冻结官方 validation | 主指标 QWK 最优 | FLAIR | 0.8341 | 0.5682 | 0.6525 |
| 冻结官方 validation | 次指标 Macro-F1 最优 | RET-CLIP | 0.8289 | 0.6338 | 0.7025 |

| 原生路由 | 结果集 | 预算 | QWK / Macro-F1 | 相对 Scout-only | 相对该结果集 QWK 最佳单模型 | corrected / introduced / net | 估算成本 |
|---|---|---:|---:|---|---|---:|---:|
| KeepFIT → FLAIR | 内部 validation | 30% | 0.8522 / 0.6835 | QWK +7.53 pp | QWK +2.78 pp | 6 / 0 / +6 | 1.125 ms/图 |
| KeepFIT → FLAIR | 冻结官方 validation | 30% | 0.8465 / 0.6346 | QWK +1.88 pp | QWK +1.25 pp | 6 / 1 / +5 | 1.127 ms/图 |
| FLAIR + RETFound-Green → Swin-Tiny | 内部 validation | 5% | 0.8293 / 0.6838 | Macro-F1 +2.85 pp；Scout QWK 未单独保存 | QWK +0.49 pp | 2 / 0 / +2 | 2.301 ms/图 |
| FLAIR + RETFound-Green → Swin-Tiny | 冻结官方 validation | 5% | 0.8370 / 0.5702 | Macro-F1 +0.19 pp；Scout QWK 未单独保存 | QWK +0.29 pp | 1 / 0 / +1 | 2.301 ms/图 |

结论：原生适配相对冻结迁移提升明显。单 Scout 路由在冻结结果集仍保留小幅且方向一致的增益；multi-scout 的 validation 增益没有稳定迁移。

## 4. Glaucoma_fundus 三级分类

模型池：ConvNeXt-Tiny、Swin-Tiny、ViT-B、ViT-L、RETFound-DINOv2。QWK 与 Macro-F1 的最佳单模型在 validation 和 Test 均为 RETFound-DINOv2：

| 结果集 | QWK | Macro-F1 | Accuracy | H100 batch16 成本 |
|---|---:|---:|---:|---:|
| Validation | 0.9086 | 0.8819 | 0.9070 | 3.630 ms/图 |
| 冻结 Test | 0.8748 | 0.8346 | 0.8621 | 3.630 ms/图 |

| 冻结路由 | 结果集 | 预算 | QWK / Macro-F1 | 相对 Scout-only | 相对 QWK 最佳单模型 | corrected / introduced / net | 估算成本 |
|---|---|---:|---:|---|---|---:|---:|
| RETFound-DINOv2 → Swin-Tiny | Validation | 5% | 未保存 / 0.9000 | Macro-F1 +1.80 pp | QWK 不可由现有选定预算产物恢复 | 2 / 0 / +2 | 约 3.655 ms/图 |
| RETFound-DINOv2 → Swin-Tiny | Test | 5% | 0.8759 / 0.8381 | QWK +0.11 pp | QWK +0.11 pp | 4 / 2 / +2 | 约 3.655 ms/图 |
| RETFound-DINOv2 + ViT-B → Swin-Tiny | Validation | 10% | 未保存 / 0.9056 | Macro-F1 +2.37 pp | QWK 不可由现有选定预算产物恢复 | 3 / 0 / +3 | 约 4.701 ms/图 |
| RETFound-DINOv2 + ViT-B → Swin-Tiny | Test | 10% | 0.8710 / 0.8254 | Macro-F1 -0.92 pp；Scout QWK 未单独保存 | QWK -0.38 pp | 4 / 2 / +2 | 约 4.701 ms/图 |

结论：低预算 single-scout 在 Test 上保留微弱 QWK 与 Macro-F1 正增益；multi-scout 的 QWK 与 Macro-F1 均低于主指标最佳单模型，不能视为稳定有效。该任务还受历史 split 完整性限制。

## 5. RIM-ONE DL 二分类

模型池包含 5 个三级模型冻结迁移资产（ConvNeXt-Tiny、Swin-Tiny、ViT-B、ViT-L、RETFound-DINOv2）和 4 个原生 head-only 适配资产（ConvNeXt-Tiny、Swin-Tiny、ViT-B、ViT-L）。

| 口径 | 指标角色 | 模型 | Macro-F1 | Balanced Accuracy | Accuracy | 成本口径 |
|---|---|---|---:|---:|---:|---|
| Validation | 主指标 Macro-F1 最优 | 原生 Swin-Tiny | 0.8754 | 0.8657 | 0.8871 | H100 batch32：1.015 ms/图 |
| Validation | 次指标 Balanced Accuracy 最优 | 原生 Swin-Tiny | 0.8754 | 0.8657 | 0.8871 | H100 batch32：1.015 ms/图 |
| 冻结 Test | 主指标 Macro-F1 最优 | 原生 ViT-L | 0.8205 | 0.8298 | 0.8391 | H100 batch32：22.150 ms/图 |
| 冻结 Test | 次指标 Balanced Accuracy 最优 | 原生 ViT-L | 0.8205 | 0.8298 | 0.8391 | H100 batch32：22.150 ms/图 |

Validation 最优模型未在 Test 保持领先，说明 62 例 validation 的选择方差较大。

| 冻结路由 | 结果集 | 预算 | Macro-F1 / Balanced Accuracy | 相对 Scout-only | 相对 Macro-F1 最佳单模型 | corrected / introduced / net | 估算成本 |
|---|---|---:|---:|---|---|---:|---:|
| 原生 Swin-Tiny → 原生 ConvNeXt-Tiny | Validation | 5% | 0.8943 / 未保存 | Macro-F1 +1.89 pp | Macro-F1 +1.89 pp | 1 / 0 / +1 | 1.057 ms/图 |
| 同上 | Test | 5% | 0.7638 / 未保存 | Macro-F1 +0.90 pp | Macro-F1 -5.67 pp | 0 / 1 / -1 | 1.060 ms/图 |
| 原生 ConvNeXt-Tiny + 冻结 RETFound-DINOv2 → 原生 Swin-Tiny | Validation | 20% | 0.8943 / 未保存 | Macro-F1 +6.26 pp | Macro-F1 +1.89 pp | 4 / 0 / +4 | 4.688 ms/图 |
| 同上 | Test | 20% | 0.7669 / 未保存 | Macro-F1 +1.14 pp | Macro-F1 -5.36 pp | 5 / 1 / +4 | 4.696 ms/图 |

单模型 Balanced Accuracy 由既有冻结预测逐例只读计算；既有冻结路由产物未保存对应预算下的 Balanced Accuracy，且本轮禁止重跑，因此路由的该次指标不作推断。结论：路由在 Test 上虽相对各自 Scout 保留小幅 Macro-F1 增益，但均低于 Test 的 Macro-F1 最佳单模型。single-scout 的假阴性代理事件净变化转负；multi-scout 保留假阴性代理事件净减少。

RIM-ONE 当前使用第三方镜像。镜像目录和文件命名可对应公开的 by-hospital 划分说明，但缺少可核验的官方压缩包 SHA256，也缺少显式患者 ID，不能证明与官方发布逐字节一致。因此本文只报告“镜像所附 by-hospital 划分上的来源受限回顾性结果”，不称为官方 Test、官方 benchmark 或官方榜单复现。

## 6. Model Hub 已证明什么

1. 同一套 registry、prediction asset、任务评测和路由协议可以覆盖 DR 五级、青光眼三级及青光眼二分类，无需为每个数据集另建 runner。
2. 模型资产可用、任务适配完成、离线评测完成和可进入路由池是四个不同状态；当前没有任何方案因此自动获得 `route_eligible`。
3. 跨数据集直接迁移会显著掉点；使用目标数据集 train 完成原生适配后，单模型和部分路由均能恢复。
4. 路由是否有效取决于模型错误互补性、预算和预先定义的任务主指标，不是“调用专家必然更好”。
5. corrected、introduced 和 net 必须与任务主指标、关键次指标、预算和成本同时报告；只看纠正数会掩盖专家引入的新错误。

## 7. 路由有效与失效场景

**相对有效**

- DeepDRiD 原生 KeepFIT → FLAIR：冻结结果集仍提高 QWK 与 Macro-F1，并保持代理事件净减少。
- 青光眼三级低预算 single-scout：Validation 与 Test 方向一致，但 Test 增益较小。
- APTOS 的零新增代理事件方案：代理事件净减少稳定，但成本高且 Accuracy 不增。

**不稳定或失效**

- DeepDRiD 冻结迁移：域偏移下路由无法替代任务适配。
- APTOS 性能优先方案：QWK 与 Macro-F1 提高，但新增代理事件明显。
- 青光眼三级 multi-scout：Validation 增益在 Test 反转。
- RIM-ONE single-scout：Test 出现代理事件净恶化。
- RIM-ONE 模型排序：Validation 与 Test 最佳模型不一致，当前 validation 容量不足以支持强结论。

## 8. 私有模型错误风险审计的冻结基线

私有数据先只运行冻结推理和错误风险审计，不在私有结果上重选模型、阈值、预算或路由。

| 私有任务语义 | 冻结单模型基线 | 冻结路由基线 | 原图推理可执行状态 | 用途 |
|---|---|---|---|---|
| DR 五级、接近 APTOS | QWK：RETFound CFP；Macro-F1：FLAIR | RETFound CFP + RetiZero → FLAIR，20% | 组成模型均完成 H100 原图批量重放；可执行受控离线推理；正式 `task_inference_ready=false` | 同时观察 QWK、Macro-F1 与零新增标签等级低估代理事件方案 |
| DR 五级、接近 DeepDRiD | QWK：FLAIR；Macro-F1：RET-CLIP | KeepFIT → FLAIR，30% | 既有原图任务适配运行已完成，但可复用私有推理链尚未登记；当前不列入首轮私有执行集 | 保留为后续原生适配对照，不替代 APTOS 冻结基线 |
| 青光眼三级 | RETFound-DINOv2 | RETFound-DINOv2 → Swin-Tiny，5% | 组成模型均有 H100 原图任务运行证据；可执行受控离线推理；正式 `task_inference_ready=false` | 使用跨 Validation/Test 方向一致的低预算方案 |
| 青光眼二分类 | Validation 预选的原生 Swin-Tiny；ViT-L 仅作事后强参考 | 原生 ConvNeXt-Tiny + RETFound-DINOv2 → 原生 Swin-Tiny，20% | 原生 checkpoint 曾从原图生成冻结概率，但受 RIM-ONE 镜像来源限制；正式 `task_inference_ready=false` | 仅在私有任务明确为二分类且通过准入后作为来源受限敏感性基线 |

准入前必须固定：任务标签语义、类别顺序、患者级划分、重复检查、主指标、代理事件定义和可接受预算。首轮只允许使用“受控离线原图推理已验证”的冻结模型及预声明路由；不得根据私有结果重选模型、阈值或预算。私有审计结果只证明模型输出错误风险，不提供诊疗建议，也不能直接授予在线路由资格。

## 9. 成本与证据限制

- APTOS、DeepDRiD 和青光眼三级主成本为 H100 FP32 forward-only batch16，不含解码、预处理、I/O、CPU-GPU 传输和服务开销。
- RIM-ONE 原生模型成本由训练入口以 batch32 记录，只适合该任务内部相对比较；不可与 batch16 数值直接排名。
- 路由成本为已测单模型成本的场景估算，不是端到端在线延迟。
- 所有 Test 均为锁定后的回顾性结果，不是全新、盲法或 pristine confirmatory holdout。
