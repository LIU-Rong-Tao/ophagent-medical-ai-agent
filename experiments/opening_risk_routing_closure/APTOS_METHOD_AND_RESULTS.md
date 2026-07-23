# OphAgent 双任务离线验证与路由结果

## 研究问题与方法

本阶段回答三个问题：眼科模型资产能否按统一任务契约完成离线比较；模型输出错误是否存在可审计的互补性；固定专家预算下，冻结路由能否相对主 Scout 稳定改善任务指标。证据来自 APTOS2019 五级 DR 和青光眼三分类的既有 validation/test 概率资产，不重新训练或推理模型。

路由候选、策略和预算只在 validation 上选择；Test 仅执行预声明的 locked retrospective evaluation。置信区间使用 2,000 次病例级配对 bootstrap。当前资产缺少统一 patient_id，因此这些区间不能替代患者聚类区间。标签等级 corrected、introduced 和 net 仅为模型输出错误代理，不代表临床后果。

## 双任务证据

### APTOS

十模型单模型结果见 `aptos_ten_model_summary.csv`。Validation Macro-F1 最高为 RET-CLIP（0.736），Test 最高为 FLAIR（0.708）；Swin-Tiny 是稳定的强单模型基线。

冻结性能主方案 `FLAIR + RET-CLIP -> RETFound CFP`（20%）在 validation 的 Macro-F1 为 0.794（95% CI 0.743–0.838），QWK 为 0.940（0.919–0.957）；相对主 Scout 的配对差分别为 +0.079（+0.032–+0.126）和 +0.026（+0.001–+0.052）。Locked Test 的 Macro-F1 为 0.735（0.699–0.770），QWK 为 0.921（0.903–0.938）；配对差分别为 +0.028（-0.002–+0.057）和 +0.017（+0.001–+0.035）。性能增量在 Test 变弱，但 QWK 增量仍为正。

该主方案在 Test 没有纠正标签等级代理事件，却新增 18/1,100（1.64%，95% CI 0.91%–2.45%）。预声明的零新增方案 `RETFound CFP + RetiZero -> FLAIR` 保持 introduced=0，并净减少 18/1,100（1.64%，0.91%–2.45%），但 Macro-F1 配对差区间跨 0，且依赖 qualification_limited 的 RETFound CFP。

Validation bootstrap 在“冻结四方案 + Swin-Tiny”范围内，性能主方案以 QWK 排名第一的频率为 72.1%；零新增方案为 21.7%。这支持候选具有方向稳定性，但不表示十模型全空间选择概率。

类别层面，APTOS 性能主方案 Test 的 F1 为：0级 0.982、1级 0.689、2级 0.807、3级 0.477、4级 0.723。3级仍是主要薄弱类别。主 Scout 与 Expert 在 Test 上呈现 121 例“Scout 错、Expert 对”和 84 例“Scout 对、Expert 错”，说明互补性与专家引入新错误同时存在。

### 青光眼

清理口径固定为 validation 215 例、Test 464 例；原始 218/465 仅保留敏感性对照。单模型最强者为 RETFound-DINOv2：validation Accuracy/Macro-F1 为 0.907/0.882，Test 为 0.862/0.835。

冻结主方案 `RETFound-DINOv2 + ViT-B -> Swin-Tiny`（10%）在 validation 的 Macro-F1 为 0.906（0.858–0.946），相对主 Scout为 +0.024（-0.011–+0.062）；Test 为 0.825（0.786–0.862），配对差 -0.009（-0.035–+0.017）。冻结 single reference 在 Test 的配对差为 +0.004（-0.015–+0.024）。两者区间均跨 0，当前不能确认路由优于 Scout-only。

主方案 Test 纠正 4/464、引入 2/464，净减少率 0.43%（95% CI -0.65%–1.51%）。Validation bootstrap 在“冻结两方案 + RETFound-DINOv2”范围内，主方案以 Macro-F1 排名第一的频率为 60.9%，single reference 为 34.1%，说明小样本下候选身份仍不稳定。

类别层面，青光眼主方案 Test 的 F1 为：正常 0.898、早期 0.674、进展期 0.903；早期类别是主要薄弱项。Test 上有 22 例“Scout 错、Expert 对”和 30 例“Scout 对、Expert 错”，解释了 validation 增益未稳定迁移到 Test。

## 成本证据

APTOS 十模型已统一为 H100、FP32、forward-only 口径，排除图像读取和预处理。Batch=16 单图中位成本从 EyeCLIP CFP 的 0.397 ms 到 RetiZero 的 5.199 ms；ConvNeXt-Tiny 0.423 ms、Swin-Tiny 0.498 ms、FLAIR 0.872 ms、RET-CLIP 1.221 ms、RETFound CFP 3.421 ms。该表只支持同 H100 协议下的相对比较，不等同于端到端部署延迟。

青光眼五模型已补齐同一 H100、FP32、forward-only 口径。Batch=16 单图中位成本为：ConvNeXt-Tiny 0.444 ms、Swin-Tiny 0.498 ms、ViT-B 1.021 ms、ViT-L 3.413 ms、RETFound-DINOv2 3.630 ms；对应 batch=1 为 3.568、3.907、3.376、7.676、9.239 ms。该证据仅用于同硬件相对成本比较，不包含图像读取、预处理、数据传输和服务开销。完整证据见 `h100_cost_evidence.csv`。

## 确认性数据准备

当前 H100 可访问范围内没有满足准入条件的独立确认性队列，因此状态保持 `candidate_identified_not_admitted`，冻结协议未执行、未修改。DR 候选数据包括 [DeepDRiD](https://doi.org/10.5281/zenodo.6452623) 和 [IDRiD](https://idrid.grand-challenge.org/Data/)；青光眼三分类语义最接近 [GAMMA](https://gamma.grand-challenge.org/)。PAPILA 的可疑类、REFUGE/AIROGS 的二分类标签与当前正常/早期/进展期定义不直接一致，只能作为后续迁移或敏感性分析候选。

正式准入前必须同时确认：CFP 模态与冻结类别顺序一致；具有稳定 `case_id`、`patient_id` 和患者级去重；与当前 train/validation/test 无图像或患者重叠；文件清单与 SHA256 固定；许可、数据使用和伦理边界明确；标签来源与独立性可追溯；候选模型不存在未披露的数据污染；样本量满足预先设定的配对终点；在协议锁定前不读取结果。准入后也不得重新训练、校准、调阈值、改预算或替换候选。

## 覆盖缺口与下一批任务

统一覆盖矩阵见 `model_hub_coverage_matrix.csv`。OphBench 27 个 checkpoint 中，H100 当前有 17 个完成 runtime Smoke；其余主要是 VisionFM、VisionUnite 和不适用于分类路由的 DERETFound SD-Retina。APTOS 已有 10 个、青光眼已有 5 个标准离线任务资产，均完成任务评测和回顾性路由验证，但尚未通过独立确认性门禁。

下一批优先级固定为：第一，核验并接入 DeepDRiD 作为 DR 第二数据集；MESSIDOR-2 作为标签来源核清后的备选，OIA-DDR 先做预训练污染核查。第二，优先核验 GAMMA 作为青光眼正常/早期/进展期三分类第二数据集；G1020、REFUGE、RIM-ONE 和 AIROGS 只用于二分类迁移或稳健性分析，不与当前三分类结果直接合并。第三，使用 PALM 建立病理性近视二分类任务；该数据包含 1,200 张 CFP、患者级隔离的官方 train/validation/test，可最大程度复用当前分类、风险审计和路由流程。RFMiD 2.0 的多标签任务排在其后，因为当前 evaluator 还没有冻结的多标签路由契约。

私有数据只通过“脱敏 manifest → 标准 prediction asset → 结果表校验 → 模型输出错误风险审计 → validation 路由筛选 → 锁定评测”的既有 Model Hub 流程接入。原始图像、患者标识和绝对路径不进入 Git；缺少 `patient_id`、患者级去重、标签来源、数据使用许可或独立 split 时，只能完成数据审计，不授予任务或路由资格。

最小 capability/tool contract 与统一 run trace 定义在 `configs/protocols/model_hub_minimal_capability_contract.json`。系统控制者固定为人工，Qwen 不参与控制。现有多模态项目保持独立；只有完成独立复现、动作资格、事件容量、来源重叠和权限门禁后，才可登记为未来工具候选。

## 结论、限制与决策门

当前已形成“资产登记与任务准入 → 离线单模型评测 → 模型输出错误审计 → validation 路由筛选 → locked Test → 稳健性与成本审计”的开题实验闭环。APTOS 支持存在预算约束下的模型互补性，但性能方案伴随代理错误引入；青光眼未显示稳定优于 Scout-only。所有方案继续保持 `route_eligible=false`。

主要限制包括：回顾性数据；Test 并非全新确认性留出；缺少统一 patient_id；青光眼样本较小且划分完整性受限；RETFound CFP 为 metric replay 而非历史 exact replay；代理事件不是临床后果。H100 成本已经统一，但仍不是端到端部署延迟。

下一决策门是执行 `configs/protocols/dual_task_independent_confirmatory_protocol.json`：在未暴露、患者级独立数据上，锁定现有模型、预处理、候选、策略和预算，不再训练、校准或调阈值；使用患者聚类配对 bootstrap，并同时通过主指标、Accuracy 和代理错误 guardrail。确认性门禁通过前不授予路由资格，也不建议继续增加 APTOS 模型。

统计明细集中在 `dual_task_robustness.csv`，未保存 bootstrap 重采样明细或额外病例轨迹。
