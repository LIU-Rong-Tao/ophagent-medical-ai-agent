# APTOS 十模型离线验证与路由结果

## 方法框架

本阶段以冻结的 APTOS 五级分类 train/validation/test 清单和既有标准概率包为输入。十个模型先完成单模型离线任务评测；路由结构、策略和预算只在 validation 上筛选，再对预声明方案做 locked retrospective Test 评估。模型输出错误审计使用标签等级代理事件的 corrected、introduced 和 net，不代表临床后果或诊疗建议。

四个新增模型（EyeCLIP CFP、KeepFIT CFP、RET-CLIP、RetiZero）目前均为“离线概率资产可用、任务评测完成、可参与 validation 筛选”；不因本阶段结果授予在线推理或路由资格。

## 核心结果

单模型中，validation Macro-F1 最高为 RET-CLIP（0.736），其次为 Swin-Tiny（0.734）与 KeepFIT CFP（0.721）；Test Macro-F1 最高为 FLAIR（0.708），其次为 RETFound CFP（0.706）与 Swin-Tiny（0.704）。完整对比见 `aptos_ten_model_summary.csv`。

十模型 validation 扫描相对六模型的最佳多 scout 组合有增量：`FLAIR + RET-CLIP -> RETFound CFP` 在 20% 调用预算下 Macro-F1 为 0.794、QWK 为 0.940；相对原六模型最佳多 scout，Macro-F1 增加 0.016、QWK 增加 0.015，但仍引入 5 个标签等级代理错误。零新增代理错误的 validation 组合为 `RETFound CFP + RetiZero -> FLAIR`（20%，net +16）。

锁定 Test 显示，性能主方案 `FLAIR + RET-CLIP -> RETFound CFP` 的 Macro-F1 为 0.735、QWK 为 0.921，但 introduced 为 18；单 scout 参考 `FLAIR -> Swin-Tiny` 的 Macro-F1 为 0.738、introduced 为 24。零新增代理错误方案在 Test 保持 introduced 为 0、net +18，但成本较高，且依赖 qualification_limited 的 RETFound CFP。因此当前结果支持“存在模型互补性和预算权衡”，不支持把任一方案描述为已具备正式路由资格。

## 当前候选与限制

- 路由研究核心候选：FLAIR、RET-CLIP、Swin-Tiny。
- 条件性代理错误候选：RetiZero，仅用于离线 `RETFound CFP + RetiZero -> FLAIR` 审计比较。
- RETFound CFP：保留为 qualification_limited；官方预处理已核验，但历史重放为 metric_replay，非 exact replay。
- ConvNeXt-Tiny、KeepFIT CFP：保留为任务与成本参考；当前未进入锁定 Test 的最终路由优先组合。
- EyeCLIP CFP：当前单模型结果较弱，不进入路由 shortlist。
- 所有模型：`route_eligible=false`。原因包括回顾性评估、缺少独立确认性留出集、部分在线链路未持久化、RETFound CFP 来源限制，以及 Test 中代理错误引入的稳定性问题。

## 开题组织与下一步

开题可分为三部分：模型资产与任务准入、模型输出错误风险审计、预算约束下的离线路由研究。青光眼保持 `blocked_by_missing_source_assets`；4090 恢复后仅定向恢复原始数据、正式 split、checkpoint 和 validation 概率资产，再复用同一 Model Hub 流程。当前不建议继续新增 APTOS 模型；优先完成青光眼最小资产恢复或准备独立确认性评估。
