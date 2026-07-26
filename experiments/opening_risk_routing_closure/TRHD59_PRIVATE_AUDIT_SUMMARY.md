# TRHD59 私有观测标签审计

- 定位：统一冻结编码器的探索性基线；线性探针排名不代表最终模型能力，不继续优化 probe-level 路由。
- 数据：10,049 张 canonical 图像；Development/Test 为 8,542/1,507；患者级隔离未验证。
- 模型：27 个 checkpoint 经 CFP 门禁后，9 个完成正式 v2 任务适配；首次 `protocol_invalid` 批次未参与任何结论。
- 选择：只使用 Development 内部 validation；Test 在协议提交后一次性解锁。
- 单模型：RetClip 最强，Test Macro-F1 0.5821。
- 路由：10% multi 的 Test Macro-F1 0.5910，observed Hit@1 0.7113，观测一致性净纠正 +18。
- 权衡：性能方案增加高置信观测标签不一致；5% 风险约束方案保持该事件数不增加，但 Test Macro-F1 降至 0.5782。
- 边界：未观测类别不是阴性；结果不是临床风险结论；`route_eligible=false`。
- 人工复核候选：`outputs/trhd59_observed_label_v2/model_hub_locked_test/manual_review_candidates.csv`。
- 微调门禁：106 例盲化复核完成并确认标签口径前，不启动任何微调。
