# OphAgent APTOS 高能力路线简单门控复核 v0.1

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Verification Status: UNVERIFIED（需确定性复跑核验）
- Version Label: aptos_high_capability_simple_gate_v0_1

## 结论

**NO_IMPROVEMENT**

仅用 APTOS development 从 90 条路线中筛出 3 条高能力同任务路线；冻结回顾性评价中满足预声明路线成功规则 0 条。

本实验把问题限定为：Scout 与 Expert 已有较高同任务能力和开发集互补性时，简单预咨询门控能否锦上添花。它不是 APTOS→DeepDRiD 跨数据集迁移实验。

## 开发集筛选

| Scout | Expert | Scout acc | Expert acc | 增量 | corrected/introduced | 比值 | 正增益折 | 非负 net 折 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flair | swin_tiny | 0.821 | 0.872 | 0.052 | 59/34 | 1.735 | 5/5 | 5/5 |
| retfound_green | retfound_cfp | 0.806 | 0.862 | 0.056 | 55/28 | 1.964 | 5/5 | 5/5 |
| retfound_green | swin_tiny | 0.806 | 0.872 | 0.066 | 57/25 | 2.280 | 5/5 | 5/5 |

固定阈值：Scout accuracy ≥0.80、Expert accuracy ≥0.85、Expert 增量 ≥0.05、corrected/introduced ≥1.7、corrected ≥30、introduced ≥20、net ≥20；固定图像组折中至少 4/5 为正增益且 4/5 为非负 net，折间增量标准差 ≤0.12。冻结评估没有参与路线筛选。

## 冻结回顾性相同预算比较

| 路线 | 预算 | 开发锁定基线 | corrected 方法/基线 | introduced 方法/基线 | net 方法/基线 | Δnet | 图像组配对 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flair→swin_tiny | 0.100 | margin | 1/39 | 2/15 | -1/24 | -25.000 | [-39.000, -11.000] |
| flair→swin_tiny | 0.200 | margin | 3/68 | 6/32 | -3/36 | -39.000 | [-59.000, -20.000] |
| flair→swin_tiny | 0.300 | entropy | 10/80 | 19/43 | -9/37 | -46.000 | [-67.025, -23.000] |
| retfound_green→retfound_cfp | 0.100 | entropy | 1/31 | 1/5 | 0/26 | -26.000 | [-37.000, -14.000] |
| retfound_green→retfound_cfp | 0.200 | margin | 1/65 | 1/11 | 0/54 | -54.000 | [-71.025, -36.000] |
| retfound_green→retfound_cfp | 0.300 | entropy | 12/82 | 5/22 | 7/60 | -53.000 | [-73.000, -33.000] |
| retfound_green→swin_tiny | 0.100 | entropy | 0/34 | 0/8 | 0/26 | -26.000 | [-39.000, -13.000] |
| retfound_green→swin_tiny | 0.200 | margin | 0/66 | 0/19 | 0/47 | -47.000 | [-64.000, -30.000] |
| retfound_green→swin_tiny | 0.300 | margin | 25/89 | 12/28 | 13/61 | -48.000 | [-69.000, -28.000] |

每个预算的 entropy/margin 强基线只由 development OOF 锁定。Scout-only 与 Always-Expert 保留为端点参考。若合格路线没有唯一冻结 v1.1 身份，v1.1 只记录为 not applicable，不移植其他路线策略。

## 证据边界

- development 每路线 485 个去重后图像分析单位；冻结回顾性评价每路线 1036 个；
- APTOS 缺少患者、眼别和检查标识，不能把图像组 bootstrap 解释为患者级泛化；
- APTOS Test 指标在严格冻结协议建立前已存在，本轮只作回顾性评价，仍需独立未暴露患者级确认；
- corrected/introduced 是标签定义的错误代理，不是临床获益或伤害；`SafetyEligibilityGate` 仍不可绕过。

## 决策与下一步

单条路线必须在至少两个预算（含 30%）同时做到 corrected 不低、introduced 不高且 net 严格更高，并且 30% 的配对 Δnet 95% CI 下界非负；至少两条路线满足才授予 `ROUTE_SPECIFIC_GO`。

若仍为负，则停止调整简单预咨询门控；下一步转向第二意见到达后的 KEEP_SCOUT / ADOPT_SECOND_OPINION / HUMAN_REVIEW 采纳机制。

## 追溯

- 实现提交：`509392828975a69929aa3821c37aff952800accc`
- 协议 SHA256：`57ba5c6490bc6f44b2cc78b43b8e0e1e7a3513eb63402874eca5d06071aae8a2`
- Benchmark manifest SHA256：`178dc7979363fa54418e85c90c3f1be0a8dd4db5fef73bd11e77862569209a35`
- 病例表 SHA256：`2a29402027557c7638ac34888fb8e4c2cbbaf8bee09d043cec5ab4e3e42be60f`
- qualified_routes.csv SHA256：`c27b97d4a7728b175cdf820611168095019472b1404ac42c55c7292ba3b8bc0c`
- core_comparison_results.csv SHA256：`a4fc7dad020b28b044565781fc8f5c119b90e0e510fff27b788bb082d7d5a134`
- 眼底模型训练/推理：未执行；冻结 Benchmark、Test、预测资产及既有 DeepDRiD 结论：未修改。
