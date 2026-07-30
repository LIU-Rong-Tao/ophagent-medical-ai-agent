# OphAgent 高质量路线简单门控复核 v0.1

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Verification Status: UNVERIFIED（需确定性复跑核验）
- Version Label: route_specific_simple_gate_v0_1

## 结论

**NO_IMPROVEMENT**

开发集从 90 条 DeepDRiD 原生路线中筛出 2 条；冻结回顾性评价中满足预声明路线成功规则 0 条。

路线筛选没有读取冻结评估结果。该结论仍是回顾性错误代理研究，不能授予路线资格或替代 `SafetyEligibilityGate`。

## 开发集路线筛选

| Scout | Expert | Scout acc | Expert acc | 增量 | corrected/introduced | 比值 | 正增益折 | 非负 net 折 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retfound_green | convnext_tiny | 0.604 | 0.685 | 0.081 | 42/23 | 1.826 | 4/5 | 4/5 |
| retfound_green | retizero | 0.604 | 0.655 | 0.051 | 30/18 | 1.667 | 5/5 | 5/5 |

筛选阈值固定为：两个单模型 accuracy ≥0.60、Expert 增量 ≥0.05、corrected/introduced ≥1.5、两类事件各 ≥15、net ≥10、固定患者折中至少 4/5 为正增益且 4/5 为非负 net，折间增量标准差 ≤0.12。

## 冻结回顾性相同预算比较

| 路线 | 预算 | 开发锁定基线 | corrected 方法/基线 | introduced 方法/基线 | net 方法/基线 | Δnet | 患者配对 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| retfound_green→convnext_tiny | 0.100 | entropy | 0/6 | 1/8 | -1/-2 | 1.000 | [-7.000, 10.000] |
| retfound_green→convnext_tiny | 0.200 | entropy | 5/12 | 8/14 | -3/-2 | -1.000 | [-15.000, 11.025] |
| retfound_green→convnext_tiny | 0.300 | margin | 12/28 | 12/15 | 0/13 | -13.000 | [-29.000, 2.025] |
| retfound_green→retizero | 0.100 | entropy | 0/7 | 0/9 | 0/-2 | 2.000 | [-5.000, 9.000] |
| retfound_green→retizero | 0.200 | margin | 9/17 | 9/12 | 0/5 | -5.000 | [-19.000, 7.000] |
| retfound_green→retizero | 0.300 | margin | 16/29 | 13/21 | 3/8 | -5.000 | [-23.025, 13.000] |

每个预算的 entropy/margin 强基线只由 development OOF 锁定。Scout-only 与 Always-Expert 作为两端参考保留在核心结果表。两条合格路线均没有唯一冻结 v1.1 路由身份，因此 v1.1 如实记为 not applicable；未移植其他路线的 v1.1 策略。

## 决策规则与下一步

单条路线必须在至少两个预声明预算（含 30%）同时做到 corrected 不低、introduced 不高且 net 严格更高，并且 30% 预算患者配对 Δnet 95% CI 下界非负。至少两条开发合格路线满足，才授予 `ROUTE_SPECIFIC_GO`。

缩小范围后仍无稳定收益；停止继续调整简单门控，不上 RL 或复杂模型。下一步应研究获得第二意见后的 KEEP_SCOUT / ADOPT_SECOND_OPINION / HUMAN_REVIEW 采纳机制。

## 追溯

- 实现提交：`d7d36b66f232e032ca5cf498696194aed4c629b9`
- 协议 SHA256：`2966baf37c6c751dbbd292d71e73d5c3aeadf40ac13e3be3a75065ef2b48c933`
- Benchmark manifest SHA256：`178dc7979363fa54418e85c90c3f1be0a8dd4db5fef73bd11e77862569209a35`
- 病例表 SHA256：`2a29402027557c7638ac34888fb8e4c2cbbaf8bee09d043cec5ab4e3e42be60f`
- qualified_routes.csv SHA256：`1e89bbe9f817a1529931b73460efbb69a195df0531a27a17b58dedef89d923a7`
- core_comparison_results.csv SHA256：`ac7fc7c504c005700193d906d0c63cd1e4815f449c03fd407c9886dcde33ac8e`
- 眼底模型训练/推理：未执行；冻结 Benchmark、Test 和预测资产：未修改。
