# OphAgent 预咨询选择性会诊方法研究 v0.1

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Verification Status: UNVERIFIED（需以同一提交确定性复跑核验）
- Version Label: selective_consultation_method_v0_1

## 结论

**NO_IMPROVEMENT**

相对既有简单基线未达到预声明改进条件，不建议以当前方法进入独立确认或更复杂训练。

本结论属于已暴露冻结结果上的回顾性研究证据；`SafetyEligibilityGate` 仍不可绕过，方法本身不能授予路线资格或触发 Expert。

## 研究设计

- 主分析固定为 DeepDRiD 原生 `keepfit_cfp→flair`，按患者分组；
- 其余 DeepDRiD 原生路线只做开发预筛后的异质性分析；
- APTOS 仅在确认重复排除后的图像级队列做敏感性分析；
- DeepDRiD 外部迁移因缺少同域开发折而排除，未在冻结结果上拟合；
- 两个 L2 逻辑回归分别预测 corrected 与 introduced；无调参搜索；
- 双模型策略先取预测 introduced 风险最低的 2×预算安全池，再按 predicted corrected 排序；没有事后加权综合分数；
- 开发预测采用嵌套分组交叉拟合；回顾性预测只使用完整开发折拟合。

## 固定主路线：相同 Expert 预算

| 预算 | 开发锁定基线 | corrected 方法/基线 | introduced 方法/基线 | net 方法/基线 | Δnet | Δnet 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | margin | 7/7 | 2/6 | 5/1 | 4 | [-4.000, 12.000] |
| 0.10 | margin | 9/12 | 5/7 | 4/5 | -1 | [-14.000, 11.000] |
| 0.20 | margin | 19/20 | 7/16 | 12/4 | 8 | [-2.000, 19.000] |
| 0.30 | margin | 22/30 | 17/22 | 5/8 | -3 | [-16.000, 9.000] |

5% 与 20% 预算点分别得到 Δnet +4 与 +8，但患者配对区间均跨 0；5% 不属于预声明的决策预算，20% 又少保留 1 个 corrected，因此都不满足“corrected 不降低且 introduced 不增加”的支配条件。

冻结 v1.1 在 30% 预算为 corrected 25、introduced 18、net 7；双模型为 22、17、5。开发 OOF 锁定的更强 margin 基线为 30、22、8。

## 固定主路线：模型可识别性

| 数据 | 模型 | 条件队列 | 事件 | AUROC | AUPRC |
| --- | --- | --- | --- | --- | --- |
| development_oof | corrected_logistic | scout_wrong_only | 35 | 0.600 | 0.574 |
| development_oof | introduced_logistic | scout_correct_only | 29 | 0.626 | 0.282 |
| retrospective_evaluation | corrected_logistic | scout_wrong_only | 46 | 0.619 | 0.442 |
| retrospective_evaluation | introduced_logistic | scout_correct_only | 58 | 0.705 | 0.447 |

条件 AUROC 用于区分真正的 Expert 特异信号与一般 Scout 错误检测，不作为临床安全终点。introduced 条件 AUROC 从开发 OOF 0.626 外推至 0.705，但 harm-only 排序为降低错误牺牲了过多 corrected，未达到 HARM_ONLY_GO 的净效益约束。

## 固定主路线：由开发 OOF 锁定的 introduced 风险上限

| 风险上限 | 预算 方法/基线 | corrected 方法/基线 | introduced 方法/基线 | 实际风险 方法/基线 | 回顾性达标 方法/基线 |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 0.01/0.01 | 0/0 | 0/1 | 0.000/0.250 | True/False |
| 0.10 | 0.05/0.28 | 7/21 | 2/17 | 0.100/0.152 | True/False |
| 0.15 | 0.30/0.29 | 22/30 | 17/22 | 0.142/0.190 | True/False |
| 0.20 | 0.30/0.29 | 22/30 | 17/22 | 0.142/0.190 | True/True |

风险上限只在开发 OOF 上选择预算；回顾性队列不会重新选阈值。若回顾性实际风险超限，该行视为未外推成功。

## 路线异质性与敏感性

- DeepDRiD 原生路线：90；开发预筛通过 0；回顾性复现 0；
- 回顾性复现覆盖 Scout 0 种、Expert 0 种；
- APTOS 图像级敏感性：开发预筛通过 0/90，回顾性复现 不适用（0 条开发预筛路线）；
- APTOS 无患者/眼别标识，不能把图像级稳定性解释为患者级泛化。

## 失败边界

- 冻结回顾性结果此前已暴露，不能充当独立确认；
- corrected/introduced 是标签定义的模型错误代理，不是临床伤害、治疗获益或最终诊断；
- 路线共享病例和模型，90 条路线不能当作 90 个独立临床样本；
- 当前特征没有真实图像质量、多模态或临床资料；
- 更复杂模型只有在独立患者级确认集与足够事件数就绪后才有意义。

## 追溯

- 实现提交：`c2d21f5886153a3a27fbddbecccd971fa51aaca3`
- 方法协议 SHA256：`984c5e2b4d8a7b57e21b24120eab8d0b26d4de9dc356bffc5cc09c1b6b84a2ac`
- 输入 Benchmark manifest SHA256：`178dc7979363fa54418e85c90c3f1be0a8dd4db5fef73bd11e77862569209a35`
- 输入病例表 SHA256：`2a29402027557c7638ac34888fb8e4c2cbbaf8bee09d043cec5ab4e3e42be60f`
- 眼底模型训练/推理：未执行；冻结预测资产：未修改。
