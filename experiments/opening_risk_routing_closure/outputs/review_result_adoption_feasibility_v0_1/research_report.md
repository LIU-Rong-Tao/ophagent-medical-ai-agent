# OphAgent 复核结果安全采纳可行性审计 v0.1

## 结论

**NO_SIGNAL**

本研究发生在复核模型已经运行之后。模型只读取两模型完整概率、置信度/分歧、等级变化、开发折内转移画像和既有 Scout 表征，输出 KEEP_SCOUT、ADOPT_REVIEW_RESULT 或 HUMAN_REVIEW。未重新训练或运行任何眼底模型。

## 核心比较

| 路线 | 人工比例 | corrected 方法/基线 | introduced 方法/基线 | dangerous 方法/基线 | Δnet | 开发锁定基线 |
|---|---:|---:|---:|---:|---:|---|
| flair→swin_tiny | 10% | 57/68 | 25/27 | 15/13 | -9 | soft_vote |
| flair→swin_tiny | 20% | 37/58 | 14/26 | 8/13 | -9 | higher_confidence |
| flair→swin_tiny | 30% | 27/41 | 4/15 | 3/9 | -3 | higher_confidence |
| retfound_green→retfound_cfp | 10% | 61/69 | 20/18 | 13/8 | -10 | higher_confidence |
| retfound_green→retfound_cfp | 20% | 46/52 | 15/12 | 10/6 | -9 | higher_confidence |
| retfound_green→retfound_cfp | 30% | 33/34 | 11/5 | 9/3 | -7 | higher_confidence |
| retfound_green→swin_tiny | 10% | 73/76 | 29/19 | 14/9 | -13 | higher_confidence |
| retfound_green→swin_tiny | 20% | 58/49 | 19/13 | 12/7 | +3 | higher_confidence |
| retfound_green→swin_tiny | 30% | 40/35 | 16/9 | 12/6 | -2 | higher_confidence |

每个预算的正式比较器只在 development OOF 中从置信度更高者和软投票锁定。保留 Scout 与始终采用复核结果作为安全/效能端点完整保留，但不用于不可能的联合支配判定。

## 状态可识别性

| 路线 | 状态 | 事件数 | AUROC | AUPRC |
|---|---|---:|---:|---:|
| flair→swin_tiny | both_wrong | 73 | 0.781 | 0.158 |
| flair→swin_tiny | corrected | 107 | 0.951 | 0.684 |
| flair→swin_tiny | introduced | 79 | 0.927 | 0.506 |
| retfound_green→retfound_cfp | both_wrong | 111 | 0.781 | 0.252 |
| retfound_green→retfound_cfp | corrected | 109 | 0.957 | 0.658 |
| retfound_green→retfound_cfp | introduced | 42 | 0.899 | 0.256 |
| retfound_green→swin_tiny | both_wrong | 104 | 0.771 | 0.262 |
| retfound_green→swin_tiny | corrected | 116 | 0.953 | 0.667 |
| retfound_green→swin_tiny | introduced | 48 | 0.901 | 0.262 |

这些指标使用同一开发折锁定模型在冻结回顾集上的输出，只描述标签状态的回顾性可分性，不等同于部署时已知病例真值。corrected AUROC 为 0.951–0.957，introduced 为 0.899–0.927，说明谁可能纠错或引错具有排序信息；但 both_wrong AUROC 仅 0.771–0.781，AUPRC 仅 0.158–0.262。

`NO_SIGNAL` 在本协议中特指没有形成满足预声明联合安全改善的可执行三动作策略，不表示所有输入特征都没有统计判别信息。

## HUMAN_REVIEW 解释

现有资产没有真实人工复核结论，因此 HUMAN_REVIEW 只表示延期裁决：主结果不把这些病例计为已纠正，也不假定人工一定正确。相同比例比较使用完全相同的复核病例数；报告同时记录人工队列捕获的 corrected、introduced、dangerous introduced 与 both_wrong 负担。

## 判定与边界

按协议停止继续开发模型采纳方法，并重新评估开题主线；不得通过更复杂模型或假设人工全对来掩盖负结果。

APTOS 缺少患者、眼别和检查标识，结论仅属于确认完全重复剔除后的图像级回顾证据。corrected/introduced/dangerous introduced 是标签错误代理，不是临床结局；SafetyEligibilityGate 与人工最终责任均不可绕过。

## 追溯

- 实现基线提交：`7044f5978a557f5b391b95c88ad08a89cc492b63`
- 协议 SHA256：`2f59bd1be60f0821bed0703a14d2a96760f8e9cf18d1165351a1c30c47c89abb`
- Benchmark manifest SHA256：`178dc7979363fa54418e85c90c3f1be0a8dd4db5fef73bd11e77862569209a35`
- 病例表 SHA256：`2a29402027557c7638ac34888fb8e4c2cbbaf8bee09d043cec5ab4e3e42be60f`
- 成功路线：`[]`
- 冻结前台、Benchmark、v1.1、Test、预测资产及既有负结论均未修改。
