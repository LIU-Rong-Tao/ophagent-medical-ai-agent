# OphAgent Scout 视觉表征预咨询研究 v0.1

## 结论

**NO_IMPROVEMENT**

本研究只使用冻结 Scout 同一次前向的分类头前视觉表征、Scout 概率与开发折内路线画像。未读取当前病例 Expert 输出或表征，未训练 Scout/Expert，未用冻结回顾结果选择路线、特征、阈值或模型。

## 核心结果

| 路线 | 预算 | corrected 表征/基线 | introduced 表征/基线 | Δnet | 开发集锁定最强基线 |
|---|---:|---:|---:|---:|---|
| flair→swin_tiny | 10% | 0/39 | 2/15 | -26 | margin |
| flair→swin_tiny | 20% | 5/68 | 3/32 | -34 | margin |
| flair→swin_tiny | 30% | 14/80 | 11/43 | -34 | entropy |
| retfound_green→retfound_cfp | 10% | 7/31 | 1/5 | -20 | entropy |
| retfound_green→retfound_cfp | 20% | 11/65 | 5/11 | -48 | margin |
| retfound_green→retfound_cfp | 30% | 18/82 | 10/22 | -52 | entropy |
| retfound_green→swin_tiny | 10% | 4/34 | 0/8 | -22 | entropy |
| retfound_green→swin_tiny | 20% | 9/66 | 1/19 | -39 | margin |
| retfound_green→swin_tiny | 30% | 27/89 | 10/28 | -44 | margin |

最强基线在每条路线、每个预算上仅由 development OOF 在 entropy、margin 与上一版简单门控中锁定；冻结回顾集只评价。完整 5%–30% 风险—预算关系见 `risk_budget_comparison.png`。

30% 预算下，表征相对上一版简单门控的直接比较为：

- flair→swin_tiny：corrected 14/10，introduced 11/19（表征/上一版）。
- retfound_green→retfound_cfp：corrected 18/12，introduced 10/5（表征/上一版）。
- retfound_green→swin_tiny：corrected 27/25，introduced 10/12（表征/上一版）。

这说明视觉表征能改善部分旧门控排序，但仍明显落后于开发集锁定的简单不确定性基线，故不能据此授予 GO。

## 表征与成本

- flair：512 维，开发/回顾资产合计 2.94 MiB；同一次在线 Scout 前向的额外编码器调用为 0。
- retfound_green：384 维，开发/回顾资产合计 11.89 MiB；同一次在线 Scout 前向的额外编码器调用为 0。

FLAIR 旧运行未保存表征，故本研究做了一次冻结前向回提取；这属于回顾性研究成本。未来在线实现直接保留分类头输入，不产生第二次 Scout 编码。RETFound-Green 复用既有表征资产。

## 判定与边界

按预声明停止增加调用前模型复杂度；下一阶段应评估 Expert 输出到达后的 KEEP_SCOUT / ADOPT_SECOND_OPINION / HUMAN_REVIEW 采纳控制。

APTOS 缺少患者、眼别和检查标识；分析单位是确认像素级重复剔除后的图像组，不能声称患者级泛化。corrected/introduced 仍是标签定义的错误代理，不是临床获益/伤害。SafetyEligibilityGate 始终保留。

## 追溯

- 实现基线提交：`0e15b829d9a9da0ec9e8b5157acf43bf7fc8b2bd`
- 协议 SHA256：`ff9a50b497490399eba91d59ae7a64a38776be826b46da9e41e39c758dbb0e18`
- Benchmark manifest SHA256：`178dc7979363fa54418e85c90c3f1be0a8dd4db5fef73bd11e77862569209a35`
- 病例表 SHA256：`2a29402027557c7638ac34888fb8e4c2cbbaf8bee09d043cec5ab4e3e42be60f`
- 成功路线：`[]`
- 冻结 Benchmark、v1.1、Test、预测资产与既有负结论均未修改。
