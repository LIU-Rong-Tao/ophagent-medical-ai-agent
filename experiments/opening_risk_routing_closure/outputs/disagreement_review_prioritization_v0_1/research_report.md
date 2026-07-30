# OphAgent 模型分歧病例人工复核优先级审计 v0.1

## 结论

**NO_SIGNAL**

本研究只分析 Scout 与复核模型预测不一致的病例。固定模型仅预测`introduced OR both_wrong` 的有害冲突概率，用它排序人工复核队列；不自动选择模型结果，也不重新训练或运行眼底模型。

## 固定人工比例捕获结果

| 路线 | 人工比例 | harmful 方法/基线 | introduced 方法/基线 | dangerous 方法/基线 | both_wrong 方法/基线 | Δharmful | 开发锁定基线 |
|---|---:|---:|---:|---:|---:|---:|---|
| flair→swin_tiny | 10% | 18/14 | 16/11 | 4/6 | 2/3 | +4 | disagreement_js |
| flair→swin_tiny | 20% | 33/25 | 29/20 | 6/13 | 4/5 | +8 | disagreement_js |
| flair→swin_tiny | 30% | 47/31 | 40/21 | 14/12 | 7/10 | +16 | random |
| retfound_green→retfound_cfp | 10% | 9/7 | 3/1 | 0/0 | 6/6 | +2 | entropy |
| retfound_green→retfound_cfp | 20% | 18/14 | 10/10 | 2/7 | 8/4 | +4 | random |
| retfound_green→retfound_cfp | 30% | 26/19 | 16/13 | 4/8 | 10/6 | +7 | random |
| retfound_green→swin_tiny | 10% | 5/8 | 3/5 | 0/3 | 2/3 | -3 | random |
| retfound_green→swin_tiny | 20% | 13/14 | 9/11 | 1/7 | 4/3 | -1 | random |
| retfound_green→swin_tiny | 30% | 21/22 | 16/15 | 2/8 | 5/7 | -1 | random |

所有方法在同一路线和预算使用完全相同的病例数。random、Scout entropy、Scout margin 和概率 JS 分歧强度中的最强比较器只由 development OOF 锁定。

FLAIR→Swin 的主 harmful 捕获在三个预算均高于锁定基线，但 20% 配对差异 95% CI 为 [-4, 19]，并且 dangerous introduced 与 both_wrong 捕获没有同步改善。两条 RETFound-Green 路线没有复现一致排序收益，因此不授予路线特异或跨路线 GO。

## 风险可识别性

| 路线 | 目标 | 事件数 | AUROC | AUPRC |
|---|---|---:|---:|---:|
| flair→swin_tiny | both_wrong | 24 | 0.516 | 0.117 |
| flair→swin_tiny | dangerous_introduced | 36 | 0.540 | 0.190 |
| flair→swin_tiny | harmful_conflict | 103 | 0.747 | 0.726 |
| flair→swin_tiny | introduced | 79 | 0.756 | 0.650 |
| retfound_green→retfound_cfp | both_wrong | 24 | 0.557 | 0.215 |
| retfound_green→retfound_cfp | dangerous_introduced | 17 | 0.409 | 0.083 |
| retfound_green→retfound_cfp | harmful_conflict | 66 | 0.574 | 0.449 |
| retfound_green→retfound_cfp | introduced | 42 | 0.558 | 0.271 |
| retfound_green→swin_tiny | both_wrong | 21 | 0.488 | 0.115 |
| retfound_green→swin_tiny | dangerous_introduced | 21 | 0.327 | 0.082 |
| retfound_green→swin_tiny | harmful_conflict | 69 | 0.510 | 0.372 |
| retfound_green→swin_tiny | introduced | 48 | 0.519 | 0.270 |

同一个 harmful-conflict 分数用于全部子类型评价，没有为dangerous introduced 或 both_wrong 另训模型、另选阈值。

## 判定与边界

按协议停止当前多模型自动协同方法研究，将既有工作收束为风险评测与能力边界主线。

大等级差和重度阈值跨越的捕获数量保存在核心表中，属于无需真值即可观察的冲突强度代理。introduced、dangerous introduced、both_wrong 仍是冻结标签定义的回顾性错误代理。APTOS 缺少患者、眼别和检查标识，不能声称患者级泛化或临床效益。

## 追溯

- 实现基线提交：`41ef3872470f11d4b50c38545f933191cf13939e`
- 协议 SHA256：`8de84e8ec5a7ec9c635462c9d49b3438958df878c671932d083978ae973a6a2f`
- Benchmark manifest SHA256：`178dc7979363fa54418e85c90c3f1be0a8dd4db5fef73bd11e77862569209a35`
- 病例表 SHA256：`2a29402027557c7638ac34888fb8e4c2cbbaf8bee09d043cec5ab4e3e42be60f`
- 成功路线：`[]`
- 前台、冻结 Benchmark、v1.1、Test、预测资产及既有负结论均未修改。
