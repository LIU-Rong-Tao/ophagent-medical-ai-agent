# v0.6.7c Ranking Signal Mechanism Analysis

## 中文结论

v0.6.7c 的目的不是再提出新规则，而是解释 v0.6.7b 中 severity-aware signals（严重程度感知信号）为什么能超过原始 `ophagent_combined`，以及它们仍然会漏掉什么。

第一，统一比较所有排序方法后可以看到，不同错误类型对应的有效信号并不相同。
在 Top20% 复核预算下，`general_error` 的最优方法是 `margin_only`，mean recall 为 52.0%；
`large_undergrading` 的最优方法是 `expected_gap_only`，mean recall 为 66.1%，捕获 177 / 263；
`vision_threatening_dr_miss` 的最优方法是 `gated_severe_prob_mass_only`，mean recall 为 75.9%，捕获 298 / 391。
这说明一个通用风险分数不一定适合所有错误类型：通用错分更偏不确定性问题，而方向敏感的危险低估更依赖严重程度相关概率信号。

第二，Top20% overlap analysis 解释了新信号相对 combined 的增益来源。
对 `large_undergrading`，`expected_gap_only` 捕获 177 个危险样本，`ophagent_combined` 捕获 156 个；两者共同捕获 129 个，`expected_gap_only` 独有捕获 48 个，combined 独有捕获 27 个，两者都漏掉 59 个。
对 `vision_threatening_dr_miss`，`gated_severe_prob_mass_only` 捕获 298 个危险样本，`ophagent_combined` 捕获 241 个；两者共同捕获 226 个，`gated_severe_prob_mass_only` 独有捕获 72 个，combined 独有捕获 15 个，两者都漏掉 78 个。

第三，最佳 severity-aware 方法仍然存在自动放行区残余风险。
Top20% 下，`large_undergrading` 使用 `expected_gap_only` 后仍残余 86 个危险低估样本；这些残余样本的 median expected_gap 为 0.16，median severe_prob_mass 为 0.15。
`vision_threatening_dr_miss` 使用 `gated_severe_prob_mass_only` 后仍残余 93 个重症漏检样本；这些残余样本的 median severe_prob_mass 为 0.09，top2_more_severe_rate 为 63.4%。
因此，v0.6.7c 支持的结论不是“某个排序信号可以证明自动放行安全”，而是“severity-aware signals 可以显著改善复核优先级，同时 residual risk 仍需要被显式审计”。

## 输出文件说明

- `unified_ranking_method_tradeoff.csv`: 全部排序方法在多复核预算下的完整 trade-off。
- `unified_ranking_method_mean_summary.csv`: 按事件、预算、方法聚合后的平均结果。
- `top20_overlap_summary.csv`: Top20% 下最佳 severity-aware 方法与 combined 的捕获重叠统计。
- `top20_overlap_cases.csv`: overlap 中每个病例的详细特征。
- `top20_residual_profile.csv`: 最佳 severity-aware 方法漏掉样本的统计特征。
- `top20_residual_cases.csv`: Top20% 自动放行区残余危险样本明细。
