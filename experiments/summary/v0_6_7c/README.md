# v0.6.7c Ranking Signal Mechanism Analysis

本目录对应 OphAgent v0.6.7c：ranking signal mechanism analysis。

## 目的

v0.6.7b 发现，原始 ophagent_combined 规则有效，但在危险低估任务上并不是最优。v0.6.7c 进一步分析不同排序信号的作用机制，回答三个问题：

1. severity-aware signals 为什么能超过原始 ophagent_combined？
2. 不同 clinical dangerous events 是否需要不同排序信号？
3. 即使使用最佳排序信号，自动放行区仍然残留哪些危险错误？

## 输入

主要输入文件：

- experiments/summary/v0_6_7/clinical_event_cases.csv
- 各 backbone 的 test_predictions.csv

脚本会从原始 test_predictions.csv 中合并五分类概率：

- prob_No DR
- prob_Mild DR
- prob_Moderate DR
- prob_Severe DR
- prob_Proliferative DR

合并时使用 backbone + image_key，其中 image_key 为图像文件名。

排序阶段只使用模型输出信号，不使用真实标签。真实标签和 clinical event columns 仅用于后验评估。

## 方法

统一比较 10 种排序方法：

- confidence_only
- margin_only
- entropy_only
- uncertainty_rank_fusion
- ophagent_combined
- severe_prob_mass_only
- gated_severe_prob_mass_only
- expected_grade_only
- expected_gap_only
- top2_more_severe_only

复核预算包括：

- Top5%
- Top10%
- Top15%
- Top20%
- Top25%
- Top30%
- Top40%
- Top50%

重点分析 Top20% 复核预算下的机制差异，同时用多预算结果观察趋势稳定性。

## 主要结论

v0.6.7c 的主要结论是：不同错误类型对应的有效排序信号不同，单一通用风险分数不一定适合所有审计目标。

在 Top20% 复核预算下：

- general_error 的最优方法是 margin_only，说明通用错分更偏向不确定性问题。
- large_undergrading 的最优方法是 expected_gap_only，说明大幅低估主要与 expected_grade - pred_grade 有关。
- vision_threatening_dr_miss 的最优方法是 gated_severe_prob_mass_only，说明重症漏检主要与 pred_grade <= 2 条件下的 P(Severe) + P(PDR) 有关。

这说明 OphAgent 的核心价值不应表述为“某个手工 combined 分数最优”，而应表述为：基于模型输出的 post-hoc risk signals 可以针对不同 clinical dangerous events 进行复核优先级优化，并持续报告自动放行区残余危险错误。

## 输出文件

- unified_ranking_method_tradeoff.csv：全部排序方法在多复核预算下的完整 trade-off。
- unified_ranking_method_mean_summary.csv：按事件、预算、方法聚合后的平均结果。
- top20_overlap_summary.csv：Top20% 下最佳 severity-aware 方法与 ophagent_combined 的捕获重叠统计。
- top20_overlap_cases.csv：overlap 中每个病例的详细特征。
- top20_residual_profile.csv：最佳 severity-aware 方法漏掉样本的统计特征。
- top20_residual_cases.csv：Top20% 自动放行区残余危险样本明细。
- v067c_key_findings.md：中文结果解释与机制分析。

## 复现命令

运行脚本：

python scripts/analyze_v067c_ranking_signal_mechanism.py

生成结果目录：

experiments/summary/v0_6_7c/

## 注意事项

本分析仍属于公共 APTOS 五分类数据上的 clinical-risk proxy 后验审计，不等价于真实临床终点。医院真实数据阶段需要结合医生定义的 dangerous pair、urgency tier 或 clinical cost matrix 进行重新校准。
