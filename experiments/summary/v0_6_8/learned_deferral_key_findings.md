# v0.6.8 关键发现：Learned Deferral Score

## 核心问题

v0.6.8 问的是：

> 轻量级 Logistic Regression 能否把模型输出后的风险信号组合成一个更强的复核排序分数？

实验结论是：

> `learned_logistic` 有竞争力，但没有稳定超过最强的事件特异性 severity-aware signal。

## 重要计数口径

本目录中的事件总数，例如 263、391、430，均指 6600 条骨干网络特异预测记录中的事件记录数，不是独立患者数，也不是唯一图像数量。

captured / total 结果同样是按预测记录统计的捕获结果，不是患者级或图像级捕获结果。

后续如果做置信区间或显著性检验，应以 `image_key` 作为聚类单位。


## `large_undergrading`：大幅低估

目标类型：primary target

事件数量：

- prediction-record-level positive records：263
- positive unique images：81

Top20% 复核预算下，平均召回率最高的方法是 `expected_gap_only`。

| 排名方法 | mean recall | mean lift | captured / total | residual |
|---|---:|---:|---:|---:|
| `expected_gap_only` | 0.6752 | 3.3474 | 180 / 263 | 83 |
| `top2_more_severe_only` | 0.6510 | 3.2255 | 169 / 263 | 94 |
| `learned_logistic` | 0.6469 | 3.2054 | 173 / 263 | 90 |
| `gated_severe_prob_mass_only` | 0.6111 | 3.0279 | 163 / 263 | 100 |
| `ophagent_combined` | 0.5703 | 2.8253 | 152 / 263 | 111 |
| `margin_only` | 0.4988 | 2.4705 | 129 / 263 | 134 |


## `vision_threatening_dr_miss`：威胁视力 DR 漏检

目标类型：primary target

事件数量：

- prediction-record-level positive records：391
- positive unique images：104

Top20% 复核预算下，平均召回率最高的方法是 `gated_severe_prob_mass_only`。

| 排名方法 | mean recall | mean lift | captured / total | residual |
|---|---:|---:|---:|---:|
| `gated_severe_prob_mass_only` | 0.7591 | 3.7451 | 297 / 391 | 94 |
| `learned_logistic` | 0.7175 | 3.5394 | 282 / 391 | 109 |
| `top2_more_severe_only` | 0.6684 | 3.2966 | 265 / 391 | 126 |
| `ophagent_combined` | 0.6260 | 3.0881 | 246 / 391 | 145 |
| `expected_gap_only` | 0.5697 | 2.8104 | 224 / 391 | 167 |
| `margin_only` | 0.4842 | 2.3881 | 191 / 391 | 200 |


## `dangerous_undergrading`：危险低估合成目标

目标类型：secondary composite target

事件数量：

- prediction-record-level positive records：430
- positive unique images：119

Top20% 复核预算下，平均召回率最高的方法是 `gated_severe_prob_mass_only`。

| 排名方法 | mean recall | mean lift | captured / total | residual |
|---|---:|---:|---:|---:|
| `gated_severe_prob_mass_only` | 0.7072 | 3.5201 | 304 / 430 | 126 |
| `top2_more_severe_only` | 0.6627 | 3.2985 | 282 / 430 | 148 |
| `learned_logistic` | 0.6548 | 3.2595 | 281 / 430 | 149 |
| `expected_gap_only` | 0.6000 | 2.9869 | 254 / 430 | 176 |
| `ophagent_combined` | 0.5943 | 2.9581 | 256 / 430 | 174 |
| `margin_only` | 0.4464 | 2.2218 | 192 / 430 | 238 |


## 总体解释

v0.6.8 没有证明 learned score 可以替代所有规则。更准确的解释是：

- 对 `large_undergrading`，`expected_gap_only` 在 Top20% 预算下仍然最强。
- 对 `vision_threatening_dr_miss`，`gated_severe_prob_mass_only` 在 Top20% 预算下仍然最强。
- `learned_logistic` 能学习到有竞争力的复核排序分数，但不是稳定统治性的最佳方法。
- `dangerous_undergrading` 是 secondary composite target，不适合作为唯一主目标。

这支持 OphAgent 的当前主线：

> OphAgent 的核心不是训练一个通吃的风险模型，而是建立临床风险感知的模型输出审计协议，在固定复核预算下比较不同 post-hoc risk signals，并报告自动放行区的残余危险错误。

## 边界说明

- `learned_logistic` 使用 Logistic Regression 的 `decision_function` 作为 learned review score，不应解释为校准后的真实危险概率。
- `StratifiedGroupKFold` 只能近似平衡正例事件，因为同一张图像在不同 backbone 下可能产生不同事件标签。
- fold balance 应结合 `learned_deferral_fold_diagnostics.csv` 检查。
- 后续置信区间或显著性检验应使用基于 `image_key` 的 clustered resampling，而不能使用 row-wise independent bootstrap。
- 本实验仍然属于 APTOS 内部探索性分析，不是独立外部验证。
