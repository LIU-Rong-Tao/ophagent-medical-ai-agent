# v0.6.8b 关键发现：稳健性与机制审计

## 核心问题

v0.6.8b 检查的是：

> v0.6.8 的 learned_logistic 是否稳定优于事件特异性 severity-aware signal？它主要学到了什么机制？

结论是：

> learned_logistic 有竞争力，但没有稳定超过最强事件特异性规则；它主要是在重加权已有的严重程度感知信号。

## Bootstrap 结论

paired image-key clustered bootstrap 使用 per-backbone aggregate endpoint，与 v0.6.8 的评价口径一致。

等价性验证显示，权重版 bootstrap 与暴力复制版 bootstrap 完全一致，主要差异项均为 0。

### `large_undergrading`

主比较：`learned_logistic` vs `expected_gap_only`

- mean recall diff = -0.0206
- 95% CI = [-0.0779, 0.0374]
- bootstrap win rate = 0.2055
- mean captured：171.3 vs 176.7

结论：`learned_logistic` 没有稳定超过 `expected_gap_only`。对大幅低估，`expected_gap_only` 仍然是 Top20% 预算下更稳的主信号。

### `vision_threatening_dr_miss`

主比较：`learned_logistic` vs `gated_severe_prob_mass_only`

- mean recall diff = -0.0420
- 95% CI = [-0.0926, 0.0073]
- bootstrap win rate = 0.0395
- mean captured：281.0 vs 297.3

结论：`learned_logistic` 没有稳定超过 `gated_severe_prob_mass_only`。对威胁视力 DR 漏检，`gated_severe_prob_mass_only` 是更稳的主信号。

## 捕获重叠结论

Top20% 捕获重叠显示，learned score 能补到一部分 simple rule 未捕获样本，但不能替代 simple rule。

- `large_undergrading`：learned_only = 18，simple_only = 23。
- `vision_threatening_dr_miss`：learned_only = 24，simple_only = 38。
- `dangerous_undergrading`：learned_only = 22，simple_only = 45。

这说明 learned score 有补充价值，但事件特异性规则仍保留更多独有危险样本。

## Repeated split 结论

在 seeds = 42, 43, 44, 45, 46 的 repeated grouped CV 下，Top20% 结论稳定。

### `large_undergrading`

- `expected_gap_only` 排名第 1，recall_mean = 0.6730，captured_mean = 177.0。
- `learned_logistic` 排名第 3，recall_mean = 0.6494，captured_mean = 170.8。

### `vision_threatening_dr_miss`

- `gated_severe_prob_mass_only` 排名第 1，recall_mean = 0.7621，captured_mean = 298.0。
- `learned_logistic` 排名第 2，recall_mean = 0.7084，captured_mean = 277.0。

### `dangerous_undergrading`

- `gated_severe_prob_mass_only` 排名第 1，recall_mean = 0.7070，captured_mean = 304.0。
- `learned_logistic` 排名第 2，recall_mean = 0.6633，captured_mean = 285.2。

## 系数稳定性结论

Logistic 系数分析显示，`learned_logistic` 主要在重加权已有的严重程度感知信号。

稳定靠前的特征包括：

- `pred_le_2`
- `expected_grade`
- `expected_gap`
- `severe_prob_mass`
- `gated_severe_prob_mass`
- `top2_more_severe_conf`

这些特征在 repeated split 下符号整体稳定，但由于特征之间高度相关，不能解释为单个特征的因果贡献。

## 总体结论

v0.6.8b 支持以下判断：

> learned_logistic 是有竞争力的监督式复核排序基线，但当前证据不支持用它替代事件特异性 severity-aware signal。

对 OphAgent 主线来说，更稳的方向不是继续堆复杂模型，而是冻结事件定义、复核预算、排序信号和评价协议，然后进入外部数据验证。
