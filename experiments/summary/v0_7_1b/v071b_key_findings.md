# v0.7.1b 关键发现：协议补全与图像聚类置信区间

## 版本定位

v0.7.1b 用于补全 v0.7.1 的证据链，不开启新方向。

本版本不做 CAM、SHAP、attention map 或其他解释性支线；重点是验证 v0.7.1 中观察到的 VTDR miss 富集是否真正超过 random gate-only baseline。

## 研究问题

在冻结 APTOS 模型、事件定义、复核预算和排序信号后，severe-class probability mass 是否能在预测等级 gate 之外，为外部 DR 数据中的 VTDR miss 提供额外复核排序信息？

## 主比较

- Target：VTDR miss
- 定义：`true_grade >= 3 and pred_grade < 3`
- Budget：Top20%
- Method：`gated_severe_prob_mass_only`
- Comparator：`random_gate_only`
- Bootstrap：2000 次 image-clustered bootstrap
- 随机基线：2000 次 gate-only random sampling
- 聚类单位：unique image，同一图像的 6 个 backbone 记录一起重采样

## 数据规模

| dataset | unique images | prediction rows | backbones | VTDR miss per-backbone |
|---|---:|---:|---:|---:|
| IDRiD_data | 103 | 618 | 6 | 17–29 |
| MESSIDOR2 | 526 | 3156 | 6 | 21–29 |

## Primary result

Top20% 复核预算下，`gated_severe_prob_mass_only` 相对 `random_gate_only` 显示明确增量。

| dataset | Δ event recall | 95% CI | win rate | residual event count reduction | residual event rate reduction |
|---|---:|---:|---:|---:|---:|
| IDRiD_data | +0.3385 | [0.2195, 0.4742] | 1.0000 | +7.8787 | +0.0961 |
| MESSIDOR2 | +0.6268 | [0.5003, 0.7369] | 1.0000 | +16.7396 | +0.0399 |

其中：

- `Δ event recall = gated_severe_prob_mass_only - random_gate_only`
- `residual event count reduction` 为正，表示 gated 方法相比 random gate-only 少留下的残余危险事件数量；
- `win rate` 表示 bootstrap 中 `Δ > 0` 的比例，不是临床成功概率。

## Go / No-Go 判断

根据预先冻结的判断规则，本版本结果属于：

**强证据**

原因：

- 两个外部数据集的差值点估计均为正；
- MESSIDOR2 的 95% CI 不跨 0；
- IDRiD_data 方向一致，且 95% CI 也不跨 0。

## 可写结论

可以写：

> 在冻结 APTOS 模型、事件定义、复核预算和排序信号后，severe-class probability mass 在预测等级 gate 之外，为外部 DR 数据中的 VTDR miss 提供了额外复核排序信息。

展示时可表述为：

> 仅知道“模型预测未达重症”还不够；在这些候选样本内部，重症概率质量仍能进一步帮助排序危险漏检。

## 不能写的结论

不能写：

- 模型已经完成外部泛化验证；
- 可以安全自动放行；
- 这是患者级临床 VTDR 终点；
- 这是临床部署验证；
- 这是由 CAM、SHAP 或 attention map 支撑的解释性发现。

## 展示口径

统一使用：

- 未进入优先复核区的残余危险事件

不要使用：

- 自动放行区

病例页必须标注：

- 公共数据集回顾性 grade-based proxy 示例，不是患者级临床判断。

## 后续建议

v0.7.1b 已经完成首次线下对接前最关键的统计证据补全。

下一步应进入展示收口：

1. 生成 v0.7.1b 总览 HTML；
2. 更新 demo 默认首页为 VTDR miss + Top20% + 两个外部数据集汇总；
3. 准备线下讲稿和备用 PDF；
4. 不再机械开启 v0.7.1c/d；
5. 是否进入 v0.7.2，需要在线下对接和 go/no-go 审查后再决定。
