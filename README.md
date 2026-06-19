# OphAgent

眼科 AI 模型输出审计与复核优先级原型。

这个项目关注一个实际问题：模型已经给出预测结果后，哪些样本更值得优先复核，未进入优先复核区的样本里还残留多少危险错误。

当前主线已经从“继续堆分类模型”转向“模型输出后的风险审计”。

本项目用于科研、工程实践和项目展示，不用于临床诊断、治疗建议或真实医疗决策。

---

## 当前实验节点：v0.7.1b

v0.7.1b 完成了外部 DR 数据上的复核排序协议补全。核心问题是：

> 在模型把病例预测成非重症时，输出概率里残留的重症概率，是否还能帮助我们把真正危险的漏检排到更前面？

主实验设置：

- 训练来源：APTOS2019 frozen checkpoints
- 外部测试：IDRiD_data / MESSIDOR2
- 目标事件：VTDR miss，`true_grade >= 3 and pred_grade < 3`
- 复核预算：Top20%
- 方法：`gated_severe_prob_mass_only`
- 对照：`random_gate_only_expected`
- 统计：image-clustered bootstrap，同一图像的 6 个 backbone 记录一起重采样

结果：

| Dataset | Δ recall | 95% CI | Bootstrap win rate | Mean residual count reduction / backbone |
|---|---:|---:|---:|---:|
| IDRiD_data | +0.3385 | [0.2195, 0.4742] | 1.0000 | +7.8787 |
| MESSIDOR2 | +0.6268 | [0.5003, 0.7369] | 1.0000 | +16.7396 |

一句话解释：

> 只知道“模型预测为非重症”还不够；在这些候选样本内部，重症概率质量仍然能继续排序危险漏检。

补充说明：

- `Bootstrap win rate` 表示 bootstrap 中 `Δ recall > 0` 的比例。
- `Mean residual count reduction / backbone` 是六个 backbone 的平均残余危险事件减少量，不是患者数。
- random gate-only 的独立随机抽样用于估计 baseline 分布；primary bootstrap 比较使用 `random_gate_only_expected`。
- `learned_logistic` 是 v0.6.8/v0.6.8b 的内部监督式基线；原 v0.7.0 协议计划保留其外部 baseline，但当前 v0.7.1/v0.7.1b 尚未实现外部 frozen learned_logistic 推理，属于预设监督式 baseline（非 primary comparator）缺失 / protocol deviation，不影响本轮 primary gate-only comparison。


## Run

启动 demo：

```bash
streamlit run app/demo.py
```

重新生成 v0.7.1b 结果：

```bash
python scripts/evaluate_v071b_protocol_completion_ci.py \
  --predictions experiments/summary/v0_7_1/external_dr_direct_inference_predictions.csv \
  --out-dir experiments/summary/v0_7_1b \
  --n-random 2000 \
  --n-bootstrap 2000 \
  --seed 42
```

外部直接推理：

```bash
python scripts/run_v071_external_dr_direct_inference.py
```

外部复核排序评估：

```bash
python scripts/evaluate_v071_external_dr_review_ranking.py
```

---

## Main files

| Path                                                  | Description                                       |
| ----------------------------------------------------- | ------------------------------------------------- |
| `scripts/evaluate_v071b_protocol_completion_ci.py`    | v0.7.1b 协议补全、random gate-only、clustered bootstrap |
| `scripts/run_v071_external_dr_direct_inference.py`    | 外部 DR frozen checkpoint direct inference          |
| `scripts/evaluate_v071_external_dr_review_ranking.py` | 外部复核排序评估                                          |
| `scripts/precheck_v070_external_dr_datasets.py`       | 外部数据预检、重叠审计、checkpoint manifest                   |
| `experiments/summary/v0_7_1b/`                        | v0.7.1b 主结果                                       |
| `experiments/summary/v0_7_1b_seed_sensitivity/`       | 随机种子敏感性检查                                         |
| `experiments/summary/v0_7_1/`                         | v0.7.1 外部直接推理与排序结果                                |
| `experiments/summary/v0_7_0/`                         | v0.7.0 外部数据预检与协议冻结                                |

---

## Project line

| Version | Focus                                                                  |
| ------- | ---------------------------------------------------------------------- |
| v0.7.1b | 外部复核排序协议补全：random gate-only、image-clustered bootstrap、seed sensitivity |
| v0.7.1  | APTOS frozen checkpoints 直接推理 IDRiD_data / MESSIDOR2                   |
| v0.7.0  | 外部 DR 数据预检、重叠审计、协议冻结                                                   |
| v0.6.8b | learned deferral score 稳健性与机制审计                                        |
| v0.6.8  | learned deferral score                                                 |
| v0.6.7c | 排序信号机制分析                                                               |
| v0.6.7b | severity-aware signal ablation                                         |
| v0.6.7  | residual risk audit                                                    |
| v0.6.6  | 无真实标签预审风险排序                                                            |
| v0.6.5  | 医院线下展示版                                                                |

---

## Classification backbone results

APTOS2019 test set:

| Backbone         | Accuracy | Macro-F1 | Weighted-F1 |   QWK |
| ---------------- | -------: | -------: | ----------: | ----: |
| Swin-Tiny        |    0.829 |    0.657 |       0.820 | 0.898 |
| ConvNeXt-Tiny    |    0.814 |    0.650 |       0.809 | 0.862 |
| ViT-B/16         |    0.818 |    0.646 |       0.814 | 0.876 |
| RETFound-MAE-CFP |    0.804 |    0.583 |       0.789 | 0.866 |

这些模型用于后续输出审计与复核排序，不作为 leaderboard 结果。

---

## Notes

* 当前结果来自公共数据集回顾性实验。
* VTDR miss 是 grade-based proxy，不是医生定义的患者级临床终点。
* v0.7.1 外部分类性能存在明显域移压力。
* v0.7.1b 的重点是复核排序信号是否仍然有效。
* 所有输出都需要人工审核。
