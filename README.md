# OphAgent

## 眼科 AI 模型输出审计与复核优先级原型

OphAgent 是一个面向眼科医学影像的 AI 工作流项目。

项目当前重点不是继续追求单模型分类精度，而是研究：

```
模型已经给出预测结果之后，
哪些样本不应直接相信，
哪些样本应该优先交给医生复核，
自动放行的样本里还残留多少危险错误。
```

本项目仅用于科研、工程实践与项目展示，不用于临床诊断、治疗建议或真实医疗决策。

---

## 当前稳定节点：v0.6.8b

v0.6.8b 已完成 learned deferral score 的稳健性与机制审计。

当前结论：

- `learned_logistic` 是有竞争力的监督式复核排序基线；
- 但它没有稳定超过事件特异性的 severity-aware signal；
- 对 `large_undergrading`，`expected_gap_only` 仍是 Top20% 预算下更稳的主信号；
- 对 `vision_threatening_dr_miss`，`gated_severe_prob_mass_only` 仍是 Top20% 预算下更稳的主信号；
- OphAgent 当前主线应从继续堆模型转向协议冻结和外部数据验证前检查。

v0.6.8b 包含四类分析：

- paired image-key clustered bootstrap；
- Top20% 捕获重叠分析；
- Logistic 系数稳定性分析；
- repeated split sensitivity。

## 核心发现

在 Top20% 复核预算下：

| 目标事件                         | 最优排序信号                        | 解释                               |
| ---------------------------- | ----------------------------- | -------------------------------- |
| `general_error`              | `margin_only`                 | 普通错分更接近不确定性和决策边界问题               |
| `large_undergrading`         | `expected_gap_only`           | 大幅低估更依赖期望严重等级与预测等级之间的偏差          |
| `vision_threatening_dr_miss` | `gated_severe_prob_mass_only` | 重症漏检更依赖低预测等级下的 Severe / PDR 概率质量 |

这说明：在当前 APTOS 多骨干预测记录上，不同 grade-based risk proxy 对应的最优排序信号不同，不能简单指望一个通用风险分数解决所有问题。

---

## 版本脉络

| 版本            | 重点                                     | 状态       |
| ------------- | -------------------------------------- | -------- |
| v0.6.8b       | 稳健性与机制审计：bootstrap / overlap / coefficients / repeated split | 当前稳定研究节点 |
| v0.6.8        | 学习型复核分数：learned deferral score              | 已完成      |
| v0.6.7c       | 排序信号机制分析                               | 已完成      |
| v0.6.7b       | 严重程度感知信号消融                             | 已完成      |
| v0.6.7        | 临床残余风险审计                               | 已完成      |
| v0.6.6        | 无真实标签预审风险排序                            | 已完成      |
| v0.6.5        | 医院线下展示版                                | 已完成      |
| v0.6.0-v0.6.4 | 病例报告草稿、安全审计、真实 LLM 小规模安全探针             | 已完成      |
| v0.5.x        | APTOS benchmark、多 backbone、CAM adapter | 已完成      |

---

## 主要结果入口

| 路径                                                  | 内容                |
| --------------------------------------------------- | ----------------- |
| `experiments/summary/v0_6_8b/`                     | 当前主结果：稳健性与机制审计       |
| `experiments/summary/v0_6_8b/README.md`            | v0.6.8b 结果说明与输出文件索引     |
| `experiments/summary/v0_6_8b/robustness_mechanism_key_findings.md` | v0.6.8b 中文关键发现 |
| `experiments/summary/v0_6_8/`                      | 学习型复核分数实验               |
| `experiments/summary/v0_6_8/learned_deferral_key_findings.md` | v0.6.8 中文关键发现 |
| `experiments/summary/v0_6_7c/`                      | 当前主结果：排序信号机制分析    |
| `experiments/summary/v0_6_7c/README.md`             | v0.6.7c 结果说明与复现入口 |
| `experiments/summary/v0_6_7c/v067c_key_findings.md` | v0.6.7c 中文结果解释    |
| `experiments/summary/v0_6_7b/`                      | 严重程度感知信号消融        |
| `experiments/summary/v0_6_7/`                       | 临床残余风险审计          |
| `experiments/summary/v0_6_6/`                       | 预审风险排序技术验证        |
| `experiments/summary/v0_6_5/`                       | 医院线下展示版           |

---

## 快速运行

运行 Streamlit demo：

```
streamlit run app/demo.py
```

重新生成 v0.6.7c 排序信号机制分析：

```
python scripts/analyze_v067c_ranking_signal_mechanism.py
```

重新生成 v0.6.7 临床残余风险分析：

```
python scripts/evaluate_clinical_residual_risk.py
```

---

## 核心脚本

| 脚本                                                  | 作用                                  |
| --------------------------------------------------- | ----------------------------------- |
| `scripts/analyze_v068b_robustness_mechanism.py`    | v0.6.8b bootstrap 与捕获重叠分析              |
| `scripts/analyze_v068b_logistic_coefficients.py`   | v0.6.8b Logistic 系数稳定性分析               |
| `scripts/analyze_v068b_repeated_split_sensitivity.py` | v0.6.8b repeated split sensitivity 分析       |
| `scripts/analyze_v068_learned_deferral_score.py`   | v0.6.8 学习型复核分数分析                     |
| `scripts/analyze_v067c_ranking_signal_mechanism.py` | v0.6.7c 排序信号机制分析                    |
| `scripts/analyze_v067_severity_aware_baselines.py`  | v0.6.7b 严重程度感知信号消融                  |
| `scripts/evaluate_clinical_residual_risk.py`        | v0.6.7 临床残余风险审计                     |
| `scripts/build_pre_review_risk_table.py`            | v0.6.6 预审风险表生成                      |
| `scripts/evaluate_pre_review_ranking.py`            | v0.6.6 预审排序后验验证                     |
| `scripts/build_demo_risk_case_table.py`             | v0.6.5 demo 风险样本表构建                 |
| `scripts/run_case_report.py`                        | 单病例 evidence-bottleneck artifact 生成 |
| `scripts/run_real_llm_safety_probe.py`              | guarded report safety probe         |

---

## 代表性分类结果

APTOS2019 糖尿病视网膜病变分级代表性结果：

| Backbone         | 设置                   | Accuracy | Macro-F1 | Weighted-F1 |   QWK |
| ---------------- | -------------------- | -------: | -------: | ----------: | ----: |
| Swin-Tiny        | lightweight baseline |    0.829 |    0.657 |       0.820 | 0.898 |
| ConvNeXt-Tiny    | lightweight baseline |    0.814 |    0.650 |       0.809 | 0.862 |
| ViT-B/16         | lightweight baseline |    0.818 |    0.646 |       0.814 | 0.876 |
| RETFound-MAE-CFP | official-like        |    0.804 |    0.583 |       0.789 | 0.866 |

这些分类结果用于支撑后续模型输出审计与失败样本分析，不作为严格 controlled leaderboard。

---

## 文档入口

| 文档                                                  | 内容                        |
| --------------------------------------------------- | ------------------------- |
| `CHANGELOG.md`                                      | 版本更新记录                    |
| `experiments/summary/v0_6_7c/README.md`             | v0.6.7c 结果说明              |
| `experiments/summary/v0_6_7c/v067c_key_findings.md` | v0.6.7c 中文结论              |
| `notes/v0.6.7_clinical_residual_risk_protocol.md`   | 临床残余风险审计协议                |
| `notes/v0.6.6_pre_review_risk_ranking_design.md`    | 预审排序设计                    |
| `notes/v0.6.6_leakage_audit.md`                     | 泄露审计                      |
| `docs/schema/case_findings_schema_v0_6.md`          | case findings schema      |
| `docs/safety/llm_report_safety_rule_boundaries.md`  | LLM report safety rule 边界 |

---

## 后续计划

| 版本     | 方向                                            |
| ------ | --------------------------------------------- |
| v0.7.0 | 外部 DR 验证协议冻结与数据预检查 |
| v0.8.0 | lesion concept / report verbalization adapter |

---

## 项目边界

* 不是临床诊断系统。
* 不是自动医学报告生成系统。
* 不提供治疗建议。
* 不替代医生复核。
* CAM / heatmap 不是病灶标注或病灶定位。
* clinical-risk proxy 不是真实临床终点。
* 当前 APTOS 实验结果不等价于真实医院部署验证。
* 所有输出都需要人工审核。

## 当前实验节点：v0.7.1 外部 DR 直接推理与复核排序

v0.7.1 在 v0.7.0 external DR protocol freeze 的基础上，使用 APTOS-trained frozen checkpoints 直接推理 IDRiD_data / MESSIDOR2 test split，并评估复核排序信号在外部数据上的危险错误富集能力。

主要产物：

- `scripts/run_v071_external_dr_direct_inference.py`
- `scripts/evaluate_v071_external_dr_review_ranking.py`
- `experiments/summary/v0_7_1/`
- `notes/v0.7.1_external_dr_direct_inference_and_review_ranking.md`

当前结论：

- 外部分类迁移存在明显域迁移压力，尤其 MESSIDOR2 上多模型预测分布偏向 0 类。
- 在此前提下，`gated_severe_prob_mass_only` 对 `vision_threatening_dr_miss` 显示稳定外部错误富集能力。
- `expected_gap_only` 对 `large_undergrading` 有一定富集能力，但外部稳定性弱于 vision-threatening miss 目标。
- v0.7.1 应解释为 external frozen-checkpoint error enrichment / residual risk analysis，不应解释为 clinical deployment validation。

