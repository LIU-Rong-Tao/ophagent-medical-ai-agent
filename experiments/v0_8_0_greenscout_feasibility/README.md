# v0.8.0 GreenScout 可行性验证

## 1. 实验目标

当前目标是做一个 **Go/No-Go 可行性验证**，判断“低成本 Scout + 现有专家模型”这条路线是否值得继续投入。

主要回答四个问题：

1. 现有多个眼科模型的错误是否具有互补性。
2. Oracle 专家选择上限是否明显高于最佳单模型。
3. RETFound-Green 是否能够在服务器上稳定加载和推理。
4. RETFound-Green 是否在实际推理耗时、显存或存储上具有低成本优势。

如果这些条件不成立，后续不继续训练 Router。

---

## 2. 当前阶段不做什么

- 不训练复杂 Router。
- 不声称已经形成论文方法。
- 不做临床分流。
- 不定义临床阈值。
- 不把 RETFound-Green 的训练成本优势直接等同于本服务器上的部署成本。
- 不把 smoke test 当成五分类性能验证。

---

## 3. 实验目录

```text
experiments/v0_8_0_greenscout_feasibility/
├── configs/
│   └── config.yaml
├── registry/
│   ├── existing_prediction_files.txt
│   └── model_artifact_registry.csv
├── predictions/
│   └── existing_models_standardized.csv
├── complementarity/
│   ├── existing_model_complementarity.csv
│   ├── pairwise_error_overlap.csv
│   ├── unique_corrections.csv
│   └── oracle_upper_bound.csv
├── green_smoke/
│   └── retfound_green_smoke_test.csv
├── cost/
│   └── inference_cost_table.csv
├── reports/
│   └── go_no_go_summary.md
└── logs/
```

---

## 4. 第一阶段任务

### 4.1 现有模型互补性分析

先使用 OphAgent 已有的 ConvNeXt、Swin、RETFound 等模型 prediction CSV，不新增模型环境。

计算内容：

- best single model
- average ensemble
- oracle expert selection
- pairwise error overlap
- unique correction count
- disagreement cases

目的：

判断现有模型是否真的“错得不一样”。如果所有模型基本错在同一批图像上，则路由方向没有继续价值。

---

### 4.2 RETFound-Green 可运行性测试

RETFound-Green 第一阶段只做 smoke test，不直接参与五分类性能比较。

检查内容：

- checkpoint 是否能加载
- 输入图像是否能正常预处理
- embedding 是否能导出
- 输出维度是多少
- 单图推理耗时
- batch 推理耗时
- 峰值显存
- 是否能导出统一 CSV

注意：

如果 RETFound-Green 只输出 embedding，没有现成 APTOS 五分类 head，则不能直接和 ConvNeXt / RETFound-MAE 的五分类 Accuracy 比较。后续需要统一训练 linear probe 或分类头。

---

### 4.3 成本实测

必须实测本服务器上的实际成本，包括：

- 单图推理耗时
- batch 推理吞吐
- 峰值显存
- 模型大小
- 输出类型

不能只引用论文中的训练成本或 embedding 加速结果。

---

## 5. Go/No-Go 判断

### 5.1 继续条件

满足以下条件，才进入下一阶段：

1. RETFound-Green 能稳定加载并导出 embedding。
2. RETFound-Green 或其他 Scout 候选相比专家模型有明确成本优势。
3. 现有模型之间存在稳定的独有纠错样本。
4. Oracle expert selection 明显高于最佳单模型。
5. average ensemble 没有完全吃掉 Oracle 空间。

### 5.2 停止条件

出现以下情况，则停止本方向：

1. RETFound-Green 无法稳定加载或输出不稳定。
2. RETFound-Green 在实际服务器上没有明显成本优势。
3. 现有模型错误高度重叠。
4. Oracle 上限几乎不超过最佳单模型。
5. RETFound-Green 短期无法形成统一五分类 prediction CSV，且只做 embedding 无法推进后续验证。

---

## 6. 核心口径

本实验当前只是 **低成本执行前路由的可行性探索**。
