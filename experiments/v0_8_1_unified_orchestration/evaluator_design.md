# v0.8.1 Unified Orchestration Evaluator Design

## 1. 目标

构建一个统一 scout-to-expert orchestration evaluator。

目标不是继续手工扩展单个 GreenScout 组合，而是让后续新模型只需提供标准化 prediction CSV 与 forward-cost CSV，即可自动进入统一评测流程。

## 2. 编码原则

本阶段采用 minimal evaluator 设计：

- 优先复用 v0.8.0d/e 已验证逻辑；
- 不引入复杂类体系；
- 不做前端 UI；
- 不做服务化；
- 不做所有模型全组合暴力搜索；
- 输入输出全部落地为 CSV / Markdown；
- 必须保留 schema check、random baseline、oracle same-budget upper bound 和实验边界说明。

## 3. 标准输入

### 3.1 Model registry

路径：

- `configs/model_registry.csv`

字段：

- `model_name`
- `family`
- `role_hint`
- `prediction_csv`
- `cost_csv`
- `enabled`
- `notes`

### 3.2 Prediction CSV

每个模型必须提供统一字段：

- `dataset`
- `split`
- `image_key`
- `true_label`
- `model_name`
- `pred_label`
- `prob_0`
- `prob_1`
- `prob_2`
- `prob_3`
- `prob_4`
- `confidence`

Evaluator 内部补充：

- `margin`
- `entropy`
- `correct`

### 3.3 Cost CSV

每个模型成本表必须包含：

- `model_name`
- `mean_ms_per_image`
- `median_ms_per_image`
- `images_per_second`
- `pytorch_peak_allocated_mem_mb`
- `checkpoint_mb`
- `batch_size`
- `device`

## 4. 标准输出

输出目录：

- `outputs/`

输出文件：

- `single_model_summary.csv`
- `static_ensemble_summary.csv`
- `pairwise_complementarity.csv`
- `scout_candidate_ranking.csv`
- `expert_candidate_ranking.csv`
- `sparse_routing_curve.csv`
- `risk_event_enrichment.csv`
- `cost_performance_frontier.csv`
- `key_findings.md`

## 5. 第一版只做什么

v0.8.1 第一版只接入当前已有三模型：

- `retfound_green_linear_probe`
- `convnext_tiny`
- `retfound_mae_cfp_official_protocol`

第一版 evaluator 只实现：

1. 输入 schema check；
2. 单模型性能汇总；
3. 静态 ensemble；
4. pairwise complementarity；
5. scout / expert 初步 ranking；
6. sparse routing curve；
7. cost-performance frontier；
8. key findings 自动生成。

## 6. 暂不做什么

本阶段暂不做：

- Streamlit UI；
- 数据库；
- 服务部署；
- 模型自动推理；
- 外部数据集验证；
- 大规模 model zoo；
- 全模型全组合暴力搜索。

这些放到 v0.8.2 之后。
