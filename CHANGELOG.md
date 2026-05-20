# 更新日志

---

## v0.5.1 - Multi-metric Benchmark Evaluation

### 新增

- multi-metric benchmark evaluation
- QWK evaluation
- per-class F1 analysis
- prediction entropy analysis
- top1-top2 margin analysis
- confusion matrix generation

### 新增文件

```text
scripts/build_benchmark_table.py

experiments/summary/v0_5_1/benchmark_metrics.csv
experiments/summary/v0_5_1/per_class_f1.csv
experiments/summary/v0_5_1/confusion_matrices/
experiments/summary/v0_5_1/metrics_update.md
```

### Benchmark

| Backbone | Accuracy | Macro-F1 | Weighted-F1 | QWK |
|---|---:|---:|---:|---:|
| ConvNeXt-Tiny | 0.814 | 0.650 | 0.809 | 0.862 |
| Swin-Tiny | 0.829 | 0.657 | 0.820 | 0.898 |
| RETFound-MAE-CFP | 0.790 | 0.552 | 0.769 | 0.834 |

### 当前观察

- Swin-Tiny 在当前 benchmark 中表现最稳定
- ConvNeXt-Tiny 在 Severe DR 类别上表现相对更稳定
- RETFound-MAE-CFP 展现出不同的 uncertainty characteristics 与 class-wise behavior

### 当前限制

- 当前 benchmark 仍基于 single-seed evaluation
- 不同 backbone 的 training protocol 并不完全一致
- 当前结果更适合作为 representation behavior observation
- 尚未形成严格 controlled benchmark leaderboard

---

## v0.4.2 - Benchmark Infrastructure Cleanup

### 新增

- Swin-Tiny checkpoint metadata
- experiment version metadata
- benchmark artifact consistency improvements

### 修复

- 修复 Streamlit demo 版本号不一致问题
- 修复 unified evaluation metrics 路径
- 修复 summary builder 中 `Version: None`
- 修复 benchmark artifact relative path consistency

### 改进

- experiment-root relative artifact path
- summary builder version support
- benchmark release consistency
- benchmark portability 与 reproducibility

---

## v0.4.1 - Second Backbone Baseline

### 新增

- Swin-Tiny baseline
- unified evaluation schema
- backbone comparison summary

### Benchmark

| Backbone | Test Accuracy | Macro F1 |
|---|---:|---:|
| ConvNeXt-Tiny | 0.8136 | 0.6496 |
| Swin-Tiny | 0.8291 | 0.6567 |

### 当前限制

- 当前 benchmark 仍为 single-seed evaluation
- 尚未形成 formal benchmark leaderboard

---

## v0.4.0 - Experiment Summary Builder

### 新增

- `build_experiment_summary.py`
- unified experiment summary artifacts
- training/evaluation aggregation
- benchmark summary generation

### 输出

```text
summary.csv
summary.md
class_mapping.csv
training_curve_summary.csv
```

## v0.3.0 - Lightweight Agent Runner

### 新增

- reusable `run_agent(...)`
- provider abstraction
- structured findings integration
- optional OpenAI reasoning
- unified workflow pipeline

### Workflow

```text
image
  ↓
run_agent(...)
  ↓
classification
  ↓
structured findings
  ↓
reasoning
```

---

## v0.2.x - Workflow Demo

### 新增

- unified Streamlit demo
- Grad-CAM / HiResCAM gallery
- structured findings
- lightweight VL reasoning
- workflow integration