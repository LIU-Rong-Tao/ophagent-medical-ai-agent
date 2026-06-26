# v0.8.2 Model Onboarding + Automated Benchmark 关键结果

## 1. 当前阶段定位

v0.8.2 的目标不是简单加入 Swin-Tiny，而是验证 v0.8.1 unified orchestration evaluator 是否具备新增模型后的自动扩展能力。

本轮完成：

- registry-driven model onboarding；
- strict checkpoint loading；
- prediction schema validation；
- probability sanity check；
- single-model metric validation；
- multi-run forward-only benchmark；
- run-level latency diagnostic；
- 将新模型接入 v0.8.1 unified evaluator；
- 自动重算 single model、static ensemble、pairwise complementarity、sparse routing、cost frontier、candidate ranking 和 DR-specific risk enrichment。

## 2. Swin-Tiny onboarding 结果

新增模型：

- model_name: `swin_tiny`
- arch: `swin_tiny_patch4_window7_224`
- checkpoint: `experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/checkpoints/swin_tiny_patch4_window7_224.ms_in1k_best.pth`
- prediction CSV: `experiments/v0_8_1_unified_orchestration/inputs/predictions/swin_tiny_standardized.csv`

验证结果：

- strict_load_ok: True
- prediction_valid: True
- n: 1100
- Accuracy: 0.829091
- Macro-F1: 0.656707
- QWK: 0.898186
- n_error: 188

## 3. 多次 forward-only benchmark

Swin-Tiny 5 次 forward-only benchmark 结果：

- median mean_ms_per_image: 0.611007
- total_forward_ms: 672.107
- images_per_second: 1636.643
- peak allocated memory: 612.779 MB
- checkpoint size: 105.063 MB
- latency_cv_gt_10pct: False
- latency_has_run_outlier: False

说明：Swin-Tiny 成本稳定，与 ConvNeXt-Tiny 属于同一成本档，明显快于 Green 和 RETFound official-protocol。

MAD sensitive flag 可能出现，但它只表示在极稳定 latency 下的敏感统计提醒，不作为正式异常。

## 4. 四模型 unified evaluator 结果

当前模型池：

- `retfound_green_linear_probe`
- `convnext_tiny`
- `retfound_mae_cfp_official_protocol`
- `swin_tiny`

### 单模型排序

1. `retfound_mae_cfp_official_protocol`: Acc 0.848182 / Macro-F1 0.705362 / QWK 0.912891
2. `swin_tiny`: Acc 0.829091 / Macro-F1 0.656707 / QWK 0.898186
3. `convnext_tiny`: Acc 0.813636 / Macro-F1 0.649639 / QWK 0.861536
4. `retfound_green_linear_probe`: Acc 0.779091 / Macro-F1 0.639431 / QWK 0.865338

### 最佳静态 ensemble

当前最佳静态 ensemble 仍为：

- `convnext_tiny + retfound_mae_cfp_official_protocol`

结果：

- Accuracy: 0.854545
- Macro-F1: 0.718665
- QWK: 0.910170
- n_error: 160

Swin + RETFound 静态 ensemble 接近但略低：

- Accuracy: 0.853636
- n_error: 161

## 5. Sparse routing 新发现

Swin 加入后，当前最佳 sparse routing 组合变为：

- `swin_tiny -> convnext_tiny + retfound_mae_cfp_official_protocol`
- budget: 0.4
- policy: low_confidence / low_margin / high_entropy
- Accuracy: 0.857273
- Macro-F1: 0.721204
- QWK: 0.909559
- n_error: 157
- online_no_cache_ms_per_image: 2.207025
- above_random_p975: True
- oracle_accuracy: 0.870909
- gap_to_oracle_accuracy: 0.013636

这个结果超过当前最佳静态 ensemble，并且成本低于 dense expert ensemble。

## 6. Cost-performance frontier 更新

Swin 加入后，cost-performance frontier 出现新的优势路径：

| setting | accuracy | ms/image |
|---|---:|---:|
| dense ConvNeXt | 0.813636 | 0.478317 |
| ConvNeXt -> Swin | 0.830909 | 0.539418 |
| ConvNeXt -> Swin | 0.840909 | 0.600519 |
| Swin -> RETFound | 0.843636 | 0.962179 |
| ConvNeXt -> RETFound | 0.847273 | 1.180663 |
| ConvNeXt -> RETFound+Swin | 0.848182 | 1.302864 |
| Swin -> ConvNeXt+RETFound | 0.849091 | 1.409016 |
| Swin -> RETFound | 0.850000 | 1.664525 |
| Swin -> ConvNeXt+RETFound | 0.854545 | 1.808020 |
| Swin -> ConvNeXt+RETFound | 0.857273 | 2.207025 |

这说明新增 Swin 后，v0.8.1 的 cost-performance frontier 被实质性更新。

## 7. Candidate ranking 更新

### Scout ranking

当前 scout ranking：

1. `swin_tiny`
2. `convnext_tiny`
3. `retfound_green_linear_probe`

解释：

- Swin-Tiny 单模型性能高于 ConvNeXt；
- 成本仍处于轻量模型档；
- 作为 scout 接 ConvNeXt + RETFound 双专家时，形成当前最佳 sparse routing 结果。

### Expert ranking

当前 expert ranking：

1. `retfound_mae_cfp_official_protocol`
2. `swin_tiny`
3. `convnext_tiny`

解释：

- RETFound official-protocol 仍是当前最强 expert；
- Swin-Tiny 兼具较好单模型性能和低成本，具备 scout_or_expert 双重候选价值；
- ConvNeXt 仍是最低成本轻量模型，但新增 Swin 后不再是最优 scout。

## 8. DR-specific risk enrichment

新增 Swin 后，风险事件覆盖也被更新。

代表性结果：

- `large_undergrading_dr_specific`: ConvNeXt scout 在 50% budget 下可覆盖 40/41，recall=0.975610。
- `referable_dr_miss_dr_specific`: ConvNeXt scout 在 50% budget 下可覆盖 76/77，recall=0.987013。
- `severe_pdr_miss_dr_specific`: Swin scout 在 50% budget 下可覆盖 61/62，recall=0.983871。

需要注意：当前风险事件是基于 scout prediction 定义，因此不同 scout 的 event_total 不同，不能简单把 recall 跨 scout 直接解释为同一分母下的绝对医学优劣。

## 9. 当前结论

v0.8.2 证明了统一模型接入与评测框架是有效的：

1. 新模型只需通过 registry + standardized prediction + automated benchmark 接入。
2. evaluator 可以自动重算性能、成本、互补性、routing、frontier、ranking 和风险事件。
3. Swin-Tiny 的加入实质性改变了 scout ranking 和 cost-performance frontier。
4. 当前更强的模型编排方向是 `Swin-Tiny scout -> ConvNeXt + RETFound expert ensemble`。
5. GreenScout 进一步退化为一个有互补性但非成本最优、非排序最优的候选 scout。
6. 项目主线应继续表述为 scout-to-expert model orchestration，而不是 GreenScout 单点方案。

## 10. 当前边界

- 当前仍只在 APTOS2019 test 上验证。
- 当前模型均来自既有 APTOS 分类产物，不代表跨数据集泛化。
- Cost benchmark 是 single-GPU forward-only，不包含图像解码、transform、DataLoader、服务队列和并发。
- 当前 onboarding 脚本只支持 registry-declared timm classifier。
- 非 timm foundation model、embedding model、多模态模型需要单独 onboarding adapter。
