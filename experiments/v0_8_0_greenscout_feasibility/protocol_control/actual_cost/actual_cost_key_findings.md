# v0.8.0e Actual Forward-Cost Benchmark and Sparse System Estimate 关键结果

## 1. 实验设置

本轮实验在 APTOS2019 test split 上重新测量三条 online inference 链路的 forward-only cost，并基于 v0.8.0d scout ablation 的 selected_n 估算 sparse system cost。

成本口径：

- Green online：RETFound-Green encoder forward + sklearn linear-probe predict_proba。
- ConvNeXt online：ConvNeXt-Tiny classifier forward。
- RETFound online：RETFound-MAE official-protocol classifier forward，使用 timm `vit_large_patch16_224` + `global_pool='avg'` 严格加载 checkpoint。
- online no-cache：包含 scout 全量前向 + 被选中样本的 expert 前向估算。
- cached scout：假设 scout 输出已经存在，只统计被选中样本的 expert 前向估算。

当前结果仍是单 GPU forward benchmark 与 system-level estimate，不等同于完整生产服务中的 I/O、队列、并发和模型加载成本。

## 2. 单模型 online forward cost

| model_name | mean_ms_per_image | median_ms_per_image | images_per_second | pytorch_peak_allocated_mem_mb | checkpoint_mb |
| --- | --- | --- | --- | --- | --- |
| retfound_green_linear_probe | 1.8965 | 1.8825 | 528.1608 | 591.9570 | 83.3480 |
| convnext_tiny | 0.4861 | 0.4733 | 2090.6620 | 538.8994 | 106.2086 |
| retfound_mae_cfp_official_protocol | 3.5176 | 3.5115 | 284.7601 | 1479.0249 | 1157.1359 |

## 3. 关键系统成本估算

| scenario | setting | budget | selected_n | estimated_ms_per_image | estimated_images_per_second | models_called |
| --- | --- | --- | --- | --- | --- | --- |
| online_no_cache | green_only | 0.0000 | 0 | 1.8934 | 528.1608 | retfound_green_linear_probe |
| online_no_cache | convnext_only | 0.0000 | 0 | 0.4783 | 2090.6620 | convnext_tiny |
| online_no_cache | retfound_only | 0.0000 | 0 | 3.5117 | 284.7601 | retfound_mae_cfp_official_protocol |
| online_no_cache | experts_only_dense | 1.0000 | 1100 | 3.9900 | 250.6237 | convnext_tiny+retfound_mae_cfp_official_protocol |
| online_no_cache | all_three_dense | 1.0000 | 1100 | 5.8834 | 169.9695 | retfound_green_linear_probe+convnext_tiny+retfound_mae_cfp_official_protocol |
| online_no_cache | A_green_scout_to_convnext_retfound_avg | 0.5000 | 550 | 3.8884 | 257.1762 | retfound_green_linear_probe -> convnext_tiny+retfound_mae_cfp_official_protocol |
| cached_scout | A_green_scout_to_convnext_retfound_avg | 0.5000 | 550 | 1.9950 | 501.2475 | convnext_tiny+retfound_mae_cfp_official_protocol |
| online_no_cache | D_convnext_scout_to_retfound_only | 0.5000 | 550 | 2.2342 | 447.5913 | convnext_tiny -> retfound_mae_cfp_official_protocol |
| cached_scout | D_convnext_scout_to_retfound_only | 0.5000 | 550 | 1.7559 | 569.5202 | retfound_mae_cfp_official_protocol |
| online_no_cache | C_green_scout_to_retfound_only | 0.5000 | 550 | 3.6492 | 274.0307 | retfound_green_linear_probe -> retfound_mae_cfp_official_protocol |
| cached_scout | C_green_scout_to_retfound_only | 0.5000 | 550 | 1.7559 | 569.5202 | retfound_mae_cfp_official_protocol |

## 4. 当前主结论

1. 在当前 APTOS2019 test、batch size=32、single GPU forward-only benchmark 下，ConvNeXt-Tiny 是最快的 scout / classifier，约 0.478 ms/image；RETFound-Green linear probe 约 1.893 ms/image；RETFound-MAE official-protocol 约 3.512 ms/image。

2. RETFound-Green 的 checkpoint 更小，但本轮结果不支持“Green scout 具有 forward latency 优势”的说法。当前更稳妥的表述是：Green 是较小 checkpoint 的 scout 候选，但其实际 forward cost 需要与 ConvNeXt 等候选 scout 共同评估。

3. 50% expert-call budget 可以转化为 forward-cost 节省，但节省幅度取决于 scout 本身成本与 expert 组合。`D_convnext_scout_to_retfound_only` 在 50% budget 下约 2.234 ms/image，明显低于 experts-only dense 的 3.990 ms/image。

4. `A_green_scout_to_convnext_retfound_avg` 在 v0.8.0d 中 accuracy 较高，但在 online no-cache 场景下约 3.888 ms/image，已经接近 experts-only dense 的 3.990 ms/image。因此它更像高 accuracy 方案，而不是明显低成本方案。

5. 当前结果支持把问题从单一 GreenScout 叙事升级为 scout-to-expert model orchestration：不同 scout / expert / budget 组合形成不同的 accuracy-cost tradeoff，不能只凭单模型大小或单点 accuracy 判断最优部署策略。


## 5. 当前边界

- sparse system cost 基于 measured per-model forward cost 与 selected_n 估算，不是实际部署服务压测。
- 当前计时从 tensor batch 送入模型前开始，不包含 PIL 图像解码、Resize、Normalize、DataLoader workers、磁盘 I/O、请求排队、模型动态加载和并发调度成本。
- Green 与 ConvNeXt 的真实部署成本差异需要结合 batch size、显存占用和服务常驻方式解释。
