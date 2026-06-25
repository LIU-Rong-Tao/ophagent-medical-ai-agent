# v0.8.0 demo_samples 稳健成本摘要

## 说明

本表基于 demo_samples 15 张图像。由于单次测试中第一张图可能包含 CUDA kernel / 缓存残余冷启动开销，因此同时报告 raw latency 和 excluding-first latency。当前结果只用于 smoke/cost feasibility，不作为最终推理基准。

## 成本摘要

| model | role | checkpoint MB | img size | median ms raw | median ms excl. first | peak mem MB |
|---|---|---:|---:|---:|---:|---:|
| retfound_green | scout | 83.35 | 392 | 5.57 | 5.62 | 108.78 |
| convnext_tiny | existing_expert | 106.21 | 224 | 5.83 | 5.81 | 130.91 |
| retfound_mae_cfp_official_like | existing_expert | 1157.14 | 224 | 8.08 | 8.08 | 1175.76 |

## 初步判断

- RETFound-Green 已通过当前环境加载与真实图像 embedding smoke test。
- 相比 RETFound-MAE official-like，RETFound-Green 在 checkpoint 大小和显存上有明显优势。
- 相比 ConvNeXt-Tiny，RETFound-Green 的延迟同量级，显存和 checkpoint 略低；但 Green 当前输出为 embedding，不是五分类 logits。
- 当前不能把 demo_samples 15 张图的 latency 当最终部署成本，应补充重复测量或全量测试。
