# v0.8.3 Glaucoma Scout-to-Expert Routing

## 目标

本版本只验证一个最小闭环：

Glaucoma task input
-> lightweight scout
-> uncertainty / high-risk routing
-> RETFound-DINOv2 adapted glaucoma expert
-> human review fallback

## 固定设置

- task: glaucoma severity classification
- dataset: Glaucoma_fundus
- expert: retfound_dinov2_glaucoma_fundus_expert
- first scout: convnext_tiny_glaucoma_scout

## 暂不做

- 不复现 FusionFM
- 不做 DINORET / RetiZero / VisionFM 接入
- 不做 MobileNet / EfficientNet / ResNet / Swin / ViT / Green 大矩阵
- 不做自动病种识别
- 不做 learned gate
- 不做复杂 UI

## 成功标准

uncertainty-based defer 相比 random defer 能更有效捕获 scout 错误或 advanced glaucoma 风险病例，并在较低 expert 调用率下接近 expert-only 表现。
