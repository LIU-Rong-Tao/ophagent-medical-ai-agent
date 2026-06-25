# RETFound-Green Smoke Test Summary

## 结论

RETFound-Green 在当前 OphAgent 环境中可以稳定加载，并可对 demo_samples 15 张真实 CFP 图像导出 384 维 embedding。

## 实验设置

- 环境：当前 ophagent 环境
- GPU：RTX 4090
- 权重：checkpoints/retfound_green/retfoundgreen_statedict.pth
- 权重大小：约 83.35 MB
- 模型结构：vit_small_patch14_reg4_dinov2
- 输入尺寸：392 × 392
- 输出类型：embedding
- 输出维度：384
- 样本：demo_samples，5 类 × 每类 3 张，共 15 张

## 结果

- 成功样本数：15 / 15
- 成功率：100%
- embedding shape：1 × 384
- 平均单图推理耗时：约 6.02 ms
- 中位单图推理耗时：约 5.57 ms
- 最大单图推理耗时：约 7.97 ms
- 峰值显存：约 108.8 MB

## 边界

本结果只是 smoke test，证明 RETFound-Green 可加载、可推理、可导出 embedding。  
它不是 APTOS 五分类性能验证，也不能说明 GreenScout Router 已经成立。
