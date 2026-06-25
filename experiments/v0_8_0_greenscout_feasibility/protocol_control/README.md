# v0.8.0 Protocol-Control 实验说明

本目录整理 v0.8.0 阶段的 protocol-control 实验。当前主线已经从单一 GreenScout 可行性叙事，升级为 **scout-to-expert model orchestration / 模型中转台** 分析。

## 阅读入口

建议先读：

- `v0_8_0_protocol_control_summary.md`

该总结合并了两部分：

- v0.8.0d：scout ablation，不同 scout / expert / budget 组合的效果对比；
- v0.8.0e：actual forward-cost benchmark，真实模型前向成本与稀疏调用系统成本估算。

## 关键实验模块

### 1. 标准化预测文件

- `predictions/greenscout_three_model_standardized.csv`

该文件统一整理三类模型在 APTOS2019 test split 上的预测结果：

- `retfound_green_linear_probe`
- `convnext_tiny`
- `retfound_mae_cfp_official_protocol`

### 2. 稀疏调用与风险富集

- `sparse_invocation/`
- `risk_enrichment/`

这部分是早期 protocol-control 分析，用于检查 GreenScout 风格的 sparse invocation 是否能够相对随机策略富集错误样本和高风险样本。

### 3. Scout Ablation

- `scout_ablation/`

该模块比较不同 scout / expert 配置：

- A: Green scout -> ConvNeXt + RETFound official-protocol average
- B: Green scout -> ConvNeXt only
- C: Green scout -> RETFound official-protocol only
- D: ConvNeXt scout -> RETFound official-protocol only

主要文件：

- `scout_ablation/scout_ablation_key_findings.md`

核心结论：

- sparse routing 优于随机 expert invocation；
- 没有单一 scout 在所有预算和指标下全面最优；
- scout 的选择会改变 accuracy、risk capture、expert-induced errors 之间的权衡。

### 4. 实际前向成本测试

- `actual_cost/`

该模块在 APTOS2019 test split 上重新测量三条 online inference 链路的 single-GPU forward-only cost：

- RETFound-Green linear probe
- ConvNeXt-Tiny
- RETFound-MAE official-protocol

主要文件：

- `actual_cost/actual_cost_key_findings.md`

核心结论：

- 在当前 batch size=32、forward-only 设置下，ConvNeXt-Tiny 比 RETFound-Green 更快；
- RETFound-Green checkpoint 更小，但没有表现出 forward latency 优势；
- 50% expert-call budget 可以转化为 forward-cost 节省，但最优 cost-performance tradeoff 取决于 scout / expert 组合。

## 当前综合判断

当前证据不支持把 GreenScout 表述为最终低成本最优 scout。

更准确的表述是：

> v0.8.0 支持 scout-to-expert model orchestration / 模型中转台框架：不同 scout、expert 和 expert-call budget 组合，需要从 accuracy、risk capture、expert-induced errors 和 forward cost 四个维度联合评估。

## 当前边界

- actual cost benchmark 是 forward-only benchmark，不包含图像解码、Resize、Normalize、DataLoader workers、磁盘 I/O、请求调度、模型加载和并发服务成本。
- sparse system cost 是基于实测模型 forward cost 与 selected expert-call counts 的估算，不是真实服务压测。
- 当前验证范围是 APTOS2019 糖尿病视网膜病变五分类任务。
- 本目录是 protocol-control 阶段实验节点，不是正式 release。
