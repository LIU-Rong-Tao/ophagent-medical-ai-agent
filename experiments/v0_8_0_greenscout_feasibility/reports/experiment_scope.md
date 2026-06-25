# v0.8.0 GreenScout Routing Feasibility Audit 实验边界

## 核心问题

本阶段验证的是：

低成本 RETFound-Green Scout + 现有专家模型 是否存在可行的成本感知执行前路由空间。

不是验证已有六个 backbone 是否能互补，也不是训练正式 Router。

## 主实验最小模型池

1. RETFound-Green：低成本 Scout 候选
2. ConvNeXt-Tiny：现有轻量 CNN 专家 / 普通监督模型对照
3. RETFound-MAE official-like：现有大型眼科 foundation model 专家

## 已有六模型预测表的定位

已有六模型 prediction CSV 只作为辅助 dry-run：

- 验证 CSV 标准化流程
- 验证 oracle / pairwise overlap / unique correction 脚本
- 初步观察已有专家池是否完全同错

它不能作为 GreenScout 成立的主证据。

## Go/No-Go 主判断必须依赖 Green

继续条件：

1. RETFound-Green 能稳定加载并导出 embedding 或 logits；
2. RETFound-Green 相比专家模型有实际推理成本优势；
3. Green 与至少一个专家模型存在错误互补；
4. Green 加入后 Oracle 上限明显超过最佳单模型；
5. Green 的 confidence / margin / entropy 对自身错误有一定识别能力。

停止条件：

1. Green 无法稳定加载；
2. Green 在本服务器上没有明显成本优势；
3. Green 与专家模型错误高度重叠；
4. Green 加入后 Oracle 几乎不提升；
5. Green 无法形成可用于后续路由的输出信号。
