# v0.6.5 Pipeline Overview

## 一句话定位

OphAgent 是眼科模型输出审计与失败样本发现原型，不是临床诊断系统，也不是自动医学报告生成系统。

## 展示链路

1. 输入眼底图像
2. 分类模型输出 prediction.json
3. CAM overlay 作为 weak visual evidence
4. findings.json 整理结构化证据
5. validation.json 检查 claim-level traceability
6. real LLM 生成 guarded draft
7. safety_report.json 记录安全审计
8. unsafe mock 示例展示过度声明如何被拦截
9. fallback 避免输出高风险报告草稿

## 核心边界

- 模型预测不是临床诊断。
- CAM 不是病灶定位。
- report draft 不是临床报告。
- 当前系统只用于 research/demo。
- 所有输出都需要人工审核。
