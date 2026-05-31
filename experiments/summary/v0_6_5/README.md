# v0.6.5 医院线下展示版

## 版本定位

v0.6.5 不再只是静态 artifact 汇总，而是面向医院线下交流的展示版本。

核心定位：

OphAgent 是一个眼科 AI 模型输出审计工具，帮助发现“模型自信但错误”“重症被低估”“决策边界模糊”等高风险样本，并将这些样本转化为人工复核优先级。

v0.6.5 的核心价值不是把已有 JSON / HTML 文件放到一起，而是展示从“普通模型预测结果”到“人工复核优先级队列”的转化过程。

本版本不声称临床诊断能力，不声称自动报告生成能力，也不声称完成真实临床安全验证。

## 核心展示入口

- `integrated_showcase.html`：医院线下展示主页面
- `demo_risk_case_table_cn.md`：中文风险样本表
- `demo_risk_case_table.csv`：机器可读风险样本表
- `demo_risk_case_summary.json`：风险样本统计摘要
- `high_priority_case_analysis.md`：高风险样本深度分析
- `pipeline_overview.md`：系统流程说明
- `raw_vs_guarded_example.md`：安全审计示例

## 15 张 demo 样本风险表

本版本使用 `demo_samples` 中每个 DR 等级 3 张图，共 15 张样本。目录名作为 weak label，ConvNeXt baseline 作为预测模型。

结果摘要：

- total_cases: 15
- correct_count: 11
- incorrect_count: 4
- accuracy_on_demo_samples: 0.733
- high_priority_human_review: 3
- human_review_recommended: 3
- routine_review: 9

风险类型包括：

- 重症被低估
- 相邻等级混淆
- 决策边界模糊
- 低置信度-预测正确
- 常规审核

## 线下展示顺序

建议展示顺序：

1. 项目定位：模型输出审计工具，不是临床诊断系统。
2. 15 张 demo 样本风险表：展示如何发现高风险样本。
3. 典型高风险样本深度分析：解释为什么建议医生优先复核。
4. 系统工作流：图像 → 模型预测 → 证据结构化 → 安全审计 → 风险标记 / 报告草稿。
5. 安全审计示例：展示诊断越权、CAM 夸大、临床用途越权如何被拦截。
6. 医院数据对接方案：5-20 张脱敏图像 + labels.csv 即可跑最小闭环。

## 重要限制

- demo_samples 数量较小，且属于展示样本，不是正式验证集。
- 目录名被用作 weak label，仅用于演示样本级审查流程。
- risk tags 是启发式规则，不代表正式医学评估。
- CAM 只作为模型关注区域辅助参考，不是病灶定位。
- 安全审计规则不是完整 hallucination detector。
- 所有输出都需要医生或研究人员复核。
