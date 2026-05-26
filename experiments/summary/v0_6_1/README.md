# v0.6.1 Guarded LLM Report Drafting Summary

## 版本定位

v0.6.1 在 v0.6.0 的 evidence-bottleneck case report prototype 基础上，引入一个确定性的 guarded generation 层。

这个版本的目标不是接入真实 LLM API，而是先验证一条更重要的链路：LLM 风格的报告草稿可以被约束、检查、审计，并在出现 unsafe claim 时安全回退到确定性模板报告。

一句话概括：v0.6.1 的成果不是“LLM 会写报告”，而是“LLM 生成过程开始可约束、可检查、可回退、可审计”。

## 核心流程

findings.json
→ constrained prompt
→ MockLLMProvider draft
→ RuleBasedSafetyChecker
→ checked LLM report or template fallback
→ safety_report.json

## 本目录包含什么

本目录保存的是从 v0.6.1 端到端 smoke test 中抽取出来的轻量展示产物，不包含完整病例图像和 CAM 图片，避免重复提交完整 case artifact。

### Safe Mock Case

safe mock 草稿通过了确定性安全检查，因此系统保留 LLM draft，并生成 checked report 与 guarded HTML。

相关文件：

- llm_raw_safe.md：MockLLMProvider 原始输出
- llm_checked_safe.md：通过 safety checker 后被接受的草稿
- llm_guarded_safe.html：简化版 guarded report HTML
- safety_report_safe.json：safe 路径的安全审计结果

结果：

- overall_pass: True
- fallback_triggered: False
- selected_output: llm_checked_safe.md

### Unsafe CAM Mock Case

unsafe_cam mock 草稿故意夸大 CAM / heatmap 证据，例如将 CAM 描述为 lesion localization。该输出应被 safety checker 拦截，并触发 template fallback。

相关文件：

- llm_raw_unsafe_cam.md：包含 unsafe CAM overclaim 的原始 MockLLMProvider 输出
- safety_report_unsafe_cam.json：unsafe 路径的安全审计结果

结果：

- overall_pass: False
- fallback_triggered: True
- flagged_claim_count: 2
- selected_output: deterministic template fallback

## 安全策略

v0.6.1 采用保守的 full-fallback 策略：

- 只要检测到任意 unsafe claim，就不对 LLM 草稿做局部修补。
- 系统直接回退到 deterministic template report。
- 原始 LLM 输出会保留，用于审计。
- safety_report.json 会记录触发的规则、flagged claims 和最终选择结果。

## 已实现内容

- constrained prompt construction
- deterministic MockLLMProvider
- rule-based post-generation safety checking
- explicit safety trace artifact
- safe fallback to template report
- run_case_report.py 支持 --report-provider 与 --mock-llm-mode

## 暂不包含内容

- 真实 LLM API 接入
- LLM-as-judge
- 批量 case 优化
- 卡片式 guarded HTML UI
- Web demo 或部署

## 说明

v0.6.1 的 HTML 只是辅助展示，不是本版本的核心卖点。本版本真正的展示重点是 safety trace：safe case 如何通过检查，unsafe case 如何被拦截并回退。

