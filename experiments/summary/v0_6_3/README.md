# v0.6.3 Controlled Real LLM Provider Summary

## 版本定位

v0.6.3 在 v0.6.2 safety regression tests 和 audit metadata 的基础上，引入 controlled real LLM provider。

本版本目标不是证明真实 LLM 可以生成临床报告，而是验证真实 OpenAI-compatible provider 能否接入既有 guarded report pipeline，并继续经过 constrained prompt、RuleBasedSafetyChecker、safety_report.json 和 fallback 机制约束。

## 本目录包含什么

本目录保存 v0.6.3 real LLM smoke / summary run 的轻量展示产物，不包含完整病例图像、CAM 图片或完整 case artifact。

相关文件：

- `llm_raw_real_llm.md`：真实 LLM provider 原始输出
- `llm_checked_real_llm.md`：通过 RuleBasedSafetyChecker 后被接受的草稿
- `llm_guarded_real_llm.html`：由 checked draft 渲染得到的简化 guarded HTML
- `safety_report_real_llm.json`：真实 LLM 路径的 safety trace 与 audit metadata

## 运行结果

本次 real LLM summary run 的关键结果：

- provider: real_llm
- real_llm_used: True
- overall_pass: True
- fallback_triggered: False
- provider_version: v0.6.3-openai-compatible-provider
- checker_version: v0.6.2-rule-based-safety-checker
- safety_policy_version: v0.6.2-rule-based-safety-guard
- API key leakage check: contains_sk_key = False

## 安全策略

真实 LLM 输出不会直接作为可信医学报告使用。

v0.6.3 仍沿用 guarded generation 策略：

- constrained prompt 限制 evidence boundary
- RuleBasedSafetyChecker 执行生成后安全检查
- unsafe draft 触发 deterministic template fallback
- safety_report.json 记录 provider metadata、prompt_hash、checker_version、safety_policy_version 和 fallback decision
- API key 不进入 git、不进入 safety_report.json、不进入 summary artifacts

## 已实现内容

- OpenAI-compatible real LLM provider skeleton
- `--report-provider real_llm` CLI 接入
- real LLM provider 配置失败路径测试
- OpenAI-compatible response parsing 测试
- real_llm → renderer → safety_report → fallback 离线集成测试
- gpt-4o-mini 手动 smoke / summary run
- v0.6.3 real LLM summary artifacts

## 暂不包含内容

- LLM-as-judge
- 医学事实自动校验
- 图像质量自动判断
- 多轮报告编辑
- Web demo / deployment
- 临床级报告生成

## 说明

v0.6.3 的重点不是“让 LLM 写得更像医生”，而是验证真实 LLM provider 可以被纳入已有 guarded report workflow。

当前结果只能说明 controlled real LLM path 已经跑通，并且本次输出通过 rule-based safety guard。它不证明真实 LLM 输出在医学上正确，也不证明系统具备临床安全性。
