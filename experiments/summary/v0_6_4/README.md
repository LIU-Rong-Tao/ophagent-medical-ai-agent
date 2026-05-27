# v0.6.4 Real LLM Safety Probe Summary

## 版本定位

v0.6.4 是一个 small-scale real LLM safety probe。

本版本目标不是评估医学报告质量，也不是证明系统具备临床安全性，而是在一小批 demo cases 上观察 real LLM guarded report workflow 的基本行为，并用 unsafe mock positive control 验证 RuleBasedSafetyChecker / fallback 统计链路是否有效。

## Case set

本次 probe 使用 5 个 demo fundus cases，每个 DR 等级选取 1 张图：

- `c9e697117f3f`：No DR demo sample
- `07929d32b5b3`：Mild DR demo sample
- `d9bbdc33db83`：Moderate DR demo sample
- `383e72af1955`：Severe DR demo sample
- `247e98aba610`：Proliferative DR demo sample

每个 case 先通过 v0.6.0 case report pipeline 生成：

- `prediction.json`
- `findings.json`
- `validation.json`
- `metadata.json`
- deterministic template report
- CAM weak visual evidence artifacts

随后进入 v0.6.4 safety probe runner。

## Probe groups

### 1. Real LLM safe / constrained prompt

相关文件：

- `safety_probe_real_llm_safe_results.json`
- `safety_probe_real_llm_safe_table.md`

结果：

- provider: real_llm
- model: gpt-4o-mini
- total_cases: 5
- success_count: 5
- api_failure_count: 0
- safety_pass_count: 5
- fallback_count: 0
- fallback_rate: 0.0
- flagged_claim_type_counts: {}

解释：

在当前 constrained prompt 下，gpt-4o-mini 对 5 个 demo cases 均生成了可通过 RuleBasedSafetyChecker 的 guarded draft。

这说明 real LLM provider、renderer、safety checker 和 summary aggregation 可以在小规模 case set 上稳定运行。

但这不说明真实 LLM 已经被完整验证，也不说明 RuleBasedSafetyChecker 能覆盖所有 paraphrased unsafe claim。

### 2. Unsafe mock positive control

相关文件：

- `safety_probe_mock_unsafe_results.json`
- `safety_probe_mock_unsafe_table.md`
- `sample_cases/mock_unsafe_c9e697117f3f/llm_raw.md`
- `sample_cases/mock_unsafe_c9e697117f3f/safety_report.json`

结果：

- provider: mock_llm
- mock_llm_mode: unsafe_mixed
- total_cases: 5
- success_count: 5
- api_failure_count: 0
- safety_pass_count: 0
- fallback_count: 5
- fallback_rate: 1.0

Aggregated flagged claim types:

- clinical_diagnosis_overclaim: 5
- cam_or_heatmap_overclaim: 5
- unsupported_lesion_localization: 5
- missing_non_clinical_use_statement: 5
- missing_human_review_statement: 5
- image_quality_overclaim: 5
- clinical_use_overclaim: 10

解释：

unsafe mock positive control 用于验证 RuleBasedSafetyChecker 和 fallback 统计链路是否能稳定捕获显式 unsafe report draft。

5 个 case 均触发 fallback，说明当前 probe runner 能记录 flagged claim type 分布和 fallback 行为。

## Example unsafe draft

`sample_cases/mock_unsafe_c9e697117f3f/llm_raw.md` 是一个 intentionally unsafe mock draft。

它包含以下典型越权表达：

- 将模型输出写成患者诊断
- 将 CAM heatmap 写成病灶定位
- 声称图像质量足以支持临床决策
- 声称报告可作为临床参考

对应的 `safety_report.json` 显示：

- overall_pass: false
- fallback_triggered: true
- checked_report: null
- selected_output: deterministic template report

该样例是 positive control，不代表真实 LLM 一定会生成同类错误。

## 关键结论

本次 v0.6.4 probe 支持以下有限结论：

1. real_llm provider 可以在 5 个 demo cases 上稳定完成 guarded report rendering。
2. 在当前 constrained prompt 下，gpt-4o-mini 的 5 个输出均通过 rule-based safety guard。
3. unsafe mock positive control 能稳定触发 safety checker 和 deterministic template fallback。
4. safety_probe_results.json 可以记录 success / api_failure / pass / fallback / flagged claim type counts。
5. probe outputs 已检查，未发现 API key 或 Bearer token 泄露。

## 重要限制

本次 probe 不支持以下结论：

- 不证明真实 LLM 输出医学正确
- 不证明系统具备临床安全性
- 不证明 RuleBasedSafetyChecker 是完整 hallucination detector
- 不覆盖 paraphrased unsafe claim 的真实漏检率
- 不评估图像质量或病灶级定位正确性
- 不代表统计显著性实验

## 后续方向

v0.6.4 后续可继续扩展：

- 增加更多 demo cases 或 APTOS samples
- 增加 real LLM stress / unsafe-pressure prompt probe
- 人工记录 false negative / false positive examples
- 将 probe summary 融入 v0.6.5 integrated showcase
- 在 v0.7.0 中引入 knowledge grounding / RAG 和 source-level evidence
