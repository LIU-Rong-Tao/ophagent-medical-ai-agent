# OphAgent 受控编排 V2 架构决策记录

- 状态：已接受
- 版本：`ophagent.controlled_orchestration.v2`
- 基线提交：`ef531c04ec6667331011b090f41327af20a6e2f5`
- 范围：Route Qualification Benchmark v1.1、规则状态机和本地控制模型比较

## 决策

本轮在现有 Model Hub 内形成“可扩展的单体架构”。不建设通用插件平台、
微服务、消息队列、动态发现或通用 DAG；也不复制 registry、Demo、runner 或
evaluator。

统一领域实体如下：

- `TaskSpec`：任务、数据集、模态、标签空间、主指标、适配类型和研究风险语义；
- `ModelCapability`：模型资产、任务能力、prediction、推理入口和可比成本协议；
- `RouteQualification`：共享资格服务的执行层级、证据标签和阻塞原因；
- `CaseState`：稳定状态码、工具结果、预算、已完成步骤和人工决定；
- `ControllerProposal`：三动作提议、固定原因码、Schema 版本和参数。

四个稳定扩展点如下：

1. Task Profile / `TaskAdapter`：输入检查、任务指标、研究风险代理和报告标签；
2. Model Adapter / `ModelRuntimeAdapter`：复用既有预处理、推理和标准概率输出；
3. Tool Contract：复用现有工具白名单、资格、失败停止和脱敏 trace；
4. Controller Interface / `ControllerAdapter`：只接收脱敏状态并提出三动作。

`rule_controller` 与 `local_llm_controller` 使用同一 Controller Interface。4B 与
27B 复用同一个 `LocalLLMController`，只切换模型配置、endpoint 和提示模板。

## 依赖方向

```text
Streamlit UI
  → Application / Agent Runtime
  → Case State Machine + Qualification Service
  → Tool Contracts
  → Task / Model Adapters
  → Unified Model Hub Index + Existing Frozen Assets
```

强制约束：

- Gate 和 State Machine 不导入 Streamlit；
- Controller 不读取实验目录、封存 Test、原图或私有路径；
- UI 不计算资格，只读取统一 ViewModel；
- Model Adapter 不授予路由资格；
- LLM 不直接调用工具；
- 主状态机不包含具体数据集或模型名称分支。

单一事实来源：

- 任务和模型能力来自 `build_model_hub_index()` 的 typed projection；
- 路由资格只由共享 `evaluate_route_qualification()` 和 Qualification Service 裁决；
- 病例状态只由 `CaseStateStore` 原子持久化；
- 允许动作只由状态机生成；
- Controller 只提议，Gate 负责 Schema、状态、资格、预算和权限复核；
- 模型资产页、资格页、病例工作台和报告读取同一 ViewModel。

## 兼容与迁移

- v1 合同、16 条冻结路由、Test 概率、route trace 和历史结果保持只读；
- v1.1 使用新目录和新 Schema，不覆盖旧 CSV；
- 新字段缺失时使用保守默认：资格降为回放或转人工，绝不推断在线资格；
- 旧 `ophagent.case_review_state.v1` 只迁移人工审阅结论，不推断已完成工具调用；
- 所有实体记录 TaskSpec、资格政策、Controller、CaseState 和路由协议版本；
- `clinical_route_eligible` 继续只允许外部治理人工授予，本轮始终为 `false`。

## 已识别的基线问题

- v1 使用路径字符串识别 Validation，可能把名称含 `validation` 的冻结结果误作
  选择证据；v1.1 必须使用唯一协议身份和明确 stage，并审计 Validation/冻结 SHA
  是否相同；
- 不同 batch、warmup、重复次数或硬件口径的成本不得混合排名；
- TRHD59 的弱单观测标签不能使用 ordinal undergrading 代理作为研究风险语义；
- 现有工作台只持久化人工结果，管线 payload、状态机和幂等缓存不能跨重启；
- 现有 `LocalActionProposer` 只是占位，缺少 Schema 和二次门控。

## 验收

最小架构测试必须证明：

- 新 `TaskSpec` 通过 TaskAdapter 接入，无需修改状态机；
- Fake `ModelRuntimeAdapter` 接入，无需修改 Agent；
- RuleController 与 MockLocalLLMController 可互换；
- 非法 Controller 提议必被 Gate 阻断；
- 模型能力、资格页、病例工作台和报告对同一资格读取一致。

里程碑 A、B、C 均需实际运行和验证。本 ADR 只约束技术选型和扩展边界，
不得作为缩减统计验证、状态机闭环或控制器比较的理由。
