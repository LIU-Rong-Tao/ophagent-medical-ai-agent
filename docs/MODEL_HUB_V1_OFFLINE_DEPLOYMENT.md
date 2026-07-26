# OphAgent V1 离线部署与工具契约

## 边界

OphAgent V1 是“工具化模型中转台 + 离线病例审阅工作台”。运行时只读取本地模型、Registry、任务契约、冻结 prediction asset 和研究路由 trace；不访问外部 API，不下载模型，不提供诊断、治疗或患者分流建议。

已有概率资产只允许只读审计。只有 `online_case_inference_ready` 模型才能接收新病例原图。所有现有路由均为 `research_simulation`，且 `route_eligible=false`。

## 六项工具

| 工具 | V1 状态 | 复用入口 | 关键门禁 |
|---|---|---|---|
| `case_input.validate` | 已实现 | 任务契约与本地图像解码 | 禁止 Test；1–8 张图；本地可读 |
| `model_registry.inspect` | 已实现 | 现有模型发现与任务资产 Registry | 返回三层资格、成本和阻塞原因 |
| `model_inference.run` | 有条件实现 | 现有单图推理入口 | 仅 `online_case_inference_ready` |
| `prediction_asset.validate` | 已实现 | 正式 prediction asset 校验 | 只读 validation；Test 锁定 |
| `result_risk_audit.run` | 已实现 | 现有置信度、entropy、margin 计算 | 仅模型输出错误风险，不推断临床后果 |
| `routing_protocol.evaluate` | 已实现 | 冻结 validation route trace 与 protocol | `research_simulation`；不授予路由资格 |

统一请求包含 `tool_name`、`request_id`、`trace_id`、任务、场景和参数；统一响应包含状态、错误码、结果、资格证据和时间。任一工具失败后，后续调用返回 `UPSTREAM_FAILED`，不会继续越权执行。Trace 不记录原始患者标识或图像绝对路径。

资格必须独立判断：

- `analytical_asset_only`：只有冻结概率或研究资产，只能审计。
- `offline_batch_inference_ready`：具备已验证批量任务链，不等于单病例调用。
- `online_case_inference_ready`：具备新病例原图单病例推理入口。

## 启动

在项目环境中执行。

localhost 单机模式：

```bash
export OPHAGENT_OFFLINE_MODE=1
export OPHAGENT_BIND_MODE=localhost
streamlit run app/model_hub_demo.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

院内局域网模式：

```bash
export OPHAGENT_OFFLINE_MODE=1
export OPHAGENT_BIND_MODE=lan
streamlit run app/model_hub_demo.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

局域网访问仍需由院内防火墙和账号权限限制。两种模式均不得配置公网模型下载或外部服务调用。

## 数据与追溯

审阅状态、结构化报告和工具 trace 写入：

```text
experiments/model_hub/runtime/case_review_v1/
```

该运行目录不进入 Git。报告记录模型、协议、代码 commit、prediction asset/checkpoint 哈希、成本口径、研究模拟状态和人工审阅结论；不显示或导出原始患者路径。

V1 演示仅使用公开 validation 数据。APTOS 和青光眼场景不会读取未解锁 Test。TRHD59 的既有结果和人工复核包保持只读，未接入本工作台执行链。

## V2 前置缺口

- 将更多模型接入经过验证的单病例原图推理入口。
- 为工具权限、人工审批、会话恢复和并发任务建立显式状态机。
- 增加更严格的身份、访问控制、数据脱敏和部署审计。
- 在独立数据与人工复核完成前，继续保持 `route_eligible=false`。
