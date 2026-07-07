# v0.8.6 Interactive Model Hub

## 版本定位

v0.8.6 将既有 prediction replay 与受控模型接入合并为一个眼科模型中转台，包含：

- 受控服务器目录中的模型与产物状态汇总；
- 模型家族折叠、单模型操作与兼容性检查；
- 基于通用 timm recipe 的 ImageFolder 数据预检；
- 人工确认后提交独立后台微调任务，并追踪状态和日志；
- 路由模型与专家模型的受控组合 replay；
- 单路由模型与多路由模型门控；
- 0%–100% 专家调用预算曲线；
- 任务感知的研究评测与无标签临床演示隔离；
- 病例级路由解释、分页队列和模型结果对照。

本版本只对已登记的 timm recipe 开放真实微调提交；不自动启用 RETFound 等尚未验证的 Loader，不做全组合最优搜索，也不把训练任务输出写入冻结科研证据。

## 研究边界

v0.8.6 结果属于 interactive engineering replay，不作为正式科研结论。当前选择与评估都使用冻结 test predictions，不能据此宣称无偏模型选择结果。

验证集冻结、Random/Oracle/Learned Gate、bootstrap、统计检验和外部验证留到 v0.8.7。

## 多路由模型语义

多路由模型协议必须显式指定 `primary_scout_artifact_id`：

- 基础输出模型负责未调用专家时的最终输出；
- 其他路由模型只参与分歧与不确定性门控；
- 本版本不进行路由模型概率融合。

## 训练任务边界

- Streamlit 只生成配置、执行预检和提交后台任务，不在页面进程内执行训练循环。
- 数据目录必须为类别映射一致的 `train/val/test` ImageFolder。
- 微调前必须通过数据、类别数、recipe 和输出目录预检，并完成人工确认。
- 运行状态为 `queued/running/succeeded/failed/cancelled`，失败只影响当前任务。

## 成本口径

已有 prediction replay 本身不代表在线部署成本。页面展示的成本由既有 forward-only benchmark 估算：

- 总计算/顺序延迟：所有 Scout 成本之和 + Expert 调用率 × Expert 成本；
- 并行情景：Scout 成本最大值 + Expert 调用率 × Expert 成本。

并行情景是假设多执行器可并行运行的估算，并非当前单卡并发实测。100% budget 仅表示最终预测全部由 Expert 替换，在线成本仍包含已经运行的 Scout。

## 运行

```bash
python scripts/routing/run_interactive_model_hub.py \
  --config experiments/v0_8_6_interactive_model_hub/configs/protocol.yaml \
  --output-dir experiments/v0_8_6_interactive_model_hub/outputs \
  --dry-run

python scripts/routing/run_controlled_protocol.py \
  --config experiments/v0_8_6_interactive_model_hub/configs/controlled_runner.yaml \
  --resume

streamlit run app/model_hub_demo.py
```

## 固定发布产物

```text
model_hub_snapshot.csv
pairing_results.csv
case_routing_trace.csv
run_config.yaml
artifact_manifest.csv
report.html
```
