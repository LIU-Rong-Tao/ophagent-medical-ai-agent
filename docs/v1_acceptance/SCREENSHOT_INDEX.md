# OphAgent V1 页面验收索引

验收时间：2026-07-26。数据仅使用公开 validation 图像和脱敏演示编号。浏览器控制台在 localhost、院内局域网、1280×720 和 1440×900 场景均无错误。

| 截图 | 页面/场景 | 输入与操作 | 预期及实际结果 | 结论 |
|---|---|---|---|---|
| `01_queue_current_case_1280x720.png` | 病例队列与当前病例 | APTOS validation 场景 | 队列、当前病例、离线/研究边界清晰；无溢出 | 通过 |
| `02_multi_image_switch_1280x720.png` | 多图切换入口 | 选择 DR-V-0002 | 显示两张公开图像页签和缩放控件 | 通过 |
| `03_input_model_qualification_1280x720.png` | 输入、结构化信息、模型资格 | 单张 APTOS 图像 | 输入完整；模型资格、成本、阻塞原因同表展示 | 通过 |
| `04_model_results_risk_1280x720.png` | Scout 输出与错误风险 | 三个冻结 prediction asset | 显示概率输出、entropy、margin、模型分歧和任务代理 | 通过 |
| `05_research_route_simulation_1280x720.png` | Scout/Expert 研究模拟 | 冻结 validation trace | 明示 `route_eligible=false`、研究模拟、调用比例和采用输出 | 通过 |
| `06_review_actions_saved_1280x720.png` | 人工审阅 | 接受、修改、不确定、加入复核队列、保存 | 控件完整，保存反馈明确 | 通过 |
| `07_report_trace_1280x720.png` | 结构化报告与 trace | 展开调用轨迹 | 可追溯工具、协议、commit、资产哈希和成本口径 | 通过 |
| `08_refresh_persistence_1280x720.png` | 刷新持久化 | 保存后刷新并返回工作台 | 已保存审阅状态保留 | 通过 |
| `09_glaucoma_readonly_1280x720.png` | 青光眼只读场景 | 青光眼 validation 资产 | 三模型输出、分歧和风险代理可见；不读取 Test | 通过 |
| `10_structured_error_1280x720.png` | 结构化故障 | offline-only 资产请求原图推理 | 返回 `QUALIFICATION_BLOCKED`；后续为 `UPSTREAM_FAILED` | 通过 |
| `11_workstation_1440x900.png` | 常用桌面宽度 | localhost 单机模式 | 队列、图像和导航布局稳定，无异常空白或遮挡 | 通过 |
| `12_multi_image_switch_1440x900.png` | 多图显示与切换 | 切换到图像 2 | 图像完整显示，缩放与页签对应清晰 | 通过 |
| `13_lan_offline_mode_1280x720.png` | 院内局域网离线模式 | `OPHAGENT_BIND_MODE=lan` | 页面明确显示院内局域网模式，运行正常 | 通过 |

截图目录：`docs/v1_acceptance/screenshots/`

视觉自查修复：

- 将研究路由结果从易截断的 KPI 控件改为稳定宽度的结果卡。
- 缩短病例队列已保存状态文案，避免窄列换行。
- 收紧模型资格表列数，保留资格、成本和阻塞原因。
- 修复总览刷新时部分模型缺失 `task_inference_ready` 引发的页面错误。

不阻塞 V1 的限制：

- 1280×720 下长页面需要纵向滚动。
- Streamlit 原生表格在极窄窗口下仍需横向滚动。
- 多图目前使用页签切换，不提供影像配准或同步缩放。
