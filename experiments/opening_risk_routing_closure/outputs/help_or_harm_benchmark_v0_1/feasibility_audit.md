# OphAgent Help-or-Harm 病例级可行性审计与 Benchmark v0.1

结论：**CONDITIONAL_GO**。

病例概率资产可一一对齐，DeepDRiD 具备患者分组开发证据；但 APTOS 冻结模型使用的 256×256 派生输入存在跨 split 字节级重复且缺少患者/眼别标识，全部冻结 Test 也仅能回顾性使用。

## 1. 审计范围

- 共枚举 210 条有向单 Scout→单 Expert 路线：APTOS 90、DeepDRiD 冻结迁移 30、DeepDRiD 原生适配 90。
- 共核对 390 个路线×split 对齐单元；异常单元 0。
- 所有路线均由冻结概率资产做只读笛卡尔枚举；未根据 Test 筛选路线、特征、阈值或预算。

## 2. 图像、患者与划分泄漏

- 本报告中的 APTOS 图像是冻结模型实际使用的 256×256 派生输入，不是对原始高分辨率 APTOS 文件的溯源结论。
- APTOS：123 个 SHA256 完全重复组（251 个文件），其中 74 个跨 split；30 个重复组标签冲突，其中 15 个同时跨 split。患者与眼别标识不可用。
- DeepDRiD：0 个完全重复组，患者跨 split 数 0；患者、眼别和图像标识覆盖完整。
- 感知哈希只产生“待确认候选”：APTOS 486 对，DeepDRiD 24 对。它们不被当作已证实重复，只进入敏感性分析。
- 主队列不修改原始 split，而是在 Benchmark 视图中排除跨 split 完全重复、标签冲突，并将同 split 完全重复固定为一个分析单位。

## 3. 标签与任务兼容边界

- APTOS 与 DeepDRiD 均可映射到固定的 DR 0–4 有序标签，因此 corrected/introduced 的定义可复用。
- 两者标签来源、采集域、适配设计和预处理不同，禁止直接混合模型排名、成本排名或把 dataset_id 当成预测特征。
- `dangerous_introduced` 只是 grade≥3 被降为 <3 的错误代理，不是临床伤害终点。

## 4. 合法预咨询特征与基本可识别信号

- 特征仅来自 Scout 概率、熵、margin、重症概率质量、其他开发折形成的 Expert 历史画像，以及无标签 Scout 分布偏移信号。
- 开发病例使用确定性分组五折画像；DeepDRiD 按患者分组，APTOS 按完全图像组分组。回顾性病例只使用完整开发折画像。
- 当前病例 Expert 输出、Expert embedding、Test 派生特征/阈值、dataset_id、身份字段、私有路径和未计成本的额外 Scout 均不进入特征合同。
- 为避免把“识别 Scout 犯错”误称为 Help-or-Harm，主信号只看：Scout 已错病例中的 corrected，以及 Scout 正确病例中的 introduced；全病例 AUROC 仅保留为次要描述。
- `aptos_dr_5class` / `expert_history_net`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.514（IQR 0.441–0.582）。
- `aptos_dr_5class` / `expert_history_net`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.495（IQR 0.296–0.645）。
- `aptos_dr_5class` / `scout_entropy`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.576（IQR 0.542–0.624）。
- `aptos_dr_5class` / `scout_entropy`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.843（IQR 0.794–0.881）。
- `aptos_dr_5class` / `scout_inverse_margin`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.615（IQR 0.564–0.662）。
- `aptos_dr_5class` / `scout_inverse_margin`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.854（IQR 0.808–0.880）。
- `aptos_dr_5class` / `scout_reference_js_divergence`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.487（IQR 0.433–0.534）。
- `aptos_dr_5class` / `scout_reference_js_divergence`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.708（IQR 0.672–0.757）。
- `aptos_dr_5class` / `scout_severe_probability_mass`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.548（IQR 0.515–0.596）。
- `aptos_dr_5class` / `scout_severe_probability_mass`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.820（IQR 0.783–0.845）。
- `deepdrid_dr_5class_native` / `expert_history_net`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.474（IQR 0.430–0.517）。
- `deepdrid_dr_5class_native` / `expert_history_net`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.559（IQR 0.465–0.633）。
- `deepdrid_dr_5class_native` / `scout_entropy`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.550（IQR 0.501–0.602）。
- `deepdrid_dr_5class_native` / `scout_entropy`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.767（IQR 0.717–0.810）。
- `deepdrid_dr_5class_native` / `scout_inverse_margin`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.577（IQR 0.538–0.638）。
- `deepdrid_dr_5class_native` / `scout_inverse_margin`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.772（IQR 0.730–0.830）。
- `deepdrid_dr_5class_native` / `scout_reference_js_divergence`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.508（IQR 0.458–0.558）。
- `deepdrid_dr_5class_native` / `scout_reference_js_divergence`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.425（IQR 0.326–0.499）。
- `deepdrid_dr_5class_native` / `scout_severe_probability_mass`：90 条可计算路线，`corrected`@`scout_wrong_only` AUROC 中位数 0.567（IQR 0.503–0.631）。
- `deepdrid_dr_5class_native` / `scout_severe_probability_mass`：90 条可计算路线，`introduced`@`scout_correct_only` AUROC 中位数 0.714（IQR 0.672–0.763）。
- AUROC 仅是开发集描述性信号，不构成正式方法训练或独立验证。

## 5. 固定基线

- 已生成随机、entropy、margin 的 5%/10%/20%/30% 固定预算基线。v1.1 仅对具有唯一冻结单 Scout 协议身份的路线按原政策和原预算重建；未向其余候选补造 v1.1 身份。
- `aptos_dr_5class::flair__to__swin_tiny` / `development`：预算 30%，捕获 corrected 48/59，引入 introduced 17/34，net=31；资格层级 `research_replay_only`。
- `aptos_dr_5class::flair__to__swin_tiny` / `retrospective_frozen`：预算 30%，捕获 corrected 80/107，引入 introduced 43/79，net=37；资格层级 `research_replay_only`。
- `aptos_dr_5class::ret_clip__to__retfound_cfp` / `development`：预算 5%，捕获 corrected 12/41，引入 introduced 1/33，net=11；资格层级 `research_replay_only`。
- `aptos_dr_5class::ret_clip__to__retfound_cfp` / `retrospective_frozen`：预算 5%，捕获 corrected 15/84，引入 introduced 4/49，net=11；资格层级 `research_replay_only`。
- `deepdrid_dr_5class_external::convnext_tiny__to__flair` / `retrospective_external`：预算 5%，捕获 corrected 1/61，引入 introduced 3/59，net=-2；资格层级 `research_replay_only`。
- `deepdrid_dr_5class_external::retfound_cfp__to__flair` / `retrospective_external`：预算 10%，捕获 corrected 19/89，引入 introduced 6/59，net=13；资格层级 `research_replay_only`。
- `deepdrid_dr_5class_native::keepfit_cfp__to__flair` / `development`：预算 30%，捕获 corrected 19/35，引入 introduced 6/29，net=13；资格层级 `research_replay_only`。
- `deepdrid_dr_5class_native::keepfit_cfp__to__flair` / `retrospective_frozen`：预算 30%，捕获 corrected 25/46，引入 introduced 18/58，net=7；资格层级 `research_replay_only`。
- 成本仅在同一 `h100_gpu_forward_only_batch16_component_v0_1` 协议内报告；部分成本模型不参与可比成本结论。

## 6. 架构边界

- `SafetyEligibilityGate` 继续委托唯一 `evaluate_route_qualification` 服务，负责不可绕过的任务、模态、资产、隐私、权限与运行边界。
- `ConsultationPolicyBaselineV1_1` 只对病例排序，不能授予资格、改预算或直接调用 Expert，后续 Help-or-Harm 方法只能替换这一层。

## 7. Test 与确认性缺口

- APTOS frozen Test、DeepDRiD official validation 以及既有路由结果均已被观察，只能作为回顾性比较，不能承担最终独立确认。
- 进入确认性阶段前需要一套未参与模型、路线、特征、阈值和提示选择的患者级数据；须有眼别/检查关联、预声明标签映射、冻结适配器生成的 Scout/Expert 概率、实测成本，以及足够的 corrected 与 introduced 事件数。

因此本轮允许建立研究 Benchmark，但不允许据此声称已能预测真实临床获益/伤害，也不授予 deployment 或 clinical route 资格。
