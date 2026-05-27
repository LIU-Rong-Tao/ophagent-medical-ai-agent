# LLM Report Safety Rule Boundaries

## 定位

本文档说明 `RuleBasedSafetyChecker` 的覆盖范围、已知边界和后续扩展方向。

`RuleBasedSafetyChecker` 是一个确定性的 rule-based safety guard，用于拦截报告草稿中高风险、可规则化识别的 unsafe claim。

它不是完整的 hallucination detector，也不声称能覆盖所有语义层面的医学幻觉。

## 当前覆盖的 unsafe 类型

### 1. Clinical diagnosis overclaim

目标：拦截将模型预测写成临床诊断结论的表述。

典型可拦截表达：

- the patient is diagnosed with diabetic retinopathy
- diagnosed with diabetic retinopathy
- clinical diagnosis
- definitive diagnosis

已知边界：

- 可以拦截直接诊断式表述。
- 对于更委婉的诊断暗示，可能漏检。

可能漏检表达：

- highly suggestive of diabetic retinopathy
- findings are consistent with diabetic retinopathy
- the image confirms disease-related changes

### 2. CAM / heatmap overclaim

目标：拦截把 CAM 或 heatmap 描述为病灶定位、病灶分割或疾病原因区域的表述。

典型可拦截表达：

- CAM heatmap localizes retinal lesions
- heatmap shows the lesion area responsible for the disease
- CAM confirms abnormal retinal regions

已知边界：

- 可以拦截明显的 localizes / lesion area / confirms 等强表述。
- 对于更模糊的视觉 grounding 说法，可能漏检。

可能漏检表达：

- attention map highlights suspected lesion locations
- heatmap points to suspicious retinal regions
- the highlighted region corresponds to pathology

### 3. Unsupported lesion localization

目标：拦截没有病灶级标注、mask、bounding box 或 segmentation evidence 支撑的定位表述。

典型可拦截表达：

- localizes retinal lesions
- lesion area responsible for the disease
- lesion segmentation

已知边界：

- 可以拦截直接定位和分割表述。
- 对于“疑似区域”“高亮区域”等弱定位表述，覆盖有限。

### 4. Missing non-clinical-use statement

目标：确保报告草稿包含非临床用途声明。

必须出现的语义：

- not for clinical use
- research/demo draft
- not used for clinical diagnosis or treatment decisions

已知边界：

- 可以检查明确的非临床用途声明。
- 不判断声明是否足够完整，也不判断声明是否被上下文抵消。

### 5. Missing human-review statement

目标：确保报告草稿包含人工审核要求。

必须出现的语义：

- Human review is required
- requires human review
- reviewed by a clinician / expert before use

已知边界：

- 可以检查明确的人工审核声明。
- 不判断人工审核声明是否被其他临床用途表述削弱。

### 6. Image-quality overclaim

目标：拦截没有图像质量评估支撑却声称图像质量足以用于临床决策的表述。

典型可拦截表达：

- image quality is validated as sufficient for clinical decision-making
- sufficient for clinical diagnosis
- acceptable for diagnosis

已知边界：

- 可以拦截强临床用途的图像质量声明。
- 对于较弱的图像质量描述，未来需要结合 image-quality module 判断。

### 7. Clinical-use overclaim

目标：拦截将 research/demo draft 表述为可临床参考、可辅助诊断或可用于治疗决策的内容。

典型可拦截表达：

- can be used as a clinical reference
- for clinical decision-making
- treatment recommendation
- diagnostic decision

已知边界：

- 可以拦截直接临床使用表述。
- 对于间接表述或营销式措辞，可能漏检。

## 当前不覆盖的内容

当前 `RuleBasedSafetyChecker` 不覆盖：

- 细粒度医学事实错误
- 所有 paraphrased hallucination
- 所有否定句和转折句中的复杂语义
- 多句组合形成的隐含诊断结论
- 真实 LLM 输出的完整多样性
- 医学知识正确性判断
- 图像质量真实评估
- 病灶级视觉 grounding
- 真实临床报告可用性
- 医生级医学审核

## 典型 false negative 风险

以下表达可能存在漏检风险：

- The findings are highly suggestive of diabetic retinopathy.
- The fundus image is consistent with moderate diabetic retinopathy.
- The heatmap highlights suspected lesion locations.
- The highlighted region corresponds to abnormal pathology.
- The image is acceptable for diagnosis.
- The model output can support clinical assessment.

这些表达后续应加入 regression test 或真实 LLM probe case。

## 典型 false positive 风险

以下表达可能被规则误杀，需要通过测试确认：

- This report does not provide a clinical diagnosis.
- CAM should not be interpreted as lesion localization.
- The heatmap does not localize lesions.
- Image quality has not been validated for clinical decision-making.
- This draft is not suitable for clinical use.

## 当前策略

v0.6.2 采用保守策略：

- 只要检测到 unsafe claim，就触发 full fallback。
- 不对 LLM draft 做局部修补。
- 原始 draft 保留用于审计。
- safety_report.json 记录 flagged claims、规则命中、fallback 决策和审计元数据。

## 后续扩展方向

v0.6.2 优先补：

- regression tests
- paraphrased unsafe claim tests
- negated safe expression tests
- prompt_hash
- provider_version
- checker_version
- deterministic metadata

后续再考虑：

- controlled real LLM provider
- 小样本真实 LLM probe
- LLM-as-judge
- medical fact checker
- image-quality module
- segmentation / lesion detector evidence provider

## 面试解释口径

当前 checker 应描述为：

- rule-based safety guard for high-risk report overclaims

不应描述为：

- complete hallucination detector
- clinical safety verifier
- medical fact checker

更准确的说法是：

当前版本先用确定性规则固定高风险边界，确保 obvious unsafe claims 可以被拦截，并为后续真实 LLM provider、LLM-as-judge 和医学知识校验预留扩展位置。
