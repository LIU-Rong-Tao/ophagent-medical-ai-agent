"""
LLM report prompt 模板。

v0.2.2 中 OpenAI provider 是 optional。
默认仍然使用 rule_based provider。
"""

SYSTEM_PROMPT_ZH = """
你是一个医学图像 AI 系统中的报告生成模块。
你需要根据模型输出和结构化 findings 生成中文摘要。

必须遵守：
1. 不得声称已经完成临床诊断
2. 不得使用“发现病灶”“确诊”“存在 hemorrhage”等确定性表达
3. 应使用“模型关注区域”“可能相关视觉特征”“需结合医生判断”等表达
4. 输出面向研究展示，不面向临床决策
""".strip()


USER_PROMPT_TEMPLATE_ZH = """
请根据以下结构化信息生成中文 clinical-style summary：

图像路径：
{image_path}

模型预测：
{prediction}

原始类别：
{raw_class}

置信度：
{confidence}

CAM 方法：
{cam_method}

CAM 目标层：
{cam_target_layer}

CAM 输出路径：
{cam_output_path}

结构化 findings：
{findings}

请输出：
1. 模型预测摘要
2. 模型关注区域说明
3. 可能相关视觉特征说明
4. 使用限制声明
""".strip()


def build_report_prompt(case_findings) -> str:
    """把 CaseFindings 转换成 LLM user prompt。"""

    findings_text = "\n".join(
        [
            f"- {finding.display_name}: {finding.description}"
            for finding in case_findings.findings
        ]
    )

    return USER_PROMPT_TEMPLATE_ZH.format(
        image_path=case_findings.image_path,
        prediction=case_findings.prediction,
        raw_class=case_findings.raw_class,
        confidence=f"{case_findings.confidence:.4f}",
        cam_method=case_findings.cam_method,
        cam_target_layer=case_findings.cam_target_layer,
        cam_output_path=case_findings.cam_output_path,
        findings=findings_text,
    )
