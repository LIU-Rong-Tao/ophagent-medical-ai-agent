"""
OpenAI report provider.

用于将 structured findings 转成更自然的中文眼底图像分析摘要。

设计原则：
1. OpenAI provider 是可选增强，不是硬依赖。
2. API key 不写入代码，也不写入 YAML 明文。
3. 默认从环境变量 OPENAI_API_KEY 读取。
4. API 失败时由 report_generator fallback 到 rule_based provider。
5. 输出必须保持研究/工程展示定位，不得写成临床诊断。
"""

import os
from typing import Optional

from openai import OpenAI

from findings.finding_schema import CaseFindings
from reasoning.providers.base import BaseReportProvider


class OpenAIReportProvider(BaseReportProvider):
    """基于 OpenAI API 的中文报告生成 provider。"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        timeout: int = 30,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.api_key_env = api_key_env
        self.base_url = base_url

        api_key = os.getenv(api_key_env)

        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {api_key_env}. "
                f"Please set it before using OpenAIReportProvider."
            )

        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
        }

        # 预留 OpenAI-compatible endpoint，例如后续接其他兼容服务
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def generate_report(self, case_findings: CaseFindings) -> str:
        """根据结构化 findings 生成中文分析报告。"""

        findings_text = "\n".join(
            [
                f"- {finding.display_name}: {finding.description}"
                for finding in case_findings.findings
            ]
        )

        prompt = f"""
你是 OphAgent 医学图像 AI demo 中的报告生成模块。

请根据以下模型输出，生成一段中文眼底图像分析摘要。

必须遵守：
1. 不得写成临床诊断。
2. 不得使用“确诊”“发现病灶”“存在某病灶”等确定性表达。
3. 可以使用“模型预测”“模型关注区域”“可能相关视觉线索”“需结合眼科医生判断”等表述。
4. 报告应适合 Streamlit demo 展示，简洁、清晰、有医学上下文。
5. 输出中文 Markdown。

输入信息：

图像路径：
{case_findings.image_path}

模型预测：
{case_findings.prediction}

原始类别：
{case_findings.raw_class}

置信度：
{case_findings.confidence:.4f}

CAM 方法：
{case_findings.cam_method}

CAM 目标层：
{case_findings.cam_target_layer}

CAM 输出路径：
{case_findings.cam_output_path}

结构化 findings：
{findings_text}

免责声明：
{case_findings.disclaimer}

请按以下结构输出：

## OphAgent 眼底图像分析摘要

### 模型预测
简述预测类别和置信度。

### 可能相关视觉线索
结合 structured findings 总结可能相关的眼底视觉线索。

### 可解释性说明
说明 CAM 热力图的作用和限制。

### 使用限制
强调该结果仅用于研究与工程展示，不构成临床诊断。
""".strip()

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )

        return response.output_text
