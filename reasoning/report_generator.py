"""
Report generator 总入口。

根据配置选择 provider。
当前支持：
- rule_based: 稳定 fallback
- openai: API 增强报告
"""

from typing import Any, Dict, Optional

from findings.finding_schema import CaseFindings
from reasoning.providers.rule_based import RuleBasedReportProvider
from reasoning.providers.openai_provider import OpenAIReportProvider


def build_report_provider(
    provider_name: str = "rule_based",
    provider_config: Optional[Dict[str, Any]] = None,
):
    """根据 provider 名称和配置构造 provider。"""

    provider_config = provider_config or {}

    if provider_name == "rule_based":
        return RuleBasedReportProvider()

    if provider_name == "openai":
        return OpenAIReportProvider(
            model=provider_config.get("model", "gpt-4o-mini"),
            temperature=float(provider_config.get("temperature", 0.2)),
            timeout=int(provider_config.get("timeout", 30)),
            api_key_env=provider_config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=provider_config.get("base_url"),
        )

    raise ValueError(f"Unsupported report provider: {provider_name}")


def generate_report(
    case_findings: CaseFindings,
    provider_name: str = "rule_based",
    fallback_provider_name: str = "rule_based",
    provider_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    生成中文报告。

    如果主 provider 失败，则 fallback 到 rule_based。
    """

    provider_config = provider_config or {}

    try:
        provider = build_report_provider(
            provider_name=provider_name,
            provider_config=provider_config,
        )
        return provider.generate_report(case_findings)

    except Exception as exc:
        print(f"[WARN] Report provider '{provider_name}' failed: {exc}")

        if provider_name == fallback_provider_name:
            raise exc

        fallback_provider = build_report_provider(
            provider_name=fallback_provider_name,
            provider_config=provider_config,
        )
        return fallback_provider.generate_report(case_findings)
