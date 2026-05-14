"""
Report generator 总入口。

根据配置选择 provider。
当前 v0.2.2 默认使用 rule_based provider。
"""

from findings.finding_schema import CaseFindings
from reasoning.providers.rule_based import RuleBasedReportProvider


def build_report_provider(provider_name: str = "rule_based"):
    """
    根据 provider 名称构造 provider。

    v0.2.2 当前优先保证 rule_based 稳定。
    """

    if provider_name == "rule_based":
        return RuleBasedReportProvider()

    raise ValueError(f"Unsupported report provider: {provider_name}")


def generate_report(
    case_findings: CaseFindings,
    provider_name: str = "rule_based",
    fallback_provider_name: str = "rule_based",
) -> str:
    """
    生成中文报告。

    如果主 provider 失败，则 fallback 到 rule_based。
    """

    try:
        provider = build_report_provider(provider_name)
        return provider.generate_report(case_findings)

    except Exception as exc:
        if provider_name == fallback_provider_name:
            raise exc

        fallback_provider = build_report_provider(fallback_provider_name)
        return fallback_provider.generate_report(case_findings)
