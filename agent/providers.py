from dataclasses import dataclass

from reasoning.report_generator import generate_report


@dataclass
class AgentReportResult:
    report: str
    provider: str


def generate_agent_report(
    *,
    findings,
    providers,
    fallback_provider,
    report_config,
):
    """
    Thin wrapper around reasoning.report_generator.generate_report.

    Current v0.2.2 signature:
        generate_report(
            case_findings,
            provider_name=...,
            fallback_provider_name=...,
            provider_config=...,
        )
    """

    provider_name = "openai" if "openai" in providers else "rule_based"

    report = generate_report(
        findings,
        provider_name=provider_name,
        fallback_provider_name=fallback_provider,
        provider_config=report_config,
    )

    return AgentReportResult(
        report=report,
        provider=provider_name,
    )