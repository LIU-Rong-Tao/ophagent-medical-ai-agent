from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class AgentInput:
    image: Any
    image_source: str = "unknown"
    top_k: int = 3
    report_providers: Sequence[str] = ("rule_based",)
    fallback_report_provider: str = "rule_based"
    report_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopKPrediction:
    rank: int
    raw_class: str
    display_name: str
    confidence: float


@dataclass
class AgentResult:
    predicted_class: str
    predicted_display_name: str
    confidence: float
    topk: list[TopKPrediction]
    findings: Any
    report: str
    report_provider: str
    raw: dict[str, Any] = field(default_factory=dict)