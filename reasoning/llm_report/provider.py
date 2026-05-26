"""Report provider abstractions for v0.6.1.

v0.6.1 only includes deterministic providers:

- TemplateProvider
- MockLLMProvider

Real LLM providers are intentionally deferred to v0.6.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MockLLMMode = Literal["safe", "unsafe_diagnosis", "unsafe_cam", "unsafe_mixed"]


@dataclass(frozen=True)
class ReportProviderResult:
    """Container for a provider-generated report draft."""

    provider: str
    text: str
    metadata: dict[str, object]


class TemplateProvider:
    """Deterministic template provider used as the safe fallback path."""

    name = "template"


class MockLLMProvider:
    """Deterministic mock LLM provider used to test safety checking."""

    name = "mock_llm"

    def __init__(self, mode: MockLLMMode = "safe") -> None:
        self.mode = mode
