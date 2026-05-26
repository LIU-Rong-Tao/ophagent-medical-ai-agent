"""Rule-based post-generation safety checker for guarded report drafts.

The checker is designed as a deterministic first-layer guard. It targets
predefined high-risk failure modes rather than all possible hallucinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FlaggedClaim:
    """A claim flagged by the safety checker."""

    claim_type: str
    text: str
    reason: str


@dataclass
class SafetyCheckResult:
    """Structured safety check result for safety_report.json."""

    overall_pass: bool
    fallback_triggered: bool
    flagged_claims: list[FlaggedClaim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RuleBasedSafetyChecker:
    """Deterministic checker for high-risk report-generation failure modes."""

    def check(self, draft_text: str) -> SafetyCheckResult:
        """Check a generated draft and return a structured safety result."""
        raise NotImplementedError("Rule-based safety checks will be implemented in v0.6.1.")
