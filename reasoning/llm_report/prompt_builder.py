"""Build constrained prompts from structured case findings.

v0.6.1 keeps prompt construction deterministic and evidence-bound.
The prompt builder should only use structured inputs such as findings.json,
prediction.json, validation.json, and metadata.json.
"""

from __future__ import annotations

from typing import Any


def build_guarded_report_prompt(case_data: dict[str, Any]) -> str:
    """Build an evidence-bounded prompt for guarded report drafting.

    The prompt must instruct the report provider not to introduce clinical
    diagnoses, unsupported lesion localization, or claims beyond the structured
    findings.
    """
    raise NotImplementedError("Prompt builder will be implemented in v0.6.1.")
