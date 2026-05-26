"""Render guarded report artifacts.

The renderer writes intermediate and final report artifacts, including raw mock
LLM drafts, checked reports, fallback reports, and safety_report.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_guarded_report(case_dir: Path, case_data: dict[str, Any]) -> dict[str, Any]:
    """Render guarded report artifacts for a case directory."""
    raise NotImplementedError("Guarded report rendering will be implemented in v0.6.1.")
