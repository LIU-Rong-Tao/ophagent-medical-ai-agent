"""Run a small real-LLM safety probe over existing case artifact directories.

This script is intended for v0.6.4 small-scale pilot evaluation.

It does not run vision inference. It expects each case directory to already contain
the v0.6.0-style artifacts required by render_guarded_report, such as:

- prediction.json
- findings.json
- validation.json
- metadata.json
- report.md / report.html

The script copies each case directory to a temporary workspace before rendering, so
source case artifacts are not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
from reasoning.llm_report.renderer import render_guarded_report  # noqa: E402


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read a CSV manifest with at least case_id and case_dir columns."""
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "case_dir"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            case_dir = (row.get("case_dir") or "").strip()
            if not case_id or not case_dir:
                continue
            rows.append({"case_id": case_id, "case_dir": case_dir})

    if not rows:
        raise ValueError(f"No valid cases found in manifest: {path}")

    return rows


def redact_sensitive_text(text: str) -> str:
    """Redact API keys and authorization-like values before writing probe outputs."""
    patterns = [
        r"sk-proj-[A-Za-z0-9_\-]+",
        r"sk-[A-Za-z0-9_\-]+",
        r"Bearer\\s+[^\\s'\"]+",
        r"OPHAGENT_LLM_API_KEY=[^\\s'\"]+",
    ]

    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, "[REDACTED_SECRET]", redacted)
    return redacted


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def copy_text_file_redacted(src: Path, dst: Path) -> None:
    """Copy a text artifact after redacting secret-like strings."""
    if not src.exists():
        return

    text = redact_sensitive_text(src.read_text(encoding="utf-8"))
    if "sk-" in text or "Bearer " in text:
        raise RuntimeError(f"Sensitive token-like text detected while copying: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def summarize_safety_report(report: dict[str, Any]) -> dict[str, Any]:
    """Extract compact safety probe fields from one safety_report.json."""
    safety_result = report.get("safety_result") or {}
    flagged_claims = safety_result.get("flagged_claims") or []

    claim_types = [
        str(claim.get("claim_type", "unknown"))
        for claim in flagged_claims
        if isinstance(claim, dict)
    ]

    audit_metadata = report.get("audit_metadata") or {}
    provider_metadata = report.get("provider_metadata") or {}

    return {
        "provider": report.get("provider"),
        "overall_pass": bool(report.get("overall_pass")),
        "fallback_triggered": bool(report.get("fallback_triggered")),
        "flagged_claim_count": len(flagged_claims),
        "flagged_claim_type_counts": dict(Counter(claim_types)),
        "provider_type": audit_metadata.get("provider_type"),
        "provider_version": audit_metadata.get("provider_version"),
        "checker_version": audit_metadata.get("checker_version"),
        "safety_policy_version": audit_metadata.get("safety_policy_version"),
        "real_llm_used": audit_metadata.get("real_llm_used"),
        "model_name": provider_metadata.get("model_name"),
        "prompt_hash_available": bool(audit_metadata.get("prompt_hash")),
    }


def write_markdown_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a compact markdown table for probe results."""
    lines = [
        "| case_id | status | overall_pass | fallback | flagged_claims | model | error |",
        "|---|---:|---:|---:|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            "| {case_id} | {status} | {overall_pass} | {fallback_triggered} | "
            "{flagged_claim_count} | {model_name} | {error} |".format(
                case_id=row.get("case_id", ""),
                status=row.get("status", ""),
                overall_pass=row.get("overall_pass", ""),
                fallback_triggered=row.get("fallback_triggered", ""),
                flagged_claim_count=row.get("flagged_claim_count", ""),
                model_name=row.get("model_name", ""),
                error=str(row.get("error", "")).replace("|", "/"),
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(args: argparse.Namespace) -> None:
    """Run the real LLM safety probe."""
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = read_manifest(manifest_path)

    case_rows: list[dict[str, Any]] = []
    aggregate_flagged_claim_types: Counter[str] = Counter()
    api_failure_count = 0
    safety_pass_count = 0
    fallback_count = 0
    success_count = 0

    with tempfile.TemporaryDirectory(prefix="ophagent_v064_probe_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)

        for item in cases:
            case_id = item["case_id"]
            source_case_dir = Path(item["case_dir"])

            row: dict[str, Any] = {
                "case_id": case_id,
                "source_case_dir": str(source_case_dir),
                "status": "unknown",
            }

            try:
                if not source_case_dir.exists():
                    raise FileNotFoundError(f"Case directory not found: {source_case_dir}")

                work_case_dir = tmp_root / case_id
                if work_case_dir.exists():
                    shutil.rmtree(work_case_dir)
                shutil.copytree(source_case_dir, work_case_dir)

                render_guarded_report(
                    case_dir=work_case_dir,
                    provider_name=args.provider,
                    mock_llm_mode=args.mock_llm_mode,
                )

                safety_report_path = work_case_dir / "safety_report.json"
                safety_report = load_json(safety_report_path)
                compact = summarize_safety_report(safety_report)

                if args.save_samples:
                    sample_dir = output_dir / "sample_cases" / case_id
                    copy_text_file_redacted(
                        work_case_dir / "reports" / "llm_raw.md",
                        sample_dir / "llm_raw.md",
                    )
                    copy_text_file_redacted(
                        work_case_dir / "reports" / "llm_checked.md",
                        sample_dir / "llm_checked.md",
                    )
                    copy_text_file_redacted(
                        work_case_dir / "reports" / "llm_guarded.html",
                        sample_dir / "llm_guarded.html",
                    )
                    copy_text_file_redacted(
                        safety_report_path,
                        sample_dir / "safety_report.json",
                    )
                    row["sample_dir"] = str(sample_dir)

                row.update(compact)
                row["status"] = "success"

                success_count += 1
                if compact["overall_pass"]:
                    safety_pass_count += 1
                if compact["fallback_triggered"]:
                    fallback_count += 1

                aggregate_flagged_claim_types.update(compact["flagged_claim_type_counts"])

            except Exception as exc:  # noqa: BLE001 - probe should continue across cases
                row["status"] = "error"
                row["error"] = redact_sensitive_text(str(exc))
                api_failure_count += 1

            case_rows.append(row)

    total_cases = len(cases)
    result = {
        "version": "v0.6.4",
        "probe_type": "small_scale_real_llm_safety_probe",
        "provider": args.provider,
        "mock_llm_mode": args.mock_llm_mode,
        "total_cases": total_cases,
        "success_count": success_count,
        "api_failure_count": api_failure_count,
        "safety_pass_count": safety_pass_count,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / success_count if success_count else None,
        "flagged_claim_type_counts": dict(aggregate_flagged_claim_types),
        "case_results": case_rows,
    }

    results_path = output_dir / "safety_probe_results.json"
    table_path = output_dir / "safety_probe_table.md"

    results_text = json.dumps(result, ensure_ascii=False, indent=2)
    if "sk-" in results_text or "Bearer " in results_text:
        raise RuntimeError("Sensitive token-like text detected in probe results; aborting write.")
    results_path.write_text(results_text, encoding="utf-8")
    write_markdown_table(case_rows, table_path)

    print("total_cases:", total_cases)
    print("success_count:", success_count)
    print("api_failure_count:", api_failure_count)
    print("safety_pass_count:", safety_pass_count)
    print("fallback_count:", fallback_count)
    print("fallback_rate:", result["fallback_rate"])
    print("flagged_claim_type_counts:", dict(aggregate_flagged_claim_types))
    print("contains_sk_key:", "sk-" in results_text)
    print("results_path:", results_path)
    print("table_path:", table_path)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a small real-LLM safety probe over existing case artifacts."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV file with columns: case_id,case_dir",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/summary/v0_6_4",
        help="Directory for probe summary outputs.",
    )
    parser.add_argument(
        "--provider",
        default="real_llm",
        choices=["real_llm", "mock_llm", "template"],
        help="Report provider used for the probe.",
    )
    parser.add_argument(
        "--mock-llm-mode",
        default="safe",
        help="Mock LLM mode, only used when provider=mock_llm.",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save redacted per-case llm_raw / llm_checked / safety_report artifacts.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
