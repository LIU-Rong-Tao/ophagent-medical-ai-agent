"""v0.6.1 受控报告 Prompt 构建器。

本模块负责把病例目录中的结构化产物整理成 evidence package，
再生成用于 guarded report drafting 的 constrained prompt。

注意：
    PromptBuilder 不负责生成报告，也不负责安全检查。
    它只负责把允许使用的证据边界明确写进 prompt。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CASE_DATA_FILES = {
    "prediction": "prediction.json",
    "findings": "findings.json",
    "validation": "validation.json",
    "metadata": "metadata.json",
}


def load_case_data(case_dir: Path | str) -> dict[str, Any]:
    """从病例目录读取 v0.6.x case artifact 数据。

    Args:
        case_dir: 病例目录，例如 experiments/case_reports/<case_id>。

    Returns:
        dict[str, Any]: 包含 prediction、findings、validation、metadata 的字典。
            如果某个文件不存在，则对应字段为 None。
    """
    case_path = Path(case_dir)

    if not case_path.exists():
        raise FileNotFoundError(f"Case directory does not exist: {case_path}")

    case_data: dict[str, Any] = {
        "case_dir": str(case_path),
        "case_id": case_path.name,
    }

    for key, filename in CASE_DATA_FILES.items():
        file_path = case_path / filename
        case_data[key] = _read_json_if_exists(file_path)

    return case_data


def build_evidence_package(case_data: dict[str, Any]) -> dict[str, Any]:
    """构建允许报告生成器使用的最小证据包。

    该函数的目标是把原始 case_data 压缩成 LLM 可读、可审计、
    且明确受限的 evidence package。

    Args:
        case_data: load_case_data 返回的病例数据。

    Returns:
        dict[str, Any]: 受控证据包。
    """
    prediction = case_data.get("prediction") or {}
    findings = case_data.get("findings") or {}
    validation = case_data.get("validation") or {}
    metadata = case_data.get("metadata") or {}

    evidence_package = {
        "case_id": case_data.get("case_id"),
        "prediction": prediction,
        "structured_findings": findings,
        "validation_summary": {
            "schema_valid": validation.get("schema_valid"),
            "human_review_required": validation.get("human_review_required"),
            "cam_described_as_weak_evidence": validation.get("cam_described_as_weak_evidence"),
            "clinical_diagnosis_claim_present": validation.get("clinical_diagnosis_claim_present"),
            "unsupported_claim_count": validation.get("unsupported_claim_count"),
            "evidence_coverage_rate": validation.get("evidence_coverage_rate"),
            "image_quality_overclaimed": validation.get("image_quality_overclaimed"),
            "non_clinical_use_statement_present": validation.get("non_clinical_use_statement_present"),
            "report_reproducible": validation.get("report_reproducible"),
            "validation_warnings": validation.get("validation_warnings"),
        },
        "metadata": metadata,
        "allowed_content": [
            "model prediction summary",
            "structured findings already present in findings.json",
            "CAM described only as weak visual evidence",
            "quality-aware context already present in the case artifact",
            "limitations and safety boundary",
            "non-clinical-use statement",
            "human-review-required statement",
        ],
        "forbidden_content": [
            "clinical diagnosis",
            "definitive disease confirmation",
            "new lesion findings not present in findings.json",
            "lesion localization claims",
            "CAM or heatmap described as segmentation",
            "CAM or heatmap described as lesion localization",
            "claims that image quality is sufficient for clinical decision-making",
            "clinical-use or clinical-reference claims",
        ],
    }

    return evidence_package


def build_guarded_report_prompt(case_data: dict[str, Any]) -> str:
    """生成 evidence-bound constrained prompt。

    该 prompt 用于指导 MockLLMProvider 或未来真实 LLM Provider 生成报告草稿。
    它必须显式写清楚允许内容和禁止内容。

    Args:
        case_data: load_case_data 返回的病例数据。

    Returns:
        str: 受控报告生成 prompt。
    """
    evidence_package = build_evidence_package(case_data)
    evidence_json = json.dumps(evidence_package, ensure_ascii=False, indent=2, sort_keys=True)

    prompt = f"""You are generating a guarded ophthalmology case report draft for a non-clinical research/demo artifact.

Your task:
Generate a concise Markdown report draft using only the structured evidence provided below.

Strict evidence boundary:
1. Only use the provided structured evidence.
2. Do not make a clinical diagnosis.
3. Do not claim definitive disease confirmation.
4. Do not introduce new lesion findings that are not present in the structured findings.
5. CAM or heatmap output must be described as weak visual evidence only.
6. Do not claim that CAM localizes lesions.
7. Do not describe CAM or heatmap as lesion segmentation.
8. Do not claim that image quality is sufficient for clinical decision-making unless explicitly supported.
9. Do not describe the report as suitable for clinical use or clinical reference.
10. The report must state: This report is not for clinical use.
11. The report must state: Human review is required before any medical interpretation or downstream decision.

Required sections:
- Interpretation Summary
- Weak Visual Evidence
- Evidence Boundary
- Limitations and Safety Statement

Style:
- Use cautious language.
- Prefer "model prediction suggests" over diagnostic wording.
- Prefer "weak visual evidence" over localization wording.
- Keep the report concise and auditable.

BEGIN_EVIDENCE_PACKAGE_JSON
{evidence_json}
END_EVIDENCE_PACKAGE_JSON
"""

    return prompt


def _read_json_if_exists(path: Path) -> Any:
    """读取 JSON 文件；如果不存在则返回 None。"""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
