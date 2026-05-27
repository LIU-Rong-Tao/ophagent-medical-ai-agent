"""
OphAgent v0.6.0
Evidence-Bottleneck Case Report Prototype.

功能：
1. 输入单张眼底图像
2. 使用当前分类模型生成 prediction.json
3. 调用 explain.gradcam 生成 CAM artifact
4. 生成 v0.6 findings.json
5. 生成 validation.json
6. 生成 report.md / report.html
7. 保存 metadata.json

注意：
- 当前 report 是科研展示用 draft，不是临床诊断报告
- CAM 只是 weak visual evidence，不是病灶标注
- 当前未自动评估图像质量
"""

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.classifiers.builder import build_model
from reasoning.llm_report.renderer import render_guarded_report


CLASS_DISPLAY_NAMES = {
    "nodr": "No DR",
    "milddr": "Mild DR",
    "moderatedr": "Moderate DR",
    "severedr": "Severe DR",
    "proliferativedr": "Proliferative DR",
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}

DR_VISUAL_CUES = {
    "No DR": [
        "no obvious diabetic-retinopathy-related visual pattern",
    ],
    "Mild DR": [
        "mild microaneurysm-like changes",
        "subtle focal microvascular abnormality-like patterns",
    ],
    "Moderate DR": [
        "microaneurysm-like changes",
        "dot- or blot-hemorrhage-like visual cues",
        "hard-exudate-like bright regions",
        "focal retinal microvascular abnormality-like patterns",
    ],
    "Severe DR": [
        "more extensive hemorrhage-like visual cues",
        "venous-beading-like patterns",
        "prominent retinal microvascular abnormality-like patterns",
        "cotton-wool-spot-like changes",
    ],
    "Proliferative DR": [
        "neovascularization-related visual cues",
        "vitreous- or preretinal-hemorrhage-like visual patterns",
        "fibrovascular-proliferation-like changes",
    ],
}

REQUIRED_LIMITATIONS = [
    "No lesion-level annotation is available.",
    "No physician report ground truth is available.",
    "No multimodal clinical context is used.",
    "Automatic image quality assessment is not implemented in v0.6.0.",
    "CAM is not equivalent to lesion localization.",
    "This system has not been clinically validated.",
    "This report must not be used for clinical diagnosis or treatment decisions.",
    "Human review is required.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OphAgent v0.6.0 single-case report pipeline."
    )

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--class-to-idx", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--cam-method", type=str, default="gradcam")
    parser.add_argument("--cam-target-layer", type=str, default="stage3")
    parser.add_argument("--cam-smoothing", type=str, default="eigen", choices=["none", "eigen", "aug_eigen"])

    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument(
        "--report-provider",
        type=str,
        default="template",
        choices=["template", "mock_llm"],
        help=(
            "Report generation provider. "
            "'template' keeps the v0.6.0 deterministic report path; "
            "'mock_llm' enables the v0.6.1 guarded mock LLM renderer."
        ),
    )
    parser.add_argument(
        "--mock-llm-mode",
        type=str,
        default="safe",
        choices=["safe", "unsafe_diagnosis", "unsafe_cam", "unsafe_mixed"],
        help="Deterministic mock LLM mode used only when --report-provider mock_llm.",
    )

    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_mapping(path: str) -> tuple[dict[str, int], dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def run_prediction(
    *,
    image_path: Path,
    config: dict[str, Any],
    checkpoint_path: str,
    class_to_idx_path: str,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    _, idx_to_class = load_class_mapping(class_to_idx_path)

    model = build_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    image = Image.open(image_path).convert("RGB")
    transform = build_transform(int(config["image_size"]))

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].detach().cpu()

    top_probs, top_indices = torch.topk(
        probs,
        k=min(top_k, len(probs)),
    )

    topk_predictions = []

    for rank, (idx, prob) in enumerate(
        zip(top_indices.tolist(), top_probs.tolist()),
        start=1,
    ):
        raw_class = idx_to_class[int(idx)]
        display_name = CLASS_DISPLAY_NAMES.get(raw_class, raw_class)

        topk_predictions.append(
            {
                "rank": rank,
                "raw_class": raw_class,
                "display_name": display_name,
                "confidence": round(float(prob), 6),
            }
        )

    top1 = topk_predictions[0]

    return {
        "prediction_id": "pred_001",
        "raw_class": top1["raw_class"],
        "display_name": top1["display_name"],
        "confidence": top1["confidence"],
        "topk_predictions": topk_predictions,
        "task": "diabetic_retinopathy_grading",
    }


def run_cam(
    *,
    image_path: Path,
    config_path: str,
    checkpoint_path: str,
    class_to_idx_path: str,
    cam_output_dir: Path,
    method: str,
    target_layer: str,
    smoothing: str,
):
    cam_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "explain.gradcam",
        "--image",
        str(image_path),
        "--config",
        config_path,
        "--checkpoint",
        checkpoint_path,
        "--class-to-idx",
        class_to_idx_path,
        "--method",
        method,
        "--target-layer",
        target_layer,
        "--output",
        str(cam_output_dir),
    ]

    if smoothing == "eigen":
        command.append("--eigen-smooth")
    elif smoothing == "aug_eigen":
        command.extend(["--aug-smooth", "--eigen-smooth"])

    print("[INFO] Running CAM:")
    print(" ".join(command))

    subprocess.run(command, check=True)


def build_findings_json(
    *,
    case_id: str,
    image_path: Path,
    saved_input_path: Path,
    output_dir: Path,
    prediction: dict[str, Any],
    config_path: str,
    checkpoint_path: str,
    class_to_idx_path: str,
    config: dict[str, Any],
    cam_method: str,
    cam_target_layer: str,
    cam_smoothing: str,
) -> dict[str, Any]:
    display_name = prediction["display_name"]
    confidence = prediction["confidence"]

    cam_source = f"{cam_method}_{cam_target_layer}_{cam_smoothing}"
    cam_overlay_path = output_dir / "cam" / "overlay.png"

    visual_cues = DR_VISUAL_CUES.get(display_name, [])
    visual_cues_text = ", ".join(visual_cues) if visual_cues else "not specified"

    evidence = [
        {
            "evidence_id": "ev_cam_001",
            "evidence_type": "cam",
            "source": cam_source,
            "region": "model-attended fundus regions",
            "description": (
                "CAM overlay provides a weak visual reference for regions that contributed "
                "to the model prediction."
            ),
            "clinical_strength": "weak",
            "caution": "CAM is not lesion annotation and must not be interpreted as clinical lesion localization.",
            "artifact_path": str(cam_overlay_path),
            "replaceable_by": [
                "segmentation_mask",
                "bounding_box",
                "lesion_detector",
            ],
        }
    ]

    findings = [
        {
            "finding_id": "finding_001",
            "finding_type": "classification_tendency",
            "description": (
                f"The model prediction suggests a tendency toward {display_name} "
                f"with confidence {confidence:.4f}."
            ),
            "supported_by": ["pred_001"],
            "confidence_level": "model_probability",
            "caution": "This is a model prediction, not a clinical diagnosis.",
        },
        {
            "finding_id": "finding_002",
            "finding_type": "possible_visual_cue",
            "description": (
                f"For the predicted DR category, possible related visual cues may include: "
                f"{visual_cues_text}. These are class-level explanatory cues, not confirmed lesions."
            ),
            "supported_by": ["pred_001"],
            "confidence_level": "qualitative",
            "caution": "These cues are not independent lesion detections.",
        },
        {
            "finding_id": "finding_003",
            "finding_type": "cam_attention_observation",
            "description": (
                "The CAM result is included as weak visual evidence to support model interpretability. "
                "It should be reviewed only as a model attention visualization."
            ),
            "supported_by": ["ev_cam_001"],
            "confidence_level": "qualitative",
            "caution": "CAM does not provide lesion-level annotation.",
        },
    ]

    quality_control = {
        "image_quality_assessed": False,
        "image_quality_level": "unknown",
        "quality_aware_mode": "not_implemented",
        "quality_note": (
            "Automatic image quality assessment is not implemented in v0.6.0. "
            "The report should be interpreted with caution."
        ),
        "action": "caution",
        "future_upgrade": [
            "image_quality_classifier",
            "quality_aware_confidence_adjustment",
            "quality_aware_evidence_weighting",
            "non_diagnostic_image_refusal",
        ],
    }

    report_claims = [
        {
            "claim_id": "claim_001",
            "text": (
                f"The model predicts {display_name} for the input fundus image "
                f"with confidence {confidence:.4f}."
            ),
            "claim_type": "model_prediction",
            "supported_by": ["pred_001"],
            "safety_level": "informational",
            "section": "Model Prediction",
        },
        {
            "claim_id": "claim_002",
            "text": (
                "CAM-based visual evidence is provided only as weak model attention evidence "
                "and must not be interpreted as lesion annotation."
            ),
            "claim_type": "visual_evidence",
            "supported_by": ["ev_cam_001"],
            "safety_level": "caution",
            "section": "Visual Evidence",
        },
        {
            "claim_id": "claim_003",
            "text": (
                "Automatic image quality assessment is not implemented in this version; "
                "the output should be interpreted with caution."
            ),
            "claim_type": "quality_context",
            "supported_by": ["quality_control"],
            "safety_level": "caution",
            "section": "Quality-aware Context",
        },
        {
            "claim_id": "claim_004",
            "text": (
                "This report is an AI-generated research/demo draft and must not be used "
                "for clinical diagnosis or treatment decisions. Human review is required."
            ),
            "claim_type": "disclaimer",
            "supported_by": ["system_policy"],
            "safety_level": "disclaimer",
            "section": "Non-clinical-use Disclaimer",
        },
    ]

    return {
        "case_id": case_id,
        "input": {
            "image_path": str(image_path),
            "saved_input_path": str(saved_input_path),
            "image_id": case_id,
            "modality": "fundus_color_photography",
            "dataset_hint": "demo_samples" if "demo_samples" in str(image_path) else "unknown",
        },
        "prediction": prediction,
        "evidence": evidence,
        "findings": findings,
        "interpretation": {
            "summary": (
                f"The current evidence-bottleneck output supports a model-level tendency "
                f"toward {display_name}. The explanation remains limited by the absence of "
                f"lesion-level annotation, image quality assessment, and clinical metadata."
            ),
            "supported_by": ["pred_001", "ev_cam_001", "finding_001", "finding_003", "quality_control"],
        },
        "report_claims": report_claims,
        "quality_control": quality_control,
        "limitations": REQUIRED_LIMITATIONS,
        "model_info": {
            "backbone": config.get("backbone", "unknown"),
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "class_to_idx_path": class_to_idx_path,
            "cam_method": cam_method,
            "cam_target_layer": cam_target_layer,
            "cam_smoothing": cam_smoothing,
        },
        "provenance": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "workflow": "v0.6.0_case_report",
            "script": "scripts/run_case_report.py",
            "output_dir": str(output_dir),
            "ophagent_version": "v0.6.0",
            "note": "Research/demo artifact only. Not for clinical use.",
        },
    }


def validate_case_artifact(
    *,
    findings_data: dict[str, Any],
    report_md: str,
    output_dir: Path,
) -> dict[str, Any]:
    required_top_level = [
        "case_id",
        "input",
        "prediction",
        "evidence",
        "findings",
        "report_claims",
        "quality_control",
        "limitations",
        "model_info",
        "provenance",
    ]

    schema_valid = all(key in findings_data for key in required_top_level)

    required_files = [
        output_dir / "input.png",
        output_dir / "prediction.json",
        output_dir / "findings.json",
        output_dir / "report.md",
        output_dir / "report.html",
        output_dir / "metadata.json",
        output_dir / "cam" / "original.png",
        output_dir / "cam" / "heatmap.png",
        output_dir / "cam" / "overlay.png",
    ]

    required_files_present = all(path.exists() for path in required_files)

    known_support_ids = {"system_policy", "quality_control"}

    if "prediction" in findings_data:
        known_support_ids.add(findings_data["prediction"].get("prediction_id", ""))

    for item in findings_data.get("evidence", []):
        known_support_ids.add(item.get("evidence_id", ""))

    for item in findings_data.get("findings", []):
        known_support_ids.add(item.get("finding_id", ""))

    unsupported_claim_ids = []

    for claim in findings_data.get("report_claims", []):
        supported_by = claim.get("supported_by", [])
        if not supported_by or any(sid not in known_support_ids for sid in supported_by):
            unsupported_claim_ids.append(claim.get("claim_id", "unknown"))

    total_claims = len(findings_data.get("report_claims", []))
    supported_claims = total_claims - len(unsupported_claim_ids)
    evidence_coverage_rate = float(supported_claims / total_claims) if total_claims else 0.0

    lower_report = report_md.lower()

    # Detect only positive clinical diagnosis claims.
    # Do not flag safety disclaimers such as
    # "must not be used for clinical diagnosis".
    prohibited_diagnosis_patterns = [
        "confirmed diagnosis of",
        "is diagnosed with",
        "the patient has",
        "definitive diagnosis of",
        "clinical diagnosis is",
        "诊断为",
        "确诊为",
        "临床诊断为",
    ]

    clinical_diagnosis_claim_present = any(
        phrase in lower_report
        for phrase in prohibited_diagnosis_patterns
    )

    image_quality_overclaimed = any(
        phrase in lower_report
        for phrase in [
            "image quality has been automatically assessed",
            "图像质量已自动评估",
            "quality verified",
        ]
    )

    return {
        "schema_valid": schema_valid,
        "required_files_present": required_files_present,
        "required_disclaimer_present": "not be used for clinical diagnosis" in lower_report,
        "human_review_required": "human review is required" in lower_report,
        "cam_described_as_weak_evidence": "weak" in lower_report and "cam" in lower_report,
        "clinical_diagnosis_claim_present": clinical_diagnosis_claim_present,
        "unsupported_claim_count": len(unsupported_claim_ids),
        "unsupported_claim_ids": unsupported_claim_ids,
        "evidence_coverage_rate": round(evidence_coverage_rate, 6),
        "image_quality_overclaimed": image_quality_overclaimed,
        "non_clinical_use_statement_present": "research/demo" in lower_report or "research and demonstration" in lower_report,
        "report_reproducible": total_claims > 0 and len(unsupported_claim_ids) == 0,
        "validation_warnings": [],
    }


def render_report_md(findings_data: dict[str, Any]) -> str:
    prediction = findings_data["prediction"]
    model_info = findings_data["model_info"]
    quality_control = findings_data["quality_control"]
    provenance = findings_data["provenance"]

    claims_by_section: dict[str, list[str]] = {}

    for claim in findings_data["report_claims"]:
        claims_by_section.setdefault(claim["section"], []).append(claim["text"])

    lines = [
        "# OphAgent Case Analysis Report",
        "",
        "## 1. Case Overview",
        "",
        f"- Case ID: `{findings_data['case_id']}`",
        f"- Input image: `{findings_data['input']['image_path']}`",
        "- Report type: AI-generated research/demo draft",
        "",
        "## 2. Model Prediction",
        "",
    ]

    for text in claims_by_section.get("Model Prediction", []):
        lines.append(f"- {text}")

    lines.extend(
        [
            "",
            "### Top-k Predictions",
            "",
        ]
    )

    for item in prediction["topk_predictions"]:
        lines.append(
            f"- Rank {item['rank']}: {item['display_name']} "
            f"(`{item['raw_class']}`), confidence={item['confidence']:.4f}"
        )

    lines.extend(
        [
            "",
            "## 3. Visual Evidence",
            "",
        ]
    )

    for text in claims_by_section.get("Visual Evidence", []):
        lines.append(f"- {text}")

    lines.extend(
        [
            "",
            f"- CAM method: `{model_info['cam_method']}`",
            f"- CAM target layer: `{model_info['cam_target_layer']}`",
            f"- CAM smoothing: `{model_info['cam_smoothing']}`",
            f"- CAM overlay: `{findings_data['evidence'][0]['artifact_path']}`",
            "",
            "## 4. Quality-aware Context",
            "",
        ]
    )

    for text in claims_by_section.get("Quality-aware Context", []):
        lines.append(f"- {text}")

    lines.extend(
        [
            f"- Image quality assessed: `{quality_control['image_quality_assessed']}`",
            f"- Image quality level: `{quality_control['image_quality_level']}`",
            f"- Quality-aware mode: `{quality_control['quality_aware_mode']}`",
            f"- Action: `{quality_control['action']}`",
            "",
            "## 5. Structured Findings",
            "",
        ]
    )

    for finding in findings_data["findings"]:
        lines.append(f"- **{finding['finding_type']}**: {finding['description']}")
        lines.append(f"  - Supported by: `{', '.join(finding['supported_by'])}`")
        lines.append(f"  - Caution: {finding['caution']}")

    lines.extend(
        [
            "",
            "## 6. Interpretation Summary",
            "",
            findings_data["interpretation"]["summary"],
            "",
            "## 7. Limitations",
            "",
        ]
    )

    for limitation in findings_data["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(
        [
            "",
            "## 8. Non-clinical-use Disclaimer",
            "",
        ]
    )

    for text in claims_by_section.get("Non-clinical-use Disclaimer", []):
        lines.append(f"- {text}")

    lines.extend(
        [
            "",
            "## 9. Artifact Metadata",
            "",
            f"- Backbone: `{model_info['backbone']}`",
            f"- Config: `{model_info['config_path']}`",
            f"- Checkpoint: `{model_info['checkpoint_path']}`",
            f"- Generated at: `{provenance['generated_at']}`",
            f"- Workflow: `{provenance['workflow']}`",
        ]
    )

    return "\n".join(lines) + "\n"


def render_report_html(
    findings_data: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> str:
    """
    Render a card-style HTML case report from structured findings.

    Unlike the Markdown report, this HTML page is designed as the
    primary visual artifact for v0.6.0.
    """

    validation = validation or {}

    prediction = findings_data["prediction"]
    model_info = findings_data["model_info"]
    quality_control = findings_data["quality_control"]
    provenance = findings_data["provenance"]
    evidence = findings_data["evidence"][0]

    def image_to_data_uri(path_str: str) -> str:
        image_path = Path(path_str)

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        suffix = image_path.suffix.lower()
        mime = "image/png"

        if suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif suffix == ".webp":
            mime = "image/webp"

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    input_image_src = image_to_data_uri(findings_data["input"]["saved_input_path"])
    cam_overlay_src = image_to_data_uri(evidence["artifact_path"])

    topk_rows = []

    for item in prediction["topk_predictions"]:
        topk_rows.append(
            "<tr>"
            f"<td>{item['rank']}</td>"
            f"<td>{html.escape(item['display_name'])}</td>"
            f"<td><code>{html.escape(item['raw_class'])}</code></td>"
            f"<td>{item['confidence']:.4f}</td>"
            "</tr>"
        )

    finding_items = []

    for finding in findings_data["findings"]:
        supported_by = ", ".join(finding.get("supported_by", []))
        finding_items.append(
            "<div class='finding-card'>"
            f"<div class='finding-title'>{html.escape(finding['finding_type'])}</div>"
            f"<p>{html.escape(finding['description'])}</p>"
            f"<p class='small'><strong>Supported by:</strong> <code>{html.escape(supported_by)}</code></p>"
            f"<p class='caution'>{html.escape(finding['caution'])}</p>"
            "</div>"
        )

    limitation_items = "\n".join(
        f"<li>{html.escape(item)}</li>"
        for item in findings_data["limitations"]
    )

    def bool_badge(value: Any, positive_when_true: bool = True) -> str:
        value_bool = bool(value)
        ok = value_bool if positive_when_true else not value_bool
        cls = "ok" if ok else "bad"
        text = str(value)
        return f"<span class='badge {cls}'>{html.escape(text)}</span>"

    validation_html = f"""
    <div class="metric-grid">
      <div class="metric">
        <span class="metric-label">Schema valid</span>
        {bool_badge(validation.get("schema_valid", False))}
      </div>
      <div class="metric">
        <span class="metric-label">Unsupported claims</span>
        <span class="badge {'ok' if validation.get('unsupported_claim_count', 1) == 0 else 'bad'}">
          {validation.get('unsupported_claim_count', 'N/A')}
        </span>
      </div>
      <div class="metric">
        <span class="metric-label">Evidence coverage</span>
        <span class="badge ok">{validation.get('evidence_coverage_rate', 'N/A')}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Clinical diagnosis claim</span>
        {bool_badge(validation.get("clinical_diagnosis_claim_present", True), positive_when_true=False)}
      </div>
      <div class="metric">
        <span class="metric-label">Image quality overclaim</span>
        {bool_badge(validation.get("image_quality_overclaimed", True), positive_when_true=False)}
      </div>
      <div class="metric">
        <span class="metric-label">Report reproducible</span>
        {bool_badge(validation.get("report_reproducible", False))}
      </div>
    </div>
    """

    html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>OphAgent Case Analysis Report</title>
  <style>
    body {{
      margin: 0;
      background: #f3f4f6;
      color: #111827;
      font-family: Arial, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      line-height: 1.6;
    }}
    .page {{
      max-width: 1180px;
      margin: 32px auto;
      padding: 0 24px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, #111827, #374151);
      color: white;
      border-radius: 24px;
      padding: 32px;
      margin-bottom: 24px;
      box-shadow: 0 16px 36px rgba(15, 23, 42, 0.18);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    .hero p {{
      margin: 0;
      color: #d1d5db;
      font-size: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 1.1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }}
    .card h2 {{
      margin: 0 0 16px;
      font-size: 21px;
      color: #111827;
    }}
    .image-card img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid #e5e7eb;
      background: #fff;
    }}
    .caption {{
      margin-top: 12px;
      font-size: 14px;
      color: #6b7280;
    }}
    .prediction {{
      font-size: 30px;
      font-weight: 700;
      margin: 4px 0 6px;
      color: #1f2937;
    }}
    .confidence {{
      font-size: 18px;
      color: #4b5563;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 10px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 6px;
      text-align: left;
    }}
    th {{
      color: #374151;
      background: #f9fafb;
    }}
    code {{
      background: #f3f4f6;
      padding: 2px 5px;
      border-radius: 5px;
      font-size: 13px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      background: #f9fafb;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
    }}
    .metric-label {{
      color: #374151;
      font-size: 14px;
    }}
    .badge {{
      display: inline-block;
      min-width: 54px;
      text-align: center;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }}
    .badge.ok {{
      color: #065f46;
      background: #d1fae5;
    }}
    .badge.bad {{
      color: #991b1b;
      background: #fee2e2;
    }}
    .section-grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .finding-card {{
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      padding: 14px 16px;
      margin-bottom: 12px;
      background: #fbfdff;
    }}
    .finding-title {{
      font-weight: 700;
      color: #111827;
      margin-bottom: 6px;
    }}
    .small {{
      color: #4b5563;
      font-size: 14px;
    }}
    .caution {{
      color: #92400e;
      background: #fffbeb;
      border-left: 4px solid #f59e0b;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 14px;
    }}
    .disclaimer {{
      border-left: 5px solid #dc2626;
      background: #fef2f2;
    }}
    .footer {{
      color: #6b7280;
      font-size: 13px;
      margin-top: 18px;
    }}
    @media (max-width: 960px) {{
      .grid, .section-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>OphAgent Case Analysis Report</h1>
      <p>Evidence-bottleneck research/demo artifact. Not for clinical diagnosis or treatment decisions.</p>
    </div>

    <div class="grid">
      <div class="card image-card">
        <h2>Input Fundus Image</h2>
        <img src="{input_image_src}" alt="Input fundus image">
        <div class="caption">
          Case ID: <code>{html.escape(findings_data['case_id'])}</code><br>
          Source: <code>{html.escape(findings_data['input']['image_path'])}</code>
        </div>
      </div>

      <div class="card image-card">
        <h2>CAM Weak Visual Evidence</h2>
        <img src="{cam_overlay_src}" alt="CAM overlay">
        <div class="caption">
          CAM: <code>{html.escape(model_info['cam_method'])}_{html.escape(model_info['cam_target_layer'])}_{html.escape(model_info['cam_smoothing'])}</code><br>
          CAM is weak model attention evidence, not lesion annotation.
        </div>
      </div>

      <div class="card">
        <h2>Prediction and Validation</h2>
        <div class="prediction">{html.escape(prediction['display_name'])}</div>
        <div class="confidence">Confidence: {prediction['confidence']:.4f}</div>

        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Class</th>
              <th>Raw</th>
              <th>Conf.</th>
            </tr>
          </thead>
          <tbody>
            {''.join(topk_rows)}
          </tbody>
        </table>

        <h2 style="margin-top: 22px;">Validation</h2>
        {validation_html}
      </div>
    </div>

    <div class="section-grid">
      <div class="card">
        <h2>Structured Findings</h2>
        {''.join(finding_items)}
      </div>

      <div class="card">
        <h2>Quality-aware Context</h2>
        <p>{html.escape(quality_control['quality_note'])}</p>
        <p><strong>Image quality assessed:</strong> <code>{quality_control['image_quality_assessed']}</code></p>
        <p><strong>Image quality level:</strong> <code>{html.escape(quality_control['image_quality_level'])}</code></p>
        <p><strong>Action:</strong> <code>{html.escape(quality_control['action'])}</code></p>

        <h2>Evidence Summary</h2>
        <p>{html.escape(evidence['description'])}</p>
        <p class="caution">{html.escape(evidence['caution'])}</p>
      </div>
    </div>

    <div class="card">
      <h2>Interpretation Summary</h2>
      <p>{html.escape(findings_data['interpretation']['summary'])}</p>
    </div>

    <div class="card disclaimer" style="margin-top: 20px;">
      <h2>Limitations and Safety Boundary</h2>
      <ul>
        {limitation_items}
      </ul>
      <p><strong>This report is an AI-generated research/demo draft. Human review is required.</strong></p>
    </div>

    <div class="footer">
      Generated at: <code>{html.escape(provenance['generated_at'])}</code> |
      Workflow: <code>{html.escape(provenance['workflow'])}</code> |
      Script: <code>{html.escape(provenance['script'])}</code>
    </div>
  </div>
</body>
</html>
"""

    return html_doc


def write_json(path: Path, data: dict[str, Any]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)
    cam_output_dir = output_dir / "cam"
    case_id = image_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_input_path = output_dir / "input.png"
    shutil.copy2(image_path, saved_input_path)

    config = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Case ID: {case_id}")
    print(f"[INFO] Output: {output_dir}")

    prediction = run_prediction(
        image_path=image_path,
        config=config,
        checkpoint_path=args.checkpoint,
        class_to_idx_path=args.class_to_idx,
        top_k=args.top_k,
        device=device,
    )

    write_json(output_dir / "prediction.json", prediction)

    run_cam(
        image_path=image_path,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        class_to_idx_path=args.class_to_idx,
        cam_output_dir=cam_output_dir,
        method=args.cam_method,
        target_layer=args.cam_target_layer,
        smoothing=args.cam_smoothing,
    )

    findings_data = build_findings_json(
        case_id=case_id,
        image_path=image_path,
        saved_input_path=saved_input_path,
        output_dir=output_dir,
        prediction=prediction,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        class_to_idx_path=args.class_to_idx,
        config=config,
        cam_method=args.cam_method,
        cam_target_layer=args.cam_target_layer,
        cam_smoothing=args.cam_smoothing,
    )

    write_json(output_dir / "findings.json", findings_data)

    metadata = {
        "case_id": case_id,
        "created_at": findings_data["provenance"]["generated_at"],
        "workflow": "v0.6.0_case_report",
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "prediction": prediction["display_name"],
        "raw_class": prediction["raw_class"],
        "confidence": prediction["confidence"],
        "cam_method": args.cam_method,
        "cam_target_layer": args.cam_target_layer,
        "cam_smoothing": args.cam_smoothing,
        "note": "Research/demo artifact only. Not for clinical use.",
    }

    write_json(output_dir / "metadata.json", metadata)

    report_md = render_report_md(findings_data)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    # Write a preliminary HTML file so required_files_present can be checked.
    preliminary_html = render_report_html(findings_data, validation=None)
    (output_dir / "report.html").write_text(preliminary_html, encoding="utf-8")

    validation = validate_case_artifact(
        findings_data=findings_data,
        report_md=report_md,
        output_dir=output_dir,
    )

    write_json(output_dir / "validation.json", validation)

    # Rewrite final HTML with validation summary included.
    report_html = render_report_html(findings_data, validation=validation)
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")

    if args.report_provider == "mock_llm":
        render_result = render_guarded_report(
            case_dir=output_dir,
            provider_name="mock_llm",
            mock_llm_mode=args.mock_llm_mode,
        )
        metadata["report_provider"] = args.report_provider
        metadata["mock_llm_mode"] = args.mock_llm_mode
        metadata["guarded_report"] = {
            "safety_passed": render_result.safety_passed,
            "fallback_triggered": render_result.fallback_triggered,
            "safety_report_path": render_result.safety_report_path,
        }
        write_json(output_dir / "metadata.json", metadata)
        print("[INFO] Guarded LLM report renderer enabled.")
        print(f"[INFO] Safety passed: {render_result.safety_passed}")
        print(f"[INFO] Fallback triggered: {render_result.fallback_triggered}")
        print(f"[INFO] Safety report: {render_result.safety_report_path}")
    else:
        metadata["report_provider"] = "template"
        metadata["guarded_report"] = None
        write_json(output_dir / "metadata.json", metadata)

    print("[DONE] Case report artifact generated.")
    print(f"[INFO] Report MD: {output_dir / 'report.md'}")
    print(f"[INFO] Report HTML: {output_dir / 'report.html'}")
    print(f"[INFO] Validation: {output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
