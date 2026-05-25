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


def render_report_html(report_md: str) -> str:
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>OphAgent Case Analysis Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; line-height: 1.6; padding: 0 20px; }",
        "h1, h2, h3 { color: #1f2937; }",
        "code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }",
        "pre { background: #f9fafb; padding: 12px; border-radius: 8px; overflow-x: auto; }",
        "li { margin-bottom: 6px; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    for line in report_md.splitlines():
        escaped = html.escape(line)

        if line.startswith("# "):
            html_lines.append(f"<h1>{escaped[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{escaped[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{escaped[4:]}</h3>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{escaped[2:]}</li>")
        elif not line.strip():
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p>{escaped}</p>")

    html_lines.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_lines)


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

    report_html = render_report_html(report_md)
    (output_dir / "report.html").write_text(report_html, encoding="utf-8")

    validation = validate_case_artifact(
        findings_data=findings_data,
        report_md=report_md,
        output_dir=output_dir,
    )

    write_json(output_dir / "validation.json", validation)

    print("[DONE] Case report artifact generated.")
    print(f"[INFO] Report MD: {output_dir / 'report.md'}")
    print(f"[INFO] Report HTML: {output_dir / 'report.html'}")
    print(f"[INFO] Validation: {output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
