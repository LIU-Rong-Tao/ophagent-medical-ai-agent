"""从 demo_samples 构建小规模 demo 风险样本表。

本脚本用于 v0.6.5 医院线下展示前的样本级风险审查演示。

脚本会使用 demo_samples 的目录名作为 weak ground-truth label，
调用已有分类模型 checkpoint 进行预测，并导出一个小规模 risk table，
用于展示模型输出可靠性审查流程。

该输出不是临床验证结果，也不是正式医学评估。
它只是一个轻量演示：说明 OphAgent 如何组织 prediction confidence、
top-k margin、预测正确性和 risk tags，为后续人工复核提供样本优先级。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from PIL import Image

from models.classifiers.builder import build_model
from torchvision import transforms


FOLDER_TO_LABEL = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}

RAW_LABEL_TO_DISPLAY = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
    "No DR": "No DR",
    "Mild DR": "Mild DR",
    "Moderate DR": "Moderate DR",
    "Severe DR": "Severe DR",
    "Proliferative DR": "Proliferative DR",
}

LABEL_TO_GRADE = {
    "No DR": 0,
    "Mild DR": 1,
    "Moderate DR": 2,
    "Severe DR": 3,
    "Proliferative DR": 4,
}


def build_eval_transform(config: dict[str, Any]) -> transforms.Compose:
    """根据配置构建评估阶段图像预处理。"""
    image_size = (
        config.get("image_size")
        or config.get("img_size")
        or config.get("input_size")
        or 224
    )

    if isinstance(image_size, (list, tuple)):
        image_size = image_size[-1]

    image_size = int(image_size)

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


@dataclass
class PredictionRow:
    case_id: str
    image_path: str
    gt_label: str
    pred_label: str
    correct: bool
    confidence: float
    top2_label: str
    top2_confidence: float
    margin: float
    risk_types: list[str]
    recommended_action: str


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_class_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """读取 class_to_idx 映射，并返回正向与反向类别映射。"""
    class_to_idx = load_json(path)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def normalize_label(label: str) -> str:
    """将原始类别名统一转换为展示标签。"""
    return RAW_LABEL_TO_DISPLAY.get(label, label)


def infer_label_from_path(image_path: Path) -> str:
    """根据 demo_samples 子目录名推断 weak GT label。"""
    folder = image_path.parent.name
    if folder not in FOLDER_TO_LABEL:
        raise ValueError(f"Unsupported demo sample folder: {folder}")
    return FOLDER_TO_LABEL[folder]


def collect_images(samples_root: Path, max_per_class: int | None) -> list[Path]:
    """按固定顺序收集 demo sample 图像，保证结果可复现。"""
    images: list[Path] = []

    for folder in sorted(FOLDER_TO_LABEL):
        class_dir = samples_root / folder
        if not class_dir.exists():
            continue

        class_images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if max_per_class is not None:
            class_images = class_images[:max_per_class]
        images.extend(class_images)

    return images


def risk_tags(
    gt_label: str,
    pred_label: str,
    confidence: float,
    margin: float,
) -> list[str]:
    """根据预测结果、置信度和 top1-top2 margin 生成可解释风险标签。"""
    tags: list[str] = []
    correct = gt_label == pred_label

    gt_grade = LABEL_TO_GRADE.get(gt_label)
    pred_grade = LABEL_TO_GRADE.get(pred_label)

    if not correct and confidence >= 0.8:
        tags.append("high_conf_error")

    if margin <= 0.15:
        tags.append("low_margin_uncertain")

    if gt_grade is not None and pred_grade is not None:
        diff = pred_grade - gt_grade

        if not correct and abs(diff) == 1:
            tags.append("adjacent_grade_confusion")

        if gt_grade >= 3 and pred_grade < gt_grade:
            tags.append("severe_underestimate")

        if gt_grade <= 1 and pred_grade >= gt_grade + 2:
            tags.append("severe_overestimate")

    if correct and confidence < 0.6:
        tags.append("low_conf_correct")

    if not tags:
        tags.append("review_not_prioritized")

    return tags


def recommended_action(tags: list[str]) -> str:
    """根据风险标签生成建议的人工复核优先级。"""
    high_priority = {
        "high_conf_error",
        "severe_underestimate",
        "severe_overestimate",
    }
    medium_priority = {
        "low_margin_uncertain",
        "adjacent_grade_confusion",
        "low_conf_correct",
    }

    if any(tag in high_priority for tag in tags):
        return "high_priority_human_review"

    if any(tag in medium_priority for tag in tags):
        return "human_review_recommended"

    return "routine_review"


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform: Any,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> tuple[str, float, str, float, float]:
    """对单张图像进行预测，并返回 top-1 / top-2 信息。"""
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_values, top_indices = torch.topk(probs, k=2)

    top1_idx = int(top_indices[0].item())
    top2_idx = int(top_indices[1].item())

    pred_label = normalize_label(idx_to_class[top1_idx])
    top2_label = normalize_label(idx_to_class[top2_idx])

    confidence = float(top_values[0].item())
    top2_confidence = float(top_values[1].item())
    margin = confidence - top2_confidence

    return pred_label, confidence, top2_label, top2_confidence, margin


def write_csv(rows: list[PredictionRow], path: Path) -> None:
    """将风险样本表写入 CSV 文件，便于后续分析。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "image_path",
        "gt_label",
        "pred_label",
        "correct",
        "confidence",
        "top2_label",
        "top2_confidence",
        "margin",
        "risk_types",
        "recommended_action",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "case_id": row.case_id,
                "image_path": row.image_path,
                "gt_label": row.gt_label,
                "pred_label": row.pred_label,
                "correct": row.correct,
                "confidence": f"{row.confidence:.6f}",
                "top2_label": row.top2_label,
                "top2_confidence": f"{row.top2_confidence:.6f}",
                "margin": f"{row.margin:.6f}",
                "risk_types": ";".join(row.risk_types),
                "recommended_action": row.recommended_action,
            })


def write_markdown(rows: list[PredictionRow], path: Path) -> None:
    """将风险样本表写入 Markdown 文件，便于 README 和线下展示引用。"""
    lines = [
        "| case_id | gt | pred | correct | confidence | top2 | margin | risk_types | action |",
        "|---|---|---|---:|---:|---|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row.case_id} | {row.gt_label} | {row.pred_label} | {row.correct} | "
            f"{row.confidence:.3f} | {row.top2_label} ({row.top2_confidence:.3f}) | "
            f"{row.margin:.3f} | {'; '.join(row.risk_types)} | {row.recommended_action} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(rows: list[PredictionRow]) -> dict[str, Any]:
    """构建风险样本表的聚合统计摘要。"""
    total = len(rows)
    correct_count = sum(row.correct for row in rows)

    risk_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}

    for row in rows:
        for tag in row.risk_types:
            risk_counts[tag] = risk_counts.get(tag, 0) + 1
        action_counts[row.recommended_action] = action_counts.get(row.recommended_action, 0) + 1

    return {
        "version": "v0.6.5",
        "summary_type": "demo_risk_case_table",
        "total_cases": total,
        "correct_count": correct_count,
        "incorrect_count": total - correct_count,
        "accuracy_on_demo_samples": correct_count / total if total else None,
        "risk_type_counts": dict(sorted(risk_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "important_limitations": [
            "demo_samples 数量较小，且属于展示样本，不是正式验证集",
            "目录名被用作 weak label，仅用于演示样本级审查流程",
            "该结果不是临床验证结果，也不能代表真实临床性能",
            "risk tags 是启发式规则，仅用于医院线下展示前的风险样本发现演示",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a demo risk-case table from demo_samples."
    )
    parser.add_argument("--samples-root", default="demo_samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class-to-idx", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-per-class", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    samples_root = Path(args.samples_root)
    config = load_yaml(Path(args.config))
    _, idx_to_class = load_class_mapping(Path(args.class_to_idx))

    device = torch.device(args.device)

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = build_eval_transform(config)

    images = collect_images(samples_root=samples_root, max_per_class=args.max_per_class)

    rows: list[PredictionRow] = []
    for image_path in images:
        gt_label = infer_label_from_path(image_path)
        pred_label, confidence, top2_label, top2_confidence, margin = predict_image(
            model=model,
            image_path=image_path,
            transform=transform,
            idx_to_class=idx_to_class,
            device=device,
        )

        tags = risk_tags(
            gt_label=gt_label,
            pred_label=pred_label,
            confidence=confidence,
            margin=margin,
        )

        rows.append(
            PredictionRow(
                case_id=image_path.stem,
                image_path=str(image_path),
                gt_label=gt_label,
                pred_label=pred_label,
                correct=gt_label == pred_label,
                confidence=confidence,
                top2_label=top2_label,
                top2_confidence=top2_confidence,
                margin=margin,
                risk_types=tags,
                recommended_action=recommended_action(tags),
            )
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, out_dir / "demo_risk_case_table.csv")
    write_markdown(rows, out_dir / "demo_risk_case_table.md")

    summary = build_summary(rows)
    (out_dir / "demo_risk_case_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("total_cases:", summary["total_cases"])
    print("correct_count:", summary["correct_count"])
    print("incorrect_count:", summary["incorrect_count"])
    print("accuracy_on_demo_samples:", summary["accuracy_on_demo_samples"])
    print("risk_type_counts:", summary["risk_type_counts"])
    print("recommended_action_counts:", summary["recommended_action_counts"])
    print("output_dir:", out_dir)


if __name__ == "__main__":
    main()
