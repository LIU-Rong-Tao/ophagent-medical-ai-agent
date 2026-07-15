"""
OphAgent v0.2.2
Batch demo for lightweight vision-language reasoning.

当前脚本目标：
1. 从 demo_samples 中自动收集样例图像
2. 为每张图生成 structured findings
3. 生成 rule-based 中文 summary
4. 保存 case-level artifacts

注意：
当前版本仍然是 lightweight reasoning workflow demo。
这里使用文件夹类别作为 mock prediction，
后续再接入真实 infer_classifier 和 CAM pipeline。
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
from findings.finding_generator import generate_case_findings  # noqa: E402
from reasoning.report_generator import generate_report  # noqa: E402


RAW_CLASS_TO_LABEL = {
    # 标准命名
    "nodr": "No DR",
    "milddr": "Mild DR",
    "moderatedr": "Moderate DR",
    "severedr": "Severe DR",
    "proliferativedr": "Proliferative DR",

    # demo_samples 前缀版本
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",

    # 兼容其他可能命名
    "amilddr": "Mild DR",
    "amoderatedr": "Moderate DR",
    "aseveredr": "Severe DR",
    "aproliferativedr": "Proliferative DR",

    "cnodr": "No DR",
    "cmilddr": "Mild DR",
    "cseveredr": "Severe DR",
    "cproliferativedr": "Proliferative DR",
}


def collect_images(samples_dir: Path, max_cases: int = 15):
    """从 demo_samples 目录递归收集图像。"""

    image_exts = {".png", ".jpg", ".jpeg"}

    images = [
        p
        for p in samples_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ]

    images = sorted(images)

    return images[:max_cases]


def infer_mock_from_path(image_path: Path):
    """
    根据图像所在文件夹名生成 mock prediction。

    这一步不是模型推理，只是为了先跑通 batch artifact workflow。
    """

    raw_class = image_path.parent.name.lower()
    prediction = RAW_CLASS_TO_LABEL.get(raw_class, "Moderate DR")

    # 当前是 mock confidence，只用于 workflow demo。
    # 后续接真实 infer 后会替换成模型输出。
    confidence = 0.5000

    return prediction, raw_class, confidence


def save_case_artifacts(
    image_path: Path,
    case_findings,
    report: str,
    case_dir: Path,
    case_id: str,
):
    """保存单个 case 的 artifact。"""

    case_dir.mkdir(parents=True, exist_ok=True)

    original_path = case_dir / f"original{image_path.suffix.lower()}"
    findings_path = case_dir / "findings.json"
    report_path = case_dir / "report.txt"
    metadata_path = case_dir / "metadata.json"

    shutil.copy2(image_path, original_path)

    with findings_path.open("w", encoding="utf-8") as f:
        json.dump(case_findings.to_dict(), f, ensure_ascii=False, indent=2)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)

    metadata = {
        "case_id": case_id,
        "source_image": str(image_path),
        "saved_original": str(original_path),
        "prediction": case_findings.prediction,
        "raw_class": case_findings.raw_class,
        "confidence": case_findings.confidence,
        "cam_method": case_findings.cam_method,
        "cam_target_layer": case_findings.cam_target_layer,
        "cam_output_path": case_findings.cam_output_path,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "This is a lightweight VL reasoning demo artifact. "
            "Prediction/confidence are mock values in this batch script."
        ),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    samples_dir = PROJECT_ROOT / "demo_samples"
    output_root = PROJECT_ROOT / "experiments" / "v0_2_2_vl_reasoning_demo_batch"

    images = collect_images(samples_dir=samples_dir, max_cases=15)

    if not images:
        raise RuntimeError(f"No images found in {samples_dir}")

    print(f"[INFO] Found {len(images)} images for batch demo")
    print(f"[INFO] Output root: {output_root}")

    for idx, image_path in enumerate(images, start=1):
        case_id = f"case_{idx:03d}"
        case_dir = output_root / case_id

        prediction, raw_class, confidence = infer_mock_from_path(image_path)

        case_findings = generate_case_findings(
            image_path=str(image_path.relative_to(PROJECT_ROOT)),
            prediction=prediction,
            raw_class=raw_class,
            confidence=confidence,
            topk_predictions=[
                {prediction: confidence},
            ],
            cam_method=None,
            cam_target_layer=None,
            cam_output_path=None,
        )

        report = generate_report(case_findings)

        save_case_artifacts(
            image_path=image_path,
            case_findings=case_findings,
            report=report,
            case_dir=case_dir,
            case_id=case_id,
        )

        print(f"[OK] {case_id}: {image_path} -> {prediction}")

    print("[DONE] Batch VL reasoning demo finished")


if __name__ == "__main__":
    main()
