"""
OphAgent v0.2.2
Lightweight vision-language reasoning demo.

当前版本目标：
1. 串通 structured findings
2. 串通 rule-based report generation
3. 保存 findings.json 和 report.txt
4. 为后续接入真实 infer / CAM 做准备
"""

import argparse
import json
import sys
from pathlib import Path

# 保证直接运行 app/*.py 时可以导入项目模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from findings.finding_generator import generate_case_findings
from reasoning.report_generator import generate_report


def save_artifacts(case_findings, report: str, output_dir: Path) -> None:
    """保存单个 case 的结构化结果和中文报告。"""

    output_dir.mkdir(parents=True, exist_ok=True)

    findings_path = output_dir / "findings.json"
    report_path = output_dir / "report.txt"

    with findings_path.open("w", encoding="utf-8") as f:
        json.dump(case_findings.to_dict(), f, ensure_ascii=False, indent=2)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"[OK] findings saved to: {findings_path}")
    print(f"[OK] report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="OphAgent lightweight vision-language reasoning demo"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入眼底图像路径",
    )
    parser.add_argument(
        "--prediction",
        type=str,
        default="Moderate DR",
        help="分类预测结果，当前阶段可手动传入",
    )
    parser.add_argument(
        "--raw-class",
        type=str,
        default="cmoderatedr",
        help="原始类别名",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4154,
        help="分类置信度",
    )
    parser.add_argument(
        "--cam-output",
        type=str,
        default=None,
        help="CAM 输出路径",
    )
    parser.add_argument(
        "--cam-method",
        type=str,
        default="hirescam",
        help="CAM 方法",
    )
    parser.add_argument(
        "--cam-target-layer",
        type=str,
        default="stage3",
        help="CAM 目标层",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/v0_2_2_vl_reasoning_demo/case_001",
        help="artifact 输出目录",
    )

    args = parser.parse_args()

    topk_predictions = [
        {args.prediction: args.confidence},
    ]

    case_findings = generate_case_findings(
        image_path=args.image,
        prediction=args.prediction,
        raw_class=args.raw_class,
        confidence=args.confidence,
        topk_predictions=topk_predictions,
        cam_method=args.cam_method,
        cam_target_layer=args.cam_target_layer,
        cam_output_path=args.cam_output,
    )

    report = generate_report(case_findings)

    print(report)
    print()

    save_artifacts(
        case_findings=case_findings,
        report=report,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
