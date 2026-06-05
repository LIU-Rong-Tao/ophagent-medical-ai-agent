#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一键运行 v0.6.6 无真实标签预审风险排序 benchmark。

功能：
1. 对多个 backbone 的 test_predictions.csv 生成预审风险排序表；
2. 使用真实标签进行后验验证；
3. 汇总不同 backbone 的 Top-K 错误富集率、重症低估召回率和风险组错误率。

注意：
- 排序阶段不使用真实标签；
- 真实标签只在后验验证阶段使用。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_ITEMS = {
    "convnext_tiny": "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "swin_tiny": "experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_imagenet": "experiments/aptos_vit_base_patch16_imagenet/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_official_like": "experiments/aptos_vit_base_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "vit_l_official_like": "experiments/aptos_vit_large_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "retfound_official_like": "experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """不依赖 tabulate 的 DataFrame -> Markdown 表格。"""
    if df.empty:
        return ""

    df_str = df.copy()
    for col in df_str.columns:
        df_str[col] = df_str[col].map(lambda x: "" if pd.isna(x) else str(x))

    headers = list(df_str.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df_str.iterrows():
        values = [
            str(row[col]).replace("\n", " ").replace("|", "/")
            for col in headers
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def pick_top_k(metrics: dict, target_fraction: float) -> dict | None:
    """根据复核比例选择最接近的 Top-K 指标。"""
    return next(
        (
            item
            for item in metrics["top_k"]
            if abs(item["review_fraction"] - target_fraction) < 0.02
        ),
        None,
    )


def build_summary(base_dir: Path) -> pd.DataFrame:
    rows = []

    for p in sorted(base_dir.glob("*/ranking_eval_metrics.json")):
        name = p.parent.name
        metrics = json.loads(p.read_text(encoding="utf-8"))

        top10 = pick_top_k(metrics, 0.10)
        top20 = pick_top_k(metrics, 0.20)
        top30 = pick_top_k(metrics, 0.30)

        group = {g["risk_level"]: g for g in metrics.get("risk_group", [])}

        rows.append({
            "backbone": name,
            "total_n": metrics["total_n"],
            "overall_error_rate": round(metrics["total_error_rate"], 4),
            "total_severe_underestimate": metrics["total_severe_underestimate"],

            "top10_error_rate": None if top10 is None else round(top10["error_rate"], 4),
            "top10_enrichment": None if top10 is None else round(top10["enrichment_ratio"], 4),
            "top10_severe_recall": None if top10 is None or top10["severe_underestimate_recall"] is None else round(top10["severe_underestimate_recall"], 4),

            "top20_error_rate": None if top20 is None else round(top20["error_rate"], 4),
            "top20_enrichment": None if top20 is None else round(top20["enrichment_ratio"], 4),
            "top20_severe_recall": None if top20 is None or top20["severe_underestimate_recall"] is None else round(top20["severe_underestimate_recall"], 4),

            "top30_error_rate": None if top30 is None else round(top30["error_rate"], 4),
            "top30_enrichment": None if top30 is None else round(top30["enrichment_ratio"], 4),
            "top30_severe_recall": None if top30 is None or top30["severe_underestimate_recall"] is None else round(top30["severe_underestimate_recall"], 4),

            "high_n": group.get("high", {}).get("n", 0),
            "high_error_rate": None if "high" not in group else round(group["high"]["error_rate"], 4),
            "medium_n": group.get("medium", {}).get("n", 0),
            "medium_error_rate": None if "medium" not in group else round(group["medium"]["error_rate"], 4),
            "low_n": group.get("low", {}).get("n", 0),
            "low_error_rate": None if "low" not in group else round(group["low"]["error_rate"], 4),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/summary/v0_6_6/full_test_backbones"),
        help="benchmark 输出目录",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果某个 backbone 已存在 ranking_eval_metrics.json，则跳过重新计算",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, csv_path in DEFAULT_ITEMS.items():
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"[WARN] 输入 CSV 不存在，跳过 {name}: {csv_file}")
            continue

        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)

        metrics_path = out / "ranking_eval_metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"[SKIP] {name}: 已存在 {metrics_path}")
            continue

        print(f"\n[BUILD] {name}")
        subprocess.run([
            "python",
            "scripts/build_pre_review_risk_table.py",
            "--input-csv",
            str(csv_file),
            "--output-dir",
            str(out),
        ], check=True)

        print(f"[EVAL] {name}")
        subprocess.run([
            "python",
            "scripts/evaluate_pre_review_ranking.py",
            "--pre-review-csv",
            str(out / "pre_review_risk_table.csv"),
            "--labeled-csv",
            str(csv_file),
            "--output-dir",
            str(out),
        ], check=True)

    summary = build_summary(args.output_dir)

    out_csv = args.output_dir / "backbone_pre_review_ranking_summary.csv"
    out_md = args.output_dir / "backbone_pre_review_ranking_summary.md"

    summary.to_csv(out_csv, index=False)
    out_md.write_text(dataframe_to_markdown(summary), encoding="utf-8")

    print("\n[SUMMARY]")
    print(summary.to_string(index=False))
    print(f"\n[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_md}")


if __name__ == "__main__":
    main()
