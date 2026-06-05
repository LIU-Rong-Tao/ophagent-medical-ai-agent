#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate pre-review risk ranking with ground-truth labels.

Ground-truth labels are used only here, after the pre-review ranking is already built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GRADE_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "no dr": 0,
    "nodr": 0,
    "normal": 0,
    "a no dr": 0,
    "anodr": 0,
    "mild": 1,
    "mild dr": 1,
    "bmilddr": 1,
    "moderate": 2,
    "moderate dr": 2,
    "cmoderatedr": 2,
    "severe": 3,
    "severe dr": 3,
    "dseveredr": 3,
    "proliferative": 4,
    "proliferative dr": 4,
    "pdr": 4,
    "eproliferativedr": 4,
}


def parse_grade(value: Any) -> float:
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return np.nan
        return int(value)

    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text_compact = text.replace(" ", "")

    if text in GRADE_MAP:
        return GRADE_MAP[text]
    if text_compact in GRADE_MAP:
        return GRADE_MAP[text_compact]

    for key, grade in GRADE_MAP.items():
        if key in text or key.replace(" ", "") in text_compact:
            return grade

    return np.nan


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None



def ensure_case_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "case_id" in out.columns:
        return out

    path_col = find_first_column(out, ["image_path", "path", "file_path", "img_path"])
    if path_col is None:
        return out

    out["case_id"] = out[path_col].map(lambda x: Path(str(x)).stem)
    return out


def add_ground_truth_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    y_col = find_first_column(out, ["y_true", "gt_label", "true_label", "ground_truth", "label"])
    pred_col = find_first_column(out, ["pred_grade", "pred_label", "prediction", "pred"])

    if y_col is None:
        raise ValueError("No ground-truth column found. Expected one of: y_true, gt_label, true_label, ground_truth, label")

    out["y_true_grade_eval"] = out[y_col].map(parse_grade)

    if "pred_grade" in out.columns:
        out["pred_grade_eval"] = out["pred_grade"].map(parse_grade)
    elif pred_col is not None:
        out["pred_grade_eval"] = out[pred_col].map(parse_grade)
    else:
        raise ValueError("No prediction column found. Expected pred_grade / pred_label / prediction / pred")

    out["is_correct_eval"] = out["y_true_grade_eval"] == out["pred_grade_eval"]
    out["is_error_eval"] = ~out["is_correct_eval"]
    out["severe_underestimate_eval"] = (
        (out["y_true_grade_eval"] >= 3) &
        (out["pred_grade_eval"] < 3)
    )

    return out



def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to a simple GitHub-style markdown table without tabulate."""
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
        values = [str(row[col]).replace("\n", " ").replace("|", "/") for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def summarize_at_k(df: pd.DataFrame, k: int, total_error_rate: float, total_severe_under_count: int) -> dict:
    top = df.head(k)
    error_count = int(top["is_error_eval"].sum())
    severe_under_count = int(top["severe_underestimate_eval"].sum())

    error_rate = error_count / max(k, 1)
    enrichment = error_rate / total_error_rate if total_error_rate > 0 else None
    severe_recall = severe_under_count / total_severe_under_count if total_severe_under_count > 0 else None

    return {
        "k": k,
        "review_fraction": k / len(df),
        "error_count": error_count,
        "error_rate": error_rate,
        "enrichment_ratio": enrichment,
        "severe_underestimate_count": severe_under_count,
        "severe_underestimate_recall": severe_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-review-csv", required=True, type=Path)
    parser.add_argument("--labeled-csv", default=None, type=Path)
    parser.add_argument("--output-dir", default=Path("experiments/summary/v0_6_6"), type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pre = ensure_case_id(pd.read_csv(args.pre_review_csv))

    if args.labeled_csv is not None:
        labeled = ensure_case_id(pd.read_csv(args.labeled_csv))
        if "case_id" not in pre.columns or "case_id" not in labeled.columns:
            raise ValueError("When --labeled-csv is provided, both files must contain case_id or image_path.")
        df = pre.merge(labeled, on="case_id", how="left", suffixes=("", "_label_src"))
    else:
        df = pre

    df = add_ground_truth_fields(df)

    df = df.sort_values(
        by=["review_priority_rank"],
        ascending=True,
    ).reset_index(drop=True)

    total_n = len(df)
    total_errors = int(df["is_error_eval"].sum())
    total_error_rate = total_errors / max(total_n, 1)
    total_severe_under = int(df["severe_underestimate_eval"].sum())

    k_values = sorted(set([
        max(1, int(round(total_n * 0.10))),
        max(1, int(round(total_n * 0.20))),
        max(1, int(round(total_n * 0.30))),
        min(total_n, 10),
        min(total_n, 20),
        min(total_n, 50),
    ]))

    at_k = [
        summarize_at_k(df, k, total_error_rate, total_severe_under)
        for k in k_values
    ]

    group_rows = []
    if "pre_review_risk_level" in df.columns:
        for level in ["high", "medium", "low"]:
            sub = df[df["pre_review_risk_level"] == level]
            if len(sub) == 0:
                continue
            group_rows.append({
                "risk_level": level,
                "n": int(len(sub)),
                "error_count": int(sub["is_error_eval"].sum()),
                "error_rate": float(sub["is_error_eval"].mean()),
                "severe_underestimate_count": int(sub["severe_underestimate_eval"].sum()),
            })

    metrics = {
        "total_n": total_n,
        "total_errors": total_errors,
        "total_error_rate": total_error_rate,
        "total_severe_underestimate": total_severe_under,
        "top_k": at_k,
        "risk_group": group_rows,
    }

    metrics_path = args.output_dir / "ranking_eval_metrics.json"
    report_path = args.output_dir / "ranking_eval_report.md"

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = []
    md.append("# v0.6.6 Pre-review Risk Ranking Evaluation")
    md.append("")
    md.append("本报告使用真实标签进行后验验证。真实标签不参与预审风险排序。")
    md.append("")
    md.append("## Overall")
    md.append("")
    md.append(f"- Total samples: {total_n}")
    md.append(f"- Total errors: {total_errors}")
    md.append(f"- Overall error rate: {total_error_rate:.4f}")
    md.append(f"- Total severe underestimation cases: {total_severe_under}")
    md.append("")
    md.append("## Top-K Evaluation")
    md.append("")
    md.append(dataframe_to_markdown(pd.DataFrame(at_k)))
    md.append("")
    md.append("## Risk Group Error Rate")
    md.append("")
    if group_rows:
        md.append(dataframe_to_markdown(pd.DataFrame(group_rows)))
    else:
        md.append("No pre_review_risk_level column found.")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- 如果 high risk 组错误率明显高于 overall error rate，说明预审风险排序具备初步有效性。")
    md.append("- 如果 Top 10% / 20% 的 enrichment ratio > 1，说明该排序优于随机抽样。")
    md.append("- 如果 severe_underestimate_recall 在较小 review_fraction 下较高，说明该规则对重症低估风险有价值。")
    md.append("- 如果上述趋势不明显，说明需要引入校准、TTA uncertainty、多模型 disagreement 或图像质量评分。")
    md.append("")

    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] saved: {metrics_path}")
    print(f"[OK] saved: {report_path}")
    print(f"[INFO] total_n={total_n}, errors={total_errors}, error_rate={total_error_rate:.4f}")


if __name__ == "__main__":
    main()
