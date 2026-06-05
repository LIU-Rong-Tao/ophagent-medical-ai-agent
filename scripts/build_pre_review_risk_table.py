#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build pre-review risk ranking table without using ground-truth labels.

This script is for v0.6.6:
- Input: model output CSV / demo risk table / prediction table
- Output: pre-review risk table
- Important: y_true / is_correct / error_type are NOT used for ranking
"""

from __future__ import annotations

import argparse
import math
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


LABEL_CN = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


FORBIDDEN_FOR_RANKING = {
    "y_true",
    "gt_label",
    "true_label",
    "ground_truth",
    "label",
    "is_correct",
    "correct",
    "error_type",
    "severe_underestimate",
    "underestimate",
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


def find_prob_columns(df: pd.DataFrame) -> list[str]:
    # Prefer numeric class probability columns.
    candidates = []

    for i in range(5):
        for name in [
            f"prob_{i}",
            f"p_{i}",
            f"class_{i}_prob",
            f"prob_class_{i}",
            f"probability_{i}",
        ]:
            if name in df.columns:
                candidates.append(name)
                break

    if len(candidates) == 5:
        return candidates

    # Support label-name probability columns used by current evaluation output.
    label_candidates = [
        "prob_No DR",
        "prob_Mild DR",
        "prob_Moderate DR",
        "prob_Severe DR",
        "prob_Proliferative DR",
    ]
    if all(c in df.columns for c in label_candidates):
        return label_candidates

    # Case-insensitive fallback for label-name columns.
    lower_map = {c.lower(): c for c in df.columns}
    label_lower = [c.lower() for c in label_candidates]
    if all(c in lower_map for c in label_lower):
        return [lower_map[c] for c in label_lower]

    # fallback: columns containing prob and class id
    fallback = []
    for i in range(5):
        matched = [
            c for c in df.columns
            if "prob" in c.lower() and str(i) in c.lower()
        ]
        if matched:
            fallback.append(matched[0])

    if len(fallback) == 5:
        return fallback

    return []


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def compute_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum())


def normalize_entropy(entropy: float, num_classes: int = 5) -> float:
    if pd.isna(entropy):
        return np.nan
    return float(entropy / math.log(num_classes))


def infer_model_output_columns(df: pd.DataFrame) -> dict[str, str | list[str] | None]:
    return {
        "case_id": find_first_column(df, ["case_id", "image_id", "id", "sample_id"]),
        "image_path": find_first_column(df, ["image_path", "path", "file_path", "img_path"]),
        "pred_label": find_first_column(df, ["pred_label", "prediction", "pred", "pred_class"]),
        "confidence": find_first_column(df, ["confidence", "top1_confidence", "pred_confidence", "prob", "top1_prob"]),
        "top2_label": find_first_column(df, ["top2_label", "second_label", "second_pred", "top_2_label"]),
        "top2_confidence": find_first_column(df, ["top2_confidence", "top2_prob", "second_confidence", "second_prob"]),
        "margin": find_first_column(df, ["margin", "top1_top2_margin", "top1_top2_gap"]),
        "entropy": find_first_column(df, ["entropy", "pred_entropy"]),
        "prob_columns": find_prob_columns(df),
    }


def build_pre_review_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = infer_model_output_columns(df)
    prob_cols = cols["prob_columns"] or []

    out = pd.DataFrame()

    case_col = cols["case_id"]
    path_col = cols["image_path"]
    pred_col = cols["pred_label"]

    if case_col:
        out["case_id"] = df[case_col]
    elif path_col:
        out["case_id"] = df[path_col].map(lambda x: Path(str(x)).stem)
    else:
        out["case_id"] = [f"case_{i:05d}" for i in range(len(df))]

    if path_col:
        out["image_path"] = df[path_col]

    if pred_col:
        out["pred_label_raw"] = df[pred_col]
        out["pred_grade"] = df[pred_col].map(parse_grade)
    else:
        out["pred_label_raw"] = np.nan
        out["pred_grade"] = np.nan

    if len(prob_cols) == 5:
        prob_mat = df[prob_cols].astype(float).to_numpy()
        prob_sum = prob_mat.sum(axis=1, keepdims=True)
        prob_sum = np.where(prob_sum <= 0, 1.0, prob_sum)
        prob_mat = prob_mat / prob_sum

        top_order = np.argsort(-prob_mat, axis=1)
        top1 = top_order[:, 0]
        top2 = top_order[:, 1]

        out["pred_grade_from_probs"] = top1
        out["top2_grade"] = top2
        out["confidence"] = prob_mat[np.arange(len(df)), top1]
        out["top2_confidence"] = prob_mat[np.arange(len(df)), top2]
        out["margin"] = out["confidence"] - out["top2_confidence"]
        out["entropy"] = [compute_entropy(row) for row in prob_mat]
        out["entropy_norm"] = out["entropy"].map(normalize_entropy)
        out["severe_prob_mass"] = prob_mat[:, 3] + prob_mat[:, 4]

        # If pred_label is missing, trust probability argmax.
        out["pred_grade"] = out["pred_grade"].where(out["pred_grade"].notna(), out["pred_grade_from_probs"])
    else:
        conf_col = cols["confidence"]
        top2_col = cols["top2_confidence"]
        margin_col = cols["margin"]
        entropy_col = cols["entropy"]
        top2_label_col = cols["top2_label"]

        out["confidence"] = df[conf_col].map(safe_float) if conf_col else np.nan
        out["top2_confidence"] = df[top2_col].map(safe_float) if top2_col else np.nan
        out["margin"] = df[margin_col].map(safe_float) if margin_col else out["confidence"] - out["top2_confidence"]
        out["entropy"] = df[entropy_col].map(safe_float) if entropy_col else np.nan
        out["entropy_norm"] = out["entropy"].map(normalize_entropy) if entropy_col else np.nan
        out["top2_grade"] = df[top2_label_col].map(parse_grade) if top2_label_col else np.nan
        out["severe_prob_mass"] = np.nan

    out["pred_label"] = out["pred_grade"].map(lambda x: LABEL_CN.get(int(x), "Unknown") if pd.notna(x) else "Unknown")
    out["top2_label"] = out["top2_grade"].map(lambda x: LABEL_CN.get(int(x), "Unknown") if pd.notna(x) else "Unknown")

    scores = []
    reasons_list = []

    for _, row in out.iterrows():
        score = 0
        reasons = []

        margin = safe_float(row.get("margin"))
        entropy_norm = safe_float(row.get("entropy_norm"))
        confidence = safe_float(row.get("confidence"))
        top2_conf = safe_float(row.get("top2_confidence"))
        pred_grade = safe_float(row.get("pred_grade"))
        top2_grade = safe_float(row.get("top2_grade"))
        severe_prob_mass = safe_float(row.get("severe_prob_mass"))

        if pd.notna(margin):
            if margin < 0.15:
                score += 2
                reasons.append("low_margin_boundary")
            elif margin < 0.30:
                score += 1
                reasons.append("moderate_margin_boundary")

        if pd.notna(entropy_norm):
            if entropy_norm >= 0.75:
                score += 2
                reasons.append("high_entropy")
            elif entropy_norm >= 0.60:
                score += 1
                reasons.append("moderate_entropy")

        if pd.notna(pred_grade) and pd.notna(severe_prob_mass):
            if pred_grade <= 2 and severe_prob_mass >= 0.25:
                score += 3
                reasons.append("potential_severe_undergrading_signal")
            elif pred_grade <= 2 and severe_prob_mass >= 0.15:
                score += 2
                reasons.append("weak_severe_undergrading_signal")

        if pd.notna(pred_grade) and pd.notna(top2_grade) and pd.notna(top2_conf):
            if top2_grade > pred_grade and top2_conf >= 0.20:
                score += 2
                reasons.append("second_choice_more_severe")
            elif top2_grade > pred_grade and top2_conf >= 0.10:
                score += 1
                reasons.append("weak_second_choice_more_severe")

        if pd.notna(confidence) and pd.notna(margin):
            if confidence >= 0.60 and margin < 0.25:
                score += 1
                reasons.append("confident_but_close_decision")

        scores.append(score)
        reasons_list.append(";".join(reasons) if reasons else "routine_low_risk")

    out["pre_review_risk_score"] = scores
    out["risk_reasons"] = reasons_list

    def level(score: int) -> str:
        if score >= 5:
            return "high"
        if score >= 3:
            return "medium"
        return "low"

    out["pre_review_risk_level"] = out["pre_review_risk_score"].map(level)
    out = out.sort_values(
        by=["pre_review_risk_score", "entropy_norm", "confidence"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["review_priority_rank"] = np.arange(1, len(out) + 1)

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


def to_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    show_cols = [
        "review_priority_rank",
        "case_id",
        "pred_label",
        "confidence",
        "top2_label",
        "top2_confidence",
        "margin",
        "entropy_norm",
        "severe_prob_mass",
        "pre_review_risk_score",
        "pre_review_risk_level",
        "risk_reasons",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    view = df[show_cols].head(max_rows).copy()

    for c in ["confidence", "top2_confidence", "margin", "entropy_norm", "severe_prob_mass"]:
        if c in view.columns:
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")

    return dataframe_to_markdown(view)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("experiments/summary/v0_6_6"), type=Path)
    parser.add_argument("--max-md-rows", default=50, type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    used_cols = infer_model_output_columns(df)

    forbidden_present = sorted([c for c in df.columns if c.lower() in FORBIDDEN_FOR_RANKING])

    table = build_pre_review_table(df)

    csv_path = args.output_dir / "pre_review_risk_table.csv"
    md_path = args.output_dir / "pre_review_risk_table.md"

    table.to_csv(csv_path, index=False)

    md = []
    md.append("# v0.6.6 Pre-review Risk Table")
    md.append("")
    md.append("本表仅根据模型输出信号生成预审风险排序，不使用真实标签参与排序。")
    md.append("")
    md.append("## 输入字段识别")
    md.append("")
    for k, v in used_cols.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 排序阶段未使用但输入中存在的后验字段")
    md.append("")
    if forbidden_present:
        for c in forbidden_present:
            md.append(f"- `{c}`")
    else:
        md.append("- 未检测到后验字段")
    md.append("")
    md.append("## 风险排序预览")
    md.append("")
    md.append(to_markdown_table(table, max_rows=args.max_md_rows))
    md.append("")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {md_path}")
    print(f"[INFO] rows: {len(table)}")
    print("[INFO] risk level counts:")
    print(table["pre_review_risk_level"].value_counts().to_string())


if __name__ == "__main__":
    main()
