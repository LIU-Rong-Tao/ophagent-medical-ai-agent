#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.6.7b severity-aware baseline ablation.

This script does not modify v0.6.7 outputs. It reads:
  - experiments/summary/v0_6_7/clinical_event_cases.csv
  - original backbone test_predictions.csv files for class probabilities

It outputs:
  - experiments/summary/v0_6_7b/severity_aware_baseline_tradeoff.csv
  - experiments/summary/v0_6_7b/severity_aware_baseline_best_by_backbone.csv
  - experiments/summary/v0_6_7b/severity_aware_baseline_best_count_summary.csv
  - experiments/summary/v0_6_7b/severity_aware_baseline_key_findings.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/summary/v0_6_7/clinical_event_cases.csv"
OUT_DIR = ROOT / "experiments/summary/v0_6_7b"

PREDICTION_PATHS = {
    "convnext_tiny": ROOT / "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "retfound_official_like": ROOT / "experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "swin_tiny": ROOT / "experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_imagenet": ROOT / "experiments/aptos_vit_base_patch16_imagenet/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_official_like": ROOT / "experiments/aptos_vit_base_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "vit_l_official_like": ROOT / "experiments/aptos_vit_large_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
}

BUDGETS = [0.05, 0.10, 0.20, 0.30]

EVENTS = [
    "general_error",
    "any_undergrading",
    "large_undergrading",
    "referable_dr_miss",
    "vision_threatening_dr_miss",
    "high_confidence_vision_threatening_miss",
]

MAIN_EVENTS = [
    "large_undergrading",
    "vision_threatening_dr_miss",
]

CANONICAL_PROB_COLS = ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4"]


def find_prob_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        ["prob_No DR", "prob_Mild DR", "prob_Moderate DR", "prob_Severe DR", "prob_Proliferative DR"],
        ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4"],
        ["p0", "p1", "p2", "p3", "p4"],
        ["p_no_dr", "p_mild", "p_moderate", "p_severe", "p_pdr"],
    ]

    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols

    prob_like = [c for c in df.columns if "prob" in c.lower() or c.lower().startswith("p_")]
    raise ValueError(f"Cannot find probability columns. Prob-like columns: {prob_like}")


def load_prediction_probs() -> pd.DataFrame:
    rows = []

    for backbone, path in PREDICTION_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {backbone}: {path}")

        pred = pd.read_csv(path)
        prob_cols = find_prob_columns(pred)

        keep = pred[["image_path"] + prob_cols].copy()
        keep["backbone"] = backbone
        keep["image_key"] = keep["image_path"].astype(str).map(lambda x: Path(x).name)
        keep = keep.rename(columns={old: new for old, new in zip(prob_cols, CANONICAL_PROB_COLS)})
        keep = keep[["backbone", "image_key"] + CANONICAL_PROB_COLS]

        rows.append(keep)

    probs = pd.concat(rows, ignore_index=True)
    for c in CANONICAL_PROB_COLS:
        probs[c] = pd.to_numeric(probs[c], errors="coerce")

    return probs


def merge_probs(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df.columns for c in CANONICAL_PROB_COLS):
        return df

    probs = load_prediction_probs()

    before = len(df)
    df = df.copy()
    if "image_key" not in df.columns:
        df["image_key"] = df["image_path"].astype(str).map(lambda x: Path(x).name)
    else:
        df["image_key"] = df["image_key"].astype(str).map(lambda x: Path(x).name)

    merged = df.merge(
        probs,
        on=["backbone", "image_key"],
        how="left",
        validate="many_to_one",
    )
    after = len(merged)

    if before != after:
        raise RuntimeError(f"Row count changed after probability merge: {before} -> {after}")

    missing = merged[CANONICAL_PROB_COLS].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(f"Missing probability rows after merge: {missing}")

    return merged


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def add_severity_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = merge_probs(df)

    ensure_numeric(
        df,
        CANONICAL_PROB_COLS
        + [
            "pred_grade",
            "top2_grade",
            "top2_confidence",
            "severe_prob_mass",
            "review_priority_rank",
        ],
    )

    prob_mat = df[CANONICAL_PROB_COLS].to_numpy(dtype=float)
    grades = np.arange(5, dtype=float)

    df["expected_grade"] = np.nansum(prob_mat * grades.reshape(1, -1), axis=1)

    if "severe_prob_mass" not in df.columns or df["severe_prob_mass"].isna().all():
        df["severe_prob_mass"] = df["prob_3"] + df["prob_4"]

    df["expected_gap"] = df["expected_grade"] - df["pred_grade"]

    df["score_severe_prob_mass_only"] = df["severe_prob_mass"]

    df["score_gated_severe_prob_mass_only"] = np.where(
        df["pred_grade"] <= 2,
        df["severe_prob_mass"],
        -1.0,
    )

    df["score_expected_grade_only"] = df["expected_grade"]

    df["score_expected_gap_only"] = df["expected_gap"]

    top2_conf = df["top2_confidence"].fillna(0.0)
    df["score_top2_more_severe_only"] = np.where(
        df["top2_grade"] > df["pred_grade"],
        top2_conf,
        -1.0,
    )

    return df


def rank_for_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ophagent_combined":
        return df.sort_values(
            by=["review_priority_rank", "case_id"],
            ascending=[True, True],
            kind="mergesort",
        )

    score_col = f"score_{method}"
    if score_col not in df.columns:
        raise KeyError(f"Missing score column: {score_col}")

    return df.sort_values(
        by=[score_col, "case_id"],
        ascending=[False, True],
        kind="mergesort",
    )


def compute_tradeoff(df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "ophagent_combined",
        "severe_prob_mass_only",
        "gated_severe_prob_mass_only",
        "expected_grade_only",
        "expected_gap_only",
        "top2_more_severe_only",
    ]

    rows = []

    for backbone, g in df.groupby("backbone", sort=True):
        n = len(g)

        for method in methods:
            ranked = rank_for_method(g, method).reset_index(drop=True)

            for event in EVENTS:
                if event not in ranked.columns:
                    raise KeyError(f"Missing event column: {event}")

                event_values = ranked[event].astype(bool).to_numpy()
                total = int(event_values.sum())
                base_rate = total / n if n else 0.0

                for budget in BUDGETS:
                    reviewed_n = int(math.ceil(n * budget))
                    reviewed_n = max(1, min(reviewed_n, n))
                    auto_released_n = n - reviewed_n

                    captured = int(event_values[:reviewed_n].sum())
                    residual = total - captured

                    recall = captured / total if total else np.nan
                    precision = captured / reviewed_n if reviewed_n else np.nan
                    lift = precision / base_rate if base_rate else np.nan
                    residual_rate = residual / auto_released_n if auto_released_n else np.nan
                    errors_per_100_reviewed = precision * 100 if not np.isnan(precision) else np.nan
                    nnr = 1 / precision if precision and not np.isnan(precision) else np.nan

                    rows.append(
                        {
                            "backbone": backbone,
                            "ranking_method": method,
                            "clinical_event": event,
                            "review_budget": budget,
                            "reviewed_n": reviewed_n,
                            "auto_released_n": auto_released_n,
                            "dangerous_error_total": total,
                            "dangerous_error_captured": captured,
                            "dangerous_error_recall_at_k": recall,
                            "dangerous_error_precision_at_k": precision,
                            "dangerous_error_lift_vs_random": lift,
                            "residual_dangerous_error_count": residual,
                            "residual_dangerous_error_rate": residual_rate,
                            "dangerous_errors_per_100_reviewed": errors_per_100_reviewed,
                            "number_needed_to_review": nnr,
                        }
                    )

    return pd.DataFrame(rows)


def best_summaries(tradeoff: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = tradeoff.sort_values(
        by=[
            "clinical_event",
            "review_budget",
            "backbone",
            "dangerous_error_recall_at_k",
            "dangerous_error_precision_at_k",
            "residual_dangerous_error_count",
            "ranking_method",
        ],
        ascending=[True, True, True, False, False, True, True],
        kind="mergesort",
    )

    best_by_backbone = (
        sorted_df.groupby(["clinical_event", "review_budget", "backbone"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    count_summary = (
        best_by_backbone.groupby(["clinical_event", "review_budget", "ranking_method"])
        .size()
        .reset_index(name="best_backbone_count")
        .sort_values(
            by=["clinical_event", "review_budget", "best_backbone_count", "ranking_method"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)
    )

    return best_by_backbone, count_summary


def mean_summary(tradeoff: pd.DataFrame) -> pd.DataFrame:
    return (
        tradeoff.groupby(["clinical_event", "review_budget", "ranking_method"], as_index=False)
        .agg(
            mean_recall=("dangerous_error_recall_at_k", "mean"),
            mean_lift=("dangerous_error_lift_vs_random", "mean"),
            total_captured=("dangerous_error_captured", "sum"),
            total_dangerous=("dangerous_error_total", "sum"),
            total_residual=("residual_dangerous_error_count", "sum"),
        )
        .sort_values(
            by=["clinical_event", "review_budget", "mean_recall", "mean_lift"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )


def md_table(df: pd.DataFrame, cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if c == "review_budget":
                    vals.append(f"{v:.0%}")
                elif "recall" in c:
                    vals.append(f"{v:.1%}")
                elif "lift" in c:
                    vals.append(f"{v:.2f}x")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def write_key_findings(
    tradeoff: pd.DataFrame,
    best_by_backbone: pd.DataFrame,
    count_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    ms = mean_summary(tradeoff)

    lines = []
    lines.append("# v0.6.7b Severity-aware Baseline Ablation")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append(
        "This ablation checks whether the original OphAgent combined rule is more stable than "
        "single severity-aware ranking signals for dangerous undergrading events."
    )
    lines.append("")
    lines.append("The ranking stage does not use true labels. True labels are used only for posterior evaluation.")
    lines.append("")
    lines.append("## Ranking methods")
    lines.append("")
    lines.append("- `ophagent_combined`: original v0.6.7 `review_priority_rank`.")
    lines.append("- `severe_prob_mass_only`: sort by `P(Severe) + P(PDR)`.")
    lines.append("- `gated_severe_prob_mass_only`: prioritize `pred_grade <= 2`, then sort by severe probability mass.")
    lines.append("- `expected_grade_only`: sort by expected grade from class probabilities.")
    lines.append("- `expected_gap_only`: sort by `expected_grade - pred_grade`.")
    lines.append("- `top2_more_severe_only`: prioritize samples whose top-2 grade is more severe than top-1.")
    lines.append("")
    lines.append("## Main event mean summary")
    lines.append("")
    main = ms[
        ms["clinical_event"].isin(MAIN_EVENTS)
        & ms["review_budget"].isin([0.10, 0.20, 0.30])
    ].copy()
    lines.append(
        md_table(
            main,
            [
                "clinical_event",
                "review_budget",
                "ranking_method",
                "mean_recall",
                "mean_lift",
                "total_captured",
                "total_dangerous",
                "total_residual",
            ],
        )
    )
    lines.append("")
    lines.append("## Winner count by backbone")
    lines.append("")
    count_main = count_summary[
        count_summary["clinical_event"].isin(MAIN_EVENTS)
        & count_summary["review_budget"].isin([0.10, 0.20, 0.30])
    ].copy()
    lines.append(
        md_table(
            count_main,
            ["clinical_event", "review_budget", "ranking_method", "best_backbone_count"],
        )
    )
    lines.append("")
    lines.append("## Interpretation checklist")
    lines.append("")
    lines.append("1. Does `ophagent_combined` remain most stable for `large_undergrading` and `vision_threatening_dr_miss`?")
    lines.append("2. If not, which simple severity-aware signal explains most of the gain?")
    lines.append("3. Do different dangerous events require different ranking signals?")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    df = add_severity_scores(df)

    tradeoff = compute_tradeoff(df)
    best_by_backbone, count_summary = best_summaries(tradeoff)

    tradeoff.to_csv(OUT_DIR / "severity_aware_baseline_tradeoff.csv", index=False)
    best_by_backbone.to_csv(OUT_DIR / "severity_aware_baseline_best_by_backbone.csv", index=False)
    count_summary.to_csv(OUT_DIR / "severity_aware_baseline_best_count_summary.csv", index=False)

    write_key_findings(
        tradeoff,
        best_by_backbone,
        count_summary,
        OUT_DIR / "severity_aware_baseline_key_findings.md",
    )

    print(f"Input: {INPUT}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Rows tradeoff: {len(tradeoff)}")
    print("Done.")


if __name__ == "__main__":
    main()
