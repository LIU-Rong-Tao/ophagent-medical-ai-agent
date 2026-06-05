#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare OphAgent pre-review ranking against uncertainty baselines.

Internal v0.6.6 analysis.

Methods:
- random_expected: theoretical random review baseline
- random_mc_1000: Monte Carlo random review baseline, mean/std over repeated random sampling
- confidence_only: low confidence first
- margin_only: small top1-top2 margin first
- entropy_only: high entropy first
- ophagent_combined: existing OphAgent combined rule

真实标签只用于后验验证，不参与任何预审排序。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


DIR_TO_LABEL = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}


def infer_true_label_from_path(image_path: str) -> str:
    parts = Path(str(image_path)).parts
    for p in parts:
        if p in DIR_TO_LABEL:
            return DIR_TO_LABEL[p]
    raise ValueError(f"Cannot infer true label from image_path: {image_path}")


def rank_dataframe(df: pd.DataFrame, method: str) -> pd.DataFrame:
    d = df.copy()

    if method == "confidence_only":
        return d.sort_values(["confidence"], ascending=[True]).reset_index(drop=True)

    if method == "margin_only":
        return d.sort_values(["margin"], ascending=[True]).reset_index(drop=True)

    if method == "entropy_only":
        entropy_col = "entropy_norm" if "entropy_norm" in d.columns else "entropy"
        return d.sort_values([entropy_col], ascending=[False]).reset_index(drop=True)

    if method == "uncertainty_rank_fusion":
        entropy_col = "entropy_norm" if "entropy_norm" in d.columns else "entropy"

        # Lower confidence = higher risk.
        d["_rank_confidence"] = d["confidence"].rank(method="average", ascending=True)

        # Smaller margin = higher risk.
        d["_rank_margin"] = d["margin"].rank(method="average", ascending=True)

        # Higher entropy = higher risk.
        d["_rank_entropy"] = d[entropy_col].rank(method="average", ascending=False)

        # Lower fused rank = higher review priority.
        d["_uncertainty_fusion_rank"] = (
            d["_rank_confidence"] + d["_rank_margin"] + d["_rank_entropy"]
        ) / 3.0

        return d.sort_values(["_uncertainty_fusion_rank"], ascending=[True]).reset_index(drop=True)

    if method == "ophagent_combined":
        if "review_priority_rank" in d.columns:
            return d.sort_values(["review_priority_rank"], ascending=[True]).reset_index(drop=True)
        return d.sort_values(["pre_review_risk_score"], ascending=[False]).reset_index(drop=True)

    raise ValueError(f"Unknown deterministic ranking method: {method}")


def evaluate_ranked(df_ranked: pd.DataFrame, fractions: List[float]) -> Dict[str, float]:
    total_n = len(df_ranked)
    total_errors = int(df_ranked["is_error"].sum())
    overall_error_rate = total_errors / total_n if total_n else 0.0

    out: Dict[str, float] = {
        "total_n": total_n,
        "total_errors": total_errors,
        "overall_error_rate": overall_error_rate,
    }

    for frac in fractions:
        k = max(1, int(round(total_n * frac)))
        top = df_ranked.head(k)
        error_count = float(top["is_error"].sum())
        random_expected_error_count = k * overall_error_rate

        error_rate = error_count / k if k else 0.0
        extra_error_count_vs_random = error_count - random_expected_error_count
        enrichment_ratio = error_rate / overall_error_rate if overall_error_rate > 0 else None
        error_recall = error_count / total_errors if total_errors > 0 else None

        tag = f"top{int(frac * 100)}"
        out[f"{tag}_k"] = k
        out[f"{tag}_error_count"] = error_count
        out[f"{tag}_random_expected_error_count"] = random_expected_error_count
        out[f"{tag}_extra_error_count_vs_random"] = extra_error_count_vs_random
        out[f"{tag}_error_rate"] = error_rate
        out[f"{tag}_enrichment_ratio"] = enrichment_ratio
        out[f"{tag}_error_recall"] = error_recall

    return out


def evaluate_random_expected(df: pd.DataFrame, fractions: List[float]) -> Dict[str, float]:
    total_n = len(df)
    total_errors = int(df["is_error"].sum())
    overall_error_rate = total_errors / total_n if total_n else 0.0

    out: Dict[str, float] = {
        "total_n": total_n,
        "total_errors": total_errors,
        "overall_error_rate": overall_error_rate,
    }

    for frac in fractions:
        k = max(1, int(round(total_n * frac)))
        random_expected_error_count = k * overall_error_rate

        tag = f"top{int(frac * 100)}"
        out[f"{tag}_k"] = k
        out[f"{tag}_error_count"] = random_expected_error_count
        out[f"{tag}_random_expected_error_count"] = random_expected_error_count
        out[f"{tag}_extra_error_count_vs_random"] = 0.0
        out[f"{tag}_error_rate"] = overall_error_rate
        out[f"{tag}_enrichment_ratio"] = 1.0
        out[f"{tag}_error_recall"] = frac

    return out


def evaluate_random_mc(
    df: pd.DataFrame,
    fractions: List[float],
    n_runs: int = 1000,
    random_seed: int = 42,
) -> Dict[str, float]:
    total_n = len(df)
    total_errors = int(df["is_error"].sum())
    overall_error_rate = total_errors / total_n if total_n else 0.0

    out: Dict[str, float] = {
        "total_n": total_n,
        "total_errors": total_errors,
        "overall_error_rate": overall_error_rate,
    }

    for frac in fractions:
        k = max(1, int(round(total_n * frac)))
        random_expected_error_count = k * overall_error_rate
        counts = []

        for i in range(n_runs):
            sampled = df.sample(n=k, replace=False, random_state=random_seed + i)
            counts.append(float(sampled["is_error"].sum()))

        s = pd.Series(counts)
        mean_count = float(s.mean())
        std_count = float(s.std(ddof=1))
        p025 = float(s.quantile(0.025))
        p975 = float(s.quantile(0.975))

        error_rate = mean_count / k if k else 0.0
        extra_error_count_vs_random = mean_count - random_expected_error_count
        enrichment_ratio = error_rate / overall_error_rate if overall_error_rate > 0 else None
        error_recall = mean_count / total_errors if total_errors > 0 else None

        tag = f"top{int(frac * 100)}"
        out[f"{tag}_k"] = k
        out[f"{tag}_error_count"] = mean_count
        out[f"{tag}_error_count_std"] = std_count
        out[f"{tag}_error_count_p025"] = p025
        out[f"{tag}_error_count_p975"] = p975
        out[f"{tag}_random_expected_error_count"] = random_expected_error_count
        out[f"{tag}_extra_error_count_vs_random"] = extra_error_count_vs_random
        out[f"{tag}_error_rate"] = error_rate
        out[f"{tag}_enrichment_ratio"] = enrichment_ratio
        out[f"{tag}_error_recall"] = error_recall

    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---" for _ in cols]) + " |")
    for _, row in df.iterrows():
        values = []
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                values.append("")
            else:
                values.append(str(v).replace("|", "/").strip())
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def process_one_table(path: Path, fractions: List[float], n_random_runs: int) -> List[Dict[str, object]]:
    backbone = path.parent.name
    df = pd.read_csv(path)

    required = ["image_path", "pred_label", "confidence", "margin", "pre_review_risk_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["true_label"] = df["image_path"].map(infer_true_label_from_path)
    df["is_error"] = df["true_label"] != df["pred_label"]

    rows: List[Dict[str, object]] = []

    random_expected = {
        "backbone": backbone,
        "ranking_method": "random_expected",
    }
    random_expected.update(evaluate_random_expected(df, fractions))
    rows.append(random_expected)

    random_mc = {
        "backbone": backbone,
        "ranking_method": f"random_mc_{n_random_runs}",
    }
    random_mc.update(evaluate_random_mc(df, fractions, n_runs=n_random_runs))
    rows.append(random_mc)

    for method in [
        "confidence_only",
        "margin_only",
        "entropy_only",
        "uncertainty_rank_fusion",
        "ophagent_combined",
    ]:
        ranked = rank_dataframe(df, method)
        row = {
            "backbone": backbone,
            "ranking_method": method,
        }
        row.update(evaluate_ranked(ranked, fractions))
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="experiments/summary/v0_6_6/full_test_backbones",
        help="Root directory containing per-backbone pre_review_risk_table.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/summary/v0_6_6/full_test_backbones",
        help="Output directory.",
    )
    parser.add_argument(
        "--n-random-runs",
        type=int,
        default=1000,
        help="Monte Carlo random baseline repetitions.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(root.glob("*/pre_review_risk_table.csv"))
    if not paths:
        raise SystemExit(f"No pre_review_risk_table.csv found under: {root}")

    all_rows = []
    for p in paths:
        print(f"[PROCESS] {p}")
        all_rows.extend(
            process_one_table(
                p,
                fractions=[0.1, 0.2, 0.3],
                n_random_runs=args.n_random_runs,
            )
        )

    out = pd.DataFrame(all_rows)

    rounded = out.copy()
    for c in rounded.columns:
        if rounded[c].dtype.kind in "fc":
            rounded[c] = rounded[c].round(4)

    csv_path = output_dir / "baseline_ranking_comparison.csv"
    md_path = output_dir / "baseline_ranking_comparison.md"

    rounded.to_csv(csv_path, index=False)

    compact_cols = [
        "backbone",
        "ranking_method",
        "overall_error_rate",
        "top20_error_count",
        "top20_error_count_std",
        "top20_random_expected_error_count",
        "top20_extra_error_count_vs_random",
        "top20_error_rate",
        "top20_enrichment_ratio",
        "top20_error_recall",
    ]
    compact_cols = [c for c in compact_cols if c in rounded.columns]
    md_path.write_text(dataframe_to_markdown(rounded[compact_cols]), encoding="utf-8")

    print(f"[OK] saved: {csv_path}")
    print(f"[OK] saved: {md_path}")

    print("\n[TOP20 SUMMARY]")
    print(rounded[compact_cols].to_string(index=False))


if __name__ == "__main__":
    main()
