#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v0.6.8b Robustness and Mechanism Audit

评价口径必须和 v0.6.8 对齐：

- pooled-backbone training
- per-backbone test reporting

因此 bootstrap 不能把 6 个 backbone 的 6600 条记录混在一起排序。
正确做法是：

1. 每次从 image_key 层面有放回抽样；
2. 每个 backbone 内部分别做 Top20% 排序；
3. 再聚合 captured / total / residual；
4. 在同一个 bootstrap replicate 内比较 learned_logistic 和 simple rule。

本脚本输出 CSV，不自动生成 Markdown。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "summary" / "v0_6_8" / "learned_deferral_fold_predictions.csv"
OUT_DIR = ROOT / "experiments" / "summary" / "v0_6_8b"

REVIEW_BUDGET = 0.20
VERIFY_BOOTSTRAP = 50
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

COMPARISONS = [
    {
        "target": "large_undergrading",
        "learned_method": "learned_logistic",
        "simple_method": "expected_gap_only",
        "comparison_role": "primary",
    },
    {
        "target": "vision_threatening_dr_miss",
        "learned_method": "learned_logistic",
        "simple_method": "gated_severe_prob_mass_only",
        "comparison_role": "primary",
    },
    {
        "target": "large_undergrading",
        "learned_method": "learned_logistic",
        "simple_method": "ophagent_combined",
        "comparison_role": "secondary",
    },
    {
        "target": "vision_threatening_dr_miss",
        "learned_method": "learned_logistic",
        "simple_method": "ophagent_combined",
        "comparison_role": "secondary",
    },
    {
        "target": "dangerous_undergrading",
        "learned_method": "learned_logistic",
        "simple_method": "gated_severe_prob_mass_only",
        "comparison_role": "secondary",
    },
    {
        "target": "dangerous_undergrading",
        "learned_method": "learned_logistic",
        "simple_method": "ophagent_combined",
        "comparison_role": "secondary",
    },
]


def load_predictions() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "target",
        "backbone",
        "case_id",
        "image_key",
        "true_grade",
        "pred_grade",
        "confidence",
        "margin",
        "entropy_norm",
        "expected_gap",
        "gated_severe_prob_mass",
        "top2_more_severe_conf",
        "review_priority_rank",
        "score_learned_logistic",
        "large_undergrading",
        "vision_threatening_dr_miss",
        "dangerous_undergrading",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "row_id" not in df.columns:
        df = df.copy()
        df["row_id"] = np.arange(len(df))

    return df


def sorted_df(g: pd.DataFrame, method: str) -> pd.DataFrame:
    g = g.copy()
    tie_cols = ["row_id", "case_id"]

    if method == "learned_logistic":
        return g.sort_values(
            ["score_learned_logistic"] + tie_cols,
            ascending=[False, True, True],
            kind="mergesort",
        )

    if method == "expected_gap_only":
        return g.sort_values(
            ["expected_gap"] + tie_cols,
            ascending=[False, True, True],
            kind="mergesort",
        )

    if method == "gated_severe_prob_mass_only":
        g["_rank_score"] = np.where(g["pred_grade"] <= 2, g["gated_severe_prob_mass"], -1.0)
        out = g.sort_values(
            ["_rank_score"] + tie_cols,
            ascending=[False, True, True],
            kind="mergesort",
        )
        return out.drop(columns=["_rank_score"])

    if method == "ophagent_combined":
        return g.sort_values(
            ["review_priority_rank"] + tie_cols,
            ascending=[True, True, True],
            kind="mergesort",
        )

    raise ValueError(f"Unsupported method: {method}")


def evaluate_unweighted_topk(g: pd.DataFrame, target: str, method: str) -> Dict[str, float]:
    ranked = sorted_df(g, method)
    n = len(ranked)
    k = int(np.ceil(n * REVIEW_BUDGET))
    top = ranked.head(k)

    total = float(ranked[target].sum())
    captured = float(top[target].sum())
    residual = total - captured

    recall = captured / total if total > 0 else np.nan
    precision = captured / k if k > 0 else np.nan
    base_rate = total / n if n > 0 else np.nan
    lift = precision / base_rate if base_rate > 0 else np.nan

    return {
        "n_records": float(n),
        "reviewed_n": float(k),
        "total_dangerous": total,
        "captured": captured,
        "residual": residual,
        "recall": recall,
        "precision": precision,
        "lift": lift,
    }


def make_image_code_map(df: pd.DataFrame) -> Tuple[Dict[str, int], np.ndarray]:
    image_keys = np.array(sorted(df["image_key"].unique()))
    image_to_code = {k: i for i, k in enumerate(image_keys)}
    return image_to_code, image_keys


def prepare_ranked_arrays(
    g: pd.DataFrame,
    target: str,
    method: str,
    image_to_code: Dict[str, int],
) -> Dict[str, np.ndarray]:
    ranked = sorted_df(g, method)
    return {
        "codes": ranked["image_key"].map(image_to_code).to_numpy(dtype=np.int64),
        "events": ranked[target].to_numpy(dtype=np.float64),
    }


def evaluate_weighted_ranked_arrays(
    ranked_arrays: Dict[str, np.ndarray],
    counts: np.ndarray,
) -> Dict[str, float]:
    codes = ranked_arrays["codes"]
    events = ranked_arrays["events"]

    weights = counts[codes].astype(np.float64)
    weighted_n = float(weights.sum())
    k = int(np.ceil(weighted_n * REVIEW_BUDGET))

    if weighted_n == 0:
        return {
            "n_records": 0.0,
            "reviewed_n": 0.0,
            "total_dangerous": 0.0,
            "captured": 0.0,
            "residual": 0.0,
            "recall": np.nan,
            "precision": np.nan,
            "lift": np.nan,
        }

    total = float(np.sum(weights * events))

    cum = np.cumsum(weights)
    before = cum - weights

    # 严格模拟复制版 Top-K 边界：
    # 如果某条 record 的 weight=3，但 Top-K 只剩 1 个名额，只纳入 1 份。
    included = np.minimum(weights, np.maximum(0.0, k - before))
    included = np.maximum(included, 0.0)

    captured = float(np.sum(included * events))
    residual = total - captured

    recall = captured / total if total > 0 else np.nan
    precision = captured / k if k > 0 else np.nan
    base_rate = total / weighted_n if weighted_n > 0 else np.nan
    lift = precision / base_rate if base_rate > 0 else np.nan

    return {
        "n_records": weighted_n,
        "reviewed_n": float(k),
        "total_dangerous": total,
        "captured": captured,
        "residual": residual,
        "recall": recall,
        "precision": precision,
        "lift": lift,
    }


def aggregate_metric(per_backbone_rows: list[Dict[str, float]]) -> Dict[str, float]:
    total_dangerous = sum(r["total_dangerous"] for r in per_backbone_rows)
    captured = sum(r["captured"] for r in per_backbone_rows)
    residual = sum(r["residual"] for r in per_backbone_rows)
    reviewed_n = sum(r["reviewed_n"] for r in per_backbone_rows)
    n_records = sum(r["n_records"] for r in per_backbone_rows)

    recall = captured / total_dangerous if total_dangerous > 0 else np.nan
    precision = captured / reviewed_n if reviewed_n > 0 else np.nan
    base_rate = total_dangerous / n_records if n_records > 0 else np.nan
    lift = precision / base_rate if base_rate > 0 else np.nan

    mean_backbone_recall = float(np.nanmean([r["recall"] for r in per_backbone_rows]))
    mean_backbone_lift = float(np.nanmean([r["lift"] for r in per_backbone_rows]))

    return {
        "n_records": n_records,
        "reviewed_n": reviewed_n,
        "total_dangerous": total_dangerous,
        "captured": captured,
        "residual": residual,
        "recall": recall,
        "precision": precision,
        "lift": lift,
        "mean_backbone_recall": mean_backbone_recall,
        "mean_backbone_lift": mean_backbone_lift,
    }


def brute_force_per_backbone_eval(
    df_target: pd.DataFrame,
    target: str,
    method: str,
    sampled_codes: np.ndarray,
    code_to_positions_by_backbone: Dict[str, Dict[int, np.ndarray]],
) -> Dict[str, float]:
    per_backbone = []

    for backbone, code_to_positions in code_to_positions_by_backbone.items():
        positions = np.concatenate([code_to_positions[int(c)] for c in sampled_codes])
        g_boot = df_target[df_target["backbone"] == backbone].iloc[positions].copy()
        per_backbone.append(evaluate_unweighted_topk(g_boot, target, method))

    return aggregate_metric(per_backbone)


def weighted_per_backbone_eval(
    ranked_by_backbone: Dict[str, Dict[str, np.ndarray]],
    counts: np.ndarray,
) -> Dict[str, float]:
    per_backbone = [
        evaluate_weighted_ranked_arrays(arrays, counts)
        for _, arrays in sorted(ranked_by_backbone.items())
    ]
    return aggregate_metric(per_backbone)


def validate_weighted_equivalence(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Equivalence validation: per-backbone brute-force vs weighted bootstrap ===")

    image_to_code, image_keys = make_image_code_map(df)
    n_images = len(image_keys)
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    for comp in COMPARISONS:
        target = comp["target"]
        learned_method = comp["learned_method"]
        simple_method = comp["simple_method"]

        df_target = df[df["target"] == target].reset_index(drop=True).copy()
        df_target["_image_code"] = df_target["image_key"].map(image_to_code).astype(int)

        code_to_positions_by_backbone = {}
        for backbone, g in df_target.groupby("backbone", sort=True):
            g = g.reset_index(drop=True)
            codes = g["_image_code"].to_numpy()
            code_to_positions_by_backbone[backbone] = {
                code: np.flatnonzero(codes == code)
                for code in range(n_images)
            }

        ranked_cache = {}
        for method in [learned_method, simple_method]:
            ranked_cache[method] = {
                backbone: prepare_ranked_arrays(g, target, method, image_to_code)
                for backbone, g in df_target.groupby("backbone", sort=True)
            }

        print(f"validate: {target} | {learned_method} vs {simple_method}")

        for b in range(VERIFY_BOOTSTRAP):
            sampled_codes = rng.integers(0, n_images, size=n_images)
            counts = np.bincount(sampled_codes, minlength=n_images)

            for method in [learned_method, simple_method]:
                brute = brute_force_per_backbone_eval(
                    df_target=df_target,
                    target=target,
                    method=method,
                    sampled_codes=sampled_codes,
                    code_to_positions_by_backbone=code_to_positions_by_backbone,
                )
                weighted = weighted_per_backbone_eval(ranked_cache[method], counts)

                row = {
                    "bootstrap_id": b,
                    "target": target,
                    "method": method,
                    "brute_recall": brute["recall"],
                    "weighted_recall": weighted["recall"],
                    "recall_abs_diff": abs(brute["recall"] - weighted["recall"]),
                    "brute_mean_backbone_recall": brute["mean_backbone_recall"],
                    "weighted_mean_backbone_recall": weighted["mean_backbone_recall"],
                    "mean_backbone_recall_abs_diff": abs(
                        brute["mean_backbone_recall"] - weighted["mean_backbone_recall"]
                    ),
                    "brute_captured": brute["captured"],
                    "weighted_captured": weighted["captured"],
                    "captured_abs_diff": abs(brute["captured"] - weighted["captured"]),
                    "brute_total": brute["total_dangerous"],
                    "weighted_total": weighted["total_dangerous"],
                    "total_abs_diff": abs(brute["total_dangerous"] - weighted["total_dangerous"]),
                    "brute_residual": brute["residual"],
                    "weighted_residual": weighted["residual"],
                    "residual_abs_diff": abs(brute["residual"] - weighted["residual"]),
                }
                rows.append(row)

                if (
                    row["recall_abs_diff"] > 1e-12
                    or row["mean_backbone_recall_abs_diff"] > 1e-12
                    or row["captured_abs_diff"] > 1e-12
                    or row["total_abs_diff"] > 1e-12
                    or row["residual_abs_diff"] > 1e-12
                ):
                    raise AssertionError(
                        "Weighted per-backbone bootstrap does not match brute-force bootstrap. "
                        f"target={target}, method={method}, bootstrap_id={b}, row={row}"
                    )

    out = pd.DataFrame(rows)
    print("Per-backbone equivalence validation passed.")
    return out


def run_weighted_bootstrap(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== Formal weighted paired clustered bootstrap: per-backbone endpoint ===")

    image_to_code, image_keys = make_image_code_map(df)
    n_images = len(image_keys)
    rng = np.random.default_rng(RANDOM_SEED + 1000)

    replicate_rows = []
    summary_rows = []

    for comp in COMPARISONS:
        target = comp["target"]
        learned_method = comp["learned_method"]
        simple_method = comp["simple_method"]
        role = comp["comparison_role"]

        df_target = df[df["target"] == target].reset_index(drop=True).copy()

        ranked_learned = {
            backbone: prepare_ranked_arrays(g, target, learned_method, image_to_code)
            for backbone, g in df_target.groupby("backbone", sort=True)
        }
        ranked_simple = {
            backbone: prepare_ranked_arrays(g, target, simple_method, image_to_code)
            for backbone, g in df_target.groupby("backbone", sort=True)
        }

        print(f"bootstrap: {target} | {learned_method} vs {simple_method}")

        for b in range(N_BOOTSTRAP):
            sampled_codes = rng.integers(0, n_images, size=n_images)
            counts = np.bincount(sampled_codes, minlength=n_images)

            learned = weighted_per_backbone_eval(ranked_learned, counts)
            simple = weighted_per_backbone_eval(ranked_simple, counts)

            replicate_rows.append(
                {
                    "bootstrap_id": b,
                    "target": target,
                    "comparison_role": role,
                    "learned_method": learned_method,
                    "simple_method": simple_method,
                    "review_budget": REVIEW_BUDGET,
                    "endpoint_scope": "per_backbone_aggregate",
                    "learned_recall": learned["recall"],
                    "simple_recall": simple["recall"],
                    "recall_diff_learned_minus_simple": learned["recall"] - simple["recall"],
                    "learned_mean_backbone_recall": learned["mean_backbone_recall"],
                    "simple_mean_backbone_recall": simple["mean_backbone_recall"],
                    "mean_backbone_recall_diff": (
                        learned["mean_backbone_recall"] - simple["mean_backbone_recall"]
                    ),
                    "learned_residual": learned["residual"],
                    "simple_residual": simple["residual"],
                    "residual_diff_learned_minus_simple": learned["residual"] - simple["residual"],
                    "learned_lift": learned["lift"],
                    "simple_lift": simple["lift"],
                    "lift_diff_learned_minus_simple": learned["lift"] - simple["lift"],
                    "learned_mean_backbone_lift": learned["mean_backbone_lift"],
                    "simple_mean_backbone_lift": simple["mean_backbone_lift"],
                    "mean_backbone_lift_diff": (
                        learned["mean_backbone_lift"] - simple["mean_backbone_lift"]
                    ),
                    "learned_captured": learned["captured"],
                    "simple_captured": simple["captured"],
                    "total_dangerous": learned["total_dangerous"],
                    "learned_wins": int(learned["recall"] > simple["recall"]),
                    "tie": int(learned["recall"] == simple["recall"]),
                }
            )

    replicates = pd.DataFrame(replicate_rows)

    for keys, g in replicates.groupby(
        ["target", "comparison_role", "learned_method", "simple_method", "review_budget", "endpoint_scope"],
        sort=True,
    ):
        target, role, learned_method, simple_method, budget, endpoint_scope = keys

        diff = g["recall_diff_learned_minus_simple"]
        mb_diff = g["mean_backbone_recall_diff"]
        residual_diff = g["residual_diff_learned_minus_simple"]
        lift_diff = g["lift_diff_learned_minus_simple"]

        summary_rows.append(
            {
                "target": target,
                "comparison_role": role,
                "learned_method": learned_method,
                "simple_method": simple_method,
                "review_budget": budget,
                "endpoint_scope": endpoint_scope,
                "n_bootstrap": len(g),
                "mean_recall_diff": diff.mean(),
                "median_recall_diff": diff.median(),
                "ci95_low_recall_diff": diff.quantile(0.025),
                "ci95_high_recall_diff": diff.quantile(0.975),
                "mean_backbone_recall_diff": mb_diff.mean(),
                "ci95_low_mean_backbone_recall_diff": mb_diff.quantile(0.025),
                "ci95_high_mean_backbone_recall_diff": mb_diff.quantile(0.975),
                "bootstrap_win_rate": g["learned_wins"].mean(),
                "tie_rate": g["tie"].mean(),
                "mean_residual_diff": residual_diff.mean(),
                "median_residual_diff": residual_diff.median(),
                "ci95_low_residual_diff": residual_diff.quantile(0.025),
                "ci95_high_residual_diff": residual_diff.quantile(0.975),
                "mean_lift_diff": lift_diff.mean(),
                "median_lift_diff": lift_diff.median(),
                "ci95_low_lift_diff": lift_diff.quantile(0.025),
                "ci95_high_lift_diff": lift_diff.quantile(0.975),
                "mean_learned_captured": g["learned_captured"].mean(),
                "mean_simple_captured": g["simple_captured"].mean(),
                "mean_total_dangerous": g["total_dangerous"].mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return replicates, summary


def capture_set(g: pd.DataFrame, target: str, method: str) -> pd.DataFrame:
    ranked = sorted_df(g, method).reset_index(drop=True)
    k = int(np.ceil(len(ranked) * REVIEW_BUDGET))
    top = ranked.head(k).copy()
    return top[top[target] == 1].copy()


def overlap_for_comparison(
    df_target: pd.DataFrame,
    target: str,
    learned_method: str,
    simple_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    case_rows = []

    for backbone, g in df_target.groupby("backbone", sort=True):
        learned_cap = capture_set(g, target, learned_method)
        simple_cap = capture_set(g, target, simple_method)

        id_cols = ["backbone", "case_id", "image_key"]
        learned_ids = set(map(tuple, learned_cap[id_cols].to_numpy()))
        simple_ids = set(map(tuple, simple_cap[id_cols].to_numpy()))

        both = learned_ids & simple_ids
        learned_only = learned_ids - simple_ids
        simple_only = simple_ids - learned_ids

        dangerous = g[g[target] == 1].copy()
        dangerous_ids = set(map(tuple, dangerous[id_cols].to_numpy()))
        missed_by_both = dangerous_ids - (learned_ids | simple_ids)

        groups = {
            "both_captured": both,
            "learned_only": learned_only,
            "simple_only": simple_only,
            "missed_by_both": missed_by_both,
        }

        summary_rows.append(
            {
                "target": target,
                "backbone": backbone,
                "learned_method": learned_method,
                "simple_method": simple_method,
                "review_budget": REVIEW_BUDGET,
                "endpoint_scope": "per_backbone",
                "total_dangerous": len(dangerous_ids),
                "both_captured": len(both),
                "learned_only": len(learned_only),
                "simple_only": len(simple_only),
                "missed_by_both": len(missed_by_both),
                "learned_captured": len(learned_ids),
                "simple_captured": len(simple_ids),
            }
        )

        for group_name, ids in groups.items():
            if not ids:
                continue
            ids_df = pd.DataFrame(list(ids), columns=id_cols)
            merged = ids_df.merge(g, on=id_cols, how="left")
            merged["target"] = target
            merged["capture_group"] = group_name
            merged["learned_method"] = learned_method
            merged["simple_method"] = simple_method

            keep = [
                "target",
                "backbone",
                "case_id",
                "image_key",
                "capture_group",
                "learned_method",
                "simple_method",
                "true_grade",
                "pred_grade",
                "confidence",
                "margin",
                "entropy_norm",
                "expected_gap",
                "gated_severe_prob_mass",
                "top2_more_severe_conf",
                "score_learned_logistic",
                "review_priority_rank",
            ]
            case_rows.append(merged[keep])

    summary = pd.DataFrame(summary_rows)
    cases = pd.concat(case_rows, ignore_index=True) if case_rows else pd.DataFrame()
    return summary, cases


def run_overlap(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== Top20% capture overlap: per-backbone endpoint ===")

    all_summary = []
    all_cases = []

    for comp in COMPARISONS:
        target = comp["target"]
        learned_method = comp["learned_method"]
        simple_method = comp["simple_method"]
        role = comp["comparison_role"]

        print(f"overlap: {target} | {learned_method} vs {simple_method}")

        df_target = df[df["target"] == target].copy()
        summary, cases = overlap_for_comparison(
            df_target=df_target,
            target=target,
            learned_method=learned_method,
            simple_method=simple_method,
        )
        summary["comparison_role"] = role
        cases["comparison_role"] = role

        all_summary.append(summary)
        all_cases.append(cases)

    return pd.concat(all_summary, ignore_index=True), pd.concat(all_cases, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions()

    print("Loaded OOF prediction table")
    print("rows:", len(df))
    print("unique image_key:", df["image_key"].nunique())
    print("targets:", sorted(df["target"].unique()))
    print("backbones:", sorted(df["backbone"].unique()))

    validation = validate_weighted_equivalence(df)
    validation.to_csv(OUT_DIR / "paired_cluster_bootstrap_weighted_equivalence_check.csv", index=False)

    replicates, summary = run_weighted_bootstrap(df)
    replicates.to_csv(OUT_DIR / "paired_cluster_bootstrap_top20_replicates.csv", index=False)
    summary.to_csv(OUT_DIR / "paired_cluster_bootstrap_top20_summary.csv", index=False)

    overlap_summary, overlap_cases = run_overlap(df)
    overlap_summary.to_csv(OUT_DIR / "top20_capture_overlap_summary.csv", index=False)
    overlap_cases.to_csv(OUT_DIR / "top20_capture_overlap_cases.csv", index=False)

    print("\nSaved outputs to:", OUT_DIR)
    for name in [
        "paired_cluster_bootstrap_weighted_equivalence_check.csv",
        "paired_cluster_bootstrap_top20_replicates.csv",
        "paired_cluster_bootstrap_top20_summary.csv",
        "top20_capture_overlap_summary.csv",
        "top20_capture_overlap_cases.csv",
    ]:
        print(" -", OUT_DIR / name)


if __name__ == "__main__":
    main()
