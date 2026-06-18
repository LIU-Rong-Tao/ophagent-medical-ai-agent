#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v0.6.8b Repeated Split Sensitivity

目的：
检查换不同 random seed 重新做 StratifiedGroupKFold 后，
learned_logistic 的 Top20% 排序结果和系数方向是否稳定。

注意：
- 这不是外部验证。
- 这不是新模型开发。
- 这是内部 repeated grouped CV 敏感性分析。
- 输出 CSV，不自动生成 Markdown。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "summary" / "v0_6_8" / "learned_deferral_fold_predictions.csv"
OUT_DIR = ROOT / "experiments" / "summary" / "v0_6_8b"

SEEDS = [42, 43, 44, 45, 46]
N_SPLITS = 5
REVIEW_BUDGET = 0.20

PRIMARY_TARGETS = [
    "large_undergrading",
    "vision_threatening_dr_miss",
]
SECONDARY_TARGETS = [
    "dangerous_undergrading",
]
TARGETS = PRIMARY_TARGETS + SECONDARY_TARGETS

FEATURE_COLUMNS = [
    "confidence",
    "margin",
    "entropy_norm",
    "pred_grade",
    "top2_grade",
    "top2_confidence",
    "severe_prob_mass",
    "expected_grade",
    "expected_gap",
    "top2_more_severe",
    "pred_le_2",
    "gated_severe_prob_mass",
    "top2_more_severe_conf",
]

RANKING_METHODS = [
    "learned_logistic",
    "expected_gap_only",
    "gated_severe_prob_mass_only",
    "top2_more_severe_only",
    "ophagent_combined",
    "margin_only",
]


def load_data() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "target",
        "image_key",
        "backbone",
        "case_id",
        "true_grade",
        "pred_grade",
        "review_priority_rank",
    ] + FEATURE_COLUMNS + TARGETS

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT}: {missing}")

    if "row_id" not in df.columns:
        df = df.copy()
        df["row_id"] = np.arange(len(df))

    return df


def fit_oof_scores_for_target(df_target: pd.DataFrame, target: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_target = df_target.copy().reset_index(drop=True)
    y = df_target[target].astype(int).to_numpy()
    groups = df_target["image_key"].to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    df_target["repeated_seed"] = seed
    df_target["repeated_fold"] = -1
    df_target["score_repeated_learned_logistic"] = np.nan

    coef_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(df_target, y, groups), start=1):
        train = df_target.iloc[train_idx].copy()
        test = df_target.iloc[test_idx].copy()

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )

        x_train = train[FEATURE_COLUMNS]
        y_train = train[target].astype(int).to_numpy()

        model.fit(x_train, y_train)

        scores = model.decision_function(test[FEATURE_COLUMNS])
        df_target.loc[test_idx, "repeated_fold"] = fold
        df_target.loc[test_idx, "score_repeated_learned_logistic"] = scores

        clf = model.named_steps["clf"]
        coefs = clf.coef_[0]

        for feature, coef in zip(FEATURE_COLUMNS, coefs):
            coef_rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "target_role": "primary" if target in PRIMARY_TARGETS else "secondary",
                    "fold": fold,
                    "feature": feature,
                    "standardized_coefficient": float(coef),
                    "abs_standardized_coefficient": float(abs(coef)),
                    "sign": int(np.sign(coef)),
                    "train_records": int(len(train)),
                    "test_records": int(len(test)),
                    "train_positive_records": int(y_train.sum()),
                    "test_positive_records": int(test[target].sum()),
                    "train_unique_images": int(train["image_key"].nunique()),
                    "test_unique_images": int(test["image_key"].nunique()),
                }
            )

    if df_target["score_repeated_learned_logistic"].isna().any():
        raise RuntimeError(f"Missing OOF scores for target={target}, seed={seed}")

    return df_target, pd.DataFrame(coef_rows)


def sort_for_method(g: pd.DataFrame, method: str) -> pd.DataFrame:
    g = g.copy()
    tie_cols = ["row_id", "case_id"]

    if method == "learned_logistic":
        return g.sort_values(
            ["score_repeated_learned_logistic"] + tie_cols,
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

    if method == "top2_more_severe_only":
        return g.sort_values(
            ["top2_more_severe_conf"] + tie_cols,
            ascending=[False, True, True],
            kind="mergesort",
        )

    if method == "ophagent_combined":
        return g.sort_values(
            ["review_priority_rank"] + tie_cols,
            ascending=[True, True, True],
            kind="mergesort",
        )

    if method == "margin_only":
        return g.sort_values(
            ["margin"] + tie_cols,
            ascending=[True, True, True],
            kind="mergesort",
        )

    raise ValueError(f"Unsupported method: {method}")


def evaluate_one(g: pd.DataFrame, target: str, method: str) -> dict:
    ranked = sort_for_method(g, method)
    n = len(ranked)
    k = int(np.ceil(n * REVIEW_BUDGET))
    top = ranked.head(k)

    total = int(ranked[target].sum())
    captured = int(top[target].sum())
    residual = total - captured

    recall = captured / total if total > 0 else np.nan
    precision = captured / k if k > 0 else np.nan
    base_rate = total / n if n > 0 else np.nan
    lift = precision / base_rate if base_rate > 0 else np.nan

    return {
        "n_records": n,
        "reviewed_n": k,
        "total_dangerous": total,
        "captured": captured,
        "residual": residual,
        "recall": recall,
        "precision": precision,
        "lift": lift,
    }


def evaluate_per_backbone(df_target: pd.DataFrame, target: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for backbone, g in df_target.groupby("backbone", sort=True):
        for method in RANKING_METHODS:
            m = evaluate_one(g, target, method)
            rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "target_role": "primary" if target in PRIMARY_TARGETS else "secondary",
                    "backbone": backbone,
                    "ranking_method": method,
                    "review_budget": REVIEW_BUDGET,
                    **m,
                }
            )

    per_backbone = pd.DataFrame(rows)

    summary_rows = []
    for (seed_, target_, target_role, method), g in per_backbone.groupby(
        ["seed", "target", "target_role", "ranking_method"], sort=True
    ):
        total = int(g["total_dangerous"].sum())
        captured = int(g["captured"].sum())
        residual = int(g["residual"].sum())
        reviewed = int(g["reviewed_n"].sum())
        n_records = int(g["n_records"].sum())

        recall = captured / total if total > 0 else np.nan
        precision = captured / reviewed if reviewed > 0 else np.nan
        base_rate = total / n_records if n_records > 0 else np.nan
        lift = precision / base_rate if base_rate > 0 else np.nan

        summary_rows.append(
            {
                "seed": seed_,
                "target": target_,
                "target_role": target_role,
                "ranking_method": method,
                "review_budget": REVIEW_BUDGET,
                "n_backbones": int(g["backbone"].nunique()),
                "n_records": n_records,
                "reviewed_n": reviewed,
                "total_dangerous": total,
                "captured": captured,
                "residual": residual,
                "recall": recall,
                "precision": precision,
                "lift": lift,
                "mean_backbone_recall": float(g["recall"].mean()),
                "mean_backbone_lift": float(g["lift"].mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return per_backbone, summary


def summarize_repeated_cv(summary_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (target, target_role, method), g in summary_by_seed.groupby(
        ["target", "target_role", "ranking_method"], sort=True
    ):
        rows.append(
            {
                "target": target,
                "target_role": target_role,
                "ranking_method": method,
                "review_budget": REVIEW_BUDGET,
                "n_seeds": int(g["seed"].nunique()),
                "recall_mean": float(g["recall"].mean()),
                "recall_std": float(g["recall"].std(ddof=1)),
                "recall_min": float(g["recall"].min()),
                "recall_max": float(g["recall"].max()),
                "mean_backbone_recall_mean": float(g["mean_backbone_recall"].mean()),
                "mean_backbone_recall_std": float(g["mean_backbone_recall"].std(ddof=1)),
                "captured_mean": float(g["captured"].mean()),
                "captured_min": int(g["captured"].min()),
                "captured_max": int(g["captured"].max()),
                "residual_mean": float(g["residual"].mean()),
                "residual_min": int(g["residual"].min()),
                "residual_max": int(g["residual"].max()),
                "lift_mean": float(g["lift"].mean()),
                "lift_std": float(g["lift"].std(ddof=1)),
            }
        )

    out = pd.DataFrame(rows)
    out["recall_rank_mean"] = (
        out.groupby("target")["recall_mean"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return out.sort_values(["target", "recall_rank_mean", "ranking_method"])


def summarize_repeated_coefficients(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (target, target_role, feature), g in coef_df.groupby(
        ["target", "target_role", "feature"], sort=True
    ):
        values = g["standardized_coefficient"].to_numpy()
        abs_values = np.abs(values)

        positive_rate = float((values > 0).mean())
        negative_rate = float((values < 0).mean())
        zero_rate = float((values == 0).mean())

        rows.append(
            {
                "target": target,
                "target_role": target_role,
                "feature": feature,
                "n_seed_folds": int(len(g)),
                "coef_mean": float(np.mean(values)),
                "coef_median": float(np.median(values)),
                "coef_q25": float(np.quantile(values, 0.25)),
                "coef_q75": float(np.quantile(values, 0.75)),
                "coef_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "abs_coef_mean": float(np.mean(abs_values)),
                "abs_coef_median": float(np.median(abs_values)),
                "positive_rate": positive_rate,
                "negative_rate": negative_rate,
                "zero_rate": zero_rate,
                "sign_consistency": max(positive_rate, negative_rate, zero_rate),
            }
        )

    out = pd.DataFrame(rows)
    out["abs_coef_rank_by_target"] = (
        out.groupby("target")["abs_coef_median"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return out.sort_values(["target", "abs_coef_rank_by_target", "feature"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("Loaded v0.6.8 fold prediction table")
    print("rows:", len(df))
    print("unique image_key:", df["image_key"].nunique())
    print("targets:", sorted(df["target"].unique()))
    print("backbones:", sorted(df["backbone"].unique()))
    print("seeds:", SEEDS)

    all_per_backbone = []
    all_summary_by_seed = []
    all_coefficients = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")

        for target in TARGETS:
            print(f"target: {target}")

            df_target = df[df["target"] == target].copy()
            scored, coef_df = fit_oof_scores_for_target(df_target, target, seed)
            per_backbone, summary = evaluate_per_backbone(scored, target, seed)

            all_per_backbone.append(per_backbone)
            all_summary_by_seed.append(summary)
            all_coefficients.append(coef_df)

    per_backbone_all = pd.concat(all_per_backbone, ignore_index=True)
    summary_by_seed = pd.concat(all_summary_by_seed, ignore_index=True)
    coefficients_by_seed_fold = pd.concat(all_coefficients, ignore_index=True)

    repeated_cv_summary = summarize_repeated_cv(summary_by_seed)
    repeated_coef_summary = summarize_repeated_coefficients(coefficients_by_seed_fold)

    per_backbone_all.to_csv(OUT_DIR / "repeated_split_per_backbone_metrics.csv", index=False)
    summary_by_seed.to_csv(OUT_DIR / "repeated_split_metrics_by_seed.csv", index=False)
    repeated_cv_summary.to_csv(OUT_DIR / "repeated_split_cv_summary.csv", index=False)
    coefficients_by_seed_fold.to_csv(OUT_DIR / "repeated_split_coefficients_by_seed_fold.csv", index=False)
    repeated_coef_summary.to_csv(OUT_DIR / "repeated_split_coefficient_summary.csv", index=False)

    print("\nSaved:")
    for name in [
        "repeated_split_per_backbone_metrics.csv",
        "repeated_split_metrics_by_seed.csv",
        "repeated_split_cv_summary.csv",
        "repeated_split_coefficients_by_seed_fold.csv",
        "repeated_split_coefficient_summary.csv",
    ]:
        print(OUT_DIR / name)

    print("\nTop repeated split methods by target:")
    for target in TARGETS:
        print(f"\n=== {target} ===")
        sub = repeated_cv_summary[repeated_cv_summary["target"] == target].head(8)
        print(
            sub[
                [
                    "ranking_method",
                    "recall_mean",
                    "recall_std",
                    "captured_mean",
                    "residual_mean",
                    "lift_mean",
                    "recall_rank_mean",
                ]
            ].to_string(index=False)
        )

    print("\nTop repeated split coefficients by target:")
    for target in TARGETS:
        print(f"\n=== {target} ===")
        sub = repeated_coef_summary[repeated_coef_summary["target"] == target].head(8)
        print(
            sub[
                [
                    "feature",
                    "coef_median",
                    "coef_iqr",
                    "abs_coef_median",
                    "sign_consistency",
                    "abs_coef_rank_by_target",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
