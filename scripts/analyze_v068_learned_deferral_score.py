#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v0.6.8 Learned Deferral Score

定位：
    Exploratory internal grouped cross-validation.

目标：
    在按 image_key 分组、避免同图泄露的内部交叉验证下，
    验证轻量 Logistic Regression 是否能组合模型输出信号，
    提高方向敏感危险错误在有限复核预算下的富集能力。

重要边界：
    1. 不是独立临床验证。
    2. 不是外部泛化验证。
    3. 不是 unseen-backbone generalization。
    4. 统计单位是 1100 unique images 产生的 6600 backbone-specific prediction records。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

INPUT_CASES = ROOT / "experiments" / "summary" / "v0_6_7" / "clinical_event_cases.csv"
OUT_DIR = ROOT / "experiments" / "summary" / "v0_6_8"

PREDICTION_FILES: Dict[str, Path] = {
    "convnext_tiny": ROOT / "experiments" / "aptos_convnext_tiny" / "lr1e-4_bs32_seed42" / "evaluation" / "test" / "test_predictions.csv",
    "swin_tiny": ROOT / "experiments" / "aptos_swin_tiny" / "lr1e-4_bs32_seed42" / "evaluation" / "test" / "test_predictions.csv",
    "vit_b_imagenet": ROOT / "experiments" / "aptos_vit_base_patch16_imagenet" / "lr1e-4_bs32_seed42" / "evaluation" / "test" / "test_predictions.csv",
    "vit_b_official_like": ROOT / "experiments" / "aptos_vit_base_patch16_official_like" / "official_like_bs32_epoch50_seed42" / "evaluation" / "test" / "test_predictions.csv",
    "vit_l_official_like": ROOT / "experiments" / "aptos_vit_large_patch16_official_like" / "official_like_bs32_epoch50_seed42" / "evaluation" / "test" / "test_predictions.csv",
    "retfound_official_like": ROOT / "experiments" / "aptos_retfound_mae_cfp_official_like" / "official_like_bs32_epoch50_seed42" / "evaluation" / "test" / "test_predictions.csv",
}

PROB_COLS = [
    "prob_No DR",
    "prob_Mild DR",
    "prob_Moderate DR",
    "prob_Severe DR",
    "prob_Proliferative DR",
]

GRADE_VALUES = np.array([0, 1, 2, 3, 4], dtype=float)

PRIMARY_TARGETS = [
    "large_undergrading",
    "vision_threatening_dr_miss",
]

SECONDARY_TARGETS = [
    "dangerous_undergrading",
]

TARGETS = PRIMARY_TARGETS + SECONDARY_TARGETS

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MAIN_REPORT_BUDGETS = [0.10, 0.20, 0.30]

OLD_METHODS = [
    "confidence_only",
    "margin_only",
    "entropy_only",
    "uncertainty_rank_fusion",
    "ophagent_combined",
    "severe_prob_mass_only",
    "gated_severe_prob_mass_only",
    "expected_grade_only",
    "expected_gap_only",
    "top2_more_severe_only",
]

METHODS = OLD_METHODS + ["learned_logistic"]

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

LEAKAGE_COLUMNS = {
    "true_idx",
    "true_label",
    "true_grade",
    "general_error",
    "any_undergrading",
    "large_undergrading",
    "referable_dr_miss",
    "vision_threatening_dr_miss",
    "high_confidence_vision_threatening_miss",
    "dangerous_undergrading",
}


def normalize_image_key(x: object) -> str:
    return Path(str(x)).name


def to_int_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    return (
        s.astype(str)
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def load_probability_table() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for backbone, path in PREDICTION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {backbone}: {path}")

        pred = pd.read_csv(path)
        missing = [c for c in PROB_COLS if c not in pred.columns]
        if missing:
            raise ValueError(f"{path} missing probability columns: {missing}")

        pred = pred.copy()
        pred["backbone"] = backbone
        pred["image_key"] = pred["image_path"].map(normalize_image_key)

        keep = ["backbone", "image_key"] + PROB_COLS
        pred = pred[keep]

        if pred.duplicated(["backbone", "image_key"]).any():
            dup = pred[pred.duplicated(["backbone", "image_key"], keep=False)]
            raise ValueError(f"Duplicated prediction records in {path}:\n{dup.head()}")

        frames.append(pred)

    prob_df = pd.concat(frames, ignore_index=True)
    return prob_df


def load_case_table() -> pd.DataFrame:
    if not INPUT_CASES.exists():
        raise FileNotFoundError(f"Missing input case table: {INPUT_CASES}")

    cases = pd.read_csv(INPUT_CASES)
    cases = cases.copy()
    cases["image_key"] = cases["image_key"].map(normalize_image_key)

    required = [
        "backbone",
        "case_id",
        "image_key",
        "true_grade",
        "pred_grade",
        "top2_grade",
        "confidence",
        "top2_confidence",
        "margin",
        "entropy_norm",
        "severe_prob_mass",
        "review_priority_rank",
        "large_undergrading",
        "vision_threatening_dr_miss",
    ]

    missing = [c for c in required if c not in cases.columns]
    if missing:
        raise ValueError(f"{INPUT_CASES} missing required columns: {missing}")

    for col in [
        "general_error",
        "any_undergrading",
        "large_undergrading",
        "referable_dr_miss",
        "vision_threatening_dr_miss",
        "high_confidence_vision_threatening_miss",
    ]:
        if col in cases.columns:
            cases[col] = to_int_bool(cases[col])

    prob_df = load_probability_table()
    df = cases.merge(
        prob_df,
        on=["backbone", "image_key"],
        how="left",
        validate="many_to_one",
    )

    if df[PROB_COLS].isna().any().any():
        bad = df[df[PROB_COLS].isna().any(axis=1)][["backbone", "image_key", "case_id"]].head()
        raise ValueError(f"Probability merge failed for some rows:\n{bad}")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    prob = out[PROB_COLS].to_numpy(dtype=float)
    out["expected_grade"] = prob @ GRADE_VALUES
    out["expected_gap"] = out["expected_grade"] - out["pred_grade"]

    out["top2_more_severe"] = (out["top2_grade"] > out["pred_grade"]).astype(int)
    out["pred_le_2"] = (out["pred_grade"] <= 2).astype(int)
    out["gated_severe_prob_mass"] = out["severe_prob_mass"] * out["pred_le_2"]
    out["top2_more_severe_conf"] = out["top2_more_severe"] * out["top2_confidence"].fillna(0.0)

    out["dangerous_undergrading"] = (
        (out["large_undergrading"].astype(int) == 1)
        | (out["vision_threatening_dr_miss"].astype(int) == 1)
    ).astype(int)

    out["row_id"] = out["backbone"].astype(str) + "::" + out["image_key"].astype(str)

    return out


def validate_table(df: pd.DataFrame) -> None:
    if df.duplicated(["backbone", "image_key"]).any():
        dup = df[df.duplicated(["backbone", "image_key"], keep=False)][
            ["backbone", "image_key", "case_id"]
        ].head()
        raise ValueError(f"Duplicated (backbone, image_key):\n{dup}")

    n_images = df["image_key"].nunique()
    n_backbones = df["backbone"].nunique()
    expected_rows = n_images * n_backbones

    if len(df) != expected_rows:
        raise ValueError(
            f"Unexpected row count: rows={len(df)}, "
            f"unique_images={n_images}, backbones={n_backbones}, "
            f"expected_rows={expected_rows}"
        )

    counts = df.groupby("image_key")["backbone"].nunique()
    if not (counts == n_backbones).all():
        raise ValueError("Not every image_key has all backbone records.")

    overlap = sorted(set(FEATURE_COLUMNS) & LEAKAGE_COLUMNS)
    if overlap:
        raise ValueError(f"Leakage columns found in FEATURE_COLUMNS: {overlap}")


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def rank_df(g: pd.DataFrame, method: str) -> pd.DataFrame:
    g = g.copy()

    if method == "confidence_only":
        return g.sort_values(["confidence", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "margin_only":
        return g.sort_values(["margin", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "entropy_only":
        return g.sort_values(["entropy_norm", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "uncertainty_rank_fusion":
        g["_r_conf"] = g["confidence"].rank(ascending=True, method="first")
        g["_r_margin"] = g["margin"].rank(ascending=True, method="first")
        g["_r_entropy"] = g["entropy_norm"].rank(ascending=False, method="first")
        g["_score"] = g["_r_conf"] + g["_r_margin"] + g["_r_entropy"]
        return g.sort_values(["_score", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "ophagent_combined":
        return g.sort_values(["review_priority_rank", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "severe_prob_mass_only":
        return g.sort_values(["severe_prob_mass", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "gated_severe_prob_mass_only":
        g["_score"] = np.where(g["pred_grade"] <= 2, g["severe_prob_mass"], -1.0)
        return g.sort_values(["_score", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "expected_grade_only":
        return g.sort_values(["expected_grade", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "expected_gap_only":
        return g.sort_values(["expected_gap", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "top2_more_severe_only":
        g["_score"] = np.where(g["top2_more_severe"].astype(bool), g["top2_confidence"].fillna(0.0), -1.0)
        return g.sort_values(["_score", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "learned_logistic":
        if "score_learned_logistic" not in g.columns:
            raise ValueError("score_learned_logistic missing. Fit learned model first.")
        return g.sort_values(["score_learned_logistic", "case_id"], ascending=[False, True], kind="mergesort")

    raise ValueError(f"Unknown ranking method: {method}")


def get_score_for_auc(g: pd.DataFrame, method: str) -> pd.Series:
    g = g.copy()

    if method == "confidence_only":
        return -g["confidence"]
    if method == "margin_only":
        return -g["margin"]
    if method == "entropy_only":
        return g["entropy_norm"]
    if method == "uncertainty_rank_fusion":
        r_conf = g["confidence"].rank(ascending=True, method="first")
        r_margin = g["margin"].rank(ascending=True, method="first")
        r_entropy = g["entropy_norm"].rank(ascending=False, method="first")
        return -(r_conf + r_margin + r_entropy)
    if method == "ophagent_combined":
        return -g["review_priority_rank"]
    if method == "severe_prob_mass_only":
        return g["severe_prob_mass"]
    if method == "gated_severe_prob_mass_only":
        return pd.Series(np.where(g["pred_grade"] <= 2, g["severe_prob_mass"], -1.0), index=g.index)
    if method == "expected_grade_only":
        return g["expected_grade"]
    if method == "expected_gap_only":
        return g["expected_gap"]
    if method == "top2_more_severe_only":
        return pd.Series(
            np.where(g["top2_more_severe"].astype(bool), g["top2_confidence"].fillna(0.0), -1.0),
            index=g.index,
        )
    if method == "learned_logistic":
        return g["score_learned_logistic"]

    raise ValueError(f"Unknown ranking method: {method}")


def evaluate_ranked(
    g: pd.DataFrame,
    target: str,
    method: str,
    budget: float,
    fold: int,
    evaluation_scope: str,
    backbone: str,
) -> dict:
    n = len(g)
    if n == 0:
        raise ValueError("Cannot evaluate empty group.")

    total = int(g[target].sum())
    k = int(np.ceil(n * budget))
    k = max(1, min(k, n))

    ranked = rank_df(g, method).reset_index(drop=True)
    reviewed = ranked.head(k)
    released = ranked.iloc[k:]

    captured = int(reviewed[target].sum())
    residual = int(total - captured)

    precision_at_k = captured / k if k > 0 else np.nan
    recall_at_k = captured / total if total > 0 else np.nan

    base_rate = total / n if n > 0 else np.nan
    lift = precision_at_k / base_rate if base_rate and base_rate > 0 else np.nan

    auto_released_n = n - k
    residual_rate = residual / auto_released_n if auto_released_n > 0 else np.nan
    n_needed = 1 / precision_at_k if precision_at_k and precision_at_k > 0 else np.nan

    return {
        "target": target,
        "target_role": "primary" if target in PRIMARY_TARGETS else "secondary",
        "fold": fold,
        "evaluation_scope": evaluation_scope,
        "backbone": backbone,
        "ranking_method": method,
        "review_budget": budget,
        "actual_review_budget": k / n,
        "n_records": n,
        "reviewed_n": k,
        "auto_released_n": auto_released_n,
        "dangerous_error_total": total,
        "dangerous_error_captured": captured,
        "dangerous_error_recall_at_k": recall_at_k,
        "dangerous_error_precision_at_k": precision_at_k,
        "dangerous_error_lift_vs_random": lift,
        "residual_dangerous_error_count": residual,
        "residual_dangerous_error_rate": residual_rate,
        "dangerous_errors_per_100_reviewed": precision_at_k * 100 if not np.isnan(precision_at_k) else np.nan,
        "number_needed_to_review": n_needed,
    }


def evaluate_auc(
    g: pd.DataFrame,
    target: str,
    method: str,
    fold: int,
    evaluation_scope: str,
    backbone: str,
) -> dict:
    y = g[target].astype(int).to_numpy()
    score = get_score_for_auc(g, method).to_numpy(dtype=float)

    if len(np.unique(y)) < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = roc_auc_score(y, score)
        auprc = average_precision_score(y, score)

    return {
        "target": target,
        "target_role": "primary" if target in PRIMARY_TARGETS else "secondary",
        "fold": fold,
        "evaluation_scope": evaluation_scope,
        "backbone": backbone,
        "ranking_method": method,
        "n_records": len(g),
        "positive_records": int(g[target].sum()),
        "auroc": auroc,
        "auprc": auprc,
    }


def fold_diagnostics(test_df: pd.DataFrame, target: str, fold: int) -> List[dict]:
    rows = []

    scopes: List[Tuple[str, str, pd.DataFrame]] = [
        ("pooled_analysis_only", "ALL", test_df)
    ]

    for backbone, g in test_df.groupby("backbone", sort=True):
        scopes.append(("per_backbone", backbone, g))

    for scope, backbone, g in scopes:
        pos = g[g[target].astype(int) == 1]
        rows.append(
            {
                "target": target,
                "target_role": "primary" if target in PRIMARY_TARGETS else "secondary",
                "fold": fold,
                "evaluation_scope": scope,
                "backbone": backbone,
                "n_unique_images": g["image_key"].nunique(),
                "n_records": len(g),
                "positive_records": int(g[target].sum()),
                "positive_image_keys": pos["image_key"].nunique(),
                "positive_rate": float(g[target].mean()) if len(g) else np.nan,
            }
        )

    return rows


def summarize_tradeoff(tradeoff: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "target",
        "target_role",
        "evaluation_scope",
        "backbone",
        "ranking_method",
        "review_budget",
    ]

    summary = (
        tradeoff
        .groupby(group_cols, dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_recall=("dangerous_error_recall_at_k", "mean"),
            std_recall=("dangerous_error_recall_at_k", "std"),
            mean_precision=("dangerous_error_precision_at_k", "mean"),
            std_precision=("dangerous_error_precision_at_k", "std"),
            mean_lift=("dangerous_error_lift_vs_random", "mean"),
            std_lift=("dangerous_error_lift_vs_random", "std"),
            mean_captured=("dangerous_error_captured", "mean"),
            mean_total=("dangerous_error_total", "mean"),
            mean_residual=("residual_dangerous_error_count", "mean"),
            total_captured=("dangerous_error_captured", "sum"),
            total_dangerous=("dangerous_error_total", "sum"),
            total_residual=("residual_dangerous_error_count", "sum"),
        )
        .reset_index()
        .sort_values(
            ["target", "evaluation_scope", "backbone", "review_budget", "mean_recall", "mean_lift"],
            ascending=[True, True, True, True, False, False],
        )
    )

    return summary


def summarize_auc(auc_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "target",
        "target_role",
        "evaluation_scope",
        "backbone",
        "ranking_method",
    ]

    return (
        auc_df
        .groupby(group_cols, dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
            mean_positive_records=("positive_records", "mean"),
        )
        .reset_index()
        .sort_values(
            ["target", "evaluation_scope", "backbone", "mean_auprc", "mean_auroc"],
            ascending=[True, True, True, False, False],
        )
    )


def compute_winner_counts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = summary[summary["evaluation_scope"] == "per_backbone"].copy()

    for (target, budget, backbone), g in sub.groupby(["target", "review_budget", "backbone"], dropna=False):
        max_recall = g["mean_recall"].max(skipna=True)
        if pd.isna(max_recall):
            continue

        winners = g[np.isclose(g["mean_recall"], max_recall, equal_nan=False)]
        for _, row in winners.iterrows():
            rows.append(
                {
                    "target": target,
                    "review_budget": budget,
                    "backbone": backbone,
                    "ranking_method": row["ranking_method"],
                    "winner_mean_recall": row["mean_recall"],
                    "tie_count": len(winners),
                }
            )

    winner_detail = pd.DataFrame(rows)
    if winner_detail.empty:
        return winner_detail

    count = (
        winner_detail
        .groupby(["target", "review_budget", "ranking_method"], dropna=False)
        .agg(
            winner_count=("backbone", "count"),
            mean_winner_recall=("winner_mean_recall", "mean"),
            mean_tie_count=("tie_count", "mean"),
        )
        .reset_index()
    )

    total_backbones = sub["backbone"].nunique()
    count["total_backbones"] = total_backbones
    count["winner_rate"] = count["winner_count"] / total_backbones

    return count.sort_values(
        ["target", "review_budget", "winner_count", "mean_winner_recall"],
        ascending=[True, True, False, False],
    )






def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_case_table()
    df = add_derived_features(df)
    validate_table(df)

    print("Loaded case table")
    print("rows:", len(df))
    print("unique images:", df["image_key"].nunique())
    print("backbones:", sorted(df["backbone"].unique()))

    for target in TARGETS:
        print(target, "positive records:", int(df[target].sum()), "positive images:", df.loc[df[target] == 1, "image_key"].nunique())

    tradeoff_rows: List[dict] = []
    auc_rows: List[dict] = []
    diag_rows: List[dict] = []
    prediction_frames: List[pd.DataFrame] = []

    for target in TARGETS:
        print(f"\n=== Target: {target} ===")

        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        X = df[FEATURE_COLUMNS]
        y = df[target].astype(int)
        groups = df["image_key"]

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            train_images = set(train_df["image_key"])
            test_images = set(test_df["image_key"])
            assert train_images.isdisjoint(test_images), "image_key leakage across train/test fold"

            model = make_model()
            model.fit(train_df[FEATURE_COLUMNS], train_df[target].astype(int))

            test_df["score_learned_logistic"] = model.decision_function(test_df[FEATURE_COLUMNS])
            test_df["fold"] = fold
            test_df["target"] = target

            prediction_frames.append(
                test_df[
                    [
                        "target",
                        "fold",
                        "row_id",
                        "backbone",
                        "case_id",
                        "image_key",
                        "image_path",
                        "true_grade",
                        "pred_grade",
                        "top2_grade",
                        "confidence",
                        "top2_confidence",
                        "margin",
                        "entropy_norm",
                        "severe_prob_mass",
                        "expected_grade",
                        "expected_gap",
                        "top2_more_severe",
                        "pred_le_2",
                        "gated_severe_prob_mass",
                        "top2_more_severe_conf",
                        "review_priority_rank",
                        "large_undergrading",
                        "vision_threatening_dr_miss",
                        "dangerous_undergrading",
                        "score_learned_logistic",
                    ]
                ].copy()
            )

            diag_rows.extend(fold_diagnostics(test_df, target, fold))

            eval_groups: List[Tuple[str, str, pd.DataFrame]] = [
                ("pooled_analysis_only", "ALL", test_df)
            ]

            for backbone, g in test_df.groupby("backbone", sort=True):
                eval_groups.append(("per_backbone", backbone, g))

            for evaluation_scope, backbone, g in eval_groups:
                for method in METHODS:
                    for budget in BUDGETS:
                        tradeoff_rows.append(
                            evaluate_ranked(
                                g=g,
                                target=target,
                                method=method,
                                budget=budget,
                                fold=fold,
                                evaluation_scope=evaluation_scope,
                                backbone=backbone,
                            )
                        )

                    auc_rows.append(
                        evaluate_auc(
                            g=g,
                            target=target,
                            method=method,
                            fold=fold,
                            evaluation_scope=evaluation_scope,
                            backbone=backbone,
                        )
                    )

            print(
                f"fold={fold}",
                "train_images=", len(train_images),
                "test_images=", len(test_images),
                "test_positive_records=", int(test_df[target].sum()),
            )

    tradeoff = pd.DataFrame(tradeoff_rows)
    summary = summarize_tradeoff(tradeoff)
    diagnostics = pd.DataFrame(diag_rows)
    auc_df = pd.DataFrame(auc_rows)
    auc_summary = summarize_auc(auc_df)
    winner = compute_winner_counts(summary)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    tradeoff.to_csv(OUT_DIR / "learned_deferral_tradeoff.csv", index=False)
    summary.to_csv(OUT_DIR / "learned_deferral_cv_summary.csv", index=False)
    diagnostics.to_csv(OUT_DIR / "learned_deferral_fold_diagnostics.csv", index=False)
    winner.to_csv(OUT_DIR / "learned_deferral_winner_count.csv", index=False)
    auc_df.to_csv(OUT_DIR / "learned_deferral_auc_by_fold.csv", index=False)
    auc_summary.to_csv(OUT_DIR / "learned_deferral_auc_summary.csv", index=False)
    predictions.to_csv(OUT_DIR / "learned_deferral_fold_predictions.csv", index=False)


    print("\nSaved outputs to:", OUT_DIR)
    for p in [
        "learned_deferral_tradeoff.csv",
        "learned_deferral_cv_summary.csv",
        "learned_deferral_fold_diagnostics.csv",
        "learned_deferral_winner_count.csv",
        "learned_deferral_auc_by_fold.csv",
        "learned_deferral_auc_summary.csv",
        "learned_deferral_fold_predictions.csv",
    ]:
        print(" -", OUT_DIR / p)


if __name__ == "__main__":
    main()
