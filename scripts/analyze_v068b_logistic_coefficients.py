#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v0.6.8b Logistic Coefficient Stability

基于 v0.6.8 的 learned_deferral_fold_predictions.csv 重新拟合每个 target、每个 fold 的 Logistic Regression，
用于检查 learned_logistic 主要依赖哪些模型输出后风险信号。

注意：
- 本分析是机制解释，不是因果解释。
- 特征之间高度相关，不能把单个系数解释为独立贡献。
- 输出 CSV，不自动生成 Markdown。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "summary" / "v0_6_8" / "learned_deferral_fold_predictions.csv"
OUT_DIR = ROOT / "experiments" / "summary" / "v0_6_8b"

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


def load_data() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "target",
        "fold",
        "image_key",
        "backbone",
    ] + FEATURE_COLUMNS + TARGETS

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT}: {missing}")

    return df


def fit_fold_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for target in TARGETS:
        df_target = df[df["target"] == target].copy()

        for fold in sorted(df_target["fold"].unique()):
            train = df_target[df_target["fold"] != fold].copy()
            test = df_target[df_target["fold"] == fold].copy()

            x_train = train[FEATURE_COLUMNS]
            y_train = train[target].astype(int).to_numpy()

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

            model.fit(x_train, y_train)

            clf = model.named_steps["clf"]
            coefs = clf.coef_[0]

            for feature, coef in zip(FEATURE_COLUMNS, coefs):
                rows.append(
                    {
                        "target": target,
                        "target_role": "primary" if target in PRIMARY_TARGETS else "secondary",
                        "fold": int(fold),
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

    return pd.DataFrame(rows)


def summarize_coefficients(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (target, target_role, feature), g in coef_df.groupby(
        ["target", "target_role", "feature"], sort=True
    ):
        values = g["standardized_coefficient"].to_numpy()
        abs_values = np.abs(values)

        positive_rate = float((values > 0).mean())
        negative_rate = float((values < 0).mean())
        zero_rate = float((values == 0).mean())
        sign_consistency = max(positive_rate, negative_rate, zero_rate)

        rows.append(
            {
                "target": target,
                "target_role": target_role,
                "feature": feature,
                "n_folds": int(len(g)),
                "coef_mean": float(np.mean(values)),
                "coef_median": float(np.median(values)),
                "coef_q25": float(np.quantile(values, 0.25)),
                "coef_q75": float(np.quantile(values, 0.75)),
                "coef_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "abs_coef_mean": float(np.mean(abs_values)),
                "abs_coef_median": float(np.median(abs_values)),
                "abs_coef_q25": float(np.quantile(abs_values, 0.25)),
                "abs_coef_q75": float(np.quantile(abs_values, 0.75)),
                "positive_rate": positive_rate,
                "negative_rate": negative_rate,
                "zero_rate": zero_rate,
                "sign_consistency": sign_consistency,
            }
        )

    summary = pd.DataFrame(rows)

    summary["abs_coef_rank_by_target"] = (
        summary.groupby("target")["abs_coef_median"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    summary = summary.sort_values(
        ["target", "abs_coef_rank_by_target", "feature"],
        ascending=[True, True, True],
    )

    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("Loaded v0.6.8 fold prediction table")
    print("rows:", len(df))
    print("unique image_key:", df["image_key"].nunique())
    print("targets:", sorted(df["target"].unique()))
    print("backbones:", sorted(df["backbone"].unique()))

    for target in TARGETS:
        sub = df[df["target"] == target]
        print(
            target,
            "positive_records=", int(sub[target].sum()),
            "positive_images=", int(sub.loc[sub[target] == 1, "image_key"].nunique()),
        )

    coef_df = fit_fold_coefficients(df)
    summary = summarize_coefficients(coef_df)

    coef_df.to_csv(OUT_DIR / "logistic_coefficients_by_fold.csv", index=False)
    summary.to_csv(OUT_DIR / "logistic_coefficients_summary.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "logistic_coefficients_by_fold.csv")
    print(OUT_DIR / "logistic_coefficients_summary.csv")

    print("\nTop coefficients by target:")
    for target in TARGETS:
        print(f"\n=== {target} ===")
        sub = summary[summary["target"] == target].head(8)
        print(
            sub[
                [
                    "feature",
                    "coef_median",
                    "abs_coef_median",
                    "sign_consistency",
                    "abs_coef_rank_by_target",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
