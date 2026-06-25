#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v0.8.0 GreenScout Routing Feasibility Audit

只做已有 APTOS test prediction CSV 标准化。
不训练 Router，不加载 RETFound-Green，不做临床阈值。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROB_CLASS_NAME_COLS = [
    "prob_No DR",
    "prob_Mild DR",
    "prob_Moderate DR",
    "prob_Severe DR",
    "prob_Proliferative DR",
]


def normalize_image_key(x) -> str:
    return Path(str(x)).stem


def standardize_one(model_name: str, path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)

    required = [
        "image_path",
        "true_idx",
        "pred_idx",
        "confidence",
        *PROB_CLASS_NAME_COLS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name}: missing columns={missing}; actual columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["image_key"] = df["image_path"].map(normalize_image_key)
    out["true_label"] = pd.to_numeric(df["true_idx"], errors="raise").astype(int)
    out["pred_label"] = pd.to_numeric(df["pred_idx"], errors="raise").astype(int)

    for i, col in enumerate(PROB_CLASS_NAME_COLS):
        out[f"prob_{i}"] = pd.to_numeric(df[col], errors="raise").astype(float)

    out["confidence"] = pd.to_numeric(df["confidence"], errors="raise").astype(float)
    out["model_name"] = model_name

    # 校验：pred_idx 应等于 prob argmax
    prob_cols = [f"prob_{i}" for i in range(5)]
    argmax_pred = out[prob_cols].to_numpy().argmax(axis=1)
    mismatch = (argmax_pred != out["pred_label"].to_numpy())
    if mismatch.any():
        raise ValueError(
            f"{model_name}: pred_idx != argmax(prob), mismatch_count={int(mismatch.sum())}"
        )

    # 校验：confidence 应等于 max prob，允许极小浮点误差
    max_prob = out[prob_cols].max(axis=1)
    conf_diff = (max_prob - out["confidence"]).abs()
    if (conf_diff > 1e-5).any():
        raise ValueError(
            f"{model_name}: confidence != max(prob), mismatch_count={int((conf_diff > 1e-5).sum())}"
        )

    if out["image_key"].duplicated().any():
        dup_n = int(out["image_key"].duplicated().sum())
        raise ValueError(f"{model_name}: duplicated image_key count={dup_n}")

    for col in ["true_label", "pred_label"]:
        bad = ~out[col].between(0, 4)
        if bad.any():
            raise ValueError(f"{model_name}: invalid {col}, bad_count={int(bad.sum())}")

    report = {
        "model_name": model_name,
        "path": str(path),
        "n_rows": len(out),
        "image_col": "image_path",
        "true_col": "true_idx",
        "pred_col": "pred_idx",
        "prob_cols": "|".join(PROB_CLASS_NAME_COLS),
        "conf_col": "confidence",
    }
    return out, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="experiments/v0_8_0_greenscout_feasibility/registry/model_artifact_registry.csv",
    )
    parser.add_argument(
        "--output",
        default="experiments/v0_8_0_greenscout_feasibility/predictions/existing_models_standardized.csv",
    )
    parser.add_argument(
        "--report",
        default="experiments/v0_8_0_greenscout_feasibility/registry/standardization_report.csv",
    )
    args = parser.parse_args()

    registry = pd.read_csv(args.registry)
    registry = registry[registry["include_in_v080"].astype(str).isin(["1", "true", "True", "yes", "YES"])]

    all_parts = []
    reports = []

    for _, row in registry.iterrows():
        model_name = row["model_name"]
        path = Path(row["prediction_path"])

        if not path.exists():
            raise FileNotFoundError(f"{model_name}: missing file: {path}")

        part, report = standardize_one(model_name, path)
        all_parts.append(part)
        reports.append(report)
        print(f"[OK] {model_name}: rows={len(part)}")

    out = pd.concat(all_parts, ignore_index=True)

    n_models = out["model_name"].nunique()
    image_model_counts = out.groupby("image_key")["model_name"].nunique()
    bad_images = image_model_counts[image_model_counts != n_models]
    if len(bad_images) > 0:
        raise ValueError(
            f"Some images do not have all model predictions: "
            f"bad_images={len(bad_images)}, expected_models_per_image={n_models}"
        )

    # 校验同一 image_key 的 true_label 是否一致
    true_nunique = out.groupby("image_key")["true_label"].nunique()
    bad_true = true_nunique[true_nunique != 1]
    if len(bad_true) > 0:
        raise ValueError(f"true_label mismatch across models: bad_images={len(bad_true)}")

    out = out[
        [
            "image_key",
            "true_label",
            "pred_label",
            "prob_0",
            "prob_1",
            "prob_2",
            "prob_3",
            "prob_4",
            "confidence",
            "model_name",
        ]
    ].sort_values(["image_key", "model_name"]).reset_index(drop=True)

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)
    pd.DataFrame(reports).to_csv(report_path, index=False)

    print("\n[DONE]")
    print(f"saved: {output_path}")
    print(f"saved: {report_path}")
    print(f"rows={len(out)}")
    print(f"unique_images={out['image_key'].nunique()}")
    print(f"models={n_models}")
    print("\nper-model rows:")
    print(out.groupby("model_name").size().to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
