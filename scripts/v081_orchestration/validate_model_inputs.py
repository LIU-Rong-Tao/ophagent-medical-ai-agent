#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_REGISTRY_COLS = {
    "model_name",
    "family",
    "role_hint",
    "prediction_csv",
    "cost_csv",
    "enabled",
    "notes",
}

REQUIRED_PRED_COLS = {
    "image_key",
    "true_label",
    "model_name",
    "pred_label",
    "confidence",
    "prob_0",
    "prob_1",
    "prob_2",
    "prob_3",
    "prob_4",
}

OPTIONAL_PRED_COLS = {"dataset", "split"}

REQUIRED_COST_COLS = {
    "model_name",
    "mean_ms_per_image",
    "median_ms_per_image",
    "images_per_second",
    "pytorch_peak_allocated_mem_mb",
    "checkpoint_mb",
    "batch_size",
    "device",
}


def check_columns(name: str, df: pd.DataFrame, required: set[str]) -> list[str]:
    missing = sorted(required - set(df.columns))
    return [f"{name}: missing columns {missing}"] if missing else []


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def validate_registry(registry: pd.DataFrame) -> list[str]:
    errors = check_columns("registry", registry, REQUIRED_REGISTRY_COLS)
    if errors:
        return errors

    duplicated = registry["model_name"][registry["model_name"].duplicated()].tolist()
    if duplicated:
        errors.append(f"registry: duplicated model_name {duplicated}")

    return errors


def validate_prediction_table(pred: pd.DataFrame, model_name: str) -> tuple[list[str], dict]:
    errors = check_columns(f"prediction[{model_name}]", pred, REQUIRED_PRED_COLS)
    if errors:
        return errors, {}

    sub = pred[pred["model_name"] == model_name].copy()
    if sub.empty:
        return [f"prediction[{model_name}]: no rows for model_name"], {}

    prob_cols = [f"prob_{i}" for i in range(5)]
    probs = sub[prob_cols].to_numpy(dtype=float)
    prob_sum = probs.sum(axis=1)

    if np.isnan(probs).any():
        errors.append(f"prediction[{model_name}]: NaN found in probability columns")

    if ((probs < -1e-6) | (probs > 1 + 1e-6)).any():
        errors.append(f"prediction[{model_name}]: probability values outside [0, 1]")

    if not np.allclose(prob_sum, 1.0, atol=1e-3):
        errors.append(f"prediction[{model_name}]: probability rows do not sum to 1 within atol=1e-3")

    pred_from_prob = probs.argmax(axis=1)
    mismatch = int((pred_from_prob != sub["pred_label"].to_numpy(dtype=int)).sum())
    if mismatch:
        errors.append(f"prediction[{model_name}]: pred_label mismatches argmax(prob) for {mismatch} rows")

    optional_missing = sorted(OPTIONAL_PRED_COLS - set(sub.columns))

    summary = {
        "model_name": model_name,
        "n_prediction_rows": int(len(sub)),
        "n_unique_images": int(sub["image_key"].nunique()),
        "true_label_min": int(sub["true_label"].min()),
        "true_label_max": int(sub["true_label"].max()),
        "confidence_mean": float(sub["confidence"].mean()),
        "confidence_min": float(sub["confidence"].min()),
        "confidence_max": float(sub["confidence"].max()),
        "optional_missing": ";".join(optional_missing),
    }

    return errors, summary


def validate_cost_table(cost: pd.DataFrame, model_name: str) -> tuple[list[str], dict]:
    errors = check_columns(f"cost[{model_name}]", cost, REQUIRED_COST_COLS)
    if errors:
        return errors, {}

    sub = cost[cost["model_name"] == model_name].copy()
    if sub.empty:
        return [f"cost[{model_name}]: no rows for model_name"], {}

    if len(sub) > 1:
        errors.append(f"cost[{model_name}]: multiple rows found; expected one row")

    r = sub.iloc[0]
    for c in ["mean_ms_per_image", "median_ms_per_image", "images_per_second", "checkpoint_mb"]:
        if float(r[c]) <= 0:
            errors.append(f"cost[{model_name}]: {c} must be positive")

    summary = {
        "model_name": model_name,
        "mean_ms_per_image": float(r["mean_ms_per_image"]),
        "median_ms_per_image": float(r["median_ms_per_image"]),
        "images_per_second": float(r["images_per_second"]),
        "pytorch_peak_allocated_mem_mb": float(r["pytorch_peak_allocated_mem_mb"]),
        "checkpoint_mb": float(r["checkpoint_mb"]),
        "batch_size": int(r["batch_size"]),
        "device": str(r["device"]),
    }

    return errors, summary


def check_shared_image_set(pred_tables: dict[str, pd.DataFrame], enabled_models: list[str]) -> list[str]:
    errors = []
    ref_model = enabled_models[0]
    ref = pred_tables[ref_model]
    ref_sub = ref[ref["model_name"] == ref_model][["image_key", "true_label"]].sort_values("image_key").reset_index(drop=True)

    for model_name in enabled_models[1:]:
        cur = pred_tables[model_name]
        cur_sub = cur[cur["model_name"] == model_name][["image_key", "true_label"]].sort_values("image_key").reset_index(drop=True)

        if len(cur_sub) != len(ref_sub):
            errors.append(f"shared image set: {model_name} row count differs from {ref_model}")
            continue

        if not cur_sub["image_key"].equals(ref_sub["image_key"]):
            errors.append(f"shared image set: {model_name} image_key differs from {ref_model}")

        if not cur_sub["true_label"].equals(ref_sub["true_label"]):
            errors.append(f"shared image set: {model_name} true_label differs from {ref_model}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="experiments/v0_8_1_unified_orchestration/configs/model_registry.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/v0_8_1_unified_orchestration/outputs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = read_csv(args.registry)
    errors = validate_registry(registry)

    if errors:
        raise SystemExit("\n".join(errors))

    enabled = registry[registry["enabled"].astype(int) == 1].copy()
    enabled_models = enabled["model_name"].tolist()

    pred_cache: dict[str, pd.DataFrame] = {}
    cost_cache: dict[str, pd.DataFrame] = {}

    pred_summaries = []
    cost_summaries = []

    for _, r in enabled.iterrows():
        model_name = r["model_name"]
        pred_path = r["prediction_csv"]
        cost_path = r["cost_csv"]

        pred = pred_cache.setdefault(pred_path, read_csv(pred_path))
        cost = cost_cache.setdefault(cost_path, read_csv(cost_path))

        pred_errors, pred_summary = validate_prediction_table(pred, model_name)
        cost_errors, cost_summary = validate_cost_table(cost, model_name)

        errors.extend(pred_errors)
        errors.extend(cost_errors)

        if pred_summary:
            pred_summary["prediction_csv"] = pred_path
            pred_summaries.append(pred_summary)

        if cost_summary:
            cost_summary["cost_csv"] = cost_path
            cost_summaries.append(cost_summary)

    model_to_pred = {
        r["model_name"]: pred_cache[r["prediction_csv"]]
        for _, r in enabled.iterrows()
    }
    errors.extend(check_shared_image_set(model_to_pred, enabled_models))

    pred_df = pd.DataFrame(pred_summaries)
    cost_df = pd.DataFrame(cost_summaries)

    pred_df.to_csv(out_dir / "input_prediction_validation_summary.csv", index=False)
    cost_df.to_csv(out_dir / "input_cost_validation_summary.csv", index=False)

    report = []
    report.append("# v0.8.1 Input Validation Report\n")
    report.append(f"- enabled_models: {len(enabled_models)}")
    report.append(f"- status: {'PASS' if not errors else 'FAIL'}\n")

    report.append("## Enabled models\n")
    for m in enabled_models:
        report.append(f"- {m}")

    report.append("\n## Errors\n")
    if errors:
        for e in errors:
            report.append(f"- {e}")
    else:
        report.append("- None")

    report.append("\n## Notes\n")
    report.append("- `dataset` and `split` are recommended but not required for v0.8.1 first-pass compatibility.")
    report.append("- Existing v0.8.0 prediction CSVs are accepted if they contain image_key, true_label, pred_label, confidence and prob_0~prob_4.")

    (out_dir / "input_validation_report.md").write_text("\n".join(report), encoding="utf-8")

    if errors:
        print("[FAIL] input validation failed")
        for e in errors:
            print("-", e)
        raise SystemExit(1)

    print("[PASS] input validation")
    print("models:", ", ".join(enabled_models))
    print("outputs:", out_dir)


if __name__ == "__main__":
    main()
