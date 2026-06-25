#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/complementarity")
PRED_DIR = Path("experiments/v0_8_0_greenscout_feasibility/predictions")

GREEN_PRED = PRED_DIR / "retfound_green_test_predictions.csv"

EXISTING = {
    "convnext_tiny": Path("experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv"),
    "retfound_mae_cfp_official_like": Path("experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv"),
}

PROB_NAME_COLS = [
    "prob_No DR",
    "prob_Mild DR",
    "prob_Moderate DR",
    "prob_Severe DR",
    "prob_Proliferative DR",
]


def standardize_existing(model_name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["image_key"] = df["image_path"].map(lambda x: Path(str(x)).stem)
    out["true_label"] = df["true_idx"].astype(int)
    out["pred_label"] = df["pred_idx"].astype(int)
    for i, c in enumerate(PROB_NAME_COLS):
        out[f"prob_{i}"] = df[c].astype(float)
    out["confidence"] = df["confidence"].astype(float)
    out["model_name"] = model_name
    return out


def standardize_green(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["image_key"] = df["image_key"]
    out["true_label"] = df["true_label"].astype(int)
    out["pred_label"] = df["pred_label"].astype(int)
    for i in range(5):
        out[f"prob_{i}"] = df[f"prob_{i}"].astype(float)
    out["confidence"] = df["confidence"].astype(float)
    out["model_name"] = "retfound_green_linear_probe"
    return out


def metric_row(method, y_true, y_pred, type_):
    return {
        "method": method,
        "type": type_,
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n_correct": int((np.asarray(y_true) == np.asarray(y_pred)).sum()),
        "n_error": int((np.asarray(y_true) != np.asarray(y_pred)).sum()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    parts = [standardize_green(GREEN_PRED)]
    for name, path in EXISTING.items():
        parts.append(standardize_existing(name, path))

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["image_key", "model_name"]).reset_index(drop=True)

    standardized_out = PRED_DIR / "greenscout_three_model_standardized.csv"
    df.to_csv(standardized_out, index=False)

    models = sorted(df["model_name"].unique())
    pred_wide = df.pivot(index="image_key", columns="model_name", values="pred_label")
    true = df.groupby("image_key")["true_label"].first().loc[pred_wide.index]
    correct = pred_wide.eq(true, axis=0)
    error = ~correct

    # 概率 ensemble
    prob_wides = {}
    for i in range(5):
        prob_wides[i] = df.pivot(index="image_key", columns="model_name", values=f"prob_{i}").loc[pred_wide.index]

    mean_probs = np.vstack([prob_wides[i].mean(axis=1).to_numpy() for i in range(5)]).T
    avg_pred = mean_probs.argmax(axis=1)

    # Oracle：只要三者中有一个答对，就认为可选对；否则用 average ensemble 的预测
    oracle_has_correct = correct.any(axis=1)
    oracle_pred = pd.Series(avg_pred, index=pred_wide.index)
    oracle_pred.loc[oracle_has_correct] = true.loc[oracle_has_correct]

    metric_rows = []
    y_true = true.to_numpy()

    for m in models:
        metric_rows.append(metric_row(m, y_true, pred_wide[m].to_numpy(), "single_model"))

    metric_rows.append(metric_row("average_ensemble", y_true, avg_pred, "posthoc_average_ensemble"))
    metric_rows.append(metric_row("oracle_expert_selection", y_true, oracle_pred.to_numpy(), "oracle_upper_bound"))

    metrics = pd.DataFrame(metric_rows).sort_values("accuracy", ascending=False)
    metrics.to_csv(OUT_DIR / "greenscout_three_model_metrics.csv", index=False)

    # pairwise error overlap
    pair_rows = []
    for a, b in combinations(models, 2):
        a_err = error[a]
        b_err = error[b]
        both_err = a_err & b_err
        either_err = a_err | b_err

        pair_rows.append({
            "model_a": a,
            "model_b": b,
            "n_images": int(len(true)),
            "model_a_error": int(a_err.sum()),
            "model_b_error": int(b_err.sum()),
            "both_error": int(both_err.sum()),
            "either_error": int(either_err.sum()),
            "a_correct_b_wrong": int((~a_err & b_err).sum()),
            "b_correct_a_wrong": int((a_err & ~b_err).sum()),
            "error_jaccard": float(both_err.sum() / either_err.sum()) if either_err.sum() else 0.0,
            "error_overlap_over_min_error": float(both_err.sum() / min(a_err.sum(), b_err.sum())) if min(a_err.sum(), b_err.sum()) else 0.0,
        })

    pair_df = pd.DataFrame(pair_rows).sort_values("error_jaccard", ascending=False)
    pair_df.to_csv(OUT_DIR / "greenscout_pairwise_error_overlap.csv", index=False)

    # unique correction
    total_correct = correct.sum(axis=1)
    total_error = error.sum(axis=1)

    unique_rows = []
    for m in models:
        unique_correct = correct[m] & (total_correct == 1)
        unique_wrong = error[m] & (total_error == 1)
        unique_rows.append({
            "model_name": m,
            "n_images": int(len(true)),
            "correct_count": int(correct[m].sum()),
            "error_count": int(error[m].sum()),
            "unique_correct_count": int(unique_correct.sum()),
            "unique_correct_rate": float(unique_correct.mean()),
            "unique_wrong_count": int(unique_wrong.sum()),
            "unique_wrong_rate": float(unique_wrong.mean()),
        })

    unique_df = pd.DataFrame(unique_rows).sort_values("unique_correct_count", ascending=False)
    unique_df.to_csv(OUT_DIR / "greenscout_unique_corrections.csv", index=False)

    # oracle summary
    best_single = metrics[metrics["type"] == "single_model"].sort_values("accuracy", ascending=False).iloc[0]
    avg = metrics[metrics["method"] == "average_ensemble"].iloc[0]
    oracle = metrics[metrics["method"] == "oracle_expert_selection"].iloc[0]

    all_wrong = ~oracle_has_correct
    all_correct = correct.all(axis=1)
    mixed = oracle_has_correct & (~all_correct)

    oracle_summary = pd.DataFrame([
        {"item": "n_images", "value": len(true), "description": "APTOS test 图像数"},
        {"item": "models", "value": ";".join(models), "description": "参与分析的模型"},
        {"item": "best_single_model", "value": best_single["method"], "description": "Accuracy 最高单模型"},
        {"item": "best_single_accuracy", "value": best_single["accuracy"], "description": "最佳单模型 Accuracy"},
        {"item": "average_ensemble_accuracy", "value": avg["accuracy"], "description": "平均概率 ensemble Accuracy"},
        {"item": "oracle_accuracy", "value": oracle["accuracy"], "description": "Oracle expert selection Accuracy"},
        {"item": "oracle_gain_over_best_single", "value": float(oracle["accuracy"] - best_single["accuracy"]), "description": "Oracle 相对最佳单模型 Acc 增益"},
        {"item": "oracle_gain_over_average_ensemble", "value": float(oracle["accuracy"] - avg["accuracy"]), "description": "Oracle 相对平均 ensemble Acc 增益"},
        {"item": "all_models_wrong_count", "value": int(all_wrong.sum()), "description": "三个模型全部错误图像数"},
        {"item": "all_models_correct_count", "value": int(all_correct.sum()), "description": "三个模型全部正确图像数"},
        {"item": "mixed_correct_wrong_count", "value": int(mixed.sum()), "description": "至少一个模型正确且至少一个模型错误的图像数"},
    ])
    oracle_summary.to_csv(OUT_DIR / "greenscout_oracle_upper_bound.csv", index=False)

    # disagreement cases
    cases = pd.DataFrame({
        "image_key": pred_wide.index,
        "true_label": true.to_numpy(),
        "n_correct_models": correct.sum(axis=1).to_numpy(),
        "n_error_models": error.sum(axis=1).to_numpy(),
        "oracle_correct": oracle_has_correct.to_numpy(),
    }).set_index("image_key")

    for m in models:
        cases[f"{m}_pred"] = pred_wide[m]
        cases[f"{m}_correct"] = correct[m]

    cases = cases.reset_index()
    cases = cases[cases["n_correct_models"].between(1, len(models)-1)]
    cases = cases.sort_values(["n_correct_models", "image_key"])
    cases.to_csv(OUT_DIR / "greenscout_mixed_cases.csv", index=False)

    print("[DONE]")
    print("standardized:", standardized_out)
    print("metrics:", OUT_DIR / "greenscout_three_model_metrics.csv")
    print("oracle:", OUT_DIR / "greenscout_oracle_upper_bound.csv")
    print("unique:", OUT_DIR / "greenscout_unique_corrections.csv")
    print("pairwise:", OUT_DIR / "greenscout_pairwise_error_overlap.csv")
    print("mixed cases:", OUT_DIR / "greenscout_mixed_cases.csv")

    print("\n=== metrics ===")
    print(metrics.to_string(index=False))

    print("\n=== oracle summary ===")
    print(oracle_summary.to_string(index=False))

    print("\n=== unique corrections ===")
    print(unique_df.to_string(index=False))

    print("\n=== pairwise overlap ===")
    print(pair_df.to_string(index=False))


if __name__ == "__main__":
    main()
