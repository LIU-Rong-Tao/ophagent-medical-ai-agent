#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

IN_CSV = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/predictions/greenscout_three_model_standardized.csv")
OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/complementarity")

GREEN = "retfound_green_linear_probe"
CONV = "convnext_tiny"
RETF = "retfound_mae_cfp_official_protocol"
MODELS = [CONV, GREEN, RETF]


def metric_row(method, type_, y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "method": method,
        "type": type_,
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n_correct": int((y_true == y_pred).sum()),
        "n_error": int((y_true != y_pred).sum()),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    found = set(df["model_name"].unique())
    missing = set(MODELS) - found
    if missing:
        raise RuntimeError(f"missing models: {missing}; found={sorted(found)}")

    pred = df.pivot(index="image_key", columns="model_name", values="pred_label")
    true = df.groupby("image_key")["true_label"].first().loc[pred.index].astype(int)
    y_true = true.to_numpy(dtype=int)

    prob = {}
    for m in MODELS:
        sub = df[df["model_name"] == m].set_index("image_key").loc[pred.index]
        prob[m] = sub[[f"prob_{i}" for i in range(5)]].to_numpy(dtype=float)

    rows = []

    for m in MODELS:
        rows.append(metric_row(m, "single_model", y_true, pred[m].to_numpy(dtype=int)))

    experts_pred = ((prob[CONV] + prob[RETF]) / 2).argmax(axis=1)
    rows.append(metric_row("experts_only_average", "posthoc_average_ensemble", y_true, experts_pred))

    all_three_pred = ((prob[CONV] + prob[GREEN] + prob[RETF]) / 3).argmax(axis=1)
    rows.append(metric_row("all_three_average", "posthoc_average_ensemble", y_true, all_three_pred))

    correct = pred.eq(true, axis=0)
    oracle_has_correct = correct.any(axis=1)
    oracle_pred = pd.Series(all_three_pred, index=pred.index)
    oracle_pred.loc[oracle_has_correct] = true.loc[oracle_has_correct]
    rows.append(metric_row("oracle_expert_selection", "oracle_upper_bound", y_true, oracle_pred.to_numpy(dtype=int)))

    metrics = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    acc_experts = float(metrics.loc[metrics["method"] == "experts_only_average", "accuracy"].iloc[0])
    acc_all = float(metrics.loc[metrics["method"] == "all_three_average", "accuracy"].iloc[0])
    acc_oracle = float(metrics.loc[metrics["method"] == "oracle_expert_selection", "accuracy"].iloc[0])

    if abs(acc_experts - 0.8545454545454545) > 1e-9:
        raise RuntimeError(f"bad experts_only_average acc: {acc_experts}")
    if abs(acc_all - 0.8454545454545455) > 1e-9:
        raise RuntimeError(f"bad all_three_average acc: {acc_all}")
    if abs(acc_oracle - 0.9109090909090909) > 1e-9:
        raise RuntimeError(f"bad oracle acc: {acc_oracle}")

    metrics.to_csv(OUT_DIR / "greenscout_three_model_metrics.csv", index=False)

    error = ~correct
    pair_rows = []
    for a, b in combinations(MODELS, 2):
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

    pd.DataFrame(pair_rows).sort_values("error_jaccard", ascending=False).to_csv(
        OUT_DIR / "greenscout_pairwise_error_overlap.csv", index=False
    )

    total_correct = correct.sum(axis=1)
    total_error = error.sum(axis=1)

    unique_rows = []
    for m in MODELS:
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

    pd.DataFrame(unique_rows).sort_values("unique_correct_count", ascending=False).to_csv(
        OUT_DIR / "greenscout_unique_corrections.csv", index=False
    )

    all_wrong = ~oracle_has_correct
    all_correct = correct.all(axis=1)
    mixed = oracle_has_correct & (~all_correct)

    best_single = metrics[metrics["type"] == "single_model"].sort_values("accuracy", ascending=False).iloc[0]
    experts_avg = metrics[metrics["method"] == "experts_only_average"].iloc[0]
    all_avg = metrics[metrics["method"] == "all_three_average"].iloc[0]
    oracle = metrics[metrics["method"] == "oracle_expert_selection"].iloc[0]

    oracle_summary = pd.DataFrame([
        {"item": "n_images", "value": len(true), "description": "APTOS test 图像数"},
        {"item": "models", "value": ";".join(MODELS), "description": "参与分析的模型"},
        {"item": "best_single_model", "value": best_single["method"], "description": "Accuracy 最高单模型"},
        {"item": "best_single_accuracy", "value": best_single["accuracy"], "description": "最佳单模型 Accuracy"},
        {"item": "experts_only_average_accuracy", "value": experts_avg["accuracy"], "description": "ConvNeXt+RETFound 专家平均概率 ensemble Accuracy"},
        {"item": "all_three_average_accuracy", "value": all_avg["accuracy"], "description": "Green+ConvNeXt+RETFound 三模型平均概率 ensemble Accuracy"},
        {"item": "oracle_accuracy", "value": oracle["accuracy"], "description": "Oracle expert selection Accuracy，不可部署"},
        {"item": "oracle_gain_over_best_single", "value": oracle["accuracy"] - best_single["accuracy"], "description": "Oracle 相对最佳单模型 Acc 增益"},
        {"item": "all_models_wrong_count", "value": int(all_wrong.sum()), "description": "三个模型全部错误图像数"},
        {"item": "all_models_correct_count", "value": int(all_correct.sum()), "description": "三个模型全部正确图像数"},
        {"item": "mixed_correct_wrong_count", "value": int(mixed.sum()), "description": "至少一个模型正确且至少一个模型错误的图像数"},
    ])

    oracle_summary.to_csv(OUT_DIR / "greenscout_oracle_upper_bound.csv", index=False)

    print("[DONE]")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
