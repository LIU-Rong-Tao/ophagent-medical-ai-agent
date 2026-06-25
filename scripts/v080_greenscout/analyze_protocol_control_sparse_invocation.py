#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

IN_CSV = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/predictions/greenscout_three_model_standardized.csv")
OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/sparse_invocation")

GREEN = "retfound_green_linear_probe"
CONV = "convnext_tiny"
RETF = "retfound_mae_cfp_official_protocol"

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5]
POLICIES = ["low_confidence", "low_margin", "high_entropy"]
N_RANDOM = 2000
RANDOM_SEED = 2026


def metrics(method, route_signal, expert_mode, budget, y_true, y_pred, expert_calls):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "method": method,
        "route_signal": route_signal,
        "expert_mode": expert_mode,
        "budget": budget,
        "expert_calls": int(expert_calls),
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n_correct": int((y_true == y_pred).sum()),
        "n_error": int((y_true != y_pred).sum()),
    }


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def avg_pred(prob_dict, model_names):
    probs = sum(prob_dict[m] for m in model_names) / len(model_names)
    return probs.argmax(axis=1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    models = sorted(df["model_name"].unique())
    required = {GREEN, CONV, RETF}
    missing = required - set(models)
    if missing:
        raise RuntimeError(f"missing models: {missing}; found={models}")

    # wide tables
    pred = df.pivot(index="image_key", columns="model_name", values="pred_label")
    true = df.groupby("image_key")["true_label"].first().loc[pred.index].astype(int)

    prob = {}
    for m in models:
        sub = df[df["model_name"] == m].set_index("image_key").loc[pred.index]
        prob[m] = sub[[f"prob_{i}" for i in range(5)]].to_numpy(dtype=float)

    y_true = true.to_numpy(dtype=int)
    green_pred = pred[GREEN].to_numpy(dtype=int)
    conv_pred = pred[CONV].to_numpy(dtype=int)
    retf_pred = pred[RETF].to_numpy(dtype=int)

    # dense baselines
    rows = []
    rows.append(metrics("single", "none", GREEN, 0.0, y_true, green_pred, 0))
    rows.append(metrics("single", "none", CONV, 1.0, y_true, conv_pred, len(y_true)))
    rows.append(metrics("single", "none", RETF, 1.0, y_true, retf_pred, len(y_true)))
    rows.append(metrics("dense_average", "none", "experts_only_average", 1.0, y_true, avg_pred(prob, [CONV, RETF]), 2 * len(y_true)))
    rows.append(metrics("dense_average", "none", "all_three_average", 1.0, y_true, avg_pred(prob, [GREEN, CONV, RETF]), 2 * len(y_true)))

    # Green uncertainty scores: larger = more worth deferring
    gp = prob[GREEN]
    conf = gp.max(axis=1)
    sorted_gp = np.sort(gp, axis=1)
    margin = sorted_gp[:, -1] - sorted_gp[:, -2]

    scores = {
        "low_confidence": -conf,
        "low_margin": -margin,
        "high_entropy": entropy(gp),
    }

    expert_modes = {
        "convnext": [CONV],
        "retfound_official_protocol": [RETF],
        "both_experts_average": [CONV, RETF],
        "all_three_average": [GREEN, CONV, RETF],
    }

    n = len(y_true)
    for policy, score in scores.items():
        order = np.argsort(-score)
        for budget in BUDGETS:
            k = int(round(n * budget))
            defer_idx = order[:k]

            for expert_mode, mode_models in expert_modes.items():
                final_pred = green_pred.copy()
                if expert_mode == "convnext":
                    final_pred[defer_idx] = conv_pred[defer_idx]
                elif expert_mode == "retfound_official_protocol":
                    final_pred[defer_idx] = retf_pred[defer_idx]
                else:
                    expert_prob = sum(prob[m][defer_idx] for m in mode_models) / len(mode_models)
                    final_pred[defer_idx] = expert_prob.argmax(axis=1)

                calls_per_deferred = 1 if expert_mode in ["convnext", "retfound_official_protocol"] else 2
                if expert_mode == "all_three_average":
                    calls_per_deferred = 2  # Green already ran globally; experts are ConvNeXt + RETFound.

                rows.append(metrics(
                    "uncertainty_defer",
                    policy,
                    expert_mode,
                    budget,
                    y_true,
                    final_pred,
                    expert_calls=k * calls_per_deferred,
                ))

    curve = pd.DataFrame(rows)
    curve.to_csv(OUT_DIR / "greenscout_sparse_invocation_curve.csv", index=False)

    # random defer baseline
    rng = np.random.default_rng(RANDOM_SEED)
    rand_rows = []
    for budget in BUDGETS:
        k = int(round(n * budget))
        for expert_mode, mode_models in expert_modes.items():
            accs, f1s, qwks = [], [], []
            for _ in range(N_RANDOM):
                idx = rng.choice(n, size=k, replace=False)
                final_pred = green_pred.copy()

                if expert_mode == "convnext":
                    final_pred[idx] = conv_pred[idx]
                elif expert_mode == "retfound_official_protocol":
                    final_pred[idx] = retf_pred[idx]
                else:
                    expert_prob = sum(prob[m][idx] for m in mode_models) / len(mode_models)
                    final_pred[idx] = expert_prob.argmax(axis=1)

                accs.append(accuracy_score(y_true, final_pred))
                f1s.append(f1_score(y_true, final_pred, average="macro"))
                qwks.append(cohen_kappa_score(y_true, final_pred, weights="quadratic"))

            rand_rows.append({
                "method": "random_defer",
                "expert_mode": expert_mode,
                "budget": budget,
                "k": k,
                "n_random": N_RANDOM,
                "acc_mean": float(np.mean(accs)),
                "acc_p025": float(np.quantile(accs, 0.025)),
                "acc_p975": float(np.quantile(accs, 0.975)),
                "macro_f1_mean": float(np.mean(f1s)),
                "qwk_mean": float(np.mean(qwks)),
            })

    pd.DataFrame(rand_rows).to_csv(OUT_DIR / "greenscout_random_defer_summary.csv", index=False)

    # compact cost-aware summary for key deployable policies
    dense_all_acc = curve.query("method == 'dense_average' and expert_mode == 'all_three_average'")["accuracy"].iloc[0]
    dense_experts_acc = curve.query("method == 'dense_average' and expert_mode == 'experts_only_average'")["accuracy"].iloc[0]

    key = curve[
        (curve["method"] == "uncertainty_defer")
        & (curve["expert_mode"] == "both_experts_average")
        & (curve["budget"].isin([0.3, 0.4, 0.5]))
    ].copy()

    key["gain_vs_dense_all"] = key["accuracy"] - dense_all_acc
    key["gain_vs_dense_experts"] = key["accuracy"] - dense_experts_acc
    key["expert_call_fraction_vs_dense_experts"] = key["budget"]

    rand = pd.read_csv(OUT_DIR / "greenscout_random_defer_summary.csv")
    rand = rand[rand["expert_mode"] == "both_experts_average"][["budget", "acc_mean", "acc_p025", "acc_p975"]]
    key = key.merge(rand, on="budget", how="left")
    key["above_random_p975"] = key["accuracy"] > key["acc_p975"]

    key.to_csv(OUT_DIR / "greenscout_cost_aware_sparse_summary.csv", index=False)

    print("[DONE]")
    print("curve:", OUT_DIR / "greenscout_sparse_invocation_curve.csv")
    print("random:", OUT_DIR / "greenscout_random_defer_summary.csv")
    print("summary:", OUT_DIR / "greenscout_cost_aware_sparse_summary.csv")
    print()
    print("=== dense baselines ===")
    print(curve[curve["method"].isin(["single", "dense_average"])].to_string(index=False))
    print()
    print("=== key sparse summary ===")
    print(key.sort_values(["budget", "accuracy"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
