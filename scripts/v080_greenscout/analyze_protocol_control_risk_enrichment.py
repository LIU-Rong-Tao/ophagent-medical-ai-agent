#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

IN_CSV = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/predictions/greenscout_three_model_standardized.csv")
OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/risk_enrichment")

GREEN = "retfound_green_linear_probe"
CONV = "convnext_tiny"
RETF = "retfound_mae_cfp_official_protocol"

BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5]


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)

    pred = df.pivot(index="image_key", columns="model_name", values="pred_label")
    true = df.groupby("image_key")["true_label"].first().loc[pred.index].astype(int)

    prob = {}
    for m in [GREEN, CONV, RETF]:
        sub = df[df["model_name"] == m].set_index("image_key").loc[pred.index]
        prob[m] = sub[[f"prob_{i}" for i in range(5)]].to_numpy(dtype=float)

    y_true = true.to_numpy(dtype=int)
    green_pred = pred[GREEN].to_numpy(dtype=int)
    conv_pred = pred[CONV].to_numpy(dtype=int)
    retf_pred = pred[RETF].to_numpy(dtype=int)

    experts_pred = ((prob[CONV] + prob[RETF]) / 2).argmax(axis=1)

    green_error = green_pred != y_true

    # Risk proxy definitions based on Green prediction.
    severe_pdr_miss = (y_true >= 3) & (green_pred <= 2)
    large_undergrading = (y_true >= 4) & (green_pred <= 2)
    referable_dr_miss = (y_true >= 2) & (green_pred <= 1)

    experts_correct_green_error = green_error & (experts_pred == y_true)
    experts_fix_severe_pdr_miss = severe_pdr_miss & (experts_pred == y_true)
    experts_fix_large_undergrading = large_undergrading & (experts_pred == y_true)
    experts_fix_referable_dr_miss = referable_dr_miss & (experts_pred == y_true)

    gp = prob[GREEN]
    conf = gp.max(axis=1)
    sorted_gp = np.sort(gp, axis=1)
    margin = sorted_gp[:, -1] - sorted_gp[:, -2]
    ent = entropy(gp)

    scores = {
        "low_confidence": -conf,
        "low_margin": -margin,
        "high_entropy": ent,
    }

    events = {
        "green_error": green_error,
        "severe_pdr_miss": severe_pdr_miss,
        "large_undergrading": large_undergrading,
        "referable_dr_miss": referable_dr_miss,
        "experts_correct_green_error": experts_correct_green_error,
        "experts_fix_severe_pdr_miss": experts_fix_severe_pdr_miss,
        "experts_fix_large_undergrading": experts_fix_large_undergrading,
        "experts_fix_referable_dr_miss": experts_fix_referable_dr_miss,
    }

    n = len(y_true)
    rows = []

    for policy, score in scores.items():
        order = np.argsort(-score)

        for budget in BUDGETS:
            k = int(round(n * budget))
            selected = np.zeros(n, dtype=bool)
            selected[order[:k]] = True

            for event_name, mask in events.items():
                total = int(mask.sum())
                captured = int((selected & mask).sum())
                random_expected = total * budget

                rows.append({
                    "policy": policy,
                    "budget": budget,
                    "selected_n": k,
                    "event": event_name,
                    "event_total": total,
                    "captured": captured,
                    "recall_at_budget": captured / total if total else 0.0,
                    "precision_in_selected": captured / k if k else 0.0,
                    "random_expected_captured": random_expected,
                    "enrichment_vs_random": captured / random_expected if random_expected > 0 else np.nan,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "greenscout_risk_enrichment.csv", index=False)

    key_events = [
        "green_error",
        "severe_pdr_miss",
        "large_undergrading",
        "referable_dr_miss",
        "experts_correct_green_error",
        "experts_fix_severe_pdr_miss",
        "experts_fix_large_undergrading",
    ]

    key = out[
        (out["budget"].isin([0.3, 0.4, 0.5]))
        & (out["event"].isin(key_events))
    ].copy()

    key = key.sort_values(["event", "budget", "enrichment_vs_random"], ascending=[True, True, False])
    key.to_csv(OUT_DIR / "greenscout_key_risk_enrichment_summary.csv", index=False)

    cases = pd.DataFrame({
        "image_key": pred.index,
        "y_true": y_true,
        "green_pred": green_pred,
        "convnext_pred": conv_pred,
        "retfound_pred": retf_pred,
        "experts_avg_pred": experts_pred,
        "green_confidence": conf,
        "green_margin": margin,
        "green_entropy_norm": ent,
        "green_error": green_error,
        "severe_pdr_miss": severe_pdr_miss,
        "large_undergrading": large_undergrading,
        "referable_dr_miss": referable_dr_miss,
        "experts_correct_green_error": experts_correct_green_error,
        "experts_fix_severe_pdr_miss": experts_fix_severe_pdr_miss,
        "experts_fix_large_undergrading": experts_fix_large_undergrading,
        "experts_fix_referable_dr_miss": experts_fix_referable_dr_miss,
    })

    cases = cases[
        cases["green_error"]
        | cases["severe_pdr_miss"]
        | cases["large_undergrading"]
        | cases["referable_dr_miss"]
    ].copy()

    cases = cases.sort_values(
        ["large_undergrading", "severe_pdr_miss", "referable_dr_miss", "green_error"],
        ascending=False,
    )

    cases.to_csv(OUT_DIR / "greenscout_green_error_and_risk_cases.csv", index=False)

    print("[DONE]")
    print("risk:", OUT_DIR / "greenscout_risk_enrichment.csv")
    print("summary:", OUT_DIR / "greenscout_key_risk_enrichment_summary.csv")
    print("cases:", OUT_DIR / "greenscout_green_error_and_risk_cases.csv")
    print()
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
