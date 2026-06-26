#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd


OUT_DIR = Path("experiments/v0_8_1_unified_orchestration/outputs")
REGISTRY = Path("experiments/v0_8_1_unified_orchestration/configs/model_registry.csv")


def minmax_score(s: pd.Series, higher_better: bool = True) -> pd.Series:
    x = s.astype(float)
    if x.max() == x.min():
        return pd.Series([0.5] * len(x), index=x.index)
    z = (x - x.min()) / (x.max() - x.min())
    return z if higher_better else 1.0 - z


def main():
    registry = pd.read_csv(REGISTRY)
    single = pd.read_csv(OUT_DIR / "single_model_summary.csv")
    pair = pd.read_csv(OUT_DIR / "pairwise_complementarity.csv")
    sparse = pd.read_csv(OUT_DIR / "sparse_routing_curve.csv")
    cost = pd.read_csv(OUT_DIR / "input_cost_validation_summary.csv")

    reg = registry[registry["enabled"].astype(int) == 1][["model_name", "family", "role_hint"]]

    base = (
        single.rename(columns={"name": "model_name"})
        .merge(cost, on="model_name", how="left")
        .merge(reg, on="model_name", how="left")
    )

    scout_rows = []
    for _, r in base[base["role_hint"].str.lower().str.contains("scout")].iterrows():
        m = r["model_name"]
        sub = sparse[sparse["scout"] == m]
        best = sub.sort_values(
            ["accuracy", "above_random_p975", "online_no_cache_ms_per_image"],
            ascending=[False, False, True],
        ).iloc[0]

        scout_rows.append({
            "model_name": m,
            "family": r["family"],
            "role_hint": r["role_hint"],
            "single_accuracy": r["accuracy"],
            "single_macro_f1": r["macro_f1"],
            "mean_ms_per_image": r["mean_ms_per_image"],
            "checkpoint_mb": r["checkpoint_mb"],
            "best_sparse_setting": best["setting"],
            "best_sparse_accuracy": best["accuracy"],
            "best_sparse_budget": best["budget"],
            "best_sparse_policy": best["policy"],
            "best_sparse_online_ms_per_image": best["online_no_cache_ms_per_image"],
            "best_sparse_above_random_p975": bool(best["above_random_p975"]),
            "best_sparse_gap_to_oracle_accuracy": best["gap_to_oracle_accuracy"],
        })

    scout = pd.DataFrame(scout_rows)
    scout["score_accuracy"] = minmax_score(scout["best_sparse_accuracy"], True)
    scout["score_cost"] = minmax_score(scout["best_sparse_online_ms_per_image"], False)
    scout["score_oracle_gap"] = minmax_score(scout["best_sparse_gap_to_oracle_accuracy"], False)
    scout["scout_score"] = (
        0.45 * scout["score_accuracy"]
        + 0.35 * scout["score_cost"]
        + 0.20 * scout["score_oracle_gap"]
    )
    scout = scout.sort_values(["scout_score", "best_sparse_accuracy"], ascending=False)

    expert_rows = []
    for _, r in base[base["role_hint"].str.lower().str.contains("expert")].iterrows():
        m = r["model_name"]

        fix_counts = []
        fix_rates = []
        for _, pr in pair.iterrows():
            if pr["model_b"] == m:
                fix_counts.append(pr["a_wrong_b_correct"])
                fix_rates.append(pr["a_error_fixable_by_b_rate"])
            if pr["model_a"] == m:
                fix_counts.append(pr["b_wrong_a_correct"])
                fix_rates.append(pr["b_error_fixable_by_a_rate"])

        sub = sparse[sparse["experts"].str.contains(m, regex=False)]
        best = sub.sort_values(
            ["accuracy", "online_no_cache_ms_per_image"],
            ascending=[False, True],
        ).iloc[0]

        expert_rows.append({
            "model_name": m,
            "family": r["family"],
            "role_hint": r["role_hint"],
            "single_accuracy": r["accuracy"],
            "single_macro_f1": r["macro_f1"],
            "single_qwk": r["qwk"],
            "mean_ms_per_image": r["mean_ms_per_image"],
            "checkpoint_mb": r["checkpoint_mb"],
            "mean_pairwise_fix_count": sum(fix_counts) / max(1, len(fix_counts)),
            "mean_pairwise_fix_rate": sum(fix_rates) / max(1, len(fix_rates)),
            "best_sparse_setting_using_expert": best["setting"],
            "best_sparse_accuracy_using_expert": best["accuracy"],
            "best_sparse_online_ms_per_image": best["online_no_cache_ms_per_image"],
        })

    expert = pd.DataFrame(expert_rows)
    expert["score_accuracy"] = minmax_score(expert["single_accuracy"], True)
    expert["score_fix_rate"] = minmax_score(expert["mean_pairwise_fix_rate"], True)
    expert["score_cost"] = minmax_score(expert["mean_ms_per_image"], False)
    expert["expert_score"] = (
        0.45 * expert["score_accuracy"]
        + 0.35 * expert["score_fix_rate"]
        + 0.20 * expert["score_cost"]
    )
    expert = expert.sort_values(["expert_score", "single_accuracy"], ascending=False)

    scout.to_csv(OUT_DIR / "scout_candidate_ranking.csv", index=False)
    expert.to_csv(OUT_DIR / "expert_candidate_ranking.csv", index=False)

    print("[DONE] candidate ranking")
    print("\nScout ranking:")
    print(scout.to_string(index=False))
    print("\nExpert ranking:")
    print(expert.to_string(index=False))


if __name__ == "__main__":
    main()
