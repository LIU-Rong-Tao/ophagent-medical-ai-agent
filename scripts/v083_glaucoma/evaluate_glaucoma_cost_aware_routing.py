#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


MERGED_CSV = Path(
    "experiments/v0_8_3_glaucoma_scout_routing/"
    "routing_inputs/scout_expert_merged_test_predictions.csv"
)

OUT_DIR = Path(
    "experiments/v0_8_3_glaucoma_scout_routing/"
    "routing_eval_cost_aware/test"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CURVE = OUT_DIR / "cost_aware_routing_curve.csv"
OUT_BEST = OUT_DIR / "best_policy_by_budget.csv"
OUT_RANDOM = OUT_DIR / "random_defer_summary.csv"
OUT_ORACLE = OUT_DIR / "oracle_up_to_k_curve.csv"
OUT_SUMMARY = OUT_DIR / "cost_aware_routing_summary.json"
OUT_MD = OUT_DIR / "cost_aware_routing_report.md"

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
N_RANDOM = 2000
RANDOM_SEED = 42

STRATEGIES = {
    "entropy_desc": ("entropy_scout", False),
    "margin_asc": ("margin_scout", True),
    "confidence_asc": ("confidence_scout", True),
}


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "error_n": int((np.asarray(y_true) != np.asarray(y_pred)).sum()),
    }


def add_cost_fields(row, n, scout_acc, expert_acc):
    expert_n = int(row["expert_n"])
    expert_rate = expert_n / n

    row["expert_call_rate"] = float(expert_rate)
    row["expert_call_count"] = expert_n

    # 当前 v0.8.3 是 single expert，不是 both-experts average。
    # 所以 single_expert_call_equivalent = expert_call_count。
    row["single_expert_call_equivalent"] = expert_n

    # 相对 expert-only 的专家调用成本。
    row["relative_expert_cost_vs_expert_only"] = float(expert_rate)

    # 所有方法默认都先跑 scout；这里单独统计 expert 追加成本。
    row["gain_over_scout_acc"] = float(row["accuracy"] - scout_acc)
    row["gap_to_expert_only_acc"] = float(expert_acc - row["accuracy"])

    if expert_n > 0:
        row["acc_gain_per_100_expert_calls"] = float((row["accuracy"] - scout_acc) / expert_n * 100)
    else:
        row["acc_gain_per_100_expert_calls"] = None

    return row


def make_routed_pred(df, routed_keys):
    routed_mask = df["image_key"].isin(routed_keys).to_numpy()
    return np.where(
        routed_mask,
        df["pred_label_expert"].to_numpy(),
        df["pred_label_scout"].to_numpy(),
    )


def main():
    df = pd.read_csv(MERGED_CSV)

    required = [
        "image_key",
        "true_label",
        "pred_label_scout",
        "pred_label_expert",
        "entropy_scout",
        "margin_scout",
        "confidence_scout",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    n = len(df)
    y_true = df["true_label"].to_numpy()
    scout_pred = df["pred_label_scout"].to_numpy()
    expert_pred = df["pred_label_expert"].to_numpy()

    scout_metrics = compute_metrics(y_true, scout_pred)
    expert_metrics = compute_metrics(y_true, expert_pred)
    scout_acc = scout_metrics["accuracy"]
    expert_acc = expert_metrics["accuracy"]

    rows = []

    # Baselines
    for method_name, pred, expert_n in [
        ("scout_only", scout_pred, 0),
        ("expert_only", expert_pred, n),
    ]:
        row = {
            "strategy": method_name,
            "budget": expert_n / n,
            "expert_n": expert_n,
            "route_type": "baseline",
        }
        row.update(compute_metrics(y_true, pred))
        row = add_cost_fields(row, n, scout_acc, expert_acc)
        row["random_acc_mean"] = None
        row["random_acc_ci95_low"] = None
        row["random_acc_ci95_high"] = None
        row["gain_over_random_mean_acc"] = None
        rows.append(row)

    # Random defer baseline
    rng = np.random.default_rng(RANDOM_SEED)
    random_rows = []

    for budget in BUDGETS:
        k = int(round(budget * n))
        accs = []
        macro_f1s = []
        kappas = []

        for _ in range(N_RANDOM):
            idx = rng.choice(n, size=k, replace=False)
            pred = scout_pred.copy()
            pred[idx] = expert_pred[idx]

            accs.append(accuracy_score(y_true, pred))
            macro_f1s.append(f1_score(y_true, pred, average="macro"))
            kappas.append(cohen_kappa_score(y_true, pred))

        accs = np.asarray(accs)
        macro_f1s = np.asarray(macro_f1s)
        kappas = np.asarray(kappas)

        random_row = {
            "strategy": "random_defer",
            "budget": float(budget),
            "expert_n": int(k),
            "route_type": "random_baseline",
            "accuracy": float(accs.mean()),
            "macro_f1": float(macro_f1s.mean()),
            "weighted_f1": None,
            "kappa": float(kappas.mean()),
            "error_n": None,
            "random_acc_mean": float(accs.mean()),
            "random_acc_ci95_low": float(np.quantile(accs, 0.025)),
            "random_acc_ci95_high": float(np.quantile(accs, 0.975)),
            "random_macro_f1_mean": float(macro_f1s.mean()),
            "random_kappa_mean": float(kappas.mean()),
        }
        random_row = add_cost_fields(random_row, n, scout_acc, expert_acc)
        random_row["gain_over_random_mean_acc"] = 0.0

        random_rows.append(random_row)
        rows.append(random_row)

    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(OUT_RANDOM, index=False)

    random_acc_by_budget = {
        float(r["budget"]): float(r["random_acc_mean"])
        for r in random_rows
    }

    # Uncertainty defer curve
    for strategy_name, (score_col, ascending) in STRATEGIES.items():
        ranked = df.sort_values(score_col, ascending=ascending).copy()

        for budget in BUDGETS:
            k = int(round(budget * n))
            routed_keys = set(ranked.head(k)["image_key"])
            pred = make_routed_pred(df, routed_keys)

            row = {
                "strategy": strategy_name,
                "budget": float(budget),
                "expert_n": int(k),
                "route_type": "uncertainty_defer",
            }
            row.update(compute_metrics(y_true, pred))
            row = add_cost_fields(row, n, scout_acc, expert_acc)

            random_mean = random_acc_by_budget[float(budget)]
            row["random_acc_mean"] = random_mean
            row["random_acc_ci95_low"] = float(
                random_df.loc[random_df["budget"] == float(budget), "random_acc_ci95_low"].iloc[0]
            )
            row["random_acc_ci95_high"] = float(
                random_df.loc[random_df["budget"] == float(budget), "random_acc_ci95_high"].iloc[0]
            )
            row["gain_over_random_mean_acc"] = float(row["accuracy"] - random_mean)

            rows.append(row)

    # Oracle up-to-k: 后验理论上限，不可部署。
    # 按“scout 错、expert 对”的样本优先路由。
    oracle_df = df.copy()
    oracle_df["correctable"] = (
        (oracle_df["pred_label_scout"] != oracle_df["true_label"])
        & (oracle_df["pred_label_expert"] == oracle_df["true_label"])
    )
    oracle_df["harmful"] = (
        (oracle_df["pred_label_scout"] == oracle_df["true_label"])
        & (oracle_df["pred_label_expert"] != oracle_df["true_label"])
    )

    # correctable 优先，其余样本不重要，只作为预算填充。
    oracle_ranked = pd.concat([
        oracle_df[oracle_df["correctable"]],
        oracle_df[~oracle_df["correctable"]],
    ], axis=0)

    oracle_rows = []
    for budget in BUDGETS:
        k = int(round(budget * n))
        routed_keys = set(oracle_ranked.head(k)["image_key"])
        pred = make_routed_pred(df, routed_keys)

        row = {
            "strategy": "oracle_correctable_first",
            "budget": float(budget),
            "expert_n": int(k),
            "route_type": "oracle_upper_bound",
        }
        row.update(compute_metrics(y_true, pred))
        row = add_cost_fields(row, n, scout_acc, expert_acc)

        random_mean = random_acc_by_budget[float(budget)]
        row["random_acc_mean"] = random_mean
        row["random_acc_ci95_low"] = float(
            random_df.loc[random_df["budget"] == float(budget), "random_acc_ci95_low"].iloc[0]
        )
        row["random_acc_ci95_high"] = float(
            random_df.loc[random_df["budget"] == float(budget), "random_acc_ci95_high"].iloc[0]
        )
        row["gain_over_random_mean_acc"] = float(row["accuracy"] - random_mean)

        oracle_rows.append(row)
        rows.append(row)

    oracle_out = pd.DataFrame(oracle_rows)
    oracle_out.to_csv(OUT_ORACLE, index=False)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CURVE, index=False)

    # Best deployable policy by budget: only uncertainty strategies, not random/oracle/baselines.
    deployable = result[result["route_type"] == "uncertainty_defer"].copy()
    best = (
        deployable.sort_values(["budget", "accuracy", "macro_f1", "kappa"], ascending=[True, False, False, False])
        .groupby("budget", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(OUT_BEST, index=False)

    complementarity = {
        "both_correct": int(((scout_pred == y_true) & (expert_pred == y_true)).sum()),
        "scout_wrong_expert_correct": int(((scout_pred != y_true) & (expert_pred == y_true)).sum()),
        "scout_correct_expert_wrong": int(((scout_pred == y_true) & (expert_pred != y_true)).sum()),
        "both_wrong": int(((scout_pred != y_true) & (expert_pred != y_true)).sum()),
    }

    summary = {
        "n": int(n),
        "budgets": BUDGETS,
        "random_trials": N_RANDOM,
        "scout_only": scout_metrics,
        "expert_only": expert_metrics,
        "complementarity": complementarity,
        "outputs": {
            "curve": str(OUT_CURVE),
            "best_policy_by_budget": str(OUT_BEST),
            "random_defer_summary": str(OUT_RANDOM),
            "oracle_up_to_k_curve": str(OUT_ORACLE),
        },
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md = []
    md.append("# v0.8.3 Glaucoma cost-aware scout-to-expert routing\n")
    md.append("## Baselines\n")
    md.append(f"- Scout-only: Acc={scout_metrics['accuracy']:.4f}, Macro-F1={scout_metrics['macro_f1']:.4f}, Kappa={scout_metrics['kappa']:.4f}\n")
    md.append(f"- Expert-only: Acc={expert_metrics['accuracy']:.4f}, Macro-F1={expert_metrics['macro_f1']:.4f}, Kappa={expert_metrics['kappa']:.4f}\n")
    md.append("\n## Complementarity\n")
    for k, v in complementarity.items():
        md.append(f"- {k}: {v}\n")

    md.append("\n## Best deployable policy by budget\n")
    md.append(best.to_csv(index=False))
    md.append("\n\n## Full routing curve\n")
    md.append(result.to_csv(index=False))
    md.append("\n")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print("saved:", OUT_CURVE)
    print("saved:", OUT_BEST)
    print("saved:", OUT_RANDOM)
    print("saved:", OUT_ORACLE)
    print("saved:", OUT_SUMMARY)
    print("saved:", OUT_MD)
    print("\nBest deployable policy by budget:")
    print(best.to_string(index=False))
    print("\nFull cost-aware curve:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
