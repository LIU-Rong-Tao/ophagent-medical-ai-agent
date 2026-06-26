#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

OUT_DIR = Path("experiments/v0_8_2b_controlled_protocols/outputs")

PERF = OUT_DIR / "controlled_protocol_main_results.csv"
RISK = OUT_DIR / "fixed_risk_pool_coverage.csv"

EVENT_MAP = {
    "large_undergrading_union_pool": "large_undergrading",
    "referable_dr_miss_union_pool": "referable_miss",
    "severe_pdr_miss_union_pool": "severe_pdr_miss",
}

KEYS = [
    "protocol_family",
    "protocol_name",
    "role",
    "scouts",
    "experts",
    "budget",
    "policy",
]


def main():
    perf = pd.read_csv(PERF)
    risk = pd.read_csv(RISK)

    # 只保留主协议和 dense expert reference。
    perf = perf[perf["role"].isin(["main", "dense_expert_reference"])].copy()

    # fixed risk pool 当前只对 routing protocol 有意义；dense baseline 没有 selected_n/routing。
    risk = risk[risk["role"].isin(["main", "dense_expert_reference"])].copy()

    risk["event_short"] = risk["event_name"].map(EVENT_MAP)

    wide_parts = []

    for metric in [
        "event_total_fixed_pool",
        "selected_event_n",
        "residual_event_n",
        "event_recall_fixed_pool",
        "event_precision_fixed_pool",
        "event_lift_vs_budget",
    ]:
        w = risk.pivot_table(
            index=KEYS + ["selected_n"],
            columns="event_short",
            values=metric,
            aggfunc="first",
        ).reset_index()

        w.columns = [
            c if not isinstance(c, tuple) else "_".join([str(x) for x in c if x])
            for c in w.columns
        ]

        rename = {}
        for event_short in EVENT_MAP.values():
            if event_short in w.columns:
                rename[event_short] = f"{event_short}_{metric}"

        w = w.rename(columns=rename)
        wide_parts.append(w)

    risk_wide = wide_parts[0]
    for w in wide_parts[1:]:
        risk_wide = risk_wide.merge(
            w,
            on=KEYS + ["selected_n"],
            how="outer",
        )

    perf_cols = KEYS + [
        "selected_n",
        "ms_per_image",
        "accuracy",
        "macro_f1",
        "qwk",
        "n_error",
        "above_random_p975",
        "random_accuracy_p975",
        "oracle_accuracy",
        "gap_to_oracle_accuracy",
    ]

    perf_cols = [c for c in perf_cols if c in perf.columns]
    merged = perf[perf_cols].merge(
        risk_wide,
        on=KEYS + ["selected_n"],
        how="left",
    )

    merged = merged.sort_values(
        ["budget", "accuracy", "ms_per_image"],
        ascending=[True, False, True],
    )

    merged.to_csv(OUT_DIR / "controlled_performance_risk_summary.csv", index=False)

    # 核心 routing 协议：ConvNeXt / Swin / ConvNeXt+Swin -> RETFound。
    target_protocols = [
        "convnext_to_retfound",
        "swin_to_retfound",
        "convnext_swin_to_retfound",
    ]

    core = merged[merged["protocol_name"].isin(target_protocols)].copy()

    # 全预算主表：0.2 / 0.3 / 0.4 / 0.5，每个 protocol-budget 保留 accuracy 最优 policy。
    all_budget_best = (
        core
        .sort_values(
            ["protocol_name", "budget", "accuracy", "ms_per_image"],
            ascending=[True, True, False, True],
        )
        .groupby(["protocol_name", "budget"], as_index=False)
        .head(1)
        .sort_values(["budget", "accuracy", "ms_per_image"], ascending=[True, False, True])
    )

    all_budget_best.to_csv(
        OUT_DIR / "all_budget_performance_risk_best.csv",
        index=False,
    )

    # 30% / 50% 预算匹配主表。
    budget_matched = core[core["budget"].round(6).isin([0.3, 0.5])].copy()

    best_budget_matched = all_budget_best[
        all_budget_best["budget"].round(6).isin([0.3, 0.5])
    ].copy()

    best_budget_matched.to_csv(
        OUT_DIR / "budget_matched_performance_risk_best.csv",
        index=False,
    )

    budget_matched.to_csv(
        OUT_DIR / "budget_matched_performance_risk_all_policies.csv",
        index=False,
    )

    # 单独导出 30% 和 50%，避免报告时混用预算。
    best_budget_matched[best_budget_matched["budget"].round(6) == 0.3].to_csv(
        OUT_DIR / "budget_30_performance_risk_best.csv",
        index=False,
    )

    best_budget_matched[best_budget_matched["budget"].round(6) == 0.5].to_csv(
        OUT_DIR / "budget_50_performance_risk_best.csv",
        index=False,
    )

    show_cols = [
        "protocol_family",
        "protocol_name",
        "budget",
        "policy",
        "selected_n",
        "ms_per_image",
        "accuracy",
        "macro_f1",
        "qwk",
        "n_error",
        "large_undergrading_selected_event_n",
        "large_undergrading_event_total_fixed_pool",
        "large_undergrading_event_recall_fixed_pool",
        "referable_miss_selected_event_n",
        "referable_miss_event_total_fixed_pool",
        "referable_miss_event_recall_fixed_pool",
        "severe_pdr_miss_selected_event_n",
        "severe_pdr_miss_event_total_fixed_pool",
        "severe_pdr_miss_event_recall_fixed_pool",
    ]

    show_cols = [c for c in show_cols if c in best_budget_matched.columns]

    print("[DONE] controlled performance+risk summary")
    print("\nBudget-matched best rows:")
    print(best_budget_matched[show_cols].to_string(index=False))

    print("\nWrote:")
    print(OUT_DIR / "controlled_performance_risk_summary.csv")
    print(OUT_DIR / "all_budget_performance_risk_best.csv")
    print(OUT_DIR / "budget_matched_performance_risk_best.csv")
    print(OUT_DIR / "budget_matched_performance_risk_all_policies.csv")
    print(OUT_DIR / "budget_30_performance_risk_best.csv")
    print(OUT_DIR / "budget_50_performance_risk_best.csv")


if __name__ == "__main__":
    main()
