#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


OUT_DIR = Path("experiments/v0_8_2c_residual_risk_cases/outputs")

ALL_CASES = OUT_DIR / "operating_point_case_table.csv"
MISSED_ALL = OUT_DIR / "fixed_risk_cases_missed_by_all_protocols.csv"
UNIQUE_SELECTED = OUT_DIR / "fixed_risk_cases_uniquely_selected.csv"

OPS = [
    "efficiency_multiscout_30",
    "safety_convnext_50",
    "safety_swin_50",
]


def main():
    all_cases = pd.read_csv(ALL_CASES)
    missed = pd.read_csv(MISSED_ALL)
    unique = pd.read_csv(UNIQUE_SELECTED)

    hard_keys = set(missed["image_key"].unique().tolist())
    unique_keys = set(unique["image_key"].unique().tolist())
    target_keys = sorted(hard_keys | unique_keys)

    rows = []

    for image_key in target_keys:
        sub_all = all_cases[all_cases["image_key"] == image_key].copy()
        if sub_all.empty:
            continue

        base = {
            "image_key": image_key,
            "case_id": image_key,
            "true_label": int(sub_all["true_label"].iloc[0]),
            "case_group": (
                "missed_by_all_protocols"
                if image_key in hard_keys
                else "uniquely_selected_by_one_protocol"
            ),
            "fixed_events": "|".join(
                sorted(
                    set(
                        missed.loc[missed["image_key"] == image_key, "fixed_event_name"].tolist()
                        + unique.loc[unique["image_key"] == image_key, "fixed_event_name"].tolist()
                    )
                )
            ),
            "only_selected_by": "|".join(
                sorted(set(unique.loc[unique["image_key"] == image_key, "only_selected_by"].tolist()))
            ),
        }

        for op in OPS:
            r = sub_all[sub_all["operating_point"] == op]
            if r.empty:
                continue
            r = r.iloc[0]

            prefix = op
            base[f"{prefix}__scout_pred"] = r["scout_pred"]
            base[f"{prefix}__expert_pred"] = r["expert_pred"]
            base[f"{prefix}__final_pred"] = r["final_pred"]
            base[f"{prefix}__selected_for_expert"] = bool(r["selected_for_expert"])
            base[f"{prefix}__scout_confidence"] = r["scout_confidence"]
            base[f"{prefix}__scout_margin"] = r["scout_margin"]
            base[f"{prefix}__scout_entropy"] = r["scout_entropy"]
            base[f"{prefix}__routing_score"] = r["routing_score"]
            base[f"{prefix}__scout_disagreement"] = bool(r["scout_disagreement"])
            base[f"{prefix}__base_correct"] = bool(r["base_correct"])
            base[f"{prefix}__expert_correct"] = bool(r["expert_correct"])
            base[f"{prefix}__final_correct"] = bool(r["final_correct"])
            base[f"{prefix}__expert_corrected"] = bool(r["expert_corrected"])
            base[f"{prefix}__expert_induced_error"] = bool(r["expert_induced_error"])
            base[f"{prefix}__residual_risk_after_routing"] = bool(r["residual_risk_after_routing"])
            base[f"{prefix}__residual_danger_after_routing"] = bool(r["residual_danger_after_routing"])

        rows.append(base)

    review = pd.DataFrame(rows)
    review.to_csv(OUT_DIR / "hard_case_review_table.csv", index=False)

    print("[DONE] hard case review table")
    print("\nHard/unique case review:")
    compact_cols = [
        "case_group",
        "image_key",
        "true_label",
        "fixed_events",
        "only_selected_by",
    ]

    for op in OPS:
        compact_cols += [
            f"{op}__scout_pred",
            f"{op}__expert_pred",
            f"{op}__final_pred",
            f"{op}__selected_for_expert",
            f"{op}__routing_score",
        ]

    print(review[compact_cols].to_string(index=False))
    print("\nWrote:", OUT_DIR / "hard_case_review_table.csv")


if __name__ == "__main__":
    main()
