#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


OUT_DIR = Path("experiments/v0_8_2c_residual_risk_cases/outputs")
OVERLAP = OUT_DIR / "protocol_overlap_cases.csv"

OPS = [
    "efficiency_multiscout_30",
    "safety_convnext_50",
    "safety_swin_50",
]

FIXED_EVENTS = [
    "fixed_large_undergrading_union_pool",
    "fixed_referable_miss_union_pool",
    "fixed_severe_pdr_miss_union_pool",
]


def bcol(op, flag):
    return f"{op}__{flag}"


def main():
    df = pd.read_csv(OVERLAP)

    for c in df.columns:
        if c.startswith(tuple(OPS)):
            df[c] = df[c].astype(bool)

    rows = []

    for event in FIXED_EVENTS:
        sub = df[df[bcol(OPS[0], event)]].copy()
        # fixed event columns are the same across OPs; use first op's copy.

        row = {
            "event_name": event,
            "event_total": len(sub),
        }

        for op in OPS:
            row[f"{op}_selected_n"] = int(sub[bcol(op, "selected_for_expert")].sum())
            row[f"{op}_residual_n"] = int((~sub[bcol(op, "selected_for_expert")]).sum())

        selected_cols = [bcol(op, "selected_for_expert") for op in OPS]
        sub["selected_by_n_ops"] = sub[selected_cols].sum(axis=1).astype(int)

        for n in [0, 1, 2, 3]:
            row[f"selected_by_{n}_ops"] = int((sub["selected_by_n_ops"] == n).sum())

        # 谁独有抓到
        for op in OPS:
            other_ops = [x for x in OPS if x != op]
            only_mask = sub[bcol(op, "selected_for_expert")].copy()
            for other in other_ops:
                only_mask = only_mask & (~sub[bcol(other, "selected_for_expert")])
            row[f"only_{op}_selected_n"] = int(only_mask.sum())

        # 所有协议都漏掉
        missed_all = sub["selected_by_n_ops"] == 0
        row["missed_by_all_n"] = int(missed_all.sum())

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "protocol_overlap_summary_by_fixed_event.csv", index=False)

    # 导出所有协议都未选中的固定风险病例
    missed_rows = []
    for event in FIXED_EVENTS:
        sub = df[df[bcol(OPS[0], event)]].copy()
        selected_cols = [bcol(op, "selected_for_expert") for op in OPS]
        missed = sub[sub[selected_cols].sum(axis=1) == 0].copy()
        missed["fixed_event_name"] = event
        missed_rows.append(missed)

    missed_all = pd.concat(missed_rows, ignore_index=True)
    missed_all.to_csv(OUT_DIR / "fixed_risk_cases_missed_by_all_protocols.csv", index=False)

    # 导出被单一 operating point 独有选中的固定风险病例
    unique_rows = []
    for event in FIXED_EVENTS:
        sub = df[df[bcol(OPS[0], event)]].copy()
        for op in OPS:
            other_ops = [x for x in OPS if x != op]
            only_mask = sub[bcol(op, "selected_for_expert")].copy()
            for other in other_ops:
                only_mask = only_mask & (~sub[bcol(other, "selected_for_expert")])
            tmp = sub[only_mask].copy()
            tmp["fixed_event_name"] = event
            tmp["only_selected_by"] = op
            unique_rows.append(tmp)

    unique_selected = pd.concat(unique_rows, ignore_index=True)
    unique_selected.to_csv(OUT_DIR / "fixed_risk_cases_uniquely_selected.csv", index=False)

    print("[DONE] v0.8.2c overlap summary")
    print("\nOverlap summary by fixed event:")
    print(summary.to_string(index=False))
    print("\nWrote:")
    print(OUT_DIR / "protocol_overlap_summary_by_fixed_event.csv")
    print(OUT_DIR / "fixed_risk_cases_missed_by_all_protocols.csv")
    print(OUT_DIR / "fixed_risk_cases_uniquely_selected.csv")


if __name__ == "__main__":
    main()
