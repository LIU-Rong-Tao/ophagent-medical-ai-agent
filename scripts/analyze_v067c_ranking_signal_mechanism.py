#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.6.7c ranking signal mechanism analysis.

Purpose:
1. Put uncertainty baselines and severity-aware baselines into one unified comparison.
2. Explain why severity-aware signals outperform ophagent_combined for dangerous undergrading.
3. Analyze Top20% overlap and residual dangerous cases.

Inputs:
  experiments/summary/v0_6_7/clinical_event_cases.csv
  original backbone test_predictions.csv files

Outputs:
  experiments/summary/v0_6_7c/unified_ranking_method_tradeoff.csv
  experiments/summary/v0_6_7c/unified_ranking_method_mean_summary.csv
  experiments/summary/v0_6_7c/top20_overlap_summary.csv
  experiments/summary/v0_6_7c/top20_overlap_cases.csv
  experiments/summary/v0_6_7c/top20_residual_profile.csv
  experiments/summary/v0_6_7c/top20_residual_cases.csv
  experiments/summary/v0_6_7c/v067c_key_findings.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments/summary/v0_6_7/clinical_event_cases.csv"
OUT_DIR = ROOT / "experiments/summary/v0_6_7c"

PREDICTION_PATHS = {
    "convnext_tiny": ROOT / "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "retfound_official_like": ROOT / "experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "swin_tiny": ROOT / "experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_imagenet": ROOT / "experiments/aptos_vit_base_patch16_imagenet/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_official_like": ROOT / "experiments/aptos_vit_base_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "vit_l_official_like": ROOT / "experiments/aptos_vit_large_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
}

CANONICAL_PROB_COLS = ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4"]

METHODS = [
    "confidence_only",
    "margin_only",
    "entropy_only",
    "uncertainty_rank_fusion",
    "ophagent_combined",
    "severe_prob_mass_only",
    "gated_severe_prob_mass_only",
    "expected_grade_only",
    "expected_gap_only",
    "top2_more_severe_only",
]

EVENTS = [
    "general_error",
    "any_undergrading",
    "large_undergrading",
    "referable_dr_miss",
    "vision_threatening_dr_miss",
    "high_confidence_vision_threatening_miss",
]

MAIN_EVENTS = ["large_undergrading", "vision_threatening_dr_miss"]

BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def find_prob_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        ["prob_No DR", "prob_Mild DR", "prob_Moderate DR", "prob_Severe DR", "prob_Proliferative DR"],
        ["prob_0", "prob_1", "prob_2", "prob_3", "prob_4"],
    ]
    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols
    raise ValueError(f"Cannot find probability columns. Columns={df.columns.tolist()}")


def load_prediction_probs() -> pd.DataFrame:
    rows = []
    for backbone, path in PREDICTION_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)

        pred = pd.read_csv(path)
        prob_cols = find_prob_columns(pred)

        keep = pred[["image_path"] + prob_cols].copy()
        keep["backbone"] = backbone
        keep["image_key"] = keep["image_path"].astype(str).map(lambda x: Path(x).name)
        keep = keep.rename(columns={old: new for old, new in zip(prob_cols, CANONICAL_PROB_COLS)})
        keep = keep[["backbone", "image_key"] + CANONICAL_PROB_COLS]
        rows.append(keep)

    out = pd.concat(rows, ignore_index=True)
    for c in CANONICAL_PROB_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "image_key" not in df.columns:
        df["image_key"] = df["image_path"].astype(str).map(lambda x: Path(x).name)
    else:
        df["image_key"] = df["image_key"].astype(str).map(lambda x: Path(x).name)

    probs = load_prediction_probs()
    before = len(df)
    df = df.merge(probs, on=["backbone", "image_key"], how="left", validate="many_to_one")
    if len(df) != before:
        raise RuntimeError(f"Row count changed after merge: {before} -> {len(df)}")

    missing = df[CANONICAL_PROB_COLS].isna().any(axis=1).sum()
    if missing:
        raise RuntimeError(f"Missing probability rows after merge: {missing}")

    numeric_cols = [
        "pred_grade", "top2_grade", "confidence", "top2_confidence",
        "margin", "entropy_norm", "severe_prob_mass", "review_priority_rank",
    ] + CANONICAL_PROB_COLS

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    prob_mat = df[CANONICAL_PROB_COLS].to_numpy(dtype=float)
    grades = np.arange(5, dtype=float)

    df["expected_grade"] = np.nansum(prob_mat * grades.reshape(1, -1), axis=1)

    if "severe_prob_mass" not in df.columns or df["severe_prob_mass"].isna().all():
        df["severe_prob_mass"] = df["prob_3"] + df["prob_4"]

    df["expected_gap"] = df["expected_grade"] - df["pred_grade"]
    df["top2_more_severe"] = df["top2_grade"] > df["pred_grade"]
    df["row_id"] = df["backbone"].astype(str) + "::" + df["image_key"].astype(str)

    return df


def rank_df(g: pd.DataFrame, method: str) -> pd.DataFrame:
    g = g.copy()

    if method == "confidence_only":
        return g.sort_values(["confidence", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "margin_only":
        return g.sort_values(["margin", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "entropy_only":
        return g.sort_values(["entropy_norm", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "uncertainty_rank_fusion":
        g["_r_conf"] = g["confidence"].rank(method="first", ascending=True, na_option="bottom")
        g["_r_margin"] = g["margin"].rank(method="first", ascending=True, na_option="bottom")
        g["_r_entropy"] = g["entropy_norm"].rank(method="first", ascending=False, na_option="bottom")
        g["_fusion"] = g[["_r_conf", "_r_margin", "_r_entropy"]].mean(axis=1)
        return g.sort_values(["_fusion", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "ophagent_combined":
        return g.sort_values(["review_priority_rank", "case_id"], ascending=[True, True], kind="mergesort")

    if method == "severe_prob_mass_only":
        return g.sort_values(["severe_prob_mass", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "gated_severe_prob_mass_only":
        g["_score"] = np.where(g["pred_grade"] <= 2, g["severe_prob_mass"], -1.0)
        return g.sort_values(["_score", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "expected_grade_only":
        return g.sort_values(["expected_grade", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "expected_gap_only":
        return g.sort_values(["expected_gap", "case_id"], ascending=[False, True], kind="mergesort")

    if method == "top2_more_severe_only":
        g["_score"] = np.where(g["top2_more_severe"], g["top2_confidence"].fillna(0.0), -1.0)
        return g.sort_values(["_score", "case_id"], ascending=[False, True], kind="mergesort")

    raise KeyError(method)


def compute_tradeoff(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for backbone, g in df.groupby("backbone", sort=True):
        n = len(g)

        for method in METHODS:
            ranked = rank_df(g, method).reset_index(drop=True)

            for event in EVENTS:
                event_values = ranked[event].astype(bool).to_numpy()
                total = int(event_values.sum())
                base_rate = total / n if n else 0.0

                for budget in BUDGETS:
                    k = int(math.ceil(n * budget))
                    k = max(1, min(k, n))

                    captured = int(event_values[:k].sum())
                    residual = total - captured
                    auto_n = n - k

                    precision = captured / k if k else np.nan
                    recall = captured / total if total else np.nan
                    lift = precision / base_rate if base_rate else np.nan
                    residual_rate = residual / auto_n if auto_n else np.nan

                    rows.append({
                        "backbone": backbone,
                        "ranking_method": method,
                        "clinical_event": event,
                        "review_budget": budget,
                        "reviewed_n": k,
                        "auto_released_n": auto_n,
                        "dangerous_error_total": total,
                        "dangerous_error_captured": captured,
                        "dangerous_error_recall_at_k": recall,
                        "dangerous_error_precision_at_k": precision,
                        "dangerous_error_lift_vs_random": lift,
                        "residual_dangerous_error_count": residual,
                        "residual_dangerous_error_rate": residual_rate,
                        "dangerous_errors_per_100_reviewed": precision * 100 if not np.isnan(precision) else np.nan,
                        "number_needed_to_review": 1 / precision if precision and not np.isnan(precision) else np.nan,
                    })

    return pd.DataFrame(rows)


def mean_summary(tradeoff: pd.DataFrame) -> pd.DataFrame:
    return (
        tradeoff.groupby(["clinical_event", "review_budget", "ranking_method"], as_index=False)
        .agg(
            mean_recall=("dangerous_error_recall_at_k", "mean"),
            mean_lift=("dangerous_error_lift_vs_random", "mean"),
            total_captured=("dangerous_error_captured", "sum"),
            total_dangerous=("dangerous_error_total", "sum"),
            total_residual=("residual_dangerous_error_count", "sum"),
        )
        .sort_values(
            ["clinical_event", "review_budget", "mean_recall", "mean_lift"],
            ascending=[True, True, False, False],
        )
        .reset_index(drop=True)
    )


def topk_ids(g: pd.DataFrame, method: str, budget: float = 0.20) -> set[str]:
    ranked = rank_df(g, method)
    k = int(math.ceil(len(ranked) * budget))
    return set(ranked.head(k)["row_id"])


def overlap_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = [
        {
            "clinical_event": "large_undergrading",
            "best_method": "expected_gap_only",
            "baseline_method": "ophagent_combined",
        },
        {
            "clinical_event": "vision_threatening_dr_miss",
            "best_method": "gated_severe_prob_mass_only",
            "baseline_method": "ophagent_combined",
        },
    ]

    summary_rows = []
    case_rows = []

    for comp in comparisons:
        event = comp["clinical_event"]
        best_method = comp["best_method"]
        base_method = comp["baseline_method"]

        for backbone, g in df.groupby("backbone", sort=True):
            event_set = set(g[g[event].astype(bool)]["row_id"])
            best_captured = topk_ids(g, best_method) & event_set
            base_captured = topk_ids(g, base_method) & event_set

            both = best_captured & base_captured
            best_only = best_captured - base_captured
            base_only = base_captured - best_captured
            missed_by_both = event_set - (best_captured | base_captured)

            summary_rows.append({
                "clinical_event": event,
                "backbone": backbone,
                "best_method": best_method,
                "baseline_method": base_method,
                "event_total": len(event_set),
                "best_captured": len(best_captured),
                "baseline_captured": len(base_captured),
                "both_captured": len(both),
                "best_only_captured": len(best_only),
                "baseline_only_captured": len(base_only),
                "missed_by_both": len(missed_by_both),
            })

            category_map = {}
            for x in both:
                category_map[x] = "both_captured"
            for x in best_only:
                category_map[x] = "best_only_captured"
            for x in base_only:
                category_map[x] = "baseline_only_captured"
            for x in missed_by_both:
                category_map[x] = "missed_by_both"

            keep_cols = [
                "row_id", "backbone", "case_id", "image_key", "image_path",
                "true_grade", "pred_grade", "top2_grade", "confidence",
                "top2_confidence", "margin", "entropy_norm", "severe_prob_mass",
                "expected_grade", "expected_gap", "top2_more_severe",
                event,
            ]
            sub = g[g["row_id"].isin(category_map.keys())][keep_cols].copy()
            sub["clinical_event"] = event
            sub["best_method"] = best_method
            sub["baseline_method"] = base_method
            sub["overlap_category"] = sub["row_id"].map(category_map)
            case_rows.append(sub)

    return pd.DataFrame(summary_rows), pd.concat(case_rows, ignore_index=True)


def distribution_str(series: pd.Series) -> str:
    vc = series.dropna().astype(int).value_counts().sort_index()
    return ";".join([f"{k}:{v}" for k, v in vc.items()])


def residual_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    best_map = {
        "large_undergrading": "expected_gap_only",
        "vision_threatening_dr_miss": "gated_severe_prob_mass_only",
    }

    profile_rows = []
    case_rows = []

    numeric_features = [
        "confidence", "margin", "entropy_norm", "severe_prob_mass",
        "expected_grade", "expected_gap", "top2_confidence",
    ]

    for event, method in best_map.items():
        residual_parts = []

        for backbone, g in df.groupby("backbone", sort=True):
            reviewed = topk_ids(g, method, budget=0.20)
            residual = g[g[event].astype(bool) & ~g["row_id"].isin(reviewed)].copy()
            residual_parts.append(residual)

            row = {
                "clinical_event": event,
                "scope": backbone,
                "best_method": method,
                "residual_n": len(residual),
                "pred_grade_distribution": distribution_str(residual["pred_grade"]),
                "true_grade_distribution": distribution_str(residual["true_grade"]),
                "top2_more_severe_rate": residual["top2_more_severe"].mean() if len(residual) else np.nan,
                "high_confidence_rate_conf_ge_0_7": (residual["confidence"] >= 0.7).mean() if len(residual) else np.nan,
            }
            for feat in numeric_features:
                row[f"{feat}_mean"] = residual[feat].mean() if len(residual) else np.nan
                row[f"{feat}_median"] = residual[feat].median() if len(residual) else np.nan
            profile_rows.append(row)

        pooled = pd.concat(residual_parts, ignore_index=True)
        row = {
            "clinical_event": event,
            "scope": "pooled_all_backbones",
            "best_method": method,
            "residual_n": len(pooled),
            "pred_grade_distribution": distribution_str(pooled["pred_grade"]),
            "true_grade_distribution": distribution_str(pooled["true_grade"]),
            "top2_more_severe_rate": pooled["top2_more_severe"].mean() if len(pooled) else np.nan,
            "high_confidence_rate_conf_ge_0_7": (pooled["confidence"] >= 0.7).mean() if len(pooled) else np.nan,
        }
        for feat in numeric_features:
            row[f"{feat}_mean"] = pooled[feat].mean() if len(pooled) else np.nan
            row[f"{feat}_median"] = pooled[feat].median() if len(pooled) else np.nan
        profile_rows.append(row)

        keep_cols = [
            "row_id", "backbone", "case_id", "image_key", "image_path",
            "true_grade", "pred_grade", "top2_grade", "confidence",
            "top2_confidence", "margin", "entropy_norm", "severe_prob_mass",
            "expected_grade", "expected_gap", "top2_more_severe", event,
        ]
        pooled_cases = pooled[keep_cols].copy()
        pooled_cases["clinical_event"] = event
        pooled_cases["best_method"] = method
        case_rows.append(pooled_cases)

    return pd.DataFrame(profile_rows), pd.concat(case_rows, ignore_index=True)


def fmt_pct(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.1%}"


def fmt_float(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.2f}"


def write_findings(
    mean_df: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    residual_profile: pd.DataFrame,
    out_path: Path,
) -> None:
    def top_line(event: str, budget: float) -> pd.Series:
        sub = mean_df[(mean_df["clinical_event"] == event) & (mean_df["review_budget"] == budget)]
        return sub.sort_values(["mean_recall", "mean_lift"], ascending=[False, False]).iloc[0]

    ge20 = top_line("general_error", 0.20)
    lu20 = top_line("large_undergrading", 0.20)
    vt20 = top_line("vision_threatening_dr_miss", 0.20)

    lu_overlap = overlap_summary[
        (overlap_summary["clinical_event"] == "large_undergrading")
    ][["event_total", "best_captured", "baseline_captured", "both_captured", "best_only_captured", "baseline_only_captured", "missed_by_both"]].sum()

    vt_overlap = overlap_summary[
        (overlap_summary["clinical_event"] == "vision_threatening_dr_miss")
    ][["event_total", "best_captured", "baseline_captured", "both_captured", "best_only_captured", "baseline_only_captured", "missed_by_both"]].sum()

    lu_res = residual_profile[
        (residual_profile["clinical_event"] == "large_undergrading")
        & (residual_profile["scope"] == "pooled_all_backbones")
    ].iloc[0]

    vt_res = residual_profile[
        (residual_profile["clinical_event"] == "vision_threatening_dr_miss")
        & (residual_profile["scope"] == "pooled_all_backbones")
    ].iloc[0]

    lines = []
    lines.append("# v0.6.7c Ranking Signal Mechanism Analysis")
    lines.append("")
    lines.append("## 中文结论")
    lines.append("")
    lines.append("v0.6.7c 的目的不是再提出新规则，而是解释 v0.6.7b 中 severity-aware signals（严重程度感知信号）为什么能超过原始 `ophagent_combined`，以及它们仍然会漏掉什么。")
    lines.append("")
    lines.append("第一，统一比较所有排序方法后可以看到，不同错误类型对应的有效信号并不相同。")
    lines.append(f"在 Top20% 复核预算下，`general_error` 的最优方法是 `{ge20['ranking_method']}`，mean recall 为 {fmt_pct(ge20['mean_recall'])}；")
    lines.append(f"`large_undergrading` 的最优方法是 `{lu20['ranking_method']}`，mean recall 为 {fmt_pct(lu20['mean_recall'])}，捕获 {int(lu20['total_captured'])} / {int(lu20['total_dangerous'])}；")
    lines.append(f"`vision_threatening_dr_miss` 的最优方法是 `{vt20['ranking_method']}`，mean recall 为 {fmt_pct(vt20['mean_recall'])}，捕获 {int(vt20['total_captured'])} / {int(vt20['total_dangerous'])}。")
    lines.append("这说明一个通用风险分数不一定适合所有错误类型：通用错分更偏不确定性问题，而方向敏感的危险低估更依赖严重程度相关概率信号。")
    lines.append("")
    lines.append("第二，Top20% overlap analysis 解释了新信号相对 combined 的增益来源。")
    lines.append(f"对 `large_undergrading`，`expected_gap_only` 捕获 {int(lu_overlap['best_captured'])} 个危险样本，`ophagent_combined` 捕获 {int(lu_overlap['baseline_captured'])} 个；两者共同捕获 {int(lu_overlap['both_captured'])} 个，`expected_gap_only` 独有捕获 {int(lu_overlap['best_only_captured'])} 个，combined 独有捕获 {int(lu_overlap['baseline_only_captured'])} 个，两者都漏掉 {int(lu_overlap['missed_by_both'])} 个。")
    lines.append(f"对 `vision_threatening_dr_miss`，`gated_severe_prob_mass_only` 捕获 {int(vt_overlap['best_captured'])} 个危险样本，`ophagent_combined` 捕获 {int(vt_overlap['baseline_captured'])} 个；两者共同捕获 {int(vt_overlap['both_captured'])} 个，`gated_severe_prob_mass_only` 独有捕获 {int(vt_overlap['best_only_captured'])} 个，combined 独有捕获 {int(vt_overlap['baseline_only_captured'])} 个，两者都漏掉 {int(vt_overlap['missed_by_both'])} 个。")
    lines.append("")
    lines.append("第三，最佳 severity-aware 方法仍然存在自动放行区残余风险。")
    lines.append(f"Top20% 下，`large_undergrading` 使用 `expected_gap_only` 后仍残余 {int(lu_res['residual_n'])} 个危险低估样本；这些残余样本的 median expected_gap 为 {fmt_float(lu_res['expected_gap_median'])}，median severe_prob_mass 为 {fmt_float(lu_res['severe_prob_mass_median'])}。")
    lines.append(f"`vision_threatening_dr_miss` 使用 `gated_severe_prob_mass_only` 后仍残余 {int(vt_res['residual_n'])} 个重症漏检样本；这些残余样本的 median severe_prob_mass 为 {fmt_float(vt_res['severe_prob_mass_median'])}，top2_more_severe_rate 为 {fmt_pct(vt_res['top2_more_severe_rate'])}。")
    lines.append("因此，v0.6.7c 支持的结论不是“某个排序信号可以证明自动放行安全”，而是“severity-aware signals 可以显著改善复核优先级，同时 residual risk 仍需要被显式审计”。")
    lines.append("")
    lines.append("## 输出文件说明")
    lines.append("")
    lines.append("- `unified_ranking_method_tradeoff.csv`: 全部排序方法在多复核预算下的完整 trade-off。")
    lines.append("- `unified_ranking_method_mean_summary.csv`: 按事件、预算、方法聚合后的平均结果。")
    lines.append("- `top20_overlap_summary.csv`: Top20% 下最佳 severity-aware 方法与 combined 的捕获重叠统计。")
    lines.append("- `top20_overlap_cases.csv`: overlap 中每个病例的详细特征。")
    lines.append("- `top20_residual_profile.csv`: 最佳 severity-aware 方法漏掉样本的统计特征。")
    lines.append("- `top20_residual_cases.csv`: Top20% 自动放行区残余危险样本明细。")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)
    df = add_features(df)

    tradeoff = compute_tradeoff(df)
    mean_df = mean_summary(tradeoff)
    overlap_summary, overlap_cases = overlap_analysis(df)
    residual_profile, residual_cases = residual_analysis(df)

    tradeoff.to_csv(OUT_DIR / "unified_ranking_method_tradeoff.csv", index=False)
    mean_df.to_csv(OUT_DIR / "unified_ranking_method_mean_summary.csv", index=False)
    overlap_summary.to_csv(OUT_DIR / "top20_overlap_summary.csv", index=False)
    overlap_cases.to_csv(OUT_DIR / "top20_overlap_cases.csv", index=False)
    residual_profile.to_csv(OUT_DIR / "top20_residual_profile.csv", index=False)
    residual_cases.to_csv(OUT_DIR / "top20_residual_cases.csv", index=False)

    write_findings(mean_df, overlap_summary, residual_profile, OUT_DIR / "v067c_key_findings.md")

    print(f"Input: {INPUT}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Rows unified tradeoff: {len(tradeoff)}")
    print("Done.")


if __name__ == "__main__":
    main()
