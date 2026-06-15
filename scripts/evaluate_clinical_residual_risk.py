from pathlib import Path
import pandas as pd


OUT_DIR = Path("experiments/summary/v0_6_7")
V066_DIR = Path("experiments/summary/v0_6_6/full_test_backbones")

BACKBONE_PREDICTIONS = {
    "convnext_tiny": "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "swin_tiny": "experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_imagenet": "experiments/aptos_vit_base_patch16_imagenet/lr1e-4_bs32_seed42/evaluation/test/test_predictions.csv",
    "vit_b_official_like": "experiments/aptos_vit_base_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "vit_l_official_like": "experiments/aptos_vit_large_patch16_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
    "retfound_official_like": "experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/evaluation/test/test_predictions.csv",
}


EVENT_DEFINITIONS = [
    {
        "event": "general_error",
        "definition": "pred_grade != true_grade",
        "clinical_meaning": "整体分类错误",
    },
    {
        "event": "any_undergrading",
        "definition": "pred_grade < true_grade",
        "clinical_meaning": "任意程度低估",
    },
    {
        "event": "large_undergrading",
        "definition": "true_grade - pred_grade >= 2",
        "clinical_meaning": "跨两级及以上低估",
    },
    {
        "event": "referable_dr_miss",
        "definition": "true_grade >= 2 and pred_grade < 2",
        "clinical_meaning": "可转诊 DR 被预测为非转诊级别",
    },
    {
        "event": "vision_threatening_dr_miss",
        "definition": "true_grade >= 3 and pred_grade < 3",
        "clinical_meaning": "重症或增殖期 DR 被低估到非重症",
    },
    {
        "event": "high_confidence_vision_threatening_miss",
        "definition": "vision_threatening_dr_miss and confidence >= 0.7",
        "clinical_meaning": "高置信重症低估",
    },
]


def image_key(value: str) -> str:
    return Path(str(value)).name


def load_merged_cases(backbone: str, pred_path: str) -> pd.DataFrame:
    risk_path = V066_DIR / backbone / "pre_review_risk_table.csv"

    risk = pd.read_csv(risk_path)
    pred = pd.read_csv(pred_path)

    risk["image_key"] = risk["image_path"].map(image_key)
    pred["image_key"] = pred["image_path"].map(image_key)

    pred_keep = pred[
        [
            "image_key",
            "true_idx",
            "true_label",
            "pred_idx",
            "pred_label",
            "confidence",
        ]
    ].rename(
        columns={
            "pred_idx": "pred_grade_from_source",
            "pred_label": "pred_label_from_source",
            "confidence": "confidence_from_source",
        }
    )

    df = risk.merge(pred_keep, on="image_key", how="left", validate="one_to_one")

    if df["true_idx"].isna().any():
        missing = int(df["true_idx"].isna().sum())
        raise ValueError(f"{backbone}: {missing} rows failed to merge true labels.")

    df.insert(0, "backbone", backbone)

    df["true_grade"] = df["true_idx"].astype(int)
    df["pred_grade"] = df["pred_grade"].astype(int)
    df["pred_grade_from_source"] = df["pred_grade_from_source"].astype(int)

    df["general_error"] = df["pred_grade"] != df["true_grade"]
    df["any_undergrading"] = df["pred_grade"] < df["true_grade"]
    df["large_undergrading"] = (df["true_grade"] - df["pred_grade"]) >= 2
    df["referable_dr_miss"] = (df["true_grade"] >= 2) & (df["pred_grade"] < 2)
    df["vision_threatening_dr_miss"] = (df["true_grade"] >= 3) & (df["pred_grade"] < 3)
    df["high_confidence_vision_threatening_miss"] = (
        df["vision_threatening_dr_miss"] & (df["confidence"] >= 0.7)
    )

    return df



REVIEW_BUDGETS = [0.05, 0.10, 0.20, 0.30]

REVIEW_EVENTS = [
    "general_error",
    "large_undergrading",
    "referable_dr_miss",
    "vision_threatening_dr_miss",
    "high_confidence_vision_threatening_miss",
]

RESIDUAL_CASE_EVENTS = [
    "large_undergrading",
    "referable_dr_miss",
    "vision_threatening_dr_miss",
    "high_confidence_vision_threatening_miss",
]

RANKING_METHODS = [
    "confidence_only",
    "margin_only",
    "entropy_only",
    "uncertainty_rank_fusion",
    "ophagent_combined",
]


def rank_cases_for_review_budget(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """排序输出：越靠前，越优先进入医生复核。"""
    d = df.copy()

    if method == "confidence_only":
        return d.sort_values(["confidence", "case_id"], ascending=[True, True]).reset_index(drop=True)

    if method == "margin_only":
        return d.sort_values(["margin", "case_id"], ascending=[True, True]).reset_index(drop=True)

    if method == "entropy_only":
        entropy_col = "entropy_norm" if "entropy_norm" in d.columns else "entropy"
        return d.sort_values([entropy_col, "case_id"], ascending=[False, True]).reset_index(drop=True)

    if method == "uncertainty_rank_fusion":
        entropy_col = "entropy_norm" if "entropy_norm" in d.columns else "entropy"
        d["_rank_confidence"] = d["confidence"].rank(method="average", ascending=True)
        d["_rank_margin"] = d["margin"].rank(method="average", ascending=True)
        d["_rank_entropy"] = d[entropy_col].rank(method="average", ascending=False)
        d["_fusion_rank"] = (
            d["_rank_confidence"] + d["_rank_margin"] + d["_rank_entropy"]
        ) / 3.0
        return d.sort_values(["_fusion_rank", "case_id"], ascending=[True, True]).reset_index(drop=True)

    if method == "ophagent_combined":
        return d.sort_values(["review_priority_rank", "case_id"], ascending=[True, True]).reset_index(drop=True)

    raise ValueError(f"Unknown ranking method: {method}")


def evaluate_review_burden(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for backbone, df_b in cases.groupby("backbone"):
        total_n = len(df_b)

        for method in RANKING_METHODS:
            ranked = rank_cases_for_review_budget(df_b, method)

            for event in REVIEW_EVENTS:
                event_total = int(ranked[event].sum())

                for budget in REVIEW_BUDGETS:
                    reviewed_n = max(1, int(round(total_n * budget)))
                    reviewed = ranked.iloc[:reviewed_n]
                    released = ranked.iloc[reviewed_n:]

                    captured = int(reviewed[event].sum())
                    residual = int(released[event].sum())
                    auto_released_n = len(released)
                    random_expected = reviewed_n * (event_total / total_n) if total_n else 0.0

                    rows.append({
                        "backbone": backbone,
                        "ranking_method": method,
                        "clinical_event": event,
                        "review_budget": budget,
                        "reviewed_n": reviewed_n,
                        "auto_released_n": auto_released_n,
                        "dangerous_error_total": event_total,
                        "dangerous_error_captured": captured,
                        "dangerous_error_recall_at_k": captured / event_total if event_total else None,
                        "dangerous_error_precision_at_k": captured / reviewed_n if reviewed_n else None,
                        "dangerous_error_lift_vs_random": captured / random_expected if random_expected > 0 else None,
                        "residual_dangerous_error_count": residual,
                        "residual_dangerous_error_rate": residual / auto_released_n if auto_released_n else None,
                        "dangerous_errors_per_100_reviewed": captured / reviewed_n * 100 if reviewed_n else None,
                        "number_needed_to_review": reviewed_n / captured if captured > 0 else None,
                    })

    return pd.DataFrame(rows)




def build_best_method_summaries(review_burden: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总每个 clinical event / review budget / backbone 下的最优排序方法。"""
    best_rows = []

    grouped = review_burden.groupby(["clinical_event", "review_budget", "backbone"])

    for (event, budget, backbone), group in grouped:
        best = group.sort_values(
            ["dangerous_error_recall_at_k", "dangerous_error_precision_at_k"],
            ascending=[False, False],
        ).iloc[0]

        best_rows.append({
            "clinical_event": event,
            "review_budget": budget,
            "backbone": backbone,
            "best_method": best["ranking_method"],
            "best_recall": best["dangerous_error_recall_at_k"],
            "captured": int(best["dangerous_error_captured"]),
            "total": int(best["dangerous_error_total"]),
            "residual": int(best["residual_dangerous_error_count"]),
        })

    best_by_backbone = pd.DataFrame(best_rows)

    best_count_summary = (
        best_by_backbone
        .groupby(["clinical_event", "review_budget", "best_method"])
        .size()
        .reset_index(name="num_backbones")
        .sort_values(
            ["clinical_event", "review_budget", "num_backbones"],
            ascending=[True, True, False],
        )
    )

    return best_by_backbone, best_count_summary




def build_residual_dangerous_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """输出复核后仍被自动放行的 clinical-risk proxy 错误样本。"""
    rows = []

    keep_cols = [
        "backbone",
        "case_id",
        "image_path",
        "true_grade",
        "true_label",
        "pred_grade",
        "pred_label",
        "confidence",
        "margin",
        "entropy",
        "entropy_norm",
        "top2_grade",
        "top2_label",
        "severe_prob_mass",
        "pre_review_risk_score",
        "pre_review_risk_level",
        "review_priority_rank",
        "risk_reasons",
    ]

    for backbone, df_b in cases.groupby("backbone"):
        total_n = len(df_b)

        for method in RANKING_METHODS:
            ranked = rank_cases_for_review_budget(df_b, method).copy()
            ranked["review_rank_by_method"] = range(1, len(ranked) + 1)

            for budget in REVIEW_BUDGETS:
                reviewed_n = max(1, int(round(total_n * budget)))
                released = ranked.iloc[reviewed_n:].copy()

                for event in RESIDUAL_CASE_EVENTS:
                    leaked = released[released[event]].copy()
                    if leaked.empty:
                        continue

                    leaked["ranking_method"] = method
                    leaked["review_budget"] = budget
                    leaked["reviewed_n"] = reviewed_n
                    leaked["clinical_event"] = event

                    cols = [
                        "ranking_method",
                        "review_budget",
                        "reviewed_n",
                        "clinical_event",
                        "review_rank_by_method",
                    ] + keep_cols

                    rows.append(leaked[cols])

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(EVENT_DEFINITIONS).to_csv(
        OUT_DIR / "clinical_event_definitions.csv", index=False
    )

    all_cases = []
    count_rows = []
    check_rows = []

    for backbone, pred_path in BACKBONE_PREDICTIONS.items():
        df = load_merged_cases(backbone, pred_path)
        all_cases.append(df)

        mismatch = int((df["pred_grade"] != df["pred_grade_from_source"]).sum())
        check_rows.append(
            {
                "backbone": backbone,
                "n": len(df),
                "pred_grade_mismatch_with_source": mismatch,
            }
        )

        for event_def in EVENT_DEFINITIONS:
            event = event_def["event"]
            count = int(df[event].sum())
            count_rows.append(
                {
                    "backbone": backbone,
                    "event": event,
                    "n": len(df),
                    "event_count": count,
                    "event_rate": count / len(df),
                }
            )

    cases = pd.concat(all_cases, ignore_index=True)
    cases.to_csv(OUT_DIR / "clinical_event_cases.csv", index=False)

    review_burden = evaluate_review_burden(cases)
    review_burden.to_csv(OUT_DIR / "review_burden_tradeoff.csv", index=False)
    best_by_backbone, best_count_summary = build_best_method_summaries(review_burden)
    best_by_backbone.to_csv(OUT_DIR / "best_method_by_event_budget_backbone.csv", index=False)
    best_count_summary.to_csv(OUT_DIR / "best_method_count_summary.csv", index=False)
    residual_cases = build_residual_dangerous_cases(cases)
    residual_cases.to_csv(OUT_DIR / "residual_dangerous_cases.csv", index=False)

    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT_DIR / "clinical_event_counts.csv", index=False)

    checks = pd.DataFrame(check_rows)
    checks.to_csv(OUT_DIR / "merge_checks.csv", index=False)

    print("Saved:")
    print(" -", OUT_DIR / "clinical_event_definitions.csv")
    print(" -", OUT_DIR / "clinical_event_cases.csv")
    print(" -", OUT_DIR / "clinical_event_counts.csv")
    print(" -", OUT_DIR / "review_burden_tradeoff.csv")
    print(" -", OUT_DIR / "best_method_by_event_budget_backbone.csv")
    print(" -", OUT_DIR / "best_method_count_summary.csv")
    print(" -", OUT_DIR / "residual_dangerous_cases.csv")
    print(" -", OUT_DIR / "merge_checks.csv")

    print("\nMerge checks:")
    print(checks.to_string(index=False))

    print("\nClinical event counts:")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
