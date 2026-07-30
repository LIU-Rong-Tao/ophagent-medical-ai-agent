from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "disagreement_review_prioritization_v0_1"
)
EXPECTED_ROUTES = {
    "aptos_dr_5class::flair__to__swin_tiny",
    "aptos_dr_5class::retfound_green__to__retfound_cfp",
    "aptos_dr_5class::retfound_green__to__swin_tiny",
}


def test_output_set_and_decision_are_frozen() -> None:
    assert {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()} == {
        "core_results.csv",
        "research_report.md",
    }
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    assert set(core["route_id"]) == EXPECTED_ROUTES
    assert set(core["study_decision"]) == {"NO_SIGNAL"}
    assert len(core.loc[core["record_type"].eq("priority_performance")]) == 90
    assert (
        len(core.loc[core["record_type"].eq("priority_discrimination")]) == 24
    )


def test_all_policies_use_the_same_disagreement_review_count() -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    performance = core.loc[core["record_type"].eq("priority_performance")]
    assert set(performance["policy"]) == {
        "random",
        "entropy",
        "margin",
        "disagreement_js",
        "learned_harmful_conflict",
    }
    selected_counts = performance.groupby(
        ["route_id", "analysis_split", "requested_review_fraction"]
    )["selected_n"].nunique()
    assert selected_counts.eq(1).all()
    assert performance["cohort"].eq(
        "scout_review_prediction_disagreement_only"
    ).all()
    assert not performance[
        "current_case_ground_truth_used_for_priority"
    ].astype(bool).any()
    assert not performance[
        "retrospective_outcome_used_for_fit_selection_or_threshold"
    ].astype(bool).any()


def test_no_route_meets_joint_capture_rule() -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    method = core.loc[
        core["record_type"].eq("priority_performance")
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq("learned_harmful_conflict")
    ]
    dominant = (
        method["delta_harmful_conflict_captured"].gt(0)
        & method["delta_introduced_captured"].ge(0)
        & method["delta_dangerous_introduced_captured"].ge(0)
        & method["delta_both_wrong_captured"].ge(0)
    )
    assert int(dominant.sum()) == 1
    at_twenty = method.loc[method["requested_review_fraction"].eq(0.2)]
    assert at_twenty["harmful_capture_difference_ci_lower"].lt(0).all()


def test_report_preserves_partial_signal_and_stop_boundary() -> None:
    report = (OUTPUT_DIR / "research_report.md").read_text(encoding="utf-8")
    assert "NO_SIGNAL" in report
    assert "20% 配对差异 95% CI 为 [-4, 19]" in report
    assert "dangerous introduced 与 both_wrong 捕获没有同步改善" in report
    assert "停止当前多模型自动协同方法研究" in report
    assert "不重新训练或运行眼底模型" in report
