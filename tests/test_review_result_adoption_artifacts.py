from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "review_result_adoption_feasibility_v0_1"
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
    assert len(core.loc[core["record_type"].eq("policy_performance")]) == 120
    assert len(core.loc[core["record_type"].eq("outcome_discrimination")]) == 18


def test_same_human_review_budget_is_exact() -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    performance = core.loc[core["record_type"].eq("policy_performance")]
    counts = performance.groupby(
        [
            "route_id",
            "analysis_split",
            "requested_human_review_fraction",
        ]
    )["human_review_n"].nunique()
    assert counts.eq(1).all()
    assert not performance[
        "human_review_resolution_assumed_correct"
    ].astype(bool).any()
    assert not performance[
        "current_case_ground_truth_used_for_action"
    ].astype(bool).any()
    assert not performance[
        "retrospective_outcome_used_for_fit_selection_or_threshold"
    ].astype(bool).any()


def test_no_route_meets_joint_safety_rule() -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    method = core.loc[
        core["record_type"].eq("policy_performance")
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq("learned_safe_adoption")
        & core["requested_human_review_fraction"].isin([0.1, 0.2, 0.3])
    ]
    dominant = (
        method["delta_corrected_retained"].ge(0)
        & method["delta_introduced_auto"].le(0)
        & method["delta_dangerous_introduced_auto"].le(0)
        & method["delta_net_retained"].gt(0)
    )
    assert not dominant.any()
    at_twenty = method.loc[
        method["requested_human_review_fraction"].eq(0.2)
    ]
    assert at_twenty["net_difference_ci_lower"].lt(0).all()


def test_discrimination_and_report_preserve_interpretation() -> None:
    core = pd.read_csv(OUTPUT_DIR / "core_results.csv")
    discrimination = core.loc[
        core["record_type"].eq("outcome_discrimination")
        & core["analysis_split"].eq("retrospective_evaluation")
    ]
    assert set(discrimination["outcome"]) == {
        "corrected",
        "introduced",
        "both_wrong",
    }
    assert discrimination["outcome_auroc"].between(0.5, 1.0).all()
    report = (OUTPUT_DIR / "research_report.md").read_text(encoding="utf-8")
    assert "NO_SIGNAL" in report
    assert "不表示所有输入特征都没有统计判别信息" in report
    assert "HUMAN_REVIEW 只表示延期裁决" in report
    assert "未重新训练或运行任何眼底模型" in report
    assert "不能宣称患者级泛化" not in report
    assert "图像级回顾证据" in report
