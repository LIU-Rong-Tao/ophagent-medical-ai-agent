from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "aptos_high_capability_simple_gate_v0_1"
)
SOURCE_COMMIT = "509392828975a69929aa3821c37aff952800accc"
PROTOCOL_SHA = "57ba5c6490bc6f44b2cc78b43b8e0e1e7a3513eb63402874eca5d06071aae8a2"
ROUTES = {
    "aptos_dr_5class::flair__to__swin_tiny",
    "aptos_dr_5class::retfound_green__to__retfound_cfp",
    "aptos_dr_5class::retfound_green__to__swin_tiny",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_output_set_is_exactly_the_three_requested_artifacts() -> None:
    assert {path.name for path in OUTPUT.iterdir() if path.is_file()} == {
        "qualified_routes.csv",
        "core_comparison_results.csv",
        "research_report.md",
    }


def test_three_high_capability_routes_are_development_selected() -> None:
    routes = pd.read_csv(OUTPUT / "qualified_routes.csv", low_memory=False)

    assert len(routes) == 3
    assert set(routes["route_id"]) == ROUTES
    assert routes["qualified"].astype(bool).all()
    assert set(routes["screening_data_scope"]) == {"development_only"}
    assert not routes["evaluation_used_for_screening"].astype(bool).any()
    assert routes["scout_accuracy"].ge(0.80).all()
    assert routes["expert_accuracy"].ge(0.85).all()
    assert routes["expert_accuracy_gain"].ge(0.05).all()
    assert routes["corrected_to_introduced_ratio"].ge(1.7).all()
    assert routes["corrected_events"].ge(30).all()
    assert routes["introduced_events"].ge(20).all()
    assert routes["positive_gain_folds"].ge(4).all()
    assert routes["nonnegative_net_folds"].ge(4).all()
    assert set(routes["source_commit_sha"]) == {SOURCE_COMMIT}
    assert set(routes["protocol_sha256"]) == {PROTOCOL_SHA}


def test_core_results_freeze_no_improvement_and_research_boundaries() -> None:
    core = pd.read_csv(OUTPUT / "core_comparison_results.csv", low_memory=False)
    forbidden = {
        "patient_id",
        "patient_group_id",
        "resampling_group_id",
        "case_id",
        "image_id",
        "image_sha256",
        "image_path",
        "private_path",
    }

    assert len(core) == 72
    assert forbidden.isdisjoint(core.columns)
    assert set(core["study_decision"]) == {"NO_IMPROVEMENT"}
    assert set(core["source_commit_sha"]) == {SOURCE_COMMIT}
    assert set(core["protocol_sha256"]) == {PROTOCOL_SHA}
    assert not core[
        "test_used_for_fit_threshold_budget_or_route_selection"
    ].astype(bool).any()
    assert core["route_selected_from_development_only"].astype(bool).all()
    assert core["safety_eligibility_gate_required"].astype(bool).all()


def test_simple_gate_sacrifices_corrected_on_all_operating_points() -> None:
    core = pd.read_csv(OUTPUT / "core_comparison_results.csv", low_memory=False)
    rows = core.loc[
        core["record_type"].eq("policy_performance")
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq("dual_logistic_harm_screened_help")
    ].sort_values(["route_id", "requested_budget"])

    assert len(rows) == 9
    assert rows["requested_budget"].tolist() == [0.1, 0.2, 0.3] * 3
    assert rows["delta_corrected_selected"].astype(int).tolist() == [
        -38,
        -65,
        -70,
        -30,
        -64,
        -70,
        -34,
        -66,
        -64,
    ]
    assert rows["delta_introduced_selected"].astype(int).tolist() == [
        -13,
        -26,
        -24,
        -4,
        -10,
        -17,
        -8,
        -19,
        -16,
    ]
    assert rows["delta_net_selected"].astype(int).tolist() == [
        -25,
        -39,
        -46,
        -26,
        -54,
        -53,
        -26,
        -47,
        -48,
    ]
    assert rows["delta_corrected_selected"].lt(0).all()
    assert rows["delta_introduced_selected"].lt(0).all()
    assert rows["delta_net_selected"].lt(0).all()


def test_thirty_percent_net_intervals_are_strictly_negative() -> None:
    core = pd.read_csv(OUTPUT / "core_comparison_results.csv", low_memory=False)
    rows = core.loc[
        core["record_type"].eq("policy_performance")
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq("dual_logistic_harm_screened_help")
        & core["requested_budget"].eq(0.3)
    ]

    assert len(rows) == 3
    assert rows["net_difference_ci_upper"].lt(0).all()


def test_v1_1_is_not_transplanted_to_unregistered_routes() -> None:
    core = pd.read_csv(OUTPUT / "core_comparison_results.csv", low_memory=False)
    rows = core.loc[
        core["policy"].eq("consultation_policy_baseline_v1_1")
    ]

    assert len(rows) == 6
    available = rows.loc[rows["comparison_status"].eq("available")]
    unavailable = rows.loc[
        rows["comparison_status"].eq(
            "not_applicable_no_frozen_route_identity"
        )
    ]
    assert len(available) == 2
    assert set(available["route_id"]) == {
        "aptos_dr_5class::flair__to__swin_tiny"
    }
    assert available["requested_budget"].eq(0.3).all()
    assert len(unavailable) == 4
    assert unavailable["selected_n"].isna().all()


def test_report_and_hashes_are_frozen() -> None:
    report = (OUTPUT / "research_report.md").read_text(encoding="utf-8")

    assert "**NO_IMPROVEMENT**" in report
    assert "不是 APTOS→DeepDRiD 跨数据集迁移实验" in report
    assert "KEEP_SCOUT / ADOPT_SECOND_OPINION / HUMAN_REVIEW" in report
    assert _sha256(OUTPUT / "qualified_routes.csv") == (
        "c27b97d4a7728b175cdf820611168095019472b1404ac42c55c7292ba3b8bc0c"
    )
    assert _sha256(OUTPUT / "core_comparison_results.csv") == (
        "a4fc7dad020b28b044565781fc8f5c119b90e0e510fff27b788bb082d7d5a134"
    )
    assert _sha256(OUTPUT / "research_report.md") == (
        "a7454e806a5a47978cbe938361a161e8da46f0d86daee28a275fe316d1073e6b"
    )
