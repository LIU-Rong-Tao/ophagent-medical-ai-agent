from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "route_specific_simple_gate_v0_1"
)
PROTOCOL = (
    ROOT
    / "experiments/opening_risk_routing_closure/configs/protocols/"
    "route_specific_simple_gate_v0_1.json"
)
SOURCE_COMMIT = "d7d36b66f232e032ca5cf498696194aed4c629b9"
PROTOCOL_SHA = "2966baf37c6c751dbbd292d71e73d5c3aeadf40ac13e3be3a75065ef2b48c933"
ROUTES = {
    "deepdrid_dr_5class_native::retfound_green__to__convnext_tiny",
    "deepdrid_dr_5class_native::retfound_green__to__retizero",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_formal_output_set_is_exactly_the_three_requested_artifacts() -> None:
    assert {path.name for path in OUTPUT.iterdir() if path.is_file()} == {
        "qualified_routes.csv",
        "core_comparison_results.csv",
        "research_report.md",
    }


def test_qualified_routes_are_selected_from_development_only() -> None:
    routes = pd.read_csv(OUTPUT / "qualified_routes.csv", low_memory=False)

    assert len(routes) == 2
    assert set(routes["route_id"]) == ROUTES
    assert routes["qualified"].astype(bool).all()
    assert set(routes["screening_data_scope"]) == {"development_only"}
    assert not routes["evaluation_used_for_screening"].astype(bool).any()
    assert routes["scout_accuracy"].ge(0.60).all()
    assert routes["expert_accuracy"].ge(0.60).all()
    assert routes["expert_accuracy_gain"].ge(0.05).all()
    assert routes["corrected_to_introduced_ratio"].ge(1.5).all()
    assert routes["corrected_events"].ge(15).all()
    assert routes["introduced_events"].ge(15).all()
    assert routes["positive_gain_folds"].ge(4).all()
    assert routes["nonnegative_net_folds"].ge(4).all()
    assert set(routes["source_commit_sha"]) == {SOURCE_COMMIT}
    assert set(routes["protocol_sha256"]) == {PROTOCOL_SHA}


def test_core_results_freeze_no_improvement_and_no_sensitive_fields() -> None:
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

    assert len(core) == 48
    assert forbidden.isdisjoint(core.columns)
    assert set(core["study_decision"]) == {"NO_IMPROVEMENT"}
    assert set(core["source_commit_sha"]) == {SOURCE_COMMIT}
    assert set(core["protocol_sha256"]) == {PROTOCOL_SHA}
    assert not core[
        "test_used_for_fit_threshold_budget_or_route_selection"
    ].astype(bool).any()
    assert core["route_selected_from_development_only"].astype(bool).all()
    assert core["safety_eligibility_gate_required"].astype(bool).all()


def test_retrospective_method_loses_corrected_on_both_qualified_routes() -> None:
    core = pd.read_csv(OUTPUT / "core_comparison_results.csv", low_memory=False)
    rows = core.loc[
        core["record_type"].eq("policy_performance")
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq("dual_logistic_harm_screened_help")
    ].sort_values(["route_id", "requested_budget"])

    assert len(rows) == 6
    assert rows["requested_budget"].tolist() == [0.1, 0.2, 0.3] * 2
    assert rows["delta_corrected_selected"].astype(int).tolist() == [
        -6,
        -7,
        -16,
        -7,
        -8,
        -13,
    ]
    assert rows["delta_introduced_selected"].astype(int).tolist() == [
        -7,
        -6,
        -3,
        -9,
        -3,
        -8,
    ]
    assert rows["delta_net_selected"].astype(int).tolist() == [
        1,
        -1,
        -13,
        2,
        -5,
        -5,
    ]
    assert not (
        rows["delta_corrected_selected"].ge(0)
        & rows["delta_introduced_selected"].le(0)
        & rows["delta_net_selected"].gt(0)
    ).any()


def test_v1_1_is_not_transplanted_without_frozen_route_identity() -> None:
    core = pd.read_csv(OUTPUT / "core_comparison_results.csv", low_memory=False)
    rows = core.loc[
        core["policy"].eq("consultation_policy_baseline_v1_1")
    ]

    assert len(rows) == 4
    assert set(rows["comparison_status"]) == {
        "not_applicable_no_frozen_route_identity"
    }
    assert rows["selected_n"].isna().all()


def test_protocol_excludes_five_percent_from_decision() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["comparison_contract"]["operating_budgets"] == [
        0.1,
        0.2,
        0.3,
    ]
    assert protocol["analysis_scope"]["evaluation_may_select_route"] is False
    assert protocol["analysis_scope"]["evaluation_may_change_threshold"] is False
    assert protocol["decision_contract"]["minimum_successful_routes"] == 2


def test_report_and_csv_hashes_are_frozen() -> None:
    report = (OUTPUT / "research_report.md").read_text(encoding="utf-8")

    assert "**NO_IMPROVEMENT**" in report
    assert "KEEP_SCOUT / ADOPT_SECOND_OPINION / HUMAN_REVIEW" in report
    assert "not applicable" in report
    assert _sha256(OUTPUT / "qualified_routes.csv") == (
        "1e89bbe9f817a1529931b73460efbb69a195df0531a27a17b58dedef89d923a7"
    )
    assert _sha256(OUTPUT / "core_comparison_results.csv") == (
        "ac7fc7c504c005700193d906d0c63cd1e4815f449c03fd407c9886dcde33ac8e"
    )
    assert _sha256(OUTPUT / "research_report.md") == (
        "35ea1be6d91d6cdb7d52cd1d4bad5e204a0c8efed23867d6ab3ec5b64cb4eebe"
    )
