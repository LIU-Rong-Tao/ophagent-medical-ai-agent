from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "selective_consultation_method_v0_1"
)
PROTOCOL = (
    ROOT
    / "experiments/opening_risk_routing_closure/configs/protocols/"
    "selective_consultation_method_v0_1.json"
)
PRIMARY_ROUTE = "deepdrid_dr_5class_native::keepfit_cfp__to__flair"
METHOD_POLICY = "dual_logistic_harm_screened_help"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_selective_consultation_output_set_is_minimal_and_complete() -> None:
    assert {path.name for path in OUTPUT.iterdir() if path.is_file()} == {
        "artifact_manifest.json",
        "core_results.csv",
        "failure_case_audit.csv",
        "research_report.md",
        "risk_budget_curve.csv",
    }


def test_manifest_freezes_negative_decision_and_research_boundaries() -> None:
    manifest = json.loads(
        (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["decision"] == "NO_IMPROVEMENT"
    assert manifest["source_commit_sha"] == (
        "c2d21f5886153a3a27fbddbecccd971fa51aaca3"
    )
    assert manifest["protocol_sha256"] == (
        "984c5e2b4d8a7b57e21b24120eab8d0b26d4de9dc356bffc5cc09c1b6b84a2ac"
    )
    assert manifest["frozen_benchmark_modified"] is False
    assert manifest["frozen_prediction_assets_modified"] is False
    assert manifest["ophthalmic_model_training_performed"] is False
    assert manifest["ophthalmic_model_inference_performed"] is False
    assert manifest["statistical_control_model_fit_performed"] is True
    assert manifest["external_api_used"] is False
    assert (
        manifest["test_used_for_fit_threshold_budget_or_route_selection"]
        is False
    )
    assert manifest["independent_confirmation_required"] is True
    assert manifest["decision_evidence"]["primary_dominant_budgets"] == []
    assert (
        manifest["decision_evidence"]["development_prequalified_route_count"]
        == 0
    )


def test_manifest_hashes_every_declared_content_artifact() -> None:
    manifest = json.loads(
        (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    assert len(manifest["outputs"]) == 4
    for artifact in manifest["outputs"]:
        uri = str(artifact["uri"])
        assert uri.startswith("repo://")
        path = ROOT / uri.removeprefix("repo://")
        assert path.is_file()
        assert path.stat().st_size == int(artifact["size_bytes"])
        assert _sha256(path) == artifact["sha256"]


def test_artifact_row_counts_and_sensitive_field_boundaries() -> None:
    core = pd.read_csv(OUTPUT / "core_results.csv", low_memory=False)
    curve = pd.read_csv(OUTPUT / "risk_budget_curve.csv", low_memory=False)
    failures = pd.read_csv(OUTPUT / "failure_case_audit.csv", low_memory=False)
    forbidden = {
        "patient_id",
        "patient_group_id",
        "image_path",
        "private_path",
        "image_sha256",
    }

    assert len(core) == 18_006
    assert len(curve) == 69_120
    assert len(failures) == 214
    assert forbidden.isdisjoint(core.columns)
    assert forbidden.isdisjoint(curve.columns)
    assert forbidden.isdisjoint(failures.columns)
    assert not failures["patient_identity_included"].astype(bool).any()
    assert not failures["private_path_included"].astype(bool).any()
    assert set(failures["expert_output_use"]) == {"posthoc_audit_only"}


def test_primary_same_budget_results_match_frozen_evidence() -> None:
    core = pd.read_csv(OUTPUT / "core_results.csv", low_memory=False)
    rows = core.loc[
        core["route_id"].eq(PRIMARY_ROUTE)
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq(METHOD_POLICY)
    ].sort_values("requested_budget")

    assert rows["requested_budget"].tolist() == [0.05, 0.1, 0.2, 0.3]
    assert rows["corrected_selected"].astype(int).tolist() == [7, 9, 19, 22]
    assert rows["introduced_selected"].astype(int).tolist() == [2, 5, 7, 17]
    assert rows["net_selected"].astype(int).tolist() == [5, 4, 12, 5]
    assert rows["comparator_corrected_selected"].astype(int).tolist() == [
        7,
        12,
        20,
        30,
    ]
    assert rows["comparator_introduced_selected"].astype(int).tolist() == [
        6,
        7,
        16,
        22,
    ]
    assert rows["comparator_net_selected"].astype(int).tolist() == [1, 5, 4, 8]
    assert rows["net_difference_ci_lower"].astype(int).tolist() == [
        -4,
        -14,
        -2,
        -16,
    ]
    assert rows["net_difference_ci_upper"].astype(int).tolist() == [
        12,
        11,
        19,
        9,
    ]


def test_five_percent_budget_is_descriptive_not_decision_operating_scope() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["policy_contract"]["same_budget_grid"] == [
        0.05,
        0.1,
        0.2,
        0.3,
    ]
    assert protocol["decision_contract"]["operating_budgets"] == [
        0.1,
        0.2,
        0.3,
    ]


def test_no_learned_policy_uses_current_case_expert_output_or_test_selection() -> None:
    core = pd.read_csv(OUTPUT / "core_results.csv", low_memory=False)
    learned = core.loc[core["policy"].ne("oracle") | core["policy"].isna()]

    assert not learned[
        "current_case_expert_output_used_for_ranking"
    ].astype(bool).any()
    assert not core[
        "test_used_for_fit_threshold_budget_or_route_selection"
    ].astype(bool).any()
    assert core.loc[
        core["policy"].eq("oracle"),
        "current_case_expert_output_used_for_ranking",
    ].astype(bool).all()


def test_report_discloses_decision_and_primary_budget_table() -> None:
    report = (OUTPUT / "research_report.md").read_text(encoding="utf-8")

    assert "# OphAgent 预咨询选择性会诊方法研究 v0.1" in report
    assert "**NO_IMPROVEMENT**" in report
    assert "| 0.05 |" in report
    assert "| 0.30 |" in report
    assert "冻结 v1.1 在 30% 预算" in report
    assert "不能充当独立确认" in report
