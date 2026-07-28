from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from app.route_qualification import (
    RouteQualificationRequest,
    evaluate_route_qualification,
    route_qualification_request_from_row,
)
from app.route_qualification_benchmark import (
    build_leave_one_task_out_results,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "experiments/opening_risk_routing_closure/configs/protocols/"
    "route_qualification_contract_v1_1.json"
)
OUTPUT_DIR = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "route_qualification_benchmark_v1_1"
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _request(**overrides: object) -> RouteQualificationRequest:
    request = RouteQualificationRequest(
        task_id="synthetic_task",
        dataset_id="synthetic_public",
        pairing_id="scout__to__expert",
        scout_artifact_ids=("scout",),
        expert_artifact_id="expert",
        request_scope="cached_prediction_replay",
        protocol_frozen=True,
        unique_protocol_identity=True,
        validation_result_sha256="a" * 64,
        frozen_result_sha256="b" * 64,
        selection_split="validation",
        validation_main_metric_delta=0.02,
        validation_delta_vs_best_single=0.01,
        validation_corrected=6,
        validation_introduced=0,
        validation_net=6,
        stability_ci_lower=0.002,
        frozen_main_metric_delta=0.01,
        frozen_corrected=4,
        frozen_introduced=0,
        frozen_net=4,
        primary_metric_available=True,
        prediction_assets_valid=True,
        cost_protocol_complete=True,
        cost_protocol_comparable=True,
        cost_protocol_id=(
            "h100_fp32_forward_only_batch1_batch16_w10_r30_v1"
        ),
        expected_cost_ms_per_image=2.0,
        requested_budget=0.2,
        expert_budget=0.2,
        domain_shift_status="in_domain",
        adaptation_type="task_native",
        task_adapter_compatible=True,
        historical_replay_eligible=True,
    )
    return replace(request, **overrides)


def test_full_layered_gate_uses_all_constraints_without_score() -> None:
    decision = evaluate_route_qualification(
        _request(),
        contract=_contract(),
    )

    assert decision.execution_level == "research_case_simulation"
    assert decision.allow_case_simulation is True
    assert decision.clinical_route_eligible is False
    assert any(
        event.get("check") == "domain_adaptation"
        and event.get("passed") is True
        for event in decision.gate_trace
    )


def test_proxy_cost_stability_and_domain_layers_are_enforced() -> None:
    contract = _contract()
    proxy = evaluate_route_qualification(
        _request(validation_introduced=5, validation_net=-1),
        contract=contract,
    )
    cost = evaluate_route_qualification(
        _request(cost_protocol_comparable=False),
        contract=contract,
    )
    static_rank_only = evaluate_route_qualification(
        _request(
            stability_ci_lower=None,
            candidate_rank=1,
            candidate_count=4,
            candidate_rank_stability_verified=False,
        ),
        contract=contract,
    )
    domain = evaluate_route_qualification(
        _request(
            domain_shift_status=(
                "external_shift_without_in_domain_validation"
            )
        ),
        contract=contract,
    )

    assert "RQ_PROXY_NET_CONSTRAINT" in proxy.error_codes
    assert "RQ_COST_PROTOCOL_MISMATCH" in cost.error_codes
    assert "RQ_STABILITY_CONSTRAINT" in static_rank_only.error_codes
    assert "RQ_DOMAIN_SHIFT_RESTRICTED" in domain.error_codes
    assert not any(
        decision.allow_case_simulation
        for decision in (proxy, cost, static_rank_only, domain)
    )


def test_frozen_result_can_restrict_but_never_grant() -> None:
    decision = evaluate_route_qualification(
        _request(frozen_main_metric_delta=-0.001),
        contract=_contract(),
    )

    assert decision.execution_level == "research_replay_only"
    assert "RQ_FROZEN_REVERSAL" in decision.error_codes


def test_asset_identity_and_primary_metric_fail_closed() -> None:
    contract = _contract()
    missing_models = evaluate_route_qualification(
        _request(scout_artifact_ids=(), expert_artifact_id=""),
        contract=contract,
    )
    missing_sha = evaluate_route_qualification(
        _request(validation_result_sha256="", frozen_result_sha256="bad"),
        contract=contract,
    )
    fallback_only = evaluate_route_qualification(
        _request(primary_metric_available=False),
        contract=contract,
    )

    assert missing_models.execution_level == "blocked"
    assert "RQ_MODEL_IDENTITY_MISSING" in missing_models.error_codes
    assert missing_sha.execution_level == "blocked"
    assert "RQ_ASSET_SHA_MISSING" in missing_sha.error_codes
    assert fallback_only.execution_level == "research_replay_only"
    assert "RQ_PRIMARY_METRIC_UNAVAILABLE" in fallback_only.error_codes


def test_protocol_and_selection_provenance_are_explicit_and_fail_closed() -> None:
    row = pd.Series(
        {
            **asdict(_request()),
            "scout_artifact_ids": "scout",
            "protocol_sha256": "c" * 64,
            "validation_main_metric": 0.9,
        }
    ).drop(labels=["protocol_frozen", "selection_split"])
    inferred = route_qualification_request_from_row(row)
    missing = evaluate_route_qualification(
        inferred,
        contract=_contract(),
    )
    test_selected = evaluate_route_qualification(
        _request(test_used_for_selection=True),
        contract=_contract(),
    )

    assert inferred.protocol_frozen is False
    assert inferred.selection_split == ""
    assert missing.allow_case_simulation is False
    assert "RQ_PROTOCOL_NOT_FROZEN" in missing.error_codes
    assert "RQ_SELECTION_NOT_VALIDATION" in missing.error_codes
    assert test_selected.allow_case_simulation is False
    assert "RQ_TEST_USED_FOR_SELECTION" in test_selected.error_codes


def test_frozen_v1_1_artifacts_have_expected_scope_and_no_sha_collision() -> None:
    matrix = pd.read_csv(
        OUTPUT_DIR / "route_qualification_evidence_matrix.csv"
    )

    assert len(matrix) == 16
    assert matrix["task_id"].nunique() == 6
    assert not matrix["validation_result_sha256"].eq(
        matrix["frozen_result_sha256"]
    ).any()
    assert not matrix["clinical_route_eligible"].astype(bool).any()
    assert matrix["pairing_id"].is_unique
    assert matrix["protocol_frozen"].astype(bool).all()
    assert matrix["selection_split"].eq("validation").all()
    assert not matrix["test_used_for_selection"].astype(bool).any()

    native = matrix.set_index("pairing_id")
    assert abs(
        native.loc[
            "deepdrid_native_primary_single",
            "validation_main_metric",
        ]
        - 0.8521884093
    ) < 1e-9
    assert (
        native.loc[
            "deepdrid_native_primary_single",
            "validation_introduced",
        ]
        == 0
    )
    assert (
        native.loc[
            "trhd59_multi_primary",
            "risk_proxy_semantics",
        ]
        == "observed_positive_consistency_proxy_not_clinical_outcome"
    )
    assert (
        native.loc["rim_one_locked_multi", "cost_protocol_comparable"]
        in {False, 0}
    )


def test_loto_never_uses_held_out_task_for_thresholds() -> None:
    results = pd.read_csv(OUTPUT_DIR / "leave_one_task_out_results.csv")
    folds = results.loc[
        results["record_type"].eq("held_out_task_validation_only")
    ]
    aggregate = results.loc[
        results["record_type"].eq("all_out_of_task_validation_only")
    ].iloc[0]
    overlay = results.loc[
        results["record_type"].eq("post_freeze_safety_overlay")
    ].iloc[0]

    assert len(folds) == 6
    assert folds["held_out_not_used_for_thresholds"].astype(bool).all()
    assert not folds[
        "held_out_frozen_outcomes_used_for_decision"
    ].astype(bool).any()
    assert bool(
        overlay["held_out_frozen_outcomes_used_for_decision"]
    )
    assert overlay["test_reversal_interception_rate"] == 1
    assert 0 <= aggregate["false_grant_rate"] <= 1


def test_failure_case_audit_contains_route_level_loto_errors() -> None:
    audit = pd.read_csv(OUTPUT_DIR / "failure_case_audit.csv")
    primary = audit.loc[
        audit["prediction_scope"].eq(
            "held_out_task_validation_only"
        )
    ]
    overlay = audit.loc[
        audit["prediction_scope"].eq(
            "held_out_task_post_freeze_overlay"
        )
    ]

    assert len(primary) == len(overlay) == 16
    assert primary["held_out_task_id"].nunique() == 6
    assert not primary[
        "frozen_outcomes_used_for_decision"
    ].astype(bool).any()
    assert overlay[
        "frozen_outcomes_used_for_decision"
    ].astype(bool).all()
    assert set(primary["benchmark_outcome"]).issubset(
        {
            "correct_grant",
            "correct_rejection",
            "false_grant",
            "false_rejection",
        }
    )


def test_validation_only_loto_is_invariant_to_frozen_outcomes() -> None:
    matrix = pd.read_csv(
        OUTPUT_DIR / "route_qualification_evidence_matrix.csv"
    )
    baseline = build_leave_one_task_out_results(matrix, _contract())
    perturbed = matrix.copy()
    perturbed["frozen_delta_vs_scout"] = (
        -pd.to_numeric(
            perturbed["frozen_delta_vs_scout"],
            errors="coerce",
        ).fillna(1.0)
    )
    perturbed["frozen_net"] = (
        -pd.to_numeric(
            perturbed["frozen_net"],
            errors="coerce",
        ).fillna(1.0)
    )
    changed = build_leave_one_task_out_results(
        perturbed,
        _contract(),
    )
    columns = [
        "held_out_task_id",
        "beneficial_route_retention_rate",
        "false_grant_rate",
        "false_rejection_rate",
        "executable_coverage_rate",
    ]
    baseline_primary = baseline.loc[
        baseline["record_type"].isin(
            {
                "held_out_task_validation_only",
                "all_out_of_task_validation_only",
            }
        ),
        columns,
    ].reset_index(drop=True)
    changed_primary = changed.loc[
        changed["record_type"].isin(
            {
                "held_out_task_validation_only",
                "all_out_of_task_validation_only",
            }
        ),
        columns,
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        baseline_primary,
        changed_primary,
        check_dtype=False,
    )
    aptos_fold = baseline.loc[
        baseline["record_type"].eq(
            "held_out_task_validation_only"
        )
        & baseline["held_out_task_id"].eq("aptos_dr_5class")
    ].iloc[0]
    assert aptos_fold["selection_dependency_rows_excluded"] >= 1
