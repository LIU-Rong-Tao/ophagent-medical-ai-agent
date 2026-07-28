from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from app.route_qualification import (
    RouteQualificationRequest,
    evaluate_route_qualification,
    find_route_qualification_record,
)


CONTRACT = {
    "protocol_id": "route_qualification_gate_v1",
    "never_auto_grant_clinical_route_eligible": True,
}


def _request(**overrides: object) -> RouteQualificationRequest:
    request = RouteQualificationRequest(
        task_id="aptos_dr_5class",
        pairing_id="scout__to__expert",
        scout_artifact_ids=("scout",),
        expert_artifact_id="expert",
        request_scope="cached_prediction_replay",
        protocol_frozen=True,
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
        all_models_online_case_ready=True,
    )
    return replace(request, **overrides)


def test_beneficial_route_allows_research_simulation_but_not_clinical() -> None:
    decision = evaluate_route_qualification(_request(), contract=CONTRACT)

    assert decision.evidence_label == "beneficial"
    assert decision.execution_level == "research_case_simulation"
    assert decision.allow_case_simulation is True
    assert decision.clinical_route_eligible is False
    assert "RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED" in decision.error_codes


def test_same_scout_and_expert_is_hard_blocked() -> None:
    decision = evaluate_route_qualification(
        _request(expert_artifact_id="scout"),
        contract=CONTRACT,
    )

    assert decision.execution_level == "blocked"
    assert decision.allow_cached_replay is False
    assert "RQ_SAME_SCOUT_EXPERT" in decision.error_codes


def test_proxy_tradeoff_remains_research_only() -> None:
    decision = evaluate_route_qualification(
        _request(
            request_scope="new_case",
            validation_introduced=2,
            validation_net=3,
        ),
        contract=CONTRACT,
    )

    assert decision.evidence_label == "risk_tradeoff"
    assert decision.execution_level == "research_replay_only"
    assert decision.allow_new_case_route is False


def test_frozen_result_reversal_is_unstable_and_intercepted() -> None:
    decision = evaluate_route_qualification(
        _request(frozen_main_metric_delta=-0.001),
        contract=CONTRACT,
    )

    assert decision.evidence_label == "unstable"
    assert decision.execution_level == "research_replay_only"
    assert "RQ_FROZEN_REVERSAL" in decision.error_codes


def test_new_case_requires_cost_and_online_entries() -> None:
    decision = evaluate_route_qualification(
        _request(
            request_scope="new_case",
            cost_protocol_complete=False,
            all_models_online_case_ready=False,
        ),
        contract=CONTRACT,
    )

    assert decision.execution_level == "research_replay_only"
    assert decision.allow_new_case_route is False
    assert "RQ_COST_EVIDENCE_INCOMPLETE" in decision.error_codes
    assert "RQ_ONLINE_ENTRY_UNAVAILABLE" in decision.error_codes


def test_fully_qualified_new_case_is_only_a_deployment_candidate() -> None:
    decision = evaluate_route_qualification(
        _request(request_scope="new_case"),
        contract=CONTRACT,
    )

    assert decision.execution_level == "deployment_candidate"
    assert decision.allow_new_case_route is True
    assert decision.clinical_route_eligible is False


def test_route_record_falls_back_to_unique_registered_model_tuple(
    tmp_path: Path,
) -> None:
    contract_path = (
        tmp_path
        / "experiments/opening_risk_routing_closure/configs/protocols/"
        "route_qualification_contract_v1.json"
    )
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text(json.dumps(CONTRACT), encoding="utf-8")
    output_dir = (
        tmp_path
        / "experiments/opening_risk_routing_closure/outputs/"
        "route_qualification_gate_v1"
    )
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "task_id": "aptos_dr_5class",
                "pairing_id": "frozen_primary",
                "scout_artifact_ids": "flair|ret_clip",
                "expert_artifact_id": "retfound_cfp",
                "routing_policy": "disagreement_then_uncertainty",
                "requested_budget": 0.2,
                "allow_cached_replay": True,
                    "cost_protocol_complete": True,
                    "all_models_online_case_ready": False,
                    "protocol_sha256": "protocol",
                    "protocol_frozen": True,
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "validation_main_metric": 0.9,
                "validation_delta_vs_scout": 0.01,
                "validation_introduced": 1,
                "validation_net": 2,
                "stability_ci_lower": 0.001,
                "frozen_delta_vs_scout": 0.005,
                "frozen_introduced": 1,
                "frozen_net": 1,
                "primary_metric_available": True,
                "input_asset_fingerprint": "assets",
            }
        ]
    ).to_csv(output_dir / "route_qualification_matrix.csv", index=False)

    record = find_route_qualification_record(
        tmp_path,
        task_id="aptos_dr_5class",
        pairing_id="runtime_generated_pairing_id",
        scout_artifact_ids=("flair", "ret_clip"),
        expert_artifact_id="retfound_cfp",
        routing_policy="disagreement_then_uncertainty",
        requested_budget=0.2,
    )

    assert record is not None
    decision, evidence = record
    assert decision.execution_level == "research_case_simulation"
    assert evidence["pairing_id"] == "frozen_primary"
