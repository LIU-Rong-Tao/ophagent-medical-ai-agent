from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "experiments/opening_risk_routing_closure/configs/protocols/"
    "aptos_high_capability_simple_gate_v0_1.json"
)


def test_aptos_protocol_targets_high_capability_same_task_routes() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    scope = protocol["analysis_scope"]
    screening = protocol["route_screening"]

    assert scope["task_id"] == "aptos_dr_5class"
    assert scope["same_task_adaptation_not_cross_dataset_transfer"] is True
    assert scope["patient_level_claim_allowed"] is False
    assert scope["evaluation_may_select_route"] is False
    assert screening["minimum_scout_accuracy"] == 0.8
    assert screening["minimum_expert_accuracy"] == 0.85
    assert screening["minimum_expert_accuracy_gain"] == 0.05
    assert screening["minimum_corrected_to_introduced_ratio"] == 1.7


def test_aptos_protocol_keeps_test_and_expert_outputs_out_of_selection() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["method_contract"]["current_case_expert_output_allowed"] is False
    assert protocol["method_contract"]["test_fit_or_selection_allowed"] is False
    assert protocol["method_contract"]["dataset_id_as_predictor_allowed"] is False
    assert protocol["comparison_contract"]["operating_budgets"] == [
        0.1,
        0.2,
        0.3,
    ]
