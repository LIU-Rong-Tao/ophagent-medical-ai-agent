from __future__ import annotations

from dataclasses import replace

from app.model_hub_agent import (
    ControlledAgentRequest,
    ControlledAgentRuntime,
    redacted_controller_context,
)
from app.route_qualification import (
    RouteQualificationRequest,
    evaluate_route_qualification,
)


CONTRACT = {"protocol_id": "route_qualification_gate_v1"}


def _qualification(
    *,
    request_scope: str = "cached_prediction_replay",
    online: bool = False,
    frozen_delta: float = 0.01,
):
    return evaluate_route_qualification(
        RouteQualificationRequest(
            task_id="aptos_dr_5class",
            pairing_id="scout__to__expert",
            scout_artifact_ids=("scout",),
            expert_artifact_id="expert",
            request_scope=request_scope,
            protocol_frozen=True,
            selection_split="validation",
            validation_main_metric_delta=0.02,
            validation_corrected=4,
            validation_introduced=0,
            validation_net=4,
            stability_ci_lower=0.001,
            frozen_main_metric_delta=frozen_delta,
            frozen_corrected=3,
            frozen_introduced=0,
            frozen_net=3,
            primary_metric_available=True,
            prediction_assets_valid=True,
            cost_protocol_complete=True,
            all_models_online_case_ready=online,
        ),
        contract=CONTRACT,
    )


def _request(**overrides: object) -> ControlledAgentRequest:
    request = ControlledAgentRequest(
        task_id="aptos_dr_5class",
        case_alias="DR-V-0001",
        case_scope="cached_prediction_replay",
        scout_artifact_ids=("scout",),
        expert_artifact_id="expert",
        protocol_requests_expert=False,
        expected_route_cost_ms_per_image=1.2,
        max_cost_ms_per_image=2.0,
        idempotency_key="case-1",
    )
    return replace(request, **overrides)


def _tools(*, failed: bool = False, disagreement: bool = False):
    return {
        "input": {"ok": not failed},
        "audit": {
            "ok": not failed,
            "data": {"model_disagreement": disagreement},
        },
        "route": {"ok": not failed},
    }


def test_low_risk_qualified_replay_keeps_scout() -> None:
    decision = ControlledAgentRuntime().decide(
        _request(),
        qualification=_qualification(),
        tool_payload=_tools(),
    )

    assert decision.action == "KEEP_SCOUT"
    assert decision.human_confirmation_required is True


def test_high_risk_protocol_requests_expert() -> None:
    decision = ControlledAgentRuntime().decide(
        _request(protocol_requests_expert=True),
        qualification=_qualification(),
        tool_payload=_tools(disagreement=True),
    )

    assert decision.action == "REQUEST_EXPERT"
    assert decision.risk_state == "protocol_requests_expert"


def test_historical_replay_cannot_be_used_for_new_case() -> None:
    decision = ControlledAgentRuntime().decide(
        _request(case_scope="new_case"),
        qualification=_qualification(request_scope="new_case", online=False),
        tool_payload=_tools(),
    )

    assert decision.action == "REFER_TO_HUMAN"
    assert decision.code == "AGENT_ROUTE_BLOCKED"


def test_cost_over_budget_refers_to_human() -> None:
    decision = ControlledAgentRuntime().decide(
        _request(max_cost_ms_per_image=1.0),
        qualification=_qualification(),
        tool_payload=_tools(),
    )

    assert decision.action == "REFER_TO_HUMAN"
    assert decision.code == "AGENT_COST_BUDGET_EXCEEDED"


def test_tool_failure_stops_downstream_action() -> None:
    decision = ControlledAgentRuntime().decide(
        _request(),
        qualification=_qualification(),
        tool_payload=_tools(failed=True),
    )

    assert decision.action == "REFER_TO_HUMAN"
    assert decision.code == "AGENT_UPSTREAM_FAILED"
    assert decision.trace[-1]["stage"] == "await_human_confirmation"


def test_duplicate_request_is_idempotent_and_conflicts_are_blocked() -> None:
    runtime = ControlledAgentRuntime()
    first = runtime.decide(
        _request(),
        qualification=_qualification(),
        tool_payload=_tools(),
    )
    replay = runtime.decide(
        _request(),
        qualification=_qualification(),
        tool_payload=_tools(),
    )
    conflict = runtime.decide(
        _request(protocol_requests_expert=True),
        qualification=_qualification(),
        tool_payload=_tools(),
    )

    assert replay.decision_id == first.decision_id
    assert replay.replayed is True
    assert conflict.action == "REFER_TO_HUMAN"
    assert conflict.code == "AGENT_IDEMPOTENCY_CONFLICT"


def test_local_controller_context_excludes_sensitive_case_content() -> None:
    context = redacted_controller_context(
        _request(),
        _qualification(),
        model_disagreement=True,
    )
    serialized = str(context).lower()

    assert "case_alias" not in context
    assert "patient" not in serialized
    assert "image" not in serialized
    assert "path" not in serialized
    assert context["qualification_is_external_to_controller"] is True
