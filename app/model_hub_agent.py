"""Controlled three-action state loop for the offline Model Hub workstation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Protocol

from app.route_qualification import RouteQualificationDecision


AGENT_ACTIONS = (
    "KEEP_SCOUT",
    "REQUEST_EXPERT",
    "REFER_TO_HUMAN",
)

AGENT_MESSAGES = {
    "AGENT_OK": "受控动作已生成，等待人工确认",
    "AGENT_UPSTREAM_FAILED": "上游工具失败，已停止后续动作",
    "AGENT_ROUTE_BLOCKED": "路由资格仅允许回放或已被门禁阻止",
    "AGENT_NEW_CASE_BLOCKED": "当前资格不允许对新病例执行该路由",
    "AGENT_COST_BUDGET_EXCEEDED": "预计成本超过当前请求预算",
    "AGENT_IDEMPOTENCY_CONFLICT": "同一幂等键对应了不同请求",
}


class LocalActionProposer(Protocol):
    """Future local-model interface; proposals never decide qualification."""

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        """Return only an action proposal and schema-constrained parameters."""


@dataclass(frozen=True)
class ControlledAgentRequest:
    task_id: str
    case_alias: str
    case_scope: str
    scout_artifact_ids: tuple[str, ...]
    expert_artifact_id: str
    protocol_requests_expert: bool
    expected_route_cost_ms_per_image: float | None = None
    max_cost_ms_per_image: float | None = None
    idempotency_key: str = ""


@dataclass(frozen=True)
class ControlledAgentDecision:
    decision_id: str
    action: str
    code: str
    message: str
    qualification_level: str
    evidence_label: str
    risk_state: str
    human_confirmation_required: bool
    replayed: bool
    trace: tuple[dict[str, Any], ...]
    qualification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _fingerprint(
    request: ControlledAgentRequest,
    qualification: RouteQualificationDecision,
    *,
    tool_failed: bool,
    model_disagreement: bool,
) -> str:
    payload = {
        "task_id": request.task_id,
        "case_scope": request.case_scope,
        "scout_artifact_ids": request.scout_artifact_ids,
        "expert_artifact_id": request.expert_artifact_id,
        "protocol_requests_expert": request.protocol_requests_expert,
        "expected_route_cost_ms_per_image": request.expected_route_cost_ms_per_image,
        "max_cost_ms_per_image": request.max_cost_ms_per_image,
        "qualification_fingerprint": qualification.evidence_fingerprint,
        "tool_failed": tool_failed,
        "model_disagreement": model_disagreement,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def redacted_controller_context(
    request: ControlledAgentRequest,
    qualification: RouteQualificationDecision,
    *,
    model_disagreement: bool,
) -> dict[str, Any]:
    """Return the maximum context a future local controller may receive."""

    return {
        "task_id": request.task_id,
        "case_scope": request.case_scope,
        "scout_count": len(request.scout_artifact_ids),
        "expert_registered": bool(request.expert_artifact_id),
        "protocol_requests_expert": request.protocol_requests_expert,
        "model_disagreement": model_disagreement,
        "qualification_level": qualification.execution_level,
        "evidence_label": qualification.evidence_label,
        "allowed_actions": list(AGENT_ACTIONS),
        "qualification_is_external_to_controller": True,
    }


class ControlledAgentRuntime:
    """Idempotent rule controller constrained by the shared route gate."""

    def __init__(self) -> None:
        self._decisions: dict[str, tuple[str, ControlledAgentDecision]] = {}

    def decide(
        self,
        request: ControlledAgentRequest,
        *,
        qualification: RouteQualificationDecision,
        tool_payload: dict[str, Any],
    ) -> ControlledAgentDecision:
        tool_failed = any(
            isinstance(response, dict) and response.get("ok") is False
            for response in tool_payload.values()
            if isinstance(response, dict) and "ok" in response
        )
        audit = tool_payload.get("audit", {})
        audit_data = audit.get("data", {}) if isinstance(audit, dict) else {}
        model_disagreement = bool(audit_data.get("model_disagreement", False))
        fingerprint = _fingerprint(
            request,
            qualification,
            tool_failed=tool_failed,
            model_disagreement=model_disagreement,
        )
        idempotency_key = request.idempotency_key or fingerprint
        cached = self._decisions.get(idempotency_key)
        if cached is not None:
            cached_fingerprint, decision = cached
            if cached_fingerprint == fingerprint:
                return replace(decision, replayed=True)
            return self._decision(
                fingerprint=fingerprint,
                action="REFER_TO_HUMAN",
                code="AGENT_IDEMPOTENCY_CONFLICT",
                qualification=qualification,
                risk_state="not_evaluated",
                trace=(
                    {
                        "stage": "read_case_state",
                        "status": "blocked",
                        "code": "AGENT_IDEMPOTENCY_CONFLICT",
                    },
                ),
            )

        trace: list[dict[str, Any]] = [
            {
                "stage": "read_case_state",
                "status": "passed",
                "case_scope": request.case_scope,
            },
            {
                "stage": "check_route_qualification",
                "status": (
                    "passed"
                    if qualification.execution_level
                    in {"research_case_simulation", "deployment_candidate"}
                    else "restricted"
                ),
                "execution_level": qualification.execution_level,
            },
        ]
        if tool_failed:
            action = "REFER_TO_HUMAN"
            code = "AGENT_UPSTREAM_FAILED"
            risk_state = "not_evaluated"
            trace.append(
                {
                    "stage": "run_or_read_scout",
                    "status": "failed",
                    "code": code,
                }
            )
        elif qualification.execution_level in {"blocked", "research_replay_only"}:
            action = "REFER_TO_HUMAN"
            code = "AGENT_ROUTE_BLOCKED"
            risk_state = "qualification_restricted"
            trace.append(
                {
                    "stage": "apply_route_gate",
                    "status": "blocked",
                    "code": code,
                }
            )
        elif request.case_scope == "new_case" and not qualification.allow_new_case_route:
            action = "REFER_TO_HUMAN"
            code = "AGENT_NEW_CASE_BLOCKED"
            risk_state = "qualification_restricted"
            trace.append(
                {
                    "stage": "apply_route_gate",
                    "status": "blocked",
                    "code": code,
                }
            )
        elif (
            request.max_cost_ms_per_image is not None
            and request.expected_route_cost_ms_per_image is not None
            and request.expected_route_cost_ms_per_image
            > request.max_cost_ms_per_image
        ):
            action = "REFER_TO_HUMAN"
            code = "AGENT_COST_BUDGET_EXCEEDED"
            risk_state = "cost_restricted"
            trace.append(
                {
                    "stage": "apply_route_gate",
                    "status": "blocked",
                    "code": code,
                    "expected_cost_ms_per_image": (
                        request.expected_route_cost_ms_per_image
                    ),
                    "max_cost_ms_per_image": request.max_cost_ms_per_image,
                }
            )
        else:
            risk_state = (
                "protocol_requests_expert"
                if request.protocol_requests_expert
                else (
                    "model_disagreement_observed"
                    if model_disagreement
                    else "protocol_keeps_scout"
                )
            )
            action = (
                "REQUEST_EXPERT"
                if request.protocol_requests_expert
                else "KEEP_SCOUT"
            )
            code = "AGENT_OK"
            trace.extend(
                [
                    {
                        "stage": "run_or_read_scout",
                        "status": "passed",
                        "mode": request.case_scope,
                    },
                    {
                        "stage": "audit_risk",
                        "status": "passed",
                        "risk_state": risk_state,
                        "model_disagreement": model_disagreement,
                    },
                    {
                        "stage": "apply_route_gate",
                        "status": "passed",
                        "execution_level": qualification.execution_level,
                    },
                ]
            )
        trace.extend(
            [
                {
                    "stage": "select_action",
                    "status": "completed",
                    "action": action,
                    "code": code,
                },
                {
                    "stage": "build_structured_report",
                    "status": "completed",
                },
                {
                    "stage": "await_human_confirmation",
                    "status": "waiting",
                },
            ]
        )
        decision = self._decision(
            fingerprint=fingerprint,
            action=action,
            code=code,
            qualification=qualification,
            risk_state=risk_state,
            trace=tuple(trace),
        )
        self._decisions[idempotency_key] = (fingerprint, decision)
        return decision

    @staticmethod
    def _decision(
        *,
        fingerprint: str,
        action: str,
        code: str,
        qualification: RouteQualificationDecision,
        risk_state: str,
        trace: tuple[dict[str, Any], ...],
    ) -> ControlledAgentDecision:
        return ControlledAgentDecision(
            decision_id=f"agent-{fingerprint[:16]}",
            action=action,
            code=code,
            message=AGENT_MESSAGES[code],
            qualification_level=qualification.execution_level,
            evidence_label=qualification.evidence_label,
            risk_state=risk_state,
            human_confirmation_required=True,
            replayed=False,
            trace=trace,
            qualification=qualification.to_dict(),
        )
