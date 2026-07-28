"""Persistent rule/local-LLM controller runtime for OphAgent V2."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterator
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.orchestration_contracts import (
    AgentAction,
    ActorRole,
    CASE_STATE_SCHEMA_VERSION,
    CONTROLLER_PROPOSAL_SCHEMA_VERSION,
    CaseState,
    CaseStatus,
    ControllerAdapter,
    ControllerProposal,
    RouteQualification,
    redact_structured_value,
    sanitize_controller_context,
)


CONTROLLER_REASON_CODES = {
    "LOW_RISK_KEEP_SCOUT",
    "HIGH_RISK_REQUEST_EXPERT",
    "QUALIFICATION_RESTRICTED",
    "TOOL_FAILURE",
    "COST_LIMIT",
    "MODEL_DISAGREEMENT",
    "STATE_REQUIRES_HUMAN",
    "INVALID_PROPOSAL",
}

TERMINAL_STATES = {
    CaseStatus.CLOSED,
    CaseStatus.BLOCKED,
    CaseStatus.CANCELLED,
}

STATE_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.CASE_RECEIVED: {
        CaseStatus.CASE_VALIDATED,
        CaseStatus.BLOCKED,
        CaseStatus.FAILED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.CASE_VALIDATED: {
        CaseStatus.SCOUT_COMPLETED,
        CaseStatus.BLOCKED,
        CaseStatus.FAILED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.SCOUT_COMPLETED: {
        CaseStatus.RISK_AUDITED,
        CaseStatus.FAILED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.RISK_AUDITED: {
        CaseStatus.EXPERT_PENDING_APPROVAL,
        CaseStatus.REVIEW_PENDING,
        CaseStatus.BLOCKED,
        CaseStatus.FAILED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.EXPERT_PENDING_APPROVAL: {
        CaseStatus.EXPERT_COMPLETED,
        CaseStatus.REVIEW_PENDING,
        CaseStatus.FAILED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.EXPERT_COMPLETED: {
        CaseStatus.REVIEW_PENDING,
        CaseStatus.FAILED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.REVIEW_PENDING: {
        CaseStatus.CLOSED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.FAILED: {
        CaseStatus.CASE_RECEIVED,
        CaseStatus.CASE_VALIDATED,
        CaseStatus.SCOUT_COMPLETED,
        CaseStatus.EXPERT_PENDING_APPROVAL,
        CaseStatus.CANCELLED,
    },
    CaseStatus.BLOCKED: {CaseStatus.CLOSED},
    CaseStatus.CLOSED: set(),
    CaseStatus.CANCELLED: set(),
}

STATE_ACTIONS: dict[CaseStatus, tuple[str, ...]] = {
    CaseStatus.RISK_AUDITED: tuple(action.value for action in AgentAction),
    CaseStatus.EXPERT_PENDING_APPROVAL: (
        AgentAction.REQUEST_EXPERT,
        AgentAction.REFER_TO_HUMAN,
    ),
    CaseStatus.REVIEW_PENDING: (AgentAction.REFER_TO_HUMAN,),
}

ROLE_PERMISSIONS = {
    ActorRole.OPERATOR: {
        "case.submit",
        "case.retry",
        "case.cancel",
    },
    ActorRole.REVIEWER: {
        "expert.decide",
        "review.confirm",
    },
    ActorRole.ADMIN: {
        "case.submit",
        "case.retry",
        "case.cancel",
        "expert.decide",
        "review.confirm",
        "protocol.inspect",
    },
}

RETRYABLE_TOOL_CODES = {
    "TOOL_EXECUTION_FAILED",
    "UPSTREAM_FAILED",
    "LOCAL_RUNTIME_UNAVAILABLE",
    "EXPERT_TOOL_EXECUTION_FAILED",
}

class StateTransitionError(RuntimeError):
    pass


class PermissionDenied(RuntimeError):
    pass


class IdempotencyConflict(RuntimeError):
    pass


class ControllerUnavailable(RuntimeError):
    pass


class CaseBusy(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def redact_sensitive_payload(value: Any) -> Any:
    """Remove identity fields and real paths before persistence or prompting."""

    return redact_structured_value(value)


def authorize(role: str, permission: str) -> None:
    try:
        actor = ActorRole(role)
    except ValueError as exc:
        raise PermissionDenied("RBAC_UNKNOWN_ROLE") from exc
    if permission not in ROLE_PERMISSIONS[actor]:
        raise PermissionDenied(f"RBAC_DENIED:{permission}")


def allowed_actions_for_state(state: str) -> tuple[str, ...]:
    try:
        return STATE_ACTIONS.get(CaseStatus(state), ())
    except ValueError:
        return ()


def transition_state(
    state: CaseState,
    target: str,
    *,
    event: str,
    code: str = "OK",
    details: dict[str, Any] | None = None,
) -> CaseState:
    source = CaseStatus(state.current_state)
    destination = CaseStatus(target)
    if destination not in STATE_TRANSITIONS[source]:
        raise StateTransitionError(
            f"STATE_TRANSITION_INVALID:{source.value}->{destination.value}"
        )
    state.current_state = destination.value
    state.allowed_actions = allowed_actions_for_state(destination.value)
    state.updated_at = _utc_now()
    state.trace.append(
        {
            "sequence": len(state.trace) + 1,
            "event": event,
            "from_state": source.value,
            "to_state": destination.value,
            "code": code,
            "at": state.updated_at,
            "details": redact_sensitive_payload(details or {}),
        }
    )
    return state


class CaseStateStore:
    """Atomic JSON store; a new instance restores service state."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _safe_case_id(case_id: str) -> str:
        if re.search(
            r"(?i)(?:mrn|patient|hospital|medical.?record|"
            r"住院|病历|身份证)",
            case_id,
        ):
            raise ValueError("CASE_ID_APPEARS_SENSITIVE")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", case_id).strip(".-")
        if not safe:
            raise ValueError("case_id has no safe characters")
        return safe

    def path_for(self, case_id: str) -> Path:
        return self.root / f"{self._safe_case_id(case_id)}.json"

    @contextmanager
    def case_lock(
        self,
        case_id: str,
        *,
        timeout_seconds: float = 120.0,
    ) -> Iterator[None]:
        """Serialize a case across threads/processes; OS releases on crash."""

        lock_path = self.path_for(case_id).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        locked = False
        try:
            deadline = time.monotonic() + timeout_seconds
            while not locked:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0, os.SEEK_END)
                        if handle.tell() == 0:
                            handle.write(b"\0")
                            handle.flush()
                        handle.seek(0)
                        msvcrt.locking(
                            handle.fileno(),
                            msvcrt.LK_NBLCK,
                            1,
                        )
                    else:
                        import fcntl

                        fcntl.flock(
                            handle.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    locked = True
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        raise CaseBusy("CASE_BUSY") from None
                    time.sleep(0.05)
            yield
        finally:
            try:
                if locked and os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif locked:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def load(self, case_id: str) -> CaseState | None:
        path = self.path_for(case_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CaseState.from_dict(payload)

    def save(self, state: CaseState) -> Path:
        state.updated_at = _utc_now()
        if not state.created_at:
            state.created_at = state.updated_at
        payload = redact_sensitive_payload(state.to_dict())
        payload["schema_version"] = CASE_STATE_SCHEMA_VERSION
        path = self.path_for(state.case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path


@dataclass(frozen=True)
class LocalLLMControllerConfig:
    model_id: str
    endpoint: str = ""
    prompt_template: str = (
        "你是受控研究编排控制器。仅输出一个 JSON 对象，字段为 action、"
        "reason_code、parameters、schema_version。可用动作："
        "KEEP_SCOUT、REQUEST_EXPERT、REFER_TO_HUMAN。\n状态：{context}"
    )
    few_shot_examples: tuple[dict[str, Any], ...] = ()
    timeout_seconds: float = 30.0
    generation_max_new_tokens: int = 128


class RuleController(ControllerAdapter):
    controller_type = "rule_controller"

    def propose(self, context: dict[str, Any]) -> ControllerProposal:
        safe = sanitize_controller_context(context)
        codes = tuple(str(value) for value in safe.get("tool_return_codes", []))
        qualification = dict(safe.get("model_qualification", {}))
        risk = dict(safe.get("risk_result", {}))
        if any(code not in {"OK", "UPSTREAM_SKIPPED"} for code in codes):
            action = AgentAction.REFER_TO_HUMAN
            reason = "TOOL_FAILURE"
        elif qualification.get("execution_level") == "blocked":
            action = AgentAction.REFER_TO_HUMAN
            reason = "QUALIFICATION_RESTRICTED"
        elif bool(risk.get("protocol_requests_expert", False)):
            action = AgentAction.REQUEST_EXPERT
            reason = "HIGH_RISK_REQUEST_EXPERT"
        elif bool(risk.get("model_disagreement", False)):
            action = AgentAction.REQUEST_EXPERT
            reason = "MODEL_DISAGREEMENT"
        else:
            action = AgentAction.KEEP_SCOUT
            reason = "LOW_RISK_KEEP_SCOUT"
        return ControllerProposal(
            action=action.value,
            reason_code=reason,
            parameters={},
            controller_type=self.controller_type,
        )


class LocalLLMController(ControllerAdapter):
    """One implementation shared by 4B/27B and zero/few-shot settings."""

    controller_type = "local_llm_controller"

    def __init__(
        self,
        config: LocalLLMControllerConfig,
        *,
        inference_callable: Callable[
            [str, LocalLLMControllerConfig], str | dict[str, Any]
        ]
        | None = None,
    ):
        self.config = config
        self._inference_callable = inference_callable

    def _prompt(self, context: dict[str, Any]) -> str:
        safe = sanitize_controller_context(context)
        examples = ""
        if self.config.few_shot_examples:
            examples = "\n示例（不属于评测集）：\n" + _stable_json(
                self.config.few_shot_examples
            )
        return self.config.prompt_template.format(
            context=_stable_json(safe)
        ) + examples

    def _call_local_endpoint(self, prompt: str) -> dict[str, Any]:
        endpoint = self.config.endpoint.strip()
        parsed = urlparse(endpoint)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ControllerUnavailable(
                "LOCAL_RUNTIME_ENDPOINT_NOT_LOOPBACK"
            )
        body = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.config.generation_max_new_tokens,
        }
        request = Request(
            endpoint,
            data=_stable_json(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(  # noqa: S310 - endpoint is explicitly loopback-only
            request,
            timeout=self.config.timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        return _parse_json_object(str(content))

    def propose(self, context: dict[str, Any]) -> ControllerProposal:
        prompt = self._prompt(context)
        if self._inference_callable is not None:
            raw = self._inference_callable(prompt, self.config)
            payload = raw if isinstance(raw, dict) else _parse_json_object(raw)
        elif self.config.endpoint:
            payload = self._call_local_endpoint(prompt)
        else:
            raise ControllerUnavailable("LOCAL_RUNTIME_CONTRACT_MISSING")
        return ControllerProposal.from_dict(
            payload,
            controller_type=self.controller_type,
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("CONTROLLER_SCHEMA_NOT_JSON") from None
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("CONTROLLER_SCHEMA_NOT_OBJECT")
    return payload


def validate_controller_proposal(
    proposal: ControllerProposal,
    *,
    allowed_actions: tuple[str, ...],
) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    if proposal.schema_version != CONTROLLER_PROPOSAL_SCHEMA_VERSION:
        errors.append("PROPOSAL_SCHEMA_VERSION_INVALID")
    if proposal.action not in {action.value for action in AgentAction}:
        errors.append("PROPOSAL_ACTION_INVALID")
    elif proposal.action not in allowed_actions:
        errors.append("PROPOSAL_ACTION_NOT_ALLOWED_BY_STATE")
    if proposal.reason_code not in CONTROLLER_REASON_CODES:
        errors.append("PROPOSAL_REASON_CODE_INVALID")
    if proposal.parameters:
        errors.append("PROPOSAL_PARAMETERS_INVALID")
    return not errors, tuple(errors)


def gate_controller_proposal(
    proposal: ControllerProposal,
    *,
    qualification: RouteQualification,
    allowed_actions: tuple[str, ...],
    remaining_budget: float | None,
    expected_expert_cost: float | None,
    case_scope: str,
    tool_return_codes: tuple[str, ...],
) -> dict[str, Any]:
    """Revalidate every proposal; the controller never decides execution."""

    valid, errors = validate_controller_proposal(
        proposal,
        allowed_actions=allowed_actions,
    )
    action = proposal.action
    code = "GATE_APPROVED"
    if not valid:
        action = AgentAction.REFER_TO_HUMAN.value
        code = "GATE_CONTROLLER_PROPOSAL_INVALID"
    elif any(
        value not in {"OK", "UPSTREAM_SKIPPED"}
        for value in tool_return_codes
    ):
        action = AgentAction.REFER_TO_HUMAN.value
        code = "GATE_TOOL_FAILURE"
    elif qualification.execution_level == "blocked":
        action = AgentAction.REFER_TO_HUMAN.value
        code = "GATE_ROUTE_QUALIFICATION_BLOCKED"
    elif (
        case_scope == "cached_prediction_replay"
        and not qualification.allow_cached_replay
    ):
        action = AgentAction.REFER_TO_HUMAN.value
        code = "GATE_CACHED_REPLAY_NOT_ELIGIBLE"
    elif (
        case_scope == "research_case_simulation"
        and not qualification.allow_case_simulation
    ):
        action = AgentAction.REFER_TO_HUMAN.value
        code = "GATE_CASE_SIMULATION_NOT_ELIGIBLE"
    elif case_scope == "new_case" and not qualification.allow_new_case_route:
        action = AgentAction.REFER_TO_HUMAN.value
        code = "GATE_NEW_CASE_NOT_ELIGIBLE"
    elif (
        action == AgentAction.REQUEST_EXPERT.value
        and remaining_budget is not None
        and expected_expert_cost is not None
        and expected_expert_cost > remaining_budget
    ):
        action = AgentAction.REFER_TO_HUMAN.value
        code = "GATE_EXPERT_BUDGET_EXCEEDED"
    return {
        "approved": code == "GATE_APPROVED",
        "proposal_action": proposal.action,
        "final_action": action,
        "code": code,
        "validation_errors": list(errors),
        "gate_intercepted": action != proposal.action,
        "qualification_level": qualification.execution_level,
        "clinical_route_eligible": False,
    }

@dataclass(frozen=True)
class ControlledCaseRequest:
    case_id: str
    task_id: str
    idempotency_key: str
    case_scope: str
    case_metadata: dict[str, Any]
    remaining_budget: float | None = None
    expected_expert_cost: float | None = None
    controller_type: str = ""
    qualification_policy_version: str = ""
    route_protocol_version: str = ""


@dataclass(frozen=True)
class AgentToolBundle:
    """Sanitized application result produced through registered Tool Contracts."""

    qualification: RouteQualification
    tool_payload: dict[str, Any]
    tool_trace: dict[str, Any]


@dataclass(frozen=True)
class AgentToolStepResult:
    """One checkpointable Tool Contract step."""

    tool_payload: dict[str, Any]
    tool_trace: dict[str, Any]
    qualification: RouteQualification | None = None


@dataclass(frozen=True)
class AgentExpertResult:
    """Expert result released only after reviewer approval."""

    tool_payload: dict[str, Any]
    tool_trace: dict[str, Any]


class ControlledAgentRuntimeV2:
    """Persist every completed state-machine step and never repeat it."""

    def __init__(self, store: CaseStateStore):
        self.store = store

    @staticmethod
    def _controller_identity(controller: ControllerAdapter) -> dict[str, Any]:
        config = getattr(controller, "config", None)
        return {
            "controller_type": controller.controller_type,
            "model_id": str(getattr(config, "model_id", "")),
            "endpoint": str(getattr(config, "endpoint", "")),
            "prompt_template_sha256": _sha256(
                str(getattr(config, "prompt_template", ""))
            ),
            "few_shot_sha256": _sha256(
                getattr(config, "few_shot_examples", ())
            ),
        }

    @classmethod
    def _request_fingerprint(
        cls,
        request: ControlledCaseRequest,
        controller: ControllerAdapter,
    ) -> str:
        return _sha256(
            {
                "case_id": request.case_id,
                "task_id": request.task_id,
                "case_scope": request.case_scope,
                "case_metadata": redact_sensitive_payload(
                    request.case_metadata
                ),
                "remaining_budget": request.remaining_budget,
                "expected_expert_cost": request.expected_expert_cost,
                "qualification_policy_version": (
                    request.qualification_policy_version
                ),
                "route_protocol_version": request.route_protocol_version,
                "controller": cls._controller_identity(controller),
            }
        )

    def _load_or_create(
        self,
        request: ControlledCaseRequest,
        *,
        controller: ControllerAdapter,
    ) -> tuple[CaseState, bool]:
        if (
            request.controller_type
            and request.controller_type != controller.controller_type
        ):
            raise ValueError("CONTROLLER_TYPE_MISMATCH")
        fingerprint = self._request_fingerprint(request, controller)
        existing = self.store.load(request.case_id)
        if existing is not None:
            if (
                existing.idempotency_key != request.idempotency_key
                or existing.request_fingerprint != fingerprint
            ):
                raise IdempotencyConflict("AGENT_IDEMPOTENCY_CONFLICT")
            return existing, False

        now = _utc_now()
        state = CaseState(
            case_id=request.case_id,
            task_id=request.task_id,
            controller_type=controller.controller_type,
            idempotency_key=request.idempotency_key,
            request_fingerprint=fingerprint,
            remaining_budget=request.remaining_budget,
            expected_expert_cost=request.expected_expert_cost,
            created_at=now,
            updated_at=now,
            qualification_policy_version=(
                request.qualification_policy_version
            ),
            route_protocol_version=request.route_protocol_version,
            trace=[
                {
                    "sequence": 1,
                    "event": "case_received",
                    "from_state": "",
                    "to_state": CaseStatus.CASE_RECEIVED.value,
                    "code": "CASE_RECEIVED",
                    "at": now,
                    "details": {},
                }
            ],
        )
        self.store.save(state)
        return state, True

    def execute(
        self,
        request: ControlledCaseRequest,
        *,
        controller: ControllerAdapter,
        tool_executor: Any,
        actor_role: str = ActorRole.OPERATOR,
    ) -> tuple[CaseState, bool]:
        """Restore before tool execution so refresh/restart cannot repeat work."""

        with self.store.case_lock(request.case_id):
            return self._execute_locked(
                request,
                controller=controller,
                tool_executor=tool_executor,
                actor_role=actor_role,
            )

    def _execute_locked(
        self,
        request: ControlledCaseRequest,
        *,
        controller: ControllerAdapter,
        tool_executor: Any,
        actor_role: str,
    ) -> tuple[CaseState, bool]:
        if (
            request.controller_type
            and request.controller_type != controller.controller_type
        ):
            raise ValueError("CONTROLLER_TYPE_MISMATCH")
        if callable(getattr(tool_executor, "execute_step", None)):
            return self._execute_stepwise_locked(
                request,
                controller=controller,
                tool_executor=tool_executor,
                actor_role=actor_role,
            )
        resumable = {
            CaseStatus.CASE_RECEIVED.value,
            CaseStatus.CASE_VALIDATED.value,
            CaseStatus.SCOUT_COMPLETED.value,
            CaseStatus.RISK_AUDITED.value,
        }
        existing = self.store.load(request.case_id)
        if existing is not None:
            expected = self._request_fingerprint(request, controller)
            if (
                existing.idempotency_key != request.idempotency_key
                or existing.request_fingerprint != expected
            ):
                raise IdempotencyConflict("AGENT_IDEMPOTENCY_CONFLICT")
            if existing.current_state not in resumable:
                return existing, True
        authorize(actor_role, "case.submit")
        state, created = self._load_or_create(
            request,
            controller=controller,
        )
        if not created and state.current_state not in resumable:
            return state, True
        if state.current_state == CaseStatus.RISK_AUDITED:
            if not state.qualification or not state.runtime_payload:
                return self._fail(
                    state,
                    code="PERSISTED_AUDIT_CONTEXT_INCOMPLETE",
                    event="restore_audited_case",
                ), True
            restored_state, _ = self._run_locked(
                request,
                qualification=RouteQualification.from_dict(
                    state.qualification
                ),
                tool_payload=state.runtime_payload,
                tool_trace=state.tool_trace,
                controller=controller,
                actor_role=actor_role,
            )
            return restored_state, True
        try:
            bundle = tool_executor(state)
        except Exception as exc:  # tool adapters normalize known failures
            failed = self._fail(
                state,
                code="TOOL_EXECUTION_FAILED",
                event="execute_tool_contract",
                details={"exception_type": type(exc).__name__},
            )
            return failed, False
        return self._run_locked(
            request,
            qualification=bundle.qualification,
            tool_payload=bundle.tool_payload,
            tool_trace=bundle.tool_trace,
            controller=controller,
            actor_role=actor_role,
        )

    def _execute_stepwise_locked(
        self,
        request: ControlledCaseRequest,
        *,
        controller: ControllerAdapter,
        tool_executor: Any,
        actor_role: str,
    ) -> tuple[CaseState, bool]:
        resumable = {
            CaseStatus.CASE_RECEIVED.value,
            CaseStatus.CASE_VALIDATED.value,
            CaseStatus.SCOUT_COMPLETED.value,
            CaseStatus.RISK_AUDITED.value,
        }
        existing = self.store.load(request.case_id)
        if existing is not None:
            expected = self._request_fingerprint(request, controller)
            if (
                existing.idempotency_key != request.idempotency_key
                or existing.request_fingerprint != expected
            ):
                raise IdempotencyConflict("AGENT_IDEMPOTENCY_CONFLICT")
            if existing.current_state not in resumable:
                return existing, True
        authorize(actor_role, "case.submit")
        state, created = self._load_or_create(
            request,
            controller=controller,
        )
        replayed = not created

        if state.current_state == CaseStatus.RISK_AUDITED:
            if not state.qualification:
                return self._fail(
                    state,
                    code="PERSISTED_AUDIT_CONTEXT_INCOMPLETE",
                    event="restore_audited_case",
                ), replayed
            completed, _ = self._run_locked(
                request,
                qualification=RouteQualification.from_dict(
                    state.qualification
                ),
                tool_payload=state.runtime_payload,
                tool_trace=state.tool_trace,
                controller=controller,
                actor_role=actor_role,
            )
            return completed, True

        if state.current_state == CaseStatus.CASE_RECEIVED:
            if "input_check" not in state.completed_steps:
                result = self._execute_tool_step(
                    tool_executor,
                    state,
                    "input",
                )
                self._merge_step_result(state, result)
                input_response = dict(
                    state.runtime_payload.get("input", {})
                )
                if not input_response.get("ok", False):
                    return self._fail(
                        state,
                        code=str(
                            input_response.get(
                                "code",
                                "TOOL_EXECUTION_FAILED",
                            )
                        ),
                        event="validate_case_input",
                    ), replayed
                state.completed_steps += ("input_check",)
                self.store.save(state)

            if "model_qualification" not in state.completed_steps:
                result = self._execute_tool_step(
                    tool_executor,
                    state,
                    "registry",
                )
                self._merge_step_result(state, result)
                registry_response = dict(
                    state.runtime_payload.get("registry", {})
                )
                if not registry_response.get("ok", False):
                    return self._fail(
                        state,
                        code=str(
                            registry_response.get(
                                "code",
                                "TOOL_EXECUTION_FAILED",
                            )
                        ),
                        event="validate_model_registry",
                    ), replayed
                state.completed_steps += (
                    "task_recognition",
                    "model_qualification",
                )
                transition_state(
                    state,
                    CaseStatus.CASE_VALIDATED,
                    event="validate_case",
                    details={
                        "trace_steps": list(state.completed_steps),
                    },
                )
                self.store.save(state)

        if state.current_state == CaseStatus.CASE_VALIDATED:
            if "route_metadata" not in state.completed_steps:
                result = self._execute_tool_step(
                    tool_executor,
                    state,
                    "route_metadata",
                )
                self._merge_step_result(state, result)
                route_response = dict(
                    state.runtime_payload.get("route", {})
                )
                if not route_response.get("ok", False):
                    return self._fail(
                        state,
                        code=str(
                            route_response.get(
                                "code",
                                "TOOL_EXECUTION_FAILED",
                            )
                        ),
                        event="query_route_metadata",
                    ), replayed
                state.completed_steps += ("route_metadata",)
                self.store.save(state)

            result = self._execute_tool_step(
                tool_executor,
                state,
                "scout",
            )
            self._merge_step_result(state, result)
            scout_response = _scout_response(state.runtime_payload)
            if not scout_response.get("ok", False):
                return self._fail(
                    state,
                    code=str(
                        scout_response.get(
                            "code",
                            "TOOL_EXECUTION_FAILED",
                        )
                    ),
                    event="run_scout",
                ), replayed
            state.completed_steps += ("scout",)
            transition_state(
                state,
                CaseStatus.SCOUT_COMPLETED,
                event="run_scout",
                code=str(scout_response.get("code", "OK")),
            )
            self.store.save(state)

        if state.current_state == CaseStatus.SCOUT_COMPLETED:
            result = self._execute_tool_step(
                tool_executor,
                state,
                "audit_and_qualification",
            )
            self._merge_step_result(state, result)
            audit_response = dict(
                state.runtime_payload.get("audit", {})
            )
            if not audit_response.get("ok", False):
                return self._fail(
                    state,
                    code=str(
                        audit_response.get(
                            "code",
                            "TOOL_EXECUTION_FAILED",
                        )
                    ),
                    event="audit_risk",
                ), replayed
            if result.qualification is None:
                return self._fail(
                    state,
                    code="QUALIFICATION_RESULT_MISSING",
                    event="audit_risk",
                ), replayed
            state.qualification = result.qualification.to_dict()
            state.tool_return_codes = tuple(
                _tool_codes(state.runtime_payload)
            )
            state.completed_steps += (
                "risk_audit",
                "route_qualification",
            )
            transition_state(
                state,
                CaseStatus.RISK_AUDITED,
                event="audit_risk_and_qualification",
                details={
                    "qualification_level": (
                        result.qualification.execution_level
                    ),
                },
            )
            self.store.save(state)

        if state.current_state != CaseStatus.RISK_AUDITED:
            return state, replayed
        completed, _ = self._run_locked(
            request,
            qualification=RouteQualification.from_dict(
                state.qualification
            ),
            tool_payload=state.runtime_payload,
            tool_trace=state.tool_trace,
            controller=controller,
            actor_role=actor_role,
        )
        return completed, replayed

    def _execute_tool_step(
        self,
        tool_executor: Any,
        state: CaseState,
        step: str,
    ) -> AgentToolStepResult:
        response_key = {
            "input": "input",
            "registry": "registry",
            "route_metadata": "route",
            "scout": "inference",
            "audit_and_qualification": "audit",
        }[step]
        try:
            result = tool_executor.execute_step(state, step)
        except Exception as exc:
            return AgentToolStepResult(
                tool_payload={
                    response_key: {
                        "ok": False,
                        "code": "TOOL_EXECUTION_FAILED",
                        "message": type(exc).__name__,
                        "data": {},
                    }
                },
                tool_trace={},
            )
        if not isinstance(result, AgentToolStepResult):
            return AgentToolStepResult(
                tool_payload={
                    response_key: {
                        "ok": False,
                        "code": "TOOL_EXECUTION_FAILED",
                        "message": "TOOL_STEP_RESULT_INVALID",
                        "data": {},
                    }
                },
                tool_trace={},
            )
        return result

    @staticmethod
    def _merge_step_result(
        state: CaseState,
        result: AgentToolStepResult,
    ) -> None:
        state.runtime_payload = {
            **state.runtime_payload,
            **dict(redact_sensitive_payload(result.tool_payload)),
        }
        state.tool_trace = _merge_tool_trace(
            state.tool_trace,
            dict(redact_sensitive_payload(result.tool_trace)),
        )

    def run(
        self,
        request: ControlledCaseRequest,
        *,
        qualification: RouteQualification,
        tool_payload: dict[str, Any],
        tool_trace: dict[str, Any] | None = None,
        controller: ControllerAdapter,
        actor_role: str = ActorRole.OPERATOR,
    ) -> tuple[CaseState, bool]:
        with self.store.case_lock(request.case_id):
            return self._run_locked(
                request,
                qualification=qualification,
                tool_payload=tool_payload,
                tool_trace=tool_trace,
                controller=controller,
                actor_role=actor_role,
            )

    def _run_locked(
        self,
        request: ControlledCaseRequest,
        *,
        qualification: RouteQualification,
        tool_payload: dict[str, Any],
        tool_trace: dict[str, Any] | None,
        controller: ControllerAdapter,
        actor_role: str,
    ) -> tuple[CaseState, bool]:
        if (
            request.controller_type
            and request.controller_type != controller.controller_type
        ):
            raise ValueError("CONTROLLER_TYPE_MISMATCH")
        resumable = {
            CaseStatus.CASE_RECEIVED.value,
            CaseStatus.CASE_VALIDATED.value,
            CaseStatus.SCOUT_COMPLETED.value,
            CaseStatus.RISK_AUDITED.value,
        }
        existing = self.store.load(request.case_id)
        if existing is not None:
            expected = self._request_fingerprint(request, controller)
            if (
                existing.idempotency_key != request.idempotency_key
                or existing.request_fingerprint != expected
            ):
                raise IdempotencyConflict("AGENT_IDEMPOTENCY_CONFLICT")
            if existing.current_state not in resumable:
                if existing.qualification and (
                    str(
                        existing.qualification.get(
                            "contract_sha256",
                            "",
                        )
                    )
                    != qualification.contract_sha256
                    or str(
                        existing.qualification.get(
                            "evidence_fingerprint",
                            "",
                        )
                    )
                    != qualification.evidence_fingerprint
                ):
                    raise IdempotencyConflict(
                        "AGENT_QUALIFICATION_EVIDENCE_CHANGED"
                    )
                return existing, True
        authorize(actor_role, "case.submit")
        state, created = self._load_or_create(
            request,
            controller=controller,
        )
        if not created and state.current_state not in resumable:
            return state, True
        if state.qualification and (
            str(state.qualification.get("contract_sha256", ""))
            != qualification.contract_sha256
            or str(state.qualification.get("evidence_fingerprint", ""))
            != qualification.evidence_fingerprint
        ):
            raise IdempotencyConflict("AGENT_QUALIFICATION_EVIDENCE_CHANGED")
        state.runtime_payload = {
            **state.runtime_payload,
            **dict(redact_sensitive_payload(tool_payload)),
        }
        state.tool_trace = _merge_tool_trace(
            state.tool_trace,
            dict(redact_sensitive_payload(tool_trace or {})),
        )
        self.store.save(state)

        if state.current_state == CaseStatus.CASE_RECEIVED:
            input_response = dict(tool_payload.get("input", {}))
            registry_response = dict(tool_payload.get("registry", {}))
            if not input_response.get("ok") or not registry_response.get("ok"):
                return self._fail(
                    state,
                    code=_first_failure_code(
                        input_response,
                        registry_response,
                    ),
                    event="validate_case",
                ), False
            state.completed_steps = (
                "input_check",
                "task_recognition",
                "model_qualification",
            )
            transition_state(
                state,
                CaseStatus.CASE_VALIDATED,
                event="validate_case",
                details={
                    "trace_steps": list(state.completed_steps),
                },
            )
            self.store.save(state)

        if state.current_state == CaseStatus.CASE_VALIDATED:
            scout_response = _scout_response(tool_payload)
            if not scout_response.get("ok", False):
                return self._fail(
                    state,
                    code=str(
                        scout_response.get("code", "TOOL_EXECUTION_FAILED")
                    ),
                    event="run_scout",
                ), False
            state.completed_steps += ("scout",)
            transition_state(
                state,
                CaseStatus.SCOUT_COMPLETED,
                event="run_scout",
                code=str(scout_response.get("code", "OK")),
            )
            self.store.save(state)

        audit_response = dict(tool_payload.get("audit", {}))
        if state.current_state == CaseStatus.SCOUT_COMPLETED:
            if not audit_response.get("ok", False):
                return self._fail(
                    state,
                    code=str(
                        audit_response.get(
                            "code",
                            "TOOL_EXECUTION_FAILED",
                        )
                    ),
                    event="audit_risk",
                ), False
            state.completed_steps += (
                "risk_audit",
                "route_qualification",
            )
            transition_state(
                state,
                CaseStatus.RISK_AUDITED,
                event="audit_risk_and_qualification",
                details={
                    "qualification_level": qualification.execution_level,
                },
            )
            state.qualification = qualification.to_dict()
            state.tool_return_codes = tuple(_tool_codes(tool_payload))
            self.store.save(state)

        risk_data = dict(audit_response.get("data", {}))
        route_data = dict(tool_payload.get("route", {}).get("data", {}))
        risk_data["protocol_requests_expert"] = bool(
            route_data.get(
                "protocol_requests_expert",
                route_data.get("expert_invoked", False),
            )
        )
        context = sanitize_controller_context(
            {
                "current_state": state.current_state,
                "task_id": state.task_id,
                "allowed_actions": list(state.allowed_actions),
                "model_qualification": state.qualification,
                "risk_result": risk_data,
                "remaining_budget": state.remaining_budget,
                "tool_return_codes": list(state.tool_return_codes),
                "case_metadata": request.case_metadata,
            }
        )
        try:
            proposal = controller.propose(context)
        except Exception as exc:  # untrusted controller boundary fails closed
            proposal = ControllerProposal(
                action=AgentAction.REFER_TO_HUMAN.value,
                reason_code="INVALID_PROPOSAL",
                parameters={},
                controller_type=controller.controller_type,
            )
            state.trace.append(
                {
                    "sequence": len(state.trace) + 1,
                    "event": "controller_proposal_error",
                    "from_state": state.current_state,
                    "to_state": state.current_state,
                    "code": type(exc).__name__,
                    "at": _utc_now(),
                    "details": {},
                }
            )
        state.controller_proposal = proposal.to_dict()
        gate = gate_controller_proposal(
            proposal,
            qualification=qualification,
            allowed_actions=state.allowed_actions,
            remaining_budget=request.remaining_budget,
            expected_expert_cost=request.expected_expert_cost,
            case_scope=request.case_scope,
            tool_return_codes=state.tool_return_codes,
        )
        state.gate_decision = gate
        state.final_action = str(gate["final_action"])
        state.completed_steps += (
            "controller_proposal",
            "qualification_gate",
            "report_generation",
        )
        state.report = {
            "schema_version": "ophagent.controlled_case_report.v2",
            "case_id": state.case_id,
            "task_id": state.task_id,
            "controller_type": state.controller_type,
            "controller_proposal": state.controller_proposal,
            "gate_decision": state.gate_decision,
            "final_action": state.final_action,
            "qualification": state.qualification,
            "tool_return_codes": list(state.tool_return_codes),
            "completed_steps": list(state.completed_steps),
            "clinical_diagnosis": False,
            "test_content_used": False,
        }
        if state.final_action == AgentAction.REQUEST_EXPERT.value:
            transition_state(
                state,
                CaseStatus.EXPERT_PENDING_APPROVAL,
                event="await_expert_approval",
                code=str(gate["code"]),
            )
        else:
            transition_state(
                state,
                CaseStatus.REVIEW_PENDING,
                event="await_human_review",
                code=str(gate["code"]),
            )
        self.store.save(state)
        return state, False

    def _fail(
        self,
        state: CaseState,
        *,
        code: str,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> CaseState:
        prior_codes = tuple(state.tool_return_codes)
        state.tool_return_codes = (*prior_codes, code)
        state.final_action = AgentAction.REFER_TO_HUMAN.value
        state.report = {
            **state.report,
            "schema_version": "ophagent.controlled_case_report.v2",
            "case_id": state.case_id,
            "task_id": state.task_id,
            "final_action": state.final_action,
            "failure_code": code,
            "failure_stage": event,
            "pre_failure_tool_return_codes": list(prior_codes),
            "clinical_diagnosis": False,
            "test_content_used": False,
        }
        transition_state(
            state,
            CaseStatus.FAILED,
            event=event,
            code=code,
            details=details,
        )
        self.store.save(state)
        return state

    def decide_expert(
        self,
        case_id: str,
        *,
        approved: bool,
        actor_role: str,
        expert_executor: Callable[[CaseState], AgentExpertResult]
        | None = None,
        idempotency_key: str = "",
    ) -> CaseState:
        authorize(actor_role, "expert.decide")
        with self.store.case_lock(case_id):
            return self._decide_expert_locked(
                case_id,
                approved=approved,
                expert_executor=expert_executor,
                idempotency_key=(
                    idempotency_key
                    or f"expert-decision:{case_id}:{approved}"
                ),
            )

    def _decide_expert_locked(
        self,
        case_id: str,
        *,
        approved: bool,
        expert_executor: Callable[[CaseState], AgentExpertResult]
        | None,
        idempotency_key: str,
    ) -> CaseState:
        state = self._required(case_id)
        expected_decision = (
            "EXPERT_APPROVED" if approved else "EXPERT_REJECTED"
        )
        if state.current_state != CaseStatus.EXPERT_PENDING_APPROVAL:
            if (
                state.human_decision == expected_decision
                and state.report.get("expert_decision_idempotency_key")
                == idempotency_key
            ):
                return state
            raise StateTransitionError("EXPERT_DECISION_NOT_PENDING")
        if approved:
            if expert_executor is None:
                return self._fail(
                    state,
                    code="EXPERT_TOOL_CONTRACT_REQUIRED",
                    event="expert_approval",
                )
            try:
                result = expert_executor(state)
            except Exception as exc:
                return self._fail(
                    state,
                    code="EXPERT_TOOL_EXECUTION_FAILED",
                    event="expert_approval",
                    details={"exception_type": type(exc).__name__},
                )
            if not isinstance(result, AgentExpertResult):
                return self._fail(
                    state,
                    code="EXPERT_TOOL_RESULT_INVALID",
                    event="expert_approval",
                )
            state.runtime_payload = {
                **state.runtime_payload,
                **dict(
                    redact_sensitive_payload(result.tool_payload)
                ),
            }
            state.tool_trace = _merge_tool_trace(
                state.tool_trace,
                dict(redact_sensitive_payload(result.tool_trace)),
            )
            expert_response = dict(
                state.runtime_payload.get("expert", {})
            )
            if not expert_response.get("ok", False):
                return self._fail(
                    state,
                    code=str(
                        expert_response.get(
                            "code",
                            "EXPERT_TOOL_EXECUTION_FAILED",
                        )
                    ),
                    event="expert_approval",
                )
            state.tool_return_codes += (
                str(expert_response.get("code", "OK")),
            )
            state.human_decision = expected_decision
            transition_state(
                state,
                CaseStatus.EXPERT_COMPLETED,
                event="expert_approval",
                code="EXPERT_APPROVED",
            )
            state.completed_steps += ("expert_frozen_replay",)
            if (
                state.remaining_budget is not None
                and state.expected_expert_cost is not None
            ):
                state.remaining_budget = max(
                    0.0,
                    state.remaining_budget - state.expected_expert_cost,
                )
            transition_state(
                state,
                CaseStatus.REVIEW_PENDING,
                event="await_human_review",
            )
        else:
            state.human_decision = expected_decision
            state.final_action = AgentAction.REFER_TO_HUMAN.value
            transition_state(
                state,
                CaseStatus.REVIEW_PENDING,
                event="expert_approval",
                code="EXPERT_REJECTED",
            )
        state.report["human_decision"] = state.human_decision
        state.report["expert_decision_idempotency_key"] = idempotency_key
        state.report["final_action"] = state.final_action
        state.report["completed_steps"] = list(state.completed_steps)
        state.report["tool_return_codes"] = list(
            state.tool_return_codes
        )
        state.report["expert_tool_code"] = (
            str(state.runtime_payload.get("expert", {}).get("code", ""))
            if approved
            else "NOT_EXECUTED"
        )
        state.report["expert_execution_mode"] = (
            "approved_frozen_replay" if approved else "not_executed"
        )
        state.report["remaining_budget"] = state.remaining_budget
        self.store.save(state)
        return state

    def confirm_review(
        self,
        case_id: str,
        *,
        decision: str,
        actor_role: str,
        idempotency_key: str = "",
    ) -> CaseState:
        authorize(actor_role, "review.confirm")
        with self.store.case_lock(case_id):
            return self._confirm_review_locked(
                case_id,
                decision=decision,
                idempotency_key=(
                    idempotency_key
                    or f"review-decision:{case_id}:{_sha256(decision)}"
                ),
            )

    def _confirm_review_locked(
        self,
        case_id: str,
        *,
        decision: str,
        idempotency_key: str,
    ) -> CaseState:
        state = self._required(case_id)
        if state.current_state not in {
            CaseStatus.REVIEW_PENDING,
            CaseStatus.BLOCKED,
        }:
            if (
                state.current_state == CaseStatus.CLOSED
                and state.human_decision == decision
                and state.report.get(
                    "review_decision_idempotency_key"
                )
                == idempotency_key
            ):
                return state
            raise StateTransitionError("REVIEW_DECISION_NOT_PENDING")
        state.human_decision = decision
        transition_state(
            state,
            CaseStatus.CLOSED,
            event="human_review",
            code="REVIEW_CONFIRMED",
        )
        state.report["human_decision"] = state.human_decision
        state.report["review_decision_idempotency_key"] = idempotency_key
        state.report["current_state"] = state.current_state
        self.store.save(state)
        return state

    def cancel_case(
        self,
        case_id: str,
        *,
        actor_role: str,
    ) -> CaseState:
        authorize(actor_role, "case.cancel")
        with self.store.case_lock(case_id):
            return self._cancel_case_locked(case_id)

    def _cancel_case_locked(self, case_id: str) -> CaseState:
        state = self._required(case_id)
        if CaseStatus(state.current_state) in TERMINAL_STATES:
            raise StateTransitionError("CASE_ALREADY_TERMINAL")
        transition_state(
            state,
            CaseStatus.CANCELLED,
            event="case_cancelled",
            code="CASE_CANCELLED",
        )
        state.final_action = AgentAction.REFER_TO_HUMAN.value
        state.report["current_state"] = state.current_state
        state.report["human_decision"] = "CASE_CANCELLED"
        self.store.save(state)
        return state

    def retry_failed(
        self,
        case_id: str,
        *,
        actor_role: str,
    ) -> CaseState:
        authorize(actor_role, "case.retry")
        with self.store.case_lock(case_id):
            return self._retry_failed_locked(case_id)

    def _retry_failed_locked(self, case_id: str) -> CaseState:
        state = self._required(case_id)
        if state.current_state != CaseStatus.FAILED:
            raise StateTransitionError("CASE_NOT_RETRYABLE")
        if not set(state.tool_return_codes).intersection(RETRYABLE_TOOL_CODES):
            raise StateTransitionError("FAILURE_CODE_NOT_RETRYABLE")
        if state.retry_count >= 2:
            raise StateTransitionError("RETRY_LIMIT_REACHED")
        if state.report.get("failure_stage") == "expert_approval":
            target = CaseStatus.EXPERT_PENDING_APPROVAL
        elif "scout" in state.completed_steps:
            target = CaseStatus.SCOUT_COMPLETED
        elif "model_qualification" in state.completed_steps:
            target = CaseStatus.CASE_VALIDATED
        else:
            target = CaseStatus.CASE_RECEIVED
        state.tool_return_codes = tuple(
            str(value)
            for value in state.report.get(
                "pre_failure_tool_return_codes",
                (),
            )
        )
        state.report.pop("failure_code", None)
        state.report.pop("failure_stage", None)
        state.report.pop("pre_failure_tool_return_codes", None)
        if target == CaseStatus.EXPERT_PENDING_APPROVAL:
            state.final_action = AgentAction.REQUEST_EXPERT.value
            state.report["final_action"] = state.final_action
        state.retry_count += 1
        transition_state(
            state,
            target,
            event="retry_failed_step",
            code="RETRY_ALLOWED",
        )
        self.store.save(state)
        return state

    def _required(self, case_id: str) -> CaseState:
        state = self.store.load(case_id)
        if state is None:
            raise FileNotFoundError(case_id)
        return state


def _first_failure_code(*responses: dict[str, Any]) -> str:
    for response in responses:
        if not response.get("ok", False):
            return str(response.get("code", "TOOL_EXECUTION_FAILED"))
    return "TOOL_EXECUTION_FAILED"


def _scout_response(tool_payload: dict[str, Any]) -> dict[str, Any]:
    predictions = tool_payload.get("predictions")
    if isinstance(predictions, list) and predictions:
        return {"ok": True, "code": "OK"}
    inference = tool_payload.get("inference")
    if isinstance(inference, dict):
        return inference
    route = tool_payload.get("route")
    if isinstance(route, dict) and route.get("ok"):
        return {"ok": True, "code": "OK"}
    return {"ok": False, "code": "TOOL_EXECUTION_FAILED"}


def _tool_codes(tool_payload: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for value in tool_payload.values():
        if isinstance(value, dict) and "ok" in value:
            codes.append(str(value.get("code", "OK")))
    return codes or ["OK"]


def _merge_tool_trace(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    if not existing:
        return incoming
    if not incoming or incoming == existing:
        return existing
    prior_events = [
        dict(event) for event in existing.get("events", [])
    ]
    new_events = [
        dict(event) for event in incoming.get("events", [])
    ]
    events = prior_events + new_events
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    return {
        "schema_version": str(
            existing.get(
                "schema_version",
                incoming.get("schema_version", ""),
            )
        ),
        "trace_id": str(
            existing.get("trace_id", incoming.get("trace_id", ""))
        ),
        "offline_mode": bool(
            existing.get(
                "offline_mode",
                incoming.get("offline_mode", True),
            )
        ),
        "git_commit": str(
            incoming.get("git_commit", existing.get("git_commit", ""))
        ),
        "events": events,
        "resume_segment_count": int(
            existing.get("resume_segment_count", 0)
        )
        + 1,
    }


def state_view_model(state: CaseState) -> dict[str, Any]:
    """Single ViewModel used by pages and reports."""

    status_labels = {
        CaseStatus.CASE_RECEIVED: "病例已接收",
        CaseStatus.CASE_VALIDATED: "病例校验完成",
        CaseStatus.SCOUT_COMPLETED: "初筛模型已完成",
        CaseStatus.RISK_AUDITED: "风险审计已完成",
        CaseStatus.EXPERT_PENDING_APPROVAL: "等待专家调用确认",
        CaseStatus.EXPERT_COMPLETED: "专家模型已完成",
        CaseStatus.REVIEW_PENDING: "等待人工确认",
        CaseStatus.CLOSED: "病例已完成",
        CaseStatus.BLOCKED: "流程已阻断",
        CaseStatus.FAILED: "工具执行失败",
        CaseStatus.CANCELLED: "流程已取消",
    }
    timeline = [
        {
            "sequence": event.get("sequence"),
            "state": event.get("to_state"),
            "label": status_labels.get(
                CaseStatus(str(event.get("to_state"))),
                str(event.get("to_state")),
            ),
            "code": event.get("code"),
            "at": event.get("at"),
        }
        for event in state.trace
        if event.get("to_state") in {item.value for item in CaseStatus}
    ]
    return {
        "case_id": state.case_id,
        "task_id": state.task_id,
        "current_state": state.current_state,
        "current_state_label": status_labels[CaseStatus(state.current_state)],
        "timeline": timeline,
        "controller_type": state.controller_type,
        "controller_proposal": state.controller_proposal,
        "gate_decision": state.gate_decision,
        "final_action": state.final_action,
        "human_decision": state.human_decision,
        "qualification": state.qualification,
        "tool_return_codes": list(state.tool_return_codes),
        "completed_steps": list(state.completed_steps),
        "remaining_budget": state.remaining_budget,
        "runtime_payload": dict(state.runtime_payload),
        "tool_trace": dict(state.tool_trace),
        "trace": list(state.trace),
        "report": dict(state.report),
    }
