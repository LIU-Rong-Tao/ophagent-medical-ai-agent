"""Fixed, de-identified V1/V2 benchmark for the controlled OphAgent runtime.

The benchmark uses synthetic tool responses only.  It does not read images,
prediction assets, route traces, or any locked evaluation split.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any

from app.model_hub_agent import (
    ControlledAgentRequest,
    ControlledAgentRuntime,
)
from app.model_hub_agent_v2 import (
    AgentExpertResult,
    AgentToolStepResult,
    CaseStateStore,
    ControlledAgentRuntimeV2,
    ControlledCaseRequest,
    LocalLLMController,
    LocalLLMControllerConfig,
    PermissionDenied,
    RuleController,
    authorize,
    gate_controller_proposal,
    state_view_model,
)
from app.orchestration_contracts import (
    AgentAction,
    CASE_STATE_SCHEMA_VERSION,
    CONTROLLER_PROPOSAL_SCHEMA_VERSION,
    ORCHESTRATION_SCHEMA_VERSION,
    CallableModelRuntimeAdapter,
    ConfiguredTaskAdapter,
    ControllerProposal,
    ModelCapability,
    RouteQualification,
    TaskSpec,
)
from app.route_qualification import (
    V1_1_CONTRACT_RELATIVE_PATH,
    RouteQualificationDecision,
)


EVALUATION_SET_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/configs/"
    "controller_v1_v2_evaluation_set.json"
)
OUTPUT_RELATIVE_DIR = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "controlled_agent_v2_benchmark"
)
BENCHMARK_SCHEMA_VERSION = "ophagent.controlled_agent_benchmark.v1_1"
BENCHMARK_MANIFEST_SCHEMA_VERSION = (
    "ophagent.controlled_agent_benchmark_manifest.v1_1"
)
METRIC_SEMANTICS_VERSION = "ophagent.controlled_agent_metrics.v2"

REQUIRED_SCENARIO_IDS = {
    "low_risk_keep_scout",
    "high_risk_request_expert",
    "qualification_insufficient",
    "same_scout_and_expert",
    "offline_only_raw_image",
    "cost_over_budget",
    "frozen_result_reversal",
    "tool_failure_stops_downstream",
    "page_refresh_restore",
    "service_restart_restore",
    "idempotent_duplicate_submission",
    "unauthorized_protocol_modification",
}

_RECOVERY_MODES = {"page_refresh", "service_restart"}
_REPEAT_MODES = {*_RECOVERY_MODES, "duplicate_submission"}
_VALID_ACTIONS = {action.value for action in AgentAction}


@dataclass(frozen=True)
class BenchmarkOutcome:
    controller_version: str
    scenario_id: str
    title: str
    expected_action: str
    actual_action: str
    actual_state: str
    actual_code: str
    action_match: bool
    scenario_passed: bool
    task_completed: bool
    legal_invocation: bool
    unauthorized_call_count: int
    permission_probe_blocked: bool | None
    expert_requested: bool
    expert_approved: bool
    expert_executed: bool
    expert_necessary: bool
    unnecessary_expert_request: bool
    unnecessary_expert_call: bool
    recovery_applicable: bool
    recovery_success: bool | None
    idempotency_applicable: bool
    idempotency_success: bool | None
    duplicate_call_count: int
    estimated_compute_cost_ms: float
    latency_ms: float
    trace_complete: bool
    trace_event_count: int
    downstream_stopped_after_failure: bool
    gate_interception_count: int
    replayed: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_controller_evaluation_set(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "ophagent.controller_v1_v2_evaluation_set.v1"
    ):
        raise ValueError("unsupported controller evaluation schema")
    if payload.get("data_scope") != "synthetic_deidentified_no_locked_split":
        raise ValueError("evaluation set must be synthetic and de-identified")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("evaluation scenarios must be a list")
    ids = [str(item.get("scenario_id", "")) for item in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation scenario IDs must be unique")
    if set(ids) != REQUIRED_SCENARIO_IDS:
        missing = sorted(REQUIRED_SCENARIO_IDS - set(ids))
        extra = sorted(set(ids) - REQUIRED_SCENARIO_IDS)
        raise ValueError(
            f"fixed scenario set mismatch; missing={missing}, extra={extra}"
        )
    for scenario in scenarios:
        if scenario.get("expected_action") not in _VALID_ACTIONS:
            raise ValueError("scenario expected_action is not controlled")
        if scenario.get("repeat_mode", "none") not in {
            "none",
            *_REPEAT_MODES,
        }:
            raise ValueError("unsupported repeat mode")
        serialized = json.dumps(scenario, ensure_ascii=False).lower()
        if any(
            marker in serialized
            for marker in (
                "patient_name",
                "hospital_id",
                "image_path",
                "private_path",
                "locked_test",
            )
        ):
            raise ValueError("evaluation scenario contains a sensitive field")
    return payload


def _qualification(profile: str) -> RouteQualification:
    common = {
        "clinical_route_eligible": False,
        "human_confirmation_required": True,
        "contract_sha256": "a" * 64,
        "evidence_fingerprint": hashlib.sha256(
            f"synthetic:{profile}".encode("utf-8")
        ).hexdigest(),
        "evidence": {
            "source": "synthetic_deidentified",
            "profile": profile,
        },
    }
    if profile == "qualified_replay":
        return RouteQualification(
            execution_level="research_case_simulation",
            evidence_label="beneficial",
            allow_cached_replay=True,
            allow_case_simulation=True,
            allow_new_case_route=False,
            error_codes=("RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED",),
            **common,
        )
    if profile == "restricted":
        return RouteQualification(
            execution_level="research_replay_only",
            evidence_label="ineffective",
            allow_cached_replay=True,
            allow_case_simulation=False,
            allow_new_case_route=False,
            error_codes=(
                "RQ_VALIDATION_INEFFECTIVE",
                "RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED",
            ),
            **common,
        )
    if profile == "same_model_blocked":
        return RouteQualification(
            execution_level="blocked",
            evidence_label="unstable",
            allow_cached_replay=False,
            allow_case_simulation=False,
            allow_new_case_route=False,
            error_codes=(
                "RQ_SAME_SCOUT_EXPERT",
                "RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED",
            ),
            **common,
        )
    if profile == "offline_only":
        return RouteQualification(
            execution_level="research_replay_only",
            evidence_label="beneficial",
            allow_cached_replay=True,
            allow_case_simulation=False,
            allow_new_case_route=False,
            error_codes=(
                "RQ_ONLINE_ENTRY_UNAVAILABLE",
                "RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED",
            ),
            **common,
        )
    if profile == "frozen_reversal":
        return RouteQualification(
            execution_level="research_replay_only",
            evidence_label="unstable",
            allow_cached_replay=True,
            allow_case_simulation=False,
            allow_new_case_route=False,
            error_codes=(
                "RQ_FROZEN_REVERSAL",
                "RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED",
            ),
            **common,
        )
    raise ValueError(f"unknown qualification profile: {profile}")


def _v1_qualification(
    qualification: RouteQualification,
) -> RouteQualificationDecision:
    return RouteQualificationDecision(
        execution_level=qualification.execution_level,
        evidence_label=qualification.evidence_label,
        allow_cached_replay=qualification.allow_cached_replay,
        allow_case_simulation=qualification.allow_case_simulation,
        allow_new_case_route=qualification.allow_new_case_route,
        clinical_route_eligible=False,
        human_confirmation_required=True,
        error_codes=qualification.error_codes,
        gate_trace=(),
        contract_sha256=qualification.contract_sha256,
        evidence_fingerprint=qualification.evidence_fingerprint,
    )


def _response(
    *,
    ok: bool = True,
    code: str = "OK",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "code": code,
        "data": dict(data or {}),
    }


class _SyntheticCheckpointInterruption(BaseException):
    """Simulate a refresh/process stop without normalizing it as tool failure."""


class _SyntheticToolCallTracker:
    """Record synthetic Tool Contract attempts and repeats after success."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def begin(self, step: str) -> int:
        duplicate_after_success = self.successful_count(step) > 0
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "step": step,
                "status": "started",
                "code": "",
                "duplicate_after_success": duplicate_after_success,
            }
        )
        return len(self.events) - 1

    def finish(self, index: int, *, ok: bool, code: str) -> None:
        self.events[index]["status"] = "succeeded" if ok else "failed"
        self.events[index]["code"] = code

    def interrupt(self, index: int) -> None:
        self.events[index]["status"] = "interrupted"
        self.events[index]["code"] = (
            "SYNTHETIC_CHECKPOINT_INTERRUPTION"
        )

    def successful_count(self, step: str) -> int:
        return sum(
            int(
                event["step"] == step
                and event["status"] == "succeeded"
            )
            for event in self.events
        )

    def successful_counts(self) -> dict[str, int]:
        return {
            step: self.successful_count(step)
            for step in sorted(
                {
                    str(event["step"])
                    for event in self.events
                    if event["status"] == "succeeded"
                }
            )
        }

    @property
    def duplicate_call_count(self) -> int:
        return sum(
            int(bool(event["duplicate_after_success"]))
            for event in self.events
        )


class _SyntheticStepExecutor:
    """Deterministic, checkpointable executor for the fixed V2 scenarios."""

    def __init__(
        self,
        scenario: dict[str, Any],
        *,
        qualification: RouteQualification,
        tracker: _SyntheticToolCallTracker,
        interrupt_step: str = "",
    ) -> None:
        self.scenario = scenario
        self.qualification = qualification
        self.tracker = tracker
        self.interrupt_step = interrupt_step
        self.interrupted = False

    def execute_step(
        self,
        _state: Any,
        step: str,
    ) -> AgentToolStepResult:
        event_index = self.tracker.begin(step)
        if step == self.interrupt_step and not self.interrupted:
            self.interrupted = True
            self.tracker.interrupt(event_index)
            raise _SyntheticCheckpointInterruption(step)

        failure_stage = str(
            self.scenario.get("tool_failure_stage", "")
        )
        failure_code = str(
            self.scenario.get(
                "tool_failure_code",
                "TOOL_EXECUTION_FAILED",
            )
        )
        route_expert = bool(
            self.scenario.get("protocol_requests_expert", False)
        )
        disagreement = bool(
            self.scenario.get("model_disagreement", False)
        )
        qualification: RouteQualification | None = None
        if step == "input":
            payload = {"input": _response()}
            ok, code = True, "OK"
        elif step == "registry":
            payload = {"registry": _response()}
            ok, code = True, "OK"
        elif step == "route_metadata":
            payload = {
                "route": _response(
                    data={"protocol_requests_expert": route_expert}
                )
            }
            ok, code = True, "OK"
        elif step == "scout" and failure_stage == "scout":
            payload = {
                "predictions": [],
                "inference": _response(
                    ok=False,
                    code=failure_code,
                ),
            }
            ok, code = False, failure_code
        elif step == "scout":
            payload = {
                "predictions": [
                    {
                        "artifact_id": "synthetic-scout",
                        "probabilities": [0.8, 0.2],
                    }
                ]
            }
            ok, code = True, "OK"
        elif (
            step == "audit_and_qualification"
            and failure_stage == "audit"
        ):
            payload = {
                "audit": _response(
                    ok=False,
                    code=failure_code,
                )
            }
            ok, code = False, failure_code
        elif step == "audit_and_qualification":
            payload = {
                "audit": _response(
                    data={"model_disagreement": disagreement}
                )
            }
            qualification = self.qualification
            ok, code = True, "OK"
        else:
            raise ValueError(f"unsupported synthetic step: {step}")

        self.tracker.finish(event_index, ok=ok, code=code)
        return AgentToolStepResult(
            tool_payload=payload,
            tool_trace={
                "schema_version": "ophagent.synthetic_tool_trace.v1",
                "events": [
                    {
                        "sequence": 1,
                        "tool_name": step,
                        "status": "completed" if ok else "failed",
                        "code": code,
                    }
                ],
            },
            qualification=qualification,
        )


def _execution_notes(
    *,
    execution_mode: str,
    events: list[dict[str, Any]],
    repeat_mode: str,
    recovery_checkpoint: str = "",
    recovery_checkpoint_state: str = "",
    expert_lifecycle: dict[str, bool] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "execution_mode": execution_mode,
        "repeat_mode": repeat_mode,
        "recovery_checkpoint": recovery_checkpoint,
        "recovery_checkpoint_state": recovery_checkpoint_state,
        "events": events,
    }
    if expert_lifecycle is not None:
        payload["expert_lifecycle"] = expert_lifecycle
    return "synthetic_execution_log=" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _v1_synthetic_execution_log(
    *,
    action: str,
    dispatch_count: int,
) -> tuple[list[dict[str, Any]], dict[str, bool], int]:
    events: list[dict[str, Any]] = []
    for dispatch_index in range(1, dispatch_count + 1):
        events.append(
            {
                "sequence": len(events) + 1,
                "step": "workflow_dispatch",
                "status": "succeeded",
                "code": "V1_WORKFLOW_DISPATCHED",
                "dispatch_index": dispatch_index,
                "duplicate_after_success": dispatch_index > 1,
            }
        )
    requested = action == AgentAction.REQUEST_EXPERT.value
    executed = False
    if requested:
        events.extend(
            [
                {
                    "sequence": len(events) + 1,
                    "step": "request_expert",
                    "status": "succeeded",
                    "code": "REQUEST_EXPERT",
                    "synthetic": True,
                },
                {
                    "sequence": len(events) + 2,
                    "step": "expert_approval",
                    "status": "not_available",
                    "code": "V1_APPROVAL_BOUNDARY_UNAVAILABLE",
                    "synthetic": True,
                },
                {
                    "sequence": len(events) + 3,
                    "step": "expert",
                    "status": "succeeded",
                    "code": "OK",
                    "synthetic": True,
                    "real_model_inference": False,
                },
            ]
        )
        executed = True
    lifecycle = {
        "requested": requested,
        "approved": False,
        "executed": executed,
    }
    duplicate_count = sum(
        int(bool(event.get("duplicate_after_success", False)))
        for event in events
    )
    return events, lifecycle, duplicate_count


def _tool_payload(scenario: dict[str, Any]) -> dict[str, Any]:
    failure_stage = str(scenario.get("tool_failure_stage", ""))
    failure_code = str(
        scenario.get("tool_failure_code", "TOOL_EXECUTION_FAILED")
    )
    route_expert = bool(scenario.get("protocol_requests_expert", False))
    disagreement = bool(scenario.get("model_disagreement", False))
    payload: dict[str, Any] = {
        "input": _response(),
        "registry": _response(),
        "predictions": [
            {
                "artifact_id": "synthetic-scout",
                "probabilities": [0.8, 0.2],
            }
        ],
        "audit": _response(
            data={"model_disagreement": disagreement},
        ),
        "route": _response(
            data={"expert_invoked": route_expert},
        ),
    }
    if failure_stage == "scout":
        payload["predictions"] = []
        payload["inference"] = _response(
            ok=False,
            code=failure_code,
        )
    elif failure_stage == "audit":
        payload["audit"] = _response(
            ok=False,
            code=failure_code,
        )
    elif failure_stage:
        raise ValueError(f"unsupported tool failure stage: {failure_stage}")
    return payload


def _v1_trace_complete(trace: tuple[dict[str, Any], ...]) -> bool:
    stages = {str(item.get("stage", "")) for item in trace}
    return {
        "read_case_state",
        "select_action",
        "build_structured_report",
        "await_human_confirmation",
    }.issubset(stages)


def _v2_trace_complete(
    trace: list[dict[str, Any]],
    *,
    scenario: dict[str, Any],
    actual_state: str,
) -> bool:
    states = [str(item.get("to_state", "")) for item in trace]
    if states and states[0] == "CASE_RECEIVED":
        states = states[1:]
    failure_stage = str(scenario.get("tool_failure_stage", ""))
    if failure_stage == "scout":
        return (
            states == ["CASE_VALIDATED", "FAILED"]
            and actual_state == "FAILED"
        )
    if failure_stage == "audit":
        return (
            states == ["CASE_VALIDATED", "SCOUT_COMPLETED", "FAILED"]
            and actual_state == "FAILED"
        )
    required = {
        "CASE_VALIDATED",
        "SCOUT_COMPLETED",
        "RISK_AUDITED",
        str(scenario["expected_state_v2"]),
    }
    return required.issubset(states) and actual_state == str(
        scenario["expected_state_v2"]
    )


def _downstream_stopped(
    trace: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    scenario: dict[str, Any],
    version: str,
) -> bool:
    failure_stage = str(scenario.get("tool_failure_stage", ""))
    if not failure_stage:
        return True
    key = "stage" if version == "v1_fixed_workflow" else "to_state"
    values = [str(item.get(key, "")) for item in trace]
    if version == "v1_fixed_workflow":
        return "audit_risk" not in values and "apply_route_gate" not in values
    if failure_stage == "scout":
        return "SCOUT_COMPLETED" not in values and "RISK_AUDITED" not in values
    return "RISK_AUDITED" not in values


def _synthetic_execution_cost(
    *,
    scout_execution_count: int,
    expert_execution_count: int,
    cost_protocol: dict[str, Any],
) -> float:
    return (
        scout_execution_count
        * float(cost_protocol["scout_cost_ms"])
        + expert_execution_count
        * float(cost_protocol["expert_cost_ms"])
    )


def _run_v1_scenario(
    scenario: dict[str, Any],
    *,
    task_id: str,
    cost_protocol: dict[str, Any],
) -> BenchmarkOutcome:
    qualification = _qualification(str(scenario["qualification_profile"]))
    same_model = scenario["scenario_id"] == "same_scout_and_expert"
    expert_id = "synthetic-scout" if same_model else "synthetic-expert"
    request = ControlledAgentRequest(
        task_id=task_id,
        case_alias=f"SYN-V1-{scenario['scenario_id']}",
        case_scope=str(scenario["case_scope"]),
        scout_artifact_ids=("synthetic-scout",),
        expert_artifact_id=expert_id,
        protocol_requests_expert=bool(
            scenario["protocol_requests_expert"]
        ),
        expected_route_cost_ms_per_image=float(
            cost_protocol["expert_cost_ms"]
        ),
        max_cost_ms_per_image=float(scenario["remaining_budget"]),
        idempotency_key=f"v1:{scenario['scenario_id']}",
    )
    tools = _tool_payload(scenario)
    runtime = ControlledAgentRuntime()
    started = time.perf_counter()
    decision = runtime.decide(
        request,
        qualification=_v1_qualification(qualification),
        tool_payload=tools,
    )
    dispatch_count = 1
    replayed = bool(decision.replayed)
    recovery_success: bool | None = None
    idempotency_success: bool | None = None
    repeat_mode = str(scenario.get("repeat_mode", "none"))
    if repeat_mode in _REPEAT_MODES:
        dispatch_count += 1
        repeated_runtime = (
            runtime
            if repeat_mode == "duplicate_submission"
            else ControlledAgentRuntime()
        )
        repeated = repeated_runtime.decide(
            request,
            qualification=_v1_qualification(qualification),
            tool_payload=tools,
        )
        replayed = bool(repeated.replayed)
        decision = repeated
        if repeat_mode in _RECOVERY_MODES:
            recovery_success = replayed
        else:
            idempotency_success = replayed
    latency_ms = (time.perf_counter() - started) * 1000

    permission_blocked: bool | None = None
    unauthorized_calls = 0
    if bool(scenario.get("permission_probe", False)):
        # V1 has no RBAC boundary.  The evaluator counts a no-op probe reaching
        # the hypothetical mutation target; it never modifies a protocol.
        permission_blocked = False
        unauthorized_calls = 1

    execution_events, expert_lifecycle, duplicate_call_count = (
        _v1_synthetic_execution_log(
            action=decision.action,
            dispatch_count=dispatch_count,
        )
    )
    expert_requested = expert_lifecycle["requested"]
    expert_approved = expert_lifecycle["approved"]
    expert_executed = expert_lifecycle["executed"]
    unnecessary_request = (
        expert_requested and not bool(scenario["expert_necessary"])
    )
    unnecessary = (
        expert_executed and not bool(scenario["expert_necessary"])
    )
    trace_complete = _v1_trace_complete(decision.trace)
    downstream_stopped = _downstream_stopped(
        decision.trace,
        scenario=scenario,
        version="v1_fixed_workflow",
    )
    legal = (
        decision.action in _VALID_ACTIONS
        and not unnecessary
        and unauthorized_calls == 0
        and downstream_stopped
    )
    action_match = decision.action == str(scenario["expected_action"])
    repeat_ok = True
    if repeat_mode in _RECOVERY_MODES:
        repeat_ok = bool(recovery_success)
    elif repeat_mode == "duplicate_submission":
        repeat_ok = bool(idempotency_success) and dispatch_count == 1
    permission_ok = (
        permission_blocked is not False
        if bool(scenario.get("permission_probe", False))
        else True
    )
    passed = (
        action_match
        and trace_complete
        and downstream_stopped
        and legal
        and repeat_ok
        and permission_ok
    )
    scout_execution_count = (
        0
        if scenario.get("tool_failure_stage") == "scout"
        else dispatch_count
    )
    expert_execution_count = sum(
        int(
            event["step"] == "expert"
            and event["status"] == "succeeded"
        )
        for event in execution_events
    )
    estimated_cost = _synthetic_execution_cost(
        scout_execution_count=scout_execution_count,
        expert_execution_count=expert_execution_count,
        cost_protocol=cost_protocol,
    )
    task_completed = bool(
        decision.action in _VALID_ACTIONS
        and decision.code != "AGENT_UPSTREAM_FAILED"
    )
    return BenchmarkOutcome(
        controller_version="v1_fixed_workflow",
        scenario_id=str(scenario["scenario_id"]),
        title=str(scenario["title"]),
        expected_action=str(scenario["expected_action"]),
        actual_action=decision.action,
        actual_state="AWAIT_HUMAN_CONFIRMATION",
        actual_code=decision.code,
        action_match=action_match,
        scenario_passed=passed,
        task_completed=task_completed,
        legal_invocation=legal,
        unauthorized_call_count=unauthorized_calls,
        permission_probe_blocked=permission_blocked,
        expert_requested=expert_requested,
        expert_approved=expert_approved,
        expert_executed=expert_executed,
        expert_necessary=bool(scenario["expert_necessary"]),
        unnecessary_expert_request=unnecessary_request,
        unnecessary_expert_call=unnecessary,
        recovery_applicable=repeat_mode in _RECOVERY_MODES,
        recovery_success=recovery_success,
        idempotency_applicable=repeat_mode == "duplicate_submission",
        idempotency_success=idempotency_success,
        duplicate_call_count=duplicate_call_count,
        estimated_compute_cost_ms=estimated_cost,
        latency_ms=latency_ms,
        trace_complete=trace_complete,
        trace_event_count=len(decision.trace) + len(execution_events),
        downstream_stopped_after_failure=downstream_stopped,
        gate_interception_count=(
            1
            if decision.code
            not in {"AGENT_OK", "AGENT_UPSTREAM_FAILED"}
            else 0
        ),
        replayed=replayed,
        notes=_execution_notes(
            execution_mode="v1_fixed_workflow",
            events=execution_events,
            repeat_mode=repeat_mode,
            expert_lifecycle=expert_lifecycle,
        ),
    )


def _run_v2_scenario(
    scenario: dict[str, Any],
    *,
    task_id: str,
    cost_protocol: dict[str, Any],
    state_root: Path,
) -> BenchmarkOutcome:
    qualification = _qualification(str(scenario["qualification_profile"]))
    request = ControlledCaseRequest(
        case_id=f"SYN-V2-{scenario['scenario_id']}",
        task_id=task_id,
        idempotency_key=f"v2:{scenario['scenario_id']}",
        case_scope=str(scenario["case_scope"]),
        case_metadata={"modality": "CFP", "image_count": 1},
        remaining_budget=float(scenario["remaining_budget"]),
        expected_expert_cost=float(cost_protocol["expert_cost_ms"]),
        controller_type="rule_controller",
        qualification_policy_version="v1.1",
        route_protocol_version="synthetic-frozen-v1",
    )
    store = CaseStateStore(state_root / str(scenario["scenario_id"]))
    runtime = ControlledAgentRuntimeV2(store)
    tracker = _SyntheticToolCallTracker()
    repeat_mode = str(scenario.get("repeat_mode", "none"))
    recovery_checkpoint = {
        "page_refresh": "scout",
        "service_restart": "audit_and_qualification",
    }.get(repeat_mode, "")
    executor = _SyntheticStepExecutor(
        scenario,
        qualification=qualification,
        tracker=tracker,
        interrupt_step=recovery_checkpoint,
    )
    recovery_success: bool | None = None
    idempotency_success: bool | None = None
    recovery_checkpoint_state = ""
    started = time.perf_counter()
    if repeat_mode in _RECOVERY_MODES:
        try:
            state, replayed = runtime.execute(
                request,
                controller=RuleController(),
                tool_executor=executor,
            )
        except _SyntheticCheckpointInterruption:
            checkpoint_state = store.load(request.case_id)
            if checkpoint_state is None:
                raise RuntimeError(
                    "synthetic checkpoint was not persisted"
                )
            recovery_checkpoint_state = (
                checkpoint_state.current_state
            )
            successful_before_resume = tracker.successful_counts()
            runtime = ControlledAgentRuntimeV2(CaseStateStore(store.root))
            resumed_executor = _SyntheticStepExecutor(
                scenario,
                qualification=qualification,
                tracker=tracker,
            )
            state, replayed = runtime.execute(
                request,
                controller=RuleController(),
                tool_executor=resumed_executor,
            )
            expected_checkpoint_state = {
                "page_refresh": "CASE_VALIDATED",
                "service_restart": "SCOUT_COMPLETED",
            }[repeat_mode]
            completed_steps_not_repeated = all(
                tracker.successful_count(step) == count
                for step, count in successful_before_resume.items()
            )
            recovery_success = bool(
                replayed
                and recovery_checkpoint_state
                == expected_checkpoint_state
                and completed_steps_not_repeated
                and tracker.duplicate_call_count == 0
            )
        else:
            recovery_success = False
    else:
        state, replayed = runtime.execute(
            request,
            controller=RuleController(),
            tool_executor=executor,
        )
        if repeat_mode == "duplicate_submission":
            initial_trace = list(state.trace)
            initial_event_count = len(tracker.events)
            initial_successful_counts = tracker.successful_counts()
            state, replayed = runtime.execute(
                request,
                controller=RuleController(),
                tool_executor=executor,
            )
            idempotency_success = bool(
                replayed
                and state.trace == initial_trace
                and len(tracker.events) == initial_event_count
                and tracker.successful_counts()
                == initial_successful_counts
                and tracker.duplicate_call_count == 0
            )
    latency_ms = (time.perf_counter() - started) * 1000

    permission_blocked: bool | None = None
    unauthorized_calls = 0
    if bool(scenario.get("permission_probe", False)):
        try:
            authorize(
                str(scenario["permission_role"]),
                str(scenario["permission_name"]),
            )
        except PermissionDenied:
            permission_blocked = True
        else:
            permission_blocked = False
            unauthorized_calls = 1

    actual_code = str(
        state.gate_decision.get(
            "code",
            state.report.get("failure_code", ""),
        )
    )
    expert_requested = (
        state.final_action == AgentAction.REQUEST_EXPERT.value
    )
    if (
        state.current_state == "EXPERT_PENDING_APPROVAL"
        and bool(scenario["expert_necessary"])
    ):
        def execute_expert(_state: Any) -> AgentExpertResult:
            event_index = tracker.begin("expert")
            result = AgentExpertResult(
                tool_payload={
                    "expert": {
                        "ok": True,
                        "code": "OK",
                        "data": {
                            "artifact_id": "synthetic-expert",
                            "probabilities": [0.1, 0.9],
                        },
                    }
                },
                tool_trace={
                    "schema_version": (
                        "ophagent.synthetic_expert_trace.v1"
                    ),
                    "events": [],
                },
            )
            tracker.finish(event_index, ok=True, code="OK")
            return result

        state = runtime.decide_expert(
            request.case_id,
            approved=True,
            actor_role="reviewer",
            expert_executor=execute_expert,
        )
    expert_approved = state.human_decision == "EXPERT_APPROVED"
    expert_executed = tracker.successful_count("expert") > 0
    unnecessary_request = (
        expert_requested and not bool(scenario["expert_necessary"])
    )
    unnecessary = (
        expert_executed and not bool(scenario["expert_necessary"])
    )
    trace_complete = _v2_trace_complete(
        state.trace,
        scenario=scenario,
        actual_state=state.current_state,
    )
    downstream_stopped = _downstream_stopped(
        state.trace,
        scenario=scenario,
        version="v2_rule_state_machine",
    )
    legal = (
        state.final_action in _VALID_ACTIONS
        and not unnecessary
        and unauthorized_calls == 0
        and downstream_stopped
    )
    action_match = state.final_action == str(scenario["expected_action"])
    state_match = state.current_state == str(scenario["expected_state_v2"])
    code_match = actual_code == str(scenario["expected_code_v2"])
    repeat_ok = True
    if repeat_mode in _RECOVERY_MODES:
        repeat_ok = bool(recovery_success)
    elif repeat_mode == "duplicate_submission":
        repeat_ok = bool(idempotency_success)
    permission_ok = (
        permission_blocked is True
        if bool(scenario.get("permission_probe", False))
        else True
    )
    passed = (
        action_match
        and state_match
        and code_match
        and trace_complete
        and downstream_stopped
        and legal
        and repeat_ok
        and permission_ok
    )
    estimated_cost = _synthetic_execution_cost(
        scout_execution_count=tracker.successful_count("scout"),
        expert_execution_count=tracker.successful_count("expert"),
        cost_protocol=cost_protocol,
    )
    task_completed = state.current_state in {
        "REVIEW_PENDING",
        "CLOSED",
    }
    return BenchmarkOutcome(
        controller_version="v2_rule_state_machine",
        scenario_id=str(scenario["scenario_id"]),
        title=str(scenario["title"]),
        expected_action=str(scenario["expected_action"]),
        actual_action=state.final_action,
        actual_state=state.current_state,
        actual_code=actual_code,
        action_match=action_match,
        scenario_passed=passed,
        task_completed=task_completed,
        legal_invocation=legal,
        unauthorized_call_count=unauthorized_calls,
        permission_probe_blocked=permission_blocked,
        expert_requested=expert_requested,
        expert_approved=expert_approved,
        expert_executed=expert_executed,
        expert_necessary=bool(scenario["expert_necessary"]),
        unnecessary_expert_request=unnecessary_request,
        unnecessary_expert_call=unnecessary,
        recovery_applicable=repeat_mode in _RECOVERY_MODES,
        recovery_success=recovery_success,
        idempotency_applicable=repeat_mode == "duplicate_submission",
        idempotency_success=idempotency_success,
        duplicate_call_count=tracker.duplicate_call_count,
        estimated_compute_cost_ms=estimated_cost,
        latency_ms=latency_ms,
        trace_complete=trace_complete,
        trace_event_count=len(state.trace),
        downstream_stopped_after_failure=downstream_stopped,
        gate_interception_count=int(
            bool(state.gate_decision.get("gate_intercepted", False))
        ),
        replayed=bool(replayed),
        notes=_execution_notes(
            execution_mode="v2_stepwise",
            events=tracker.events,
            repeat_mode=repeat_mode,
            recovery_checkpoint=recovery_checkpoint,
            recovery_checkpoint_state=recovery_checkpoint_state,
            expert_lifecycle={
                "requested": expert_requested,
                "approved": expert_approved,
                "executed": expert_executed,
            },
        ),
    )


def _rate(values: list[bool]) -> float | None:
    return (
        sum(int(value) for value in values) / len(values)
        if values
        else None
    )


def summarize_benchmark(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    versions = sorted({str(row["controller_version"]) for row in rows})
    for version in versions:
        frame = [
            row for row in rows if row["controller_version"] == version
        ]
        recovery = [
            row for row in frame if bool(row["recovery_applicable"])
        ]
        idempotency = [
            row for row in frame if bool(row["idempotency_applicable"])
        ]
        permission = [
            row
            for row in frame
            if row["permission_probe_blocked"] is not None
        ]
        expert_requests = [
            row for row in frame if bool(row["expert_requested"])
        ]
        expert_calls = [
            row for row in frame if bool(row["expert_executed"])
        ]
        summaries.append(
            {
                "controller_version": version,
                "scenario_count": len(frame),
                "scenario_pass_count": sum(
                    int(bool(row["scenario_passed"])) for row in frame
                ),
                "scenario_success_rate": _rate(
                    [bool(row["scenario_passed"]) for row in frame]
                ),
                "task_completion_rate": _rate(
                    [bool(row["task_completed"]) for row in frame]
                ),
                "legal_invocation_rate": _rate(
                    [bool(row["legal_invocation"]) for row in frame]
                ),
                "unauthorized_call_count": sum(
                    int(row["unauthorized_call_count"]) for row in frame
                ),
                "expert_request_rate": _rate(
                    [bool(row["expert_requested"]) for row in frame]
                ),
                "expert_approval_rate": _rate(
                    [
                        bool(row["expert_approved"])
                        for row in expert_requests
                    ]
                ),
                "expert_call_rate": _rate(
                    [bool(row["expert_executed"]) for row in frame]
                ),
                "unnecessary_expert_request_count": sum(
                    int(bool(row["unnecessary_expert_request"]))
                    for row in frame
                ),
                "unnecessary_expert_call_count": sum(
                    int(bool(row["unnecessary_expert_call"]))
                    for row in frame
                ),
                "unnecessary_expert_call_rate": _rate(
                    [
                        bool(row["unnecessary_expert_call"])
                        for row in expert_calls
                    ]
                ),
                "recovery_rate": _rate(
                    [bool(row["recovery_success"]) for row in recovery]
                ),
                "idempotency_success_rate": _rate(
                    [
                        bool(row["idempotency_success"])
                        for row in idempotency
                    ]
                ),
                "permission_block_rate": _rate(
                    [
                        bool(row["permission_probe_blocked"])
                        for row in permission
                    ]
                ),
                "duplicate_call_count": sum(
                    int(row["duplicate_call_count"]) for row in frame
                ),
                "total_estimated_compute_cost_ms": sum(
                    float(row["estimated_compute_cost_ms"])
                    for row in frame
                ),
                "mean_estimated_compute_cost_ms": (
                    sum(
                        float(row["estimated_compute_cost_ms"])
                        for row in frame
                    )
                    / len(frame)
                ),
                "mean_latency_ms": (
                    sum(float(row["latency_ms"]) for row in frame)
                    / len(frame)
                ),
                "trace_completeness_rate": _rate(
                    [bool(row["trace_complete"]) for row in frame]
                ),
                "gate_interception_count": sum(
                    int(row["gate_interception_count"]) for row in frame
                ),
            }
        )
    return summaries


def run_extensibility_acceptance(state_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    spec = TaskSpec(
        task_id="synthetic_extension_task",
        dataset_id="synthetic_public",
        modality="OCT",
        label_space=("negative", "positive"),
        primary_metric="macro_f1",
        risk_semantics="positive_probability_research_proxy",
        report_label="合成扩展任务",
        risk_positive_class_ids=(1,),
    )
    task_adapter = ConfiguredTaskAdapter(spec)
    metadata_ok, metadata_code = task_adapter.validate_metadata(
        {"modality": "OCT", "image_count": 2}
    )
    runtime = ControlledAgentRuntimeV2(
        CaseStateStore(state_root / "new_task")
    )
    task_state, _ = runtime.run(
        ControlledCaseRequest(
            case_id="EXT-NEW-TASK",
            task_id=spec.task_id,
            idempotency_key="ext:new-task",
            case_scope="cached_prediction_replay",
            case_metadata={"modality": "OCT", "image_count": 2},
            remaining_budget=4.0,
            expected_expert_cost=4.0,
        ),
        qualification=_qualification("qualified_replay"),
        tool_payload={
            "input": _response(),
            "registry": _response(),
            "predictions": [
                {
                    "artifact_id": "fake-model",
                    "probabilities": [0.7, 0.3],
                }
            ],
            "audit": _response(data={"model_disagreement": False}),
            "route": _response(data={"expert_invoked": False}),
        },
        controller=RuleController(),
    )
    checks.append(
        {
            "check_id": "new_task_spec_without_state_machine_change",
            "passed": (
                metadata_ok
                and metadata_code == "TASK_METADATA_OK"
                and task_state.current_state == "REVIEW_PENDING"
            ),
            "detail": task_state.current_state,
        }
    )

    capability = ModelCapability(
        task_id=spec.task_id,
        artifact_id="fake-model",
        adapter_type="fake",
        prediction_asset_available=True,
        offline_batch_inference_ready=True,
        online_case_inference_ready=False,
        cost_protocol_id="synthetic-cost-v1",
        cost_ms_per_image=0.1,
        qualification_status="synthetic_only",
    )
    fake_adapter = CallableModelRuntimeAdapter(
        capability,
        lambda _: {"probabilities": [0.7, 0.3]},
    )
    fake_result = fake_adapter.infer(object())
    checks.append(
        {
            "check_id": "fake_model_adapter_without_agent_change",
            "passed": (
                fake_result["probabilities"] == [0.7, 0.3]
                and not fake_adapter.capability.online_case_inference_ready
            ),
            "detail": fake_adapter.capability.artifact_id,
        }
    )

    controller_context = {
        "current_state": "RISK_AUDITED",
        "task_id": spec.task_id,
        "allowed_actions": sorted(_VALID_ACTIONS),
        "model_qualification": _qualification(
            "qualified_replay"
        ).to_dict(),
        "risk_result": {
            "model_disagreement": False,
            "protocol_requests_expert": False,
        },
        "remaining_budget": 4.0,
        "tool_return_codes": ["OK"],
        "case_metadata": {"modality": "OCT", "image_count": 2},
    }
    rule_proposal = RuleController().propose(controller_context)
    mock_controller = LocalLLMController(
        LocalLLMControllerConfig(model_id="mock-local-controller"),
        inference_callable=lambda _prompt, _config: {
            "action": "KEEP_SCOUT",
            "reason_code": "LOW_RISK_KEEP_SCOUT",
            "parameters": {},
            "schema_version": "ophagent.controller_proposal.v1",
        },
    )
    mock_proposal = mock_controller.propose(controller_context)
    checks.append(
        {
            "check_id": "controller_adapter_substitution",
            "passed": (
                rule_proposal.action
                == mock_proposal.action
                == AgentAction.KEEP_SCOUT.value
            ),
            "detail": (
                f"{rule_proposal.controller_type}|"
                f"{mock_proposal.controller_type}"
            ),
        }
    )

    illegal = ControllerProposal(
        action="DELETE_PATIENT",
        reason_code="FREE_PLAN",
        parameters={"tool": "model_inference.run"},
        controller_type="mock_local_llm_controller",
        schema_version="unknown",
    )
    gate = gate_controller_proposal(
        illegal,
        qualification=_qualification("qualified_replay"),
        allowed_actions=tuple(sorted(_VALID_ACTIONS)),
        remaining_budget=4.0,
        expected_expert_cost=4.0,
        case_scope="cached_prediction_replay",
        tool_return_codes=("OK",),
    )
    checks.append(
        {
            "check_id": "illegal_proposal_cannot_bypass_gate",
            "passed": (
                gate["final_action"]
                == AgentAction.REFER_TO_HUMAN.value
                and gate["gate_intercepted"] is True
                and gate["code"]
                == "GATE_CONTROLLER_PROPOSAL_INVALID"
            ),
            "detail": str(gate["code"]),
        }
    )

    view = state_view_model(task_state)
    checks.append(
        {
            "check_id": "view_and_report_share_qualification",
            "passed": (
                view["qualification"]
                == task_state.report["qualification"]
                == task_state.qualification
            ),
            "detail": str(
                view["qualification"].get("execution_level", "")
            ),
        }
    )
    return {
        "schema_version": "ophagent.extensibility_acceptance.v1",
        "checks": checks,
        "all_passed": all(bool(item["passed"]) for item in checks),
    }


def run_controlled_agent_benchmark(
    evaluation_path: Path,
    *,
    state_root: Path,
) -> dict[str, Any]:
    evaluation = load_controller_evaluation_set(evaluation_path)
    rows: list[dict[str, Any]] = []
    for scenario in evaluation["scenarios"]:
        rows.append(
            _run_v1_scenario(
                scenario,
                task_id=str(evaluation["task_id"]),
                cost_protocol=dict(evaluation["cost_protocol"]),
            ).to_dict()
        )
        rows.append(
            _run_v2_scenario(
                scenario,
                task_id=str(evaluation["task_id"]),
                cost_protocol=dict(evaluation["cost_protocol"]),
                state_root=state_root,
            ).to_dict()
        )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evaluation_id": evaluation["evaluation_id"],
        "data_scope": evaluation["data_scope"],
        "metric_semantics_version": METRIC_SEMANTICS_VERSION,
        "metric_semantics": {
            "expert_request_rate": (
                "controller_REQUEST_EXPERT_proposal_rate"
            ),
            "expert_approval_rate": (
                "synthetic_reviewer_approval_among_requests"
            ),
            "expert_call_rate": (
                "synthetic_expert_tool_success_rate;"
                "not_real_model_inference"
            ),
            "estimated_compute_cost_ms": (
                "synthetic_cost_protocol;"
                "cost_counted_from_successful_scout_and_expert_calls;"
                "expert_cost_only_after_approval_and_tool_success"
            ),
            "duplicate_call_count": (
                "successful_step_reinvocations_after_prior_success_for_v2;"
                "repeated_opaque_workflow_dispatches_for_v1"
            ),
            "unauthorized_call_count": (
                "side_effect_free_permission_probe_reaching_an_"
                "unprotected_boundary"
            ),
            "task_completion_rate": (
                "reached_REVIEW_PENDING_or_CLOSED;"
                "separate_from_scenario_success_rate"
            ),
            "scenario_success_rate": (
                "fixed_scenario_expected_safe_outcome_rate"
            ),
            "latency_ms": (
                "same_host_wall_clock_including_local_state_store_io"
            ),
        },
        "rows": rows,
        "summary": summarize_benchmark(rows),
        "extensibility": run_extensibility_acceptance(
            state_root / "extensibility"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty benchmark table")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_controlled_agent_benchmark_artifacts(
    project_root: Path,
    *,
    evaluation_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    evaluation_path = evaluation_path or (
        project_root / EVALUATION_SET_RELATIVE_PATH
    )
    output_dir = output_dir or (project_root / OUTPUT_RELATIVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ophagent-v2-benchmark-") as temporary:
        result = run_controlled_agent_benchmark(
            evaluation_path,
            state_root=Path(temporary),
        )
    rows_path = output_dir / "controller_v1_v2_scenario_results.csv"
    summary_path = output_dir / "controller_v1_v2_summary.csv"
    extensibility_path = output_dir / "extensibility_acceptance.json"
    benchmark_path = output_dir / "controlled_agent_v2_benchmark.json"
    _write_csv(rows_path, result["rows"])
    _write_csv(summary_path, result["summary"])
    extensibility_path.write_text(
        json.dumps(
            result["extensibility"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    benchmark_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "metric_semantics_version": METRIC_SEMANTICS_VERSION,
        "generated_at": _utc_now(),
        "source_commit_sha": _git_commit(project_root),
        "evaluation_path": evaluation_path.relative_to(
            project_root
        ).as_posix(),
        "evaluation_sha256": _file_sha256(evaluation_path),
        "scenario_count": len(result["rows"]) // 2,
        "controller_versions": [
            item["controller_version"] for item in result["summary"]
        ],
        "qualification_contract": {
            "path": V1_1_CONTRACT_RELATIVE_PATH,
            "sha256": _file_sha256(
                project_root / V1_1_CONTRACT_RELATIVE_PATH
            ),
        },
        "schema_versions": {
            "orchestration": ORCHESTRATION_SCHEMA_VERSION,
            "case_state": CASE_STATE_SCHEMA_VERSION,
            "controller_proposal": (
                CONTROLLER_PROPOSAL_SCHEMA_VERSION
            ),
        },
        "locked_split_content_used": False,
        "real_model_inference_used": False,
        "files": {
            rows_path.name: _file_sha256(rows_path),
            summary_path.name: _file_sha256(summary_path),
            extensibility_path.name: _file_sha256(extensibility_path),
            benchmark_path.name: _file_sha256(benchmark_path),
        },
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "scenario_results": rows_path,
        "summary": summary_path,
        "extensibility": extensibility_path,
        "benchmark": benchmark_path,
        "manifest": manifest_path,
    }
