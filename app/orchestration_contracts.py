"""Stable contracts for the controlled OphAgent V2 monolith.

The module deliberately contains no Streamlit, filesystem discovery, dataset
special cases, or model-specific branches.  Task and model differences enter
through typed profiles and adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Protocol, runtime_checkable


ORCHESTRATION_SCHEMA_VERSION = "ophagent.controlled_orchestration.v2"
TASK_SPEC_SCHEMA_VERSION = "ophagent.task_spec.v1"
MODEL_CAPABILITY_SCHEMA_VERSION = "ophagent.model_capability.v1"
CASE_STATE_SCHEMA_VERSION = "ophagent.case_state.v2"
CONTROLLER_PROPOSAL_SCHEMA_VERSION = "ophagent.controller_proposal.v1"

SENSITIVE_FIELD_PARTS = {
    "patient",
    "name",
    "hospital",
    "admission",
    "raw_case",
    "source_case",
    "image_path",
    "private_path",
    "path",
    "mrn",
    "medical_record",
    "email",
    "phone",
    "birth",
    "dob",
    "address",
}


class CaseStatus(str, Enum):
    CASE_RECEIVED = "CASE_RECEIVED"
    CASE_VALIDATED = "CASE_VALIDATED"
    SCOUT_COMPLETED = "SCOUT_COMPLETED"
    RISK_AUDITED = "RISK_AUDITED"
    EXPERT_PENDING_APPROVAL = "EXPERT_PENDING_APPROVAL"
    EXPERT_COMPLETED = "EXPERT_COMPLETED"
    REVIEW_PENDING = "REVIEW_PENDING"
    CLOSED = "CLOSED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AgentAction(str, Enum):
    KEEP_SCOUT = "KEEP_SCOUT"
    REQUEST_EXPERT = "REQUEST_EXPERT"
    REFER_TO_HUMAN = "REFER_TO_HUMAN"


class ActorRole(str, Enum):
    OPERATOR = "operator"
    REVIEWER = "reviewer"
    ADMIN = "admin"


@dataclass(frozen=True)
class TaskSpec:
    """Versioned task profile projected from the unified Model Hub index."""

    task_id: str
    dataset_id: str
    modality: str
    label_space: tuple[str, ...]
    primary_metric: str
    risk_semantics: str
    report_label: str
    risk_positive_class_ids: tuple[int, ...] = ()
    adaptation_type: str = "task_native"
    version: str = TASK_SPEC_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCapability:
    """Task-scoped model capability; qualification is not granted here."""

    task_id: str
    artifact_id: str
    adapter_type: str
    prediction_asset_available: bool
    offline_batch_inference_ready: bool
    online_case_inference_ready: bool
    cost_protocol_id: str
    cost_ms_per_image: float | None
    qualification_status: str
    version: str = MODEL_CAPABILITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteQualification:
    """Stable domain projection of the shared qualification decision."""

    execution_level: str
    evidence_label: str
    allow_cached_replay: bool
    allow_case_simulation: bool
    allow_new_case_route: bool
    clinical_route_eligible: bool
    human_confirmation_required: bool
    error_codes: tuple[str, ...]
    contract_sha256: str
    evidence_fingerprint: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_decision(
        cls,
        decision: Any,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> "RouteQualification":
        return cls(
            execution_level=str(decision.execution_level),
            evidence_label=str(decision.evidence_label),
            allow_cached_replay=bool(decision.allow_cached_replay),
            allow_case_simulation=bool(decision.allow_case_simulation),
            allow_new_case_route=bool(decision.allow_new_case_route),
            clinical_route_eligible=bool(decision.clinical_route_eligible),
            human_confirmation_required=bool(
                decision.human_confirmation_required
            ),
            error_codes=tuple(str(value) for value in decision.error_codes),
            contract_sha256=str(decision.contract_sha256),
            evidence_fingerprint=str(decision.evidence_fingerprint),
            evidence=dict(evidence or {}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteQualification":
        """Restore the persisted qualification without re-evaluating policy."""

        return cls(
            execution_level=str(payload["execution_level"]),
            evidence_label=str(payload["evidence_label"]),
            allow_cached_replay=bool(payload["allow_cached_replay"]),
            allow_case_simulation=bool(payload["allow_case_simulation"]),
            allow_new_case_route=bool(payload["allow_new_case_route"]),
            clinical_route_eligible=bool(
                payload.get("clinical_route_eligible", False)
            ),
            human_confirmation_required=bool(
                payload.get("human_confirmation_required", True)
            ),
            error_codes=tuple(
                str(value) for value in payload.get("error_codes", ())
            ),
            contract_sha256=str(payload["contract_sha256"]),
            evidence_fingerprint=str(payload["evidence_fingerprint"]),
            evidence=dict(payload.get("evidence", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerProposal:
    """Only output accepted from a rule or local-language-model controller."""

    action: str
    reason_code: str
    parameters: dict[str, Any]
    controller_type: str
    schema_version: str = CONTROLLER_PROPOSAL_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        controller_type: str,
    ) -> "ControllerProposal":
        required = {
            "action",
            "reason_code",
            "parameters",
            "schema_version",
        }
        if set(payload) != required:
            raise ValueError("CONTROLLER_SCHEMA_FIELDS_INVALID")
        if not isinstance(payload["action"], str):
            raise ValueError("CONTROLLER_SCHEMA_ACTION_TYPE_INVALID")
        if not isinstance(payload["reason_code"], str):
            raise ValueError("CONTROLLER_SCHEMA_REASON_TYPE_INVALID")
        parameters = payload["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError("CONTROLLER_SCHEMA_PARAMETERS_TYPE_INVALID")
        if not isinstance(payload["schema_version"], str):
            raise ValueError("CONTROLLER_SCHEMA_VERSION_TYPE_INVALID")
        return cls(
            action=payload["action"],
            reason_code=payload["reason_code"],
            parameters=dict(parameters),
            controller_type=controller_type,
            schema_version=payload["schema_version"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseState:
    """Patient-redacted state persisted by :class:`CaseStateStore`."""

    case_id: str
    task_id: str
    current_state: str = CaseStatus.CASE_RECEIVED
    controller_type: str = "rule_controller"
    idempotency_key: str = ""
    request_fingerprint: str = ""
    remaining_budget: float | None = None
    expected_expert_cost: float | None = None
    allowed_actions: tuple[str, ...] = ()
    tool_return_codes: tuple[str, ...] = ()
    completed_steps: tuple[str, ...] = ()
    qualification: dict[str, Any] = field(default_factory=dict)
    controller_proposal: dict[str, Any] = field(default_factory=dict)
    gate_decision: dict[str, Any] = field(default_factory=dict)
    final_action: str = ""
    human_decision: str = ""
    retry_count: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    runtime_payload: dict[str, Any] = field(default_factory=dict)
    tool_trace: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    schema_version: str = CASE_STATE_SCHEMA_VERSION
    qualification_policy_version: str = ""
    controller_version: str = "2.0"
    route_protocol_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaseState":
        fields = {
            "case_id": str(payload["case_id"]),
            "task_id": str(payload["task_id"]),
            "current_state": str(
                payload.get("current_state", CaseStatus.CASE_RECEIVED)
            ),
            "controller_type": str(
                payload.get("controller_type", "rule_controller")
            ),
            "idempotency_key": str(payload.get("idempotency_key", "")),
            "request_fingerprint": str(
                payload.get("request_fingerprint", "")
            ),
            "remaining_budget": payload.get("remaining_budget"),
            "expected_expert_cost": payload.get("expected_expert_cost"),
            "allowed_actions": tuple(
                str(value) for value in payload.get("allowed_actions", [])
            ),
            "tool_return_codes": tuple(
                str(value) for value in payload.get("tool_return_codes", [])
            ),
            "completed_steps": tuple(
                str(value) for value in payload.get("completed_steps", [])
            ),
            "qualification": dict(payload.get("qualification", {})),
            "controller_proposal": dict(
                payload.get("controller_proposal", {})
            ),
            "gate_decision": dict(payload.get("gate_decision", {})),
            "final_action": str(payload.get("final_action", "")),
            "human_decision": str(payload.get("human_decision", "")),
            "retry_count": int(payload.get("retry_count", 0)),
            "trace": [
                dict(value) for value in payload.get("trace", [])
            ],
            "report": dict(payload.get("report", {})),
            "runtime_payload": dict(payload.get("runtime_payload", {})),
            "tool_trace": dict(payload.get("tool_trace", {})),
            "created_at": str(payload.get("created_at", "")),
            "updated_at": str(payload.get("updated_at", "")),
            "schema_version": str(
                payload.get("schema_version", CASE_STATE_SCHEMA_VERSION)
            ),
            "qualification_policy_version": str(
                payload.get("qualification_policy_version", "")
            ),
            "controller_version": str(
                payload.get("controller_version", "2.0")
            ),
            "route_protocol_version": str(
                payload.get("route_protocol_version", "")
            ),
        }
        return cls(**fields)


@runtime_checkable
class TaskAdapter(Protocol):
    @property
    def spec(self) -> TaskSpec:
        """Return the task profile without reading controller or UI state."""

    def validate_metadata(self, metadata: dict[str, Any]) -> tuple[bool, str]:
        """Validate non-identifying case metadata."""

    def risk_summary(
        self,
        probabilities: tuple[float, ...],
    ) -> dict[str, Any]:
        """Return a task-specific research risk proxy."""


@runtime_checkable
class ModelRuntimeAdapter(Protocol):
    @property
    def capability(self) -> ModelCapability:
        """Return the Model Hub capability record."""

    def infer(self, model_input: Any) -> dict[str, Any]:
        """Return standard probabilities; never grant route qualification."""


@runtime_checkable
class ControllerAdapter(Protocol):
    @property
    def controller_type(self) -> str:
        """Stable controller identifier."""

    def propose(self, context: dict[str, Any]) -> ControllerProposal:
        """Propose one of the three actions without executing tools."""


@dataclass(frozen=True)
class ConfiguredTaskAdapter:
    """Minimal task adapter driven entirely by a :class:`TaskSpec`."""

    spec: TaskSpec

    def validate_metadata(
        self,
        metadata: dict[str, Any],
    ) -> tuple[bool, str]:
        modality = str(metadata.get("modality", self.spec.modality))
        if modality != self.spec.modality:
            return False, "TASK_MODALITY_MISMATCH"
        image_count = int(metadata.get("image_count", 1))
        if image_count < 1 or image_count > 8:
            return False, "TASK_IMAGE_COUNT_INVALID"
        return True, "TASK_METADATA_OK"

    def risk_summary(
        self,
        probabilities: tuple[float, ...],
    ) -> dict[str, Any]:
        if len(probabilities) != len(self.spec.label_space):
            raise ValueError("probability count does not match TaskSpec")
        risk_mass = sum(
            probabilities[index]
            for index in self.spec.risk_positive_class_ids
        )
        return {
            "name": self.spec.risk_semantics,
            "value": float(risk_mass),
            "semantics": "model_output_error_risk_not_clinical_consequence",
        }


@dataclass(frozen=True)
class CallableModelRuntimeAdapter:
    """Small adapter for an existing inference callable."""

    capability: ModelCapability
    inference_callable: Callable[[Any], dict[str, Any]]

    def infer(self, model_input: Any) -> dict[str, Any]:
        result = dict(self.inference_callable(model_input))
        probabilities = result.get("probabilities")
        if not isinstance(probabilities, (list, tuple)):
            raise ValueError("model adapter must return standard probabilities")
        return result


SAFE_CONTROLLER_METADATA_FIELDS = {
    "modality",
    "image_count",
    "laterality",
    "quality_flag",
    "age_band",
}

CONTROLLER_CONTEXT_FIELDS = {
    "current_state",
    "task_id",
    "allowed_actions",
    "model_qualification",
    "risk_result",
    "remaining_budget",
    "tool_return_codes",
    "case_metadata",
}


def redact_free_text(value: str) -> str:
    """Remove common identity tokens from otherwise permitted review text."""

    redacted = value
    patterns = (
        (
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "[REDACTED_EMAIL]",
        ),
        (
            r"(?i)\b(?:MRN|medical\s*record|hospital\s*id|patient\s*id|"
            r"DOB)\s*[:=#：]?\s*[A-Z0-9._/-]+",
            "[REDACTED_IDENTIFIER]",
        ),
        (
            r"(?:姓名|患者姓名|住院号|病历号|身份证号|出生日期|"
            r"联系电话|电话|地址)\s*[:：]?\s*\S+",
            "[REDACTED_IDENTIFIER]",
        ),
        (
            r"(?<!\d)(?:(?:\+?86[- ]?)?"
            r"1[3-9]\d(?:[- ]?\d){8}|"
            r"(?:\+?\d{1,3}[- ]?)?\(?\d{2,4}\)?[- ]"
            r"\d{3,4}[- ]\d{4})(?!\d)",
            "[REDACTED_PHONE]",
        ),
        (
            r"(?i)(?:file://|[A-Z]:[\\/]|\\\\{2}|"
            r"(?<![A-Za-z0-9_])/)"
            r"(?:[^ \t\r\n,;]+[\\/])*[^ \t\r\n,;]+",
            "[REDACTED_PATH]",
        ),
    )
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)
    return redacted


def redact_structured_value(value: Any) -> Any:
    """Recursively remove identity fields and absolute/private paths."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in SENSITIVE_FIELD_PARTS):
                continue
            result[str(key)] = redact_structured_value(item)
        return result
    if isinstance(value, list):
        return [redact_structured_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_structured_value(item) for item in value)
    if isinstance(value, str):
        return redact_free_text(value)
    return value


def sanitize_controller_context(context: dict[str, Any]) -> dict[str, Any]:
    """Apply a strict allowlist before any controller sees case state."""

    sanitized = {
        key: context[key]
        for key in CONTROLLER_CONTEXT_FIELDS
        if key in context
    }
    metadata = sanitized.get("case_metadata", {})
    if isinstance(metadata, dict):
        sanitized["case_metadata"] = {
            key: metadata[key]
            for key in SAFE_CONTROLLER_METADATA_FIELDS
            if key in metadata
        }
    else:
        sanitized["case_metadata"] = {}
    redacted = redact_structured_value(sanitized)
    return dict(redacted)
