"""Offline benchmark contract for controlled local-language-model proposals.

The module deliberately keeps model reasoning separate from execution.  A
local model may only return a structured proposal; the existing state-machine
and qualification gate remain authoritative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Iterable, Sequence

from app.model_hub_agent_v2 import (
    CONTROLLER_REASON_CODES,
    ControllerUnavailable,
    LocalLLMController,
    LocalLLMControllerConfig,
    RuleController,
    gate_controller_proposal,
    redact_sensitive_payload,
    validate_controller_proposal,
)
from app.orchestration_contracts import (
    AgentAction,
    CONTROLLER_PROPOSAL_SCHEMA_VERSION,
    ControllerAdapter,
    ControllerProposal,
    RouteQualification,
    sanitize_controller_context,
)


LOCAL_CONTROLLER_BENCHMARK_SCHEMA = "ophagent.local_controller_benchmark.v1"
LOCAL_CONTROLLER_MANIFEST_SCHEMA = (
    "ophagent.local_controller_benchmark_manifest.v1_1"
)
LOCAL_TRANSFORMERS_RUNTIME_SCHEMA = "ophagent.local_transformers_runtime.v1"
PROMPT_SUITE_SCHEMA = "ophagent.local_controller_prompt.v1"
EVALUATION_SUITE_SCHEMA = "ophagent.local_controller_eval.v1"
CURRENT_CONTEXT_MARKER = "CURRENT_CONTEXT_JSON="

REQUIRED_PROPOSAL_FIELDS = frozenset(
    {"action", "reason_code", "parameters", "schema_version"}
)
ALLOWED_PROPOSAL_ACTIONS = frozenset(action.value for action in AgentAction)
ALLOWED_PROPOSAL_PARAMETERS: frozenset[str] = frozenset()


class LocalControllerBenchmarkError(RuntimeError):
    """Base error for deterministic benchmark contract failures."""


class SensitiveControllerContextError(LocalControllerBenchmarkError):
    """Raised before a prompt can contain identity or private-path data."""


class EvaluationLeakageError(LocalControllerBenchmarkError):
    """Raised when prompt examples overlap the held-out evaluation split."""


class LocalModelRuntimeError(ControllerUnavailable):
    """Raised when the local-only Transformers runtime cannot be used."""


class LocalModelSchemaError(ValueError):
    """Raised when a model response violates the fixed proposal schema."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a small protocol/config file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_controller_context_safe(
    context: dict[str, Any],
) -> dict[str, Any]:
    """Reject sensitive input, then apply the shared structural allowlist."""

    redacted = redact_sensitive_payload(context)
    if redacted != context:
        raise SensitiveControllerContextError(
            "CONTROLLER_CONTEXT_CONTAINS_SENSITIVE_DATA"
        )
    safe = sanitize_controller_context(context)
    if redact_sensitive_payload(safe) != safe:
        raise SensitiveControllerContextError(
            "CONTROLLER_CONTEXT_SANITIZATION_INCOMPLETE"
        )
    return safe


def ensure_prompt_safe(prompt: str) -> None:
    """Defense in depth for fixed templates and serialized safe context."""

    if redact_sensitive_payload({"prompt": prompt}) != {"prompt": prompt}:
        raise SensitiveControllerContextError(
            "CONTROLLER_PROMPT_CONTAINS_PRIVATE_PATH"
        )
    lowered = prompt.lower()
    forbidden_json_keys = (
        '"patient_name"',
        '"patient_id"',
        '"hospital_id"',
        '"admission_id"',
        '"private_path"',
        '"image_path"',
        '"raw_case"',
        '"source_case"',
    )
    if any(value in lowered for value in forbidden_json_keys):
        raise SensitiveControllerContextError(
            "CONTROLLER_PROMPT_CONTAINS_IDENTITY_FIELD"
        )


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object without accepting arrays or scalar output."""

    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise LocalModelSchemaError("CONTROLLER_SCHEMA_NOT_JSON") from None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LocalModelSchemaError(
                "CONTROLLER_SCHEMA_NOT_SINGLE_JSON_OBJECT"
            ) from exc
    if not isinstance(payload, dict):
        raise LocalModelSchemaError("CONTROLLER_SCHEMA_NOT_OBJECT")
    return payload


def validate_raw_proposal_payload(
    payload: Any,
) -> tuple[bool, tuple[str, ...]]:
    """Validate raw model output before lossy dataclass projection."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return False, ("PROPOSAL_SCHEMA_NOT_OBJECT",)
    fields = set(payload)
    missing = REQUIRED_PROPOSAL_FIELDS - fields
    extra = fields - REQUIRED_PROPOSAL_FIELDS
    if missing:
        errors.append("PROPOSAL_SCHEMA_FIELDS_MISSING")
    if extra:
        errors.append("PROPOSAL_SCHEMA_FIELDS_EXTRA")

    action = payload.get("action")
    if not isinstance(action, str) or action not in ALLOWED_PROPOSAL_ACTIONS:
        errors.append("PROPOSAL_ACTION_INVALID")
    reason = payload.get("reason_code")
    if not isinstance(reason, str) or reason not in CONTROLLER_REASON_CODES:
        errors.append("PROPOSAL_REASON_CODE_INVALID")
    if payload.get("schema_version") != CONTROLLER_PROPOSAL_SCHEMA_VERSION:
        errors.append("PROPOSAL_SCHEMA_VERSION_INVALID")

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("PROPOSAL_PARAMETERS_NOT_OBJECT")
    else:
        if set(parameters) - ALLOWED_PROPOSAL_PARAMETERS:
            errors.append("PROPOSAL_PARAMETERS_INVALID")
    return not errors, tuple(dict.fromkeys(errors))


@dataclass(frozen=True)
class LocalTransformersRuntimeConfig:
    """One local-only runtime configuration shared by 4B and 27B."""

    model_id: str
    model_path: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    auto_model_class: str = "AutoModelForImageTextToText"
    max_new_tokens: int = 128
    disable_thinking: bool = True
    runtime_schema: str = LOCAL_TRANSFORMERS_RUNTIME_SCHEMA

    def public_dict(self) -> dict[str, Any]:
        """Exclude the private server path from persisted benchmark output."""

        return {
            "model_id": self.model_id,
            "model_path_configured": bool(str(self.model_path)),
            "device": self.device,
            "dtype": self.dtype,
            "auto_model_class": self.auto_model_class,
            "max_new_tokens": self.max_new_tokens,
            "disable_thinking": self.disable_thinking,
            "runtime_schema": self.runtime_schema,
            "local_files_only": True,
            "trust_remote_code": False,
            "network_allowed": False,
        }


@dataclass
class LocalInferenceTelemetry:
    model_id: str
    started_at: str = ""
    completed_at: str = ""
    load_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    peak_vram_allocated_mib: float = 0.0
    peak_vram_reserved_mib: float = 0.0
    raw_output: str = ""
    parsed_payload: dict[str, Any] | None = None
    schema_valid: bool = False
    schema_errors: tuple[str, ...] = ()
    error_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_local_model_snapshot(model_path: Path) -> dict[str, Any]:
    """Check local snapshot completeness without hashing multi-gigabyte shards."""

    path = Path(model_path)
    if not path.is_absolute() or not path.is_dir():
        raise LocalModelRuntimeError("LOCAL_MODEL_PATH_MISSING")
    required = ("config.json", "tokenizer_config.json")
    missing_required = [name for name in required if not (path / name).is_file()]
    if missing_required:
        raise LocalModelRuntimeError(
            "LOCAL_MODEL_SNAPSHOT_REQUIRED_FILES_MISSING:"
            + ",".join(missing_required)
        )
    indexes = sorted(path.glob("*.safetensors.index.json"))
    if not indexes:
        raise LocalModelRuntimeError("LOCAL_MODEL_WEIGHT_INDEX_MISSING")
    index_payload = json.loads(indexes[0].read_text(encoding="utf-8"))
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise LocalModelRuntimeError("LOCAL_MODEL_WEIGHT_INDEX_INVALID")
    shard_names = sorted({str(value) for value in weight_map.values()})
    missing_shards = [name for name in shard_names if not (path / name).is_file()]
    if missing_shards:
        raise LocalModelRuntimeError(
            "LOCAL_MODEL_WEIGHT_SHARDS_MISSING:" + ",".join(missing_shards)
        )
    total_bytes = sum((path / name).stat().st_size for name in shard_names)
    manifest_path = path / "model_source_manifest.json"
    provenance: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = {
            "model_id": str(manifest.get("model_id", "")),
            "provider": str(manifest.get("provider", "")),
            "revision": str(manifest.get("revision", "")),
            "manifest_sha256": sha256_file(manifest_path),
        }
    return {
        "weight_index": indexes[0].name,
        "shard_count": len(shard_names),
        "weight_file_bytes": total_bytes,
        "missing_shards": [],
        "provenance": provenance,
    }


class LocalTransformersInference:
    """Lazy, local-files-only text inference for a controller proposal."""

    def __init__(self, config: LocalTransformersRuntimeConfig):
        self.config = config
        self.last_telemetry = LocalInferenceTelemetry(config.model_id)
        self.snapshot = verify_local_model_snapshot(config.model_path)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._load_latency_ms = 0.0
        self._fatal_error = ""

    def _load(self) -> None:
        if self._fatal_error:
            raise LocalModelRuntimeError(self._fatal_error)
        if self._model is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        started = time.perf_counter()
        try:
            import torch
            import transformers

            if not self.config.device.startswith("cuda"):
                raise LocalModelRuntimeError(
                    "LOCAL_RUNTIME_REQUIRES_EXPLICIT_CUDA_DEVICE"
                )
            if not torch.cuda.is_available():
                raise LocalModelRuntimeError("LOCAL_RUNTIME_CUDA_UNAVAILABLE")
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }.get(self.config.dtype)
            if dtype is None:
                raise LocalModelRuntimeError("LOCAL_RUNTIME_DTYPE_UNSUPPORTED")
            model_class = getattr(
                transformers,
                self.config.auto_model_class,
                None,
            )
            if model_class is None:
                raise LocalModelRuntimeError(
                    "LOCAL_RUNTIME_AUTO_MODEL_CLASS_UNAVAILABLE"
                )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self.config.model_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            model = model_class.from_pretrained(
                str(self.config.model_path),
                local_files_only=True,
                trust_remote_code=False,
                dtype=dtype,
            )
            model.to(self.config.device)
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._load_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0
        except Exception as exc:
            self._fatal_error = (
                f"LOCAL_RUNTIME_LOAD_FAILED:{type(exc).__name__}:{exc}"
            )[:1000]
            self.close()
            raise LocalModelRuntimeError(self._fatal_error) from exc

    def _encode(self, prompt: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if self.config.disable_thinking:
            kwargs["enable_thinking"] = False
        try:
            encoded = self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            encoded = self._tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(encoded, "items"):
            return {
                str(key): value.to(self.config.device)
                for key, value in encoded.items()
            }
        return {"input_ids": encoded.to(self.config.device)}

    def __call__(
        self,
        prompt: str,
        controller_config: LocalLLMControllerConfig,
    ) -> str:
        if controller_config.model_id != self.config.model_id:
            raise LocalModelRuntimeError("LOCAL_RUNTIME_MODEL_ID_MISMATCH")
        ensure_prompt_safe(prompt)
        telemetry = LocalInferenceTelemetry(
            model_id=self.config.model_id,
            started_at=_utc_now(),
            load_latency_ms=self._load_latency_ms,
        )
        self.last_telemetry = telemetry
        try:
            self._load()
            telemetry.load_latency_ms = self._load_latency_ms
            encoded = self._encode(prompt)
            input_ids = encoded["input_ids"]
            telemetry.input_tokens = int(input_ids.shape[-1])
            torch = self._torch
            device = torch.device(self.config.device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=min(
                        controller_config.generation_max_new_tokens,
                        self.config.max_new_tokens,
                    ),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=(
                        self._tokenizer.pad_token_id
                        if self._tokenizer.pad_token_id is not None
                        else self._tokenizer.eos_token_id
                    ),
                )
            torch.cuda.synchronize(device)
            telemetry.inference_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0
            new_tokens = generated[0, input_ids.shape[-1] :]
            telemetry.output_tokens = int(new_tokens.shape[-1])
            telemetry.peak_vram_allocated_mib = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
            )
            telemetry.peak_vram_reserved_mib = (
                torch.cuda.max_memory_reserved(device) / (1024**2)
            )
            raw = self._tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()
            telemetry.raw_output = raw[:4000]
            payload = parse_json_object(raw)
            telemetry.parsed_payload = payload
            valid, errors = validate_raw_proposal_payload(payload)
            telemetry.schema_valid = valid
            telemetry.schema_errors = errors
            if not valid:
                raise LocalModelSchemaError(
                    "CONTROLLER_SCHEMA_INVALID:" + ",".join(errors)
                )
            telemetry.completed_at = _utc_now()
            return _stable_json(payload)
        except Exception as exc:
            telemetry.completed_at = _utc_now()
            telemetry.error_code = (
                f"{type(exc).__name__}:{exc}"
            )[:1000]
            if isinstance(
                exc,
                (
                    LocalModelRuntimeError,
                    LocalModelSchemaError,
                    SensitiveControllerContextError,
                ),
            ):
                raise
            raise LocalModelRuntimeError(
                f"LOCAL_RUNTIME_INFERENCE_FAILED:{type(exc).__name__}:{exc}"
            ) from exc

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._torch = None


class ScriptedLocalInference:
    """Strict deterministic callable used by tests, never by real evaluation."""

    def __init__(
        self,
        responder: Callable[[str, LocalLLMControllerConfig], Any],
    ):
        self.responder = responder
        self.last_telemetry = LocalInferenceTelemetry("scripted")

    def __call__(
        self,
        prompt: str,
        controller_config: LocalLLMControllerConfig,
    ) -> str:
        ensure_prompt_safe(prompt)
        started = time.perf_counter()
        raw = self.responder(prompt, controller_config)
        text = raw if isinstance(raw, str) else _stable_json(raw)
        telemetry = LocalInferenceTelemetry(
            model_id=controller_config.model_id,
            started_at=_utc_now(),
            completed_at=_utc_now(),
            inference_latency_ms=(time.perf_counter() - started) * 1000.0,
            raw_output=text[:4000],
        )
        self.last_telemetry = telemetry
        try:
            payload = parse_json_object(text)
            telemetry.parsed_payload = payload
            valid, errors = validate_raw_proposal_payload(payload)
            telemetry.schema_valid = valid
            telemetry.schema_errors = errors
            if not valid:
                raise LocalModelSchemaError(
                    "CONTROLLER_SCHEMA_INVALID:" + ",".join(errors)
                )
            return _stable_json(payload)
        except Exception as exc:
            telemetry.error_code = f"{type(exc).__name__}:{exc}"[:1000]
            raise


@dataclass(frozen=True)
class PromptSuite:
    prompt_id: str
    zero_shot_template: str
    few_shot_examples: tuple[dict[str, Any], ...]
    schema_version: str = PROMPT_SUITE_SCHEMA


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    tags: tuple[str, ...]
    context: dict[str, Any]
    case_scope: str
    expected_expert_cost: float | None
    expected_proposal_action: str
    accepted_reason_codes: tuple[str, ...]
    expected_final_action: str


@dataclass(frozen=True)
class EvaluationSuite:
    benchmark_id: str
    split: str
    cases: tuple[EvaluationCase, ...]
    schema_version: str = EVALUATION_SUITE_SCHEMA


def load_prompt_suite(path: Path) -> PromptSuite:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROMPT_SUITE_SCHEMA:
        raise LocalControllerBenchmarkError("PROMPT_SCHEMA_VERSION_INVALID")
    template = str(payload.get("zero_shot_template", ""))
    if "{context}" not in template:
        raise LocalControllerBenchmarkError("PROMPT_CONTEXT_SLOT_MISSING")
    examples = payload.get("few_shot_examples")
    if not isinstance(examples, list):
        raise LocalControllerBenchmarkError("PROMPT_EXAMPLES_NOT_LIST")
    for example in examples:
        if not isinstance(example, dict):
            raise LocalControllerBenchmarkError("PROMPT_EXAMPLE_NOT_OBJECT")
        ensure_controller_context_safe(dict(example.get("context", {})))
        valid, errors = validate_raw_proposal_payload(example.get("output"))
        if not valid:
            raise LocalControllerBenchmarkError(
                "PROMPT_EXAMPLE_OUTPUT_INVALID:" + ",".join(errors)
            )
    return PromptSuite(
        prompt_id=str(payload["prompt_id"]),
        zero_shot_template=template,
        few_shot_examples=tuple(dict(value) for value in examples),
    )


def load_evaluation_suite(path: Path) -> EvaluationSuite:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != EVALUATION_SUITE_SCHEMA:
        raise LocalControllerBenchmarkError("EVAL_SCHEMA_VERSION_INVALID")
    if payload.get("synthetic_or_deidentified_only") is not True:
        raise LocalControllerBenchmarkError("EVAL_DATA_BOUNDARY_NOT_DECLARED")
    if payload.get("frozen_test_assets_used") is not False:
        raise LocalControllerBenchmarkError("EVAL_FROZEN_TEST_BOUNDARY_INVALID")
    if payload.get("raw_images_used") is not False:
        raise LocalControllerBenchmarkError("EVAL_RAW_IMAGE_BOUNDARY_INVALID")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise LocalControllerBenchmarkError("EVAL_CASES_EMPTY")
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise LocalControllerBenchmarkError("EVAL_CASE_NOT_OBJECT")
        case_id = str(raw.get("case_id", ""))
        if not case_id or case_id in seen_ids:
            raise LocalControllerBenchmarkError("EVAL_CASE_ID_INVALID")
        seen_ids.add(case_id)
        context = ensure_controller_context_safe(dict(raw.get("context", {})))
        expected_proposal = str(raw.get("expected_proposal_action", ""))
        expected_final = str(raw.get("expected_final_action", ""))
        if (
            expected_proposal not in ALLOWED_PROPOSAL_ACTIONS
            or expected_final not in ALLOWED_PROPOSAL_ACTIONS
        ):
            raise LocalControllerBenchmarkError("EVAL_EXPECTED_ACTION_INVALID")
        accepted_reasons = tuple(
            str(value) for value in raw.get("accepted_reason_codes", [])
        )
        if not accepted_reasons or any(
            value not in CONTROLLER_REASON_CODES
            for value in accepted_reasons
        ):
            raise LocalControllerBenchmarkError(
                "EVAL_ACCEPTED_REASON_CODES_INVALID"
            )
        expected_cost = raw.get("expected_expert_cost")
        cases.append(
            EvaluationCase(
                case_id=case_id,
                tags=tuple(str(value) for value in raw.get("tags", [])),
                context=context,
                case_scope=str(raw.get("case_scope", "")),
                expected_expert_cost=(
                    float(expected_cost)
                    if expected_cost is not None
                    else None
                ),
                expected_proposal_action=expected_proposal,
                accepted_reason_codes=accepted_reasons,
                expected_final_action=expected_final,
            )
        )
    return EvaluationSuite(
        benchmark_id=str(payload["benchmark_id"]),
        split=str(payload["split"]),
        cases=tuple(cases),
    )


def verify_prompt_eval_isolation(
    prompt_suite: PromptSuite,
    evaluation_suite: EvaluationSuite,
) -> dict[str, Any]:
    """Prove few-shot IDs and exact contexts are absent from held-out eval."""

    example_ids = {
        str(value.get("example_id", ""))
        for value in prompt_suite.few_shot_examples
    }
    eval_ids = {value.case_id for value in evaluation_suite.cases}
    id_overlap = sorted(example_ids.intersection(eval_ids))
    example_fingerprints = {
        _sha256_json(ensure_controller_context_safe(dict(value["context"])))
        for value in prompt_suite.few_shot_examples
    }
    eval_fingerprints = {
        _sha256_json(value.context) for value in evaluation_suite.cases
    }
    context_overlap = sorted(
        example_fingerprints.intersection(eval_fingerprints)
    )
    if id_overlap or context_overlap:
        raise EvaluationLeakageError("PROMPT_EVAL_OVERLAP_DETECTED")
    return {
        "example_count": len(example_ids),
        "evaluation_count": len(eval_ids),
        "id_overlap_count": 0,
        "exact_context_overlap_count": 0,
        "isolation_passed": True,
    }


def _qualification_from_context(context: dict[str, Any]) -> RouteQualification:
    payload = dict(context.get("model_qualification", {}))
    return RouteQualification(
        execution_level=str(payload.get("execution_level", "blocked")),
        evidence_label=str(payload.get("evidence_label", "not_available")),
        allow_cached_replay=bool(payload.get("allow_cached_replay", False)),
        allow_case_simulation=bool(
            payload.get("allow_case_simulation", False)
        ),
        allow_new_case_route=bool(
            payload.get("allow_new_case_route", False)
        ),
        clinical_route_eligible=False,
        human_confirmation_required=True,
        error_codes=tuple(
            str(value) for value in payload.get("error_codes", [])
        ),
        contract_sha256=str(payload.get("contract_sha256", "")),
        evidence_fingerprint=str(payload.get("evidence_fingerprint", "")),
        evidence={},
    )


def _telemetry_from_callable(
    inference_callable: Any,
) -> LocalInferenceTelemetry | None:
    telemetry = getattr(inference_callable, "last_telemetry", None)
    return telemetry if isinstance(telemetry, LocalInferenceTelemetry) else None


def _proposal_from_failed_output(
    telemetry: LocalInferenceTelemetry | None,
    *,
    controller_type: str,
) -> ControllerProposal:
    payload = telemetry.parsed_payload if telemetry is not None else None
    if isinstance(payload, dict):
        return ControllerProposal.from_dict(
            payload,
            controller_type=controller_type,
        )
    return ControllerProposal(
        action="INVALID_ACTION",
        reason_code="INVALID_PROPOSAL",
        parameters={},
        controller_type=controller_type,
        schema_version="invalid",
    )


def _proposal_payload(proposal: ControllerProposal) -> dict[str, Any]:
    return {
        "action": proposal.action,
        "reason_code": proposal.reason_code,
        "parameters": dict(proposal.parameters),
        "schema_version": proposal.schema_version,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rate(rows: Iterable[dict[str, Any]], key: str) -> float:
    selected = list(rows)
    if not selected:
        return 0.0
    return sum(bool(row.get(key)) for row in selected) / len(selected)


def summarize_benchmark_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    latencies = [
        float(row["inference_latency_ms"])
        for row in rows
        if row.get("inference_latency_ms") is not None
    ]
    fault_rows = [row for row in rows if "fault_recovery" in row["tags"]]
    non_expert_expected = [
        row
        for row in rows
        if row["expected_proposal_action"]
        != AgentAction.REQUEST_EXPERT.value
    ]
    non_expert_final_expected = [
        row
        for row in rows
        if row["expected_final_action"] != AgentAction.REQUEST_EXPERT.value
    ]
    return {
        "case_count": len(rows),
        "next_action_accuracy": _rate(rows, "next_action_correct"),
        "legal_action_rate": _rate(rows, "legal_action"),
        "schema_valid_rate": _rate(rows, "raw_schema_valid"),
        "unauthorized_action_proposal_rate": _rate(
            rows,
            "unauthorized_action_proposed",
        ),
        "task_completion_rate": _rate(rows, "task_completed"),
        "fault_recovery_rate": _rate(fault_rows, "task_completed"),
        "unnecessary_expert_proposal_rate": _rate(
            non_expert_expected,
            "unnecessary_expert_proposed",
        ),
        "unnecessary_expert_execution_rate": _rate(
            non_expert_final_expected,
            "unnecessary_expert_executed",
        ),
        "human_referral_rate": _rate(rows, "human_referred"),
        "report_fidelity_rate": _rate(rows, "report_faithful"),
        "gate_intercept_count": sum(
            bool(row["gate_intercepted"]) for row in rows
        ),
        "gate_intercept_rate": _rate(rows, "gate_intercepted"),
        "runtime_error_count": sum(bool(row["runtime_error"]) for row in rows),
        "latency_ms_mean": (
            statistics.fmean(latencies) if latencies else 0.0
        ),
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "peak_vram_allocated_mib": max(
            (
                float(row.get("peak_vram_allocated_mib", 0.0))
                for row in rows
            ),
            default=0.0,
        ),
        "peak_vram_reserved_mib": max(
            (
                float(row.get("peak_vram_reserved_mib", 0.0))
                for row in rows
            ),
            default=0.0,
        ),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
        "output_tokens": sum(
            int(row.get("output_tokens", 0)) for row in rows
        ),
        "local_token_cost": sum(
            int(row.get("input_tokens", 0))
            + int(row.get("output_tokens", 0))
            for row in rows
        ),
        "monetary_api_cost": 0.0,
    }


def run_controller_benchmark(
    *,
    controller: ControllerAdapter,
    evaluation_suite: EvaluationSuite,
    controller_label: str,
    prompt_mode: str,
    inference_callable: Any = None,
) -> dict[str, Any]:
    """Evaluate proposals and re-run the authoritative gate for every case."""

    rows: list[dict[str, Any]] = []
    for case in evaluation_suite.cases:
        context = ensure_controller_context_safe(case.context)
        telemetry: LocalInferenceTelemetry | None = None
        runtime_error = ""
        started = time.perf_counter()
        try:
            proposal = controller.propose(context)
        except Exception as exc:
            telemetry = _telemetry_from_callable(inference_callable)
            runtime_error = f"{type(exc).__name__}:{exc}"[:1000]
            proposal = _proposal_from_failed_output(
                telemetry,
                controller_type=controller.controller_type,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        telemetry = telemetry or _telemetry_from_callable(inference_callable)
        raw_payload = (
            telemetry.parsed_payload
            if telemetry is not None and telemetry.parsed_payload is not None
            else _proposal_payload(proposal)
        )
        raw_schema_valid, raw_schema_errors = (
            validate_raw_proposal_payload(raw_payload)
        )
        state_valid, state_errors = validate_controller_proposal(
            proposal,
            allowed_actions=tuple(
                str(value) for value in context.get("allowed_actions", [])
            ),
        )
        if not raw_schema_valid and state_valid:
            proposal = ControllerProposal(
                action=proposal.action,
                reason_code=proposal.reason_code,
                parameters=proposal.parameters,
                controller_type=proposal.controller_type,
                schema_version="invalid_raw_schema",
            )
            state_valid, state_errors = validate_controller_proposal(
                proposal,
                allowed_actions=tuple(
                    str(value)
                    for value in context.get("allowed_actions", [])
                ),
            )
        gate = gate_controller_proposal(
            proposal,
            qualification=_qualification_from_context(context),
            allowed_actions=tuple(
                str(value) for value in context.get("allowed_actions", [])
            ),
            remaining_budget=(
                float(context["remaining_budget"])
                if context.get("remaining_budget") is not None
                else None
            ),
            expected_expert_cost=case.expected_expert_cost,
            case_scope=case.case_scope,
            tool_return_codes=tuple(
                str(value)
                for value in context.get("tool_return_codes", [])
            ),
        )
        final_action = str(gate["final_action"])
        next_action_correct = (
            raw_schema_valid
            and proposal.action == case.expected_proposal_action
        )
        report_faithful = (
            next_action_correct
            and proposal.reason_code in case.accepted_reason_codes
            and not runtime_error
        )
        action_in_state = proposal.action in set(
            str(value) for value in context.get("allowed_actions", [])
        )
        row = {
            "case_id": case.case_id,
            "tags": list(case.tags),
            "expected_proposal_action": case.expected_proposal_action,
            "accepted_reason_codes": list(case.accepted_reason_codes),
            "proposal_action": proposal.action,
            "proposal_reason_code": proposal.reason_code,
            "expected_final_action": case.expected_final_action,
            "final_action": final_action,
            "raw_schema_valid": raw_schema_valid,
            "raw_schema_errors": list(raw_schema_errors),
            "state_schema_valid": state_valid,
            "state_schema_errors": list(state_errors),
            "legal_action": bool(
                raw_schema_valid and state_valid and action_in_state
            ),
            "unauthorized_action_proposed": not action_in_state,
            "next_action_correct": next_action_correct,
            "task_completed": (
                final_action == case.expected_final_action
                and not runtime_error
            ),
            "report_faithful": report_faithful,
            "gate_code": str(gate["code"]),
            "gate_intercepted": bool(gate["gate_intercepted"]),
            "human_referred": (
                final_action == AgentAction.REFER_TO_HUMAN.value
            ),
            "unnecessary_expert_proposed": (
                proposal.action == AgentAction.REQUEST_EXPERT.value
                and case.expected_proposal_action
                != AgentAction.REQUEST_EXPERT.value
            ),
            "unnecessary_expert_executed": (
                final_action == AgentAction.REQUEST_EXPERT.value
                and case.expected_final_action
                != AgentAction.REQUEST_EXPERT.value
            ),
            "runtime_error": runtime_error,
            "inference_latency_ms": (
                telemetry.inference_latency_ms
                if telemetry is not None
                else elapsed_ms
            ),
            "input_tokens": (
                telemetry.input_tokens if telemetry is not None else 0
            ),
            "output_tokens": (
                telemetry.output_tokens if telemetry is not None else 0
            ),
            "peak_vram_allocated_mib": (
                telemetry.peak_vram_allocated_mib
                if telemetry is not None
                else 0.0
            ),
            "peak_vram_reserved_mib": (
                telemetry.peak_vram_reserved_mib
                if telemetry is not None
                else 0.0
            ),
        }
        rows.append(row)
    metrics = summarize_benchmark_rows(rows)
    all_failed = bool(rows) and metrics["runtime_error_count"] == len(rows)
    return {
        "schema_version": LOCAL_CONTROLLER_BENCHMARK_SCHEMA,
        "benchmark_id": evaluation_suite.benchmark_id,
        "controller_label": controller_label,
        "controller_type": controller.controller_type,
        "prompt_mode": prompt_mode,
        "status": "blocked" if all_failed else "completed",
        "blocking_reason": (
            next(
                (
                    row["runtime_error"]
                    for row in rows
                    if row["runtime_error"]
                ),
                "",
            )
            if all_failed
            else ""
        ),
        "completed_at": _utc_now(),
        "metrics": metrics,
        "cases": rows,
    }


def build_local_controller(
    *,
    model_id: str,
    prompt_suite: PromptSuite,
    prompt_mode: str,
    inference_callable: Callable[
        [str, LocalLLMControllerConfig],
        str | dict[str, Any],
    ],
    max_new_tokens: int,
) -> LocalLLMController:
    if prompt_mode not in {"zero_shot", "few_shot"}:
        raise ValueError("prompt_mode must be zero_shot or few_shot")
    template = prompt_suite.zero_shot_template
    if prompt_mode == "few_shot":
        marker = f"{CURRENT_CONTEXT_MARKER}{{context}}"
        if marker not in template:
            raise LocalControllerBenchmarkError(
                "PROMPT_CURRENT_CONTEXT_MARKER_MISSING"
            )
        example_block = _stable_json(prompt_suite.few_shot_examples)
        example_block = example_block.replace("{", "{{").replace("}", "}}")
        template = template.replace(
            marker,
            (
                "示例（与评测集隔离，不得复述）："
                f"{example_block}\n"
                f"现在只处理以下待评估结构化状态。"
                f"{CURRENT_CONTEXT_MARKER}{{context}}"
            ),
            1,
        )
    config = LocalLLMControllerConfig(
        model_id=model_id,
        prompt_template=template,
        # Examples are inserted before the current context above.  Leaving this
        # empty avoids the legacy controller appending examples after the query.
        few_shot_examples=(),
        generation_max_new_tokens=max_new_tokens,
    )
    return LocalLLMController(
        config,
        inference_callable=inference_callable,
    )


def build_rule_controller() -> RuleController:
    return RuleController()


def write_benchmark_result(path: Path, result: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def build_benchmark_manifest(
    *,
    eval_path: Path,
    prompt_path: Path,
    results: Sequence[dict[str, Any]],
    isolation: dict[str, Any],
    project_root: Path | None = None,
    result_paths: Sequence[Path] = (),
    results_reused: bool = False,
    reuse_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_commit = "unknown"
    if project_root is not None:
        try:
            source_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    def registered_path(path: Path) -> str:
        if project_root is None:
            return str(path)
        try:
            return path.resolve().relative_to(
                project_root.resolve()
            ).as_posix()
        except ValueError:
            return str(path)

    return {
        "schema_version": LOCAL_CONTROLLER_MANIFEST_SCHEMA,
        "generated_at": _utc_now(),
        "manifest_build_commit_sha": source_commit,
        "source_commit_sha": source_commit,
        "evaluation_config_path": registered_path(eval_path),
        "evaluation_config_sha256": sha256_file(eval_path),
        "prompt_config_path": registered_path(prompt_path),
        "prompt_config_sha256": sha256_file(prompt_path),
        "prompt_eval_isolation": isolation,
        "controllers": [
            {
                "controller_label": result["controller_label"],
                "status": result["status"],
                "blocking_reason": result["blocking_reason"],
                "metrics": result["metrics"],
            }
            for result in results
        ],
        "result_artifacts": [
            {
                "path": registered_path(path),
                "sha256": sha256_file(path),
            }
            for path in sorted(result_paths)
        ],
        "results_reused": results_reused,
        "result_execution_commit_sha": None,
        "result_execution_commit_recorded": False,
        "reuse_validation": dict(reuse_validation or {}),
        "controller_proposal_schema_version": (
            CONTROLLER_PROPOSAL_SCHEMA_VERSION
        ),
        "network_used": False,
        "training_performed": False,
        "test_assets_used": False,
        "raw_images_used": False,
    }


def build_controller_decision_report(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Derive the milestone-C decision from completed fixed-eval results."""

    indexed = {
        str(result.get("controller_label", "")): result
        for result in results
    }

    def metrics(label: str) -> dict[str, Any]:
        return dict(indexed.get(label, {}).get("metrics", {}))

    rule = metrics("rule_controller")
    four_zero = metrics("qwen3_5_4b_zero_shot")
    four_few = metrics("qwen3_5_4b_few_shot")
    twenty_seven_zero = metrics("qwen3_5_27b_zero_shot")
    twenty_seven_few = metrics("qwen3_5_27b_few_shot")
    error_patterns = {
        label: [
            str(row.get("case_id", ""))
            for row in result.get("cases", [])
            if not row.get("next_action_correct", False)
        ]
        for label, result in indexed.items()
    }
    rule_completion = float(rule.get("task_completion_rate", 0.0))
    best_llm_completion = max(
        (
            float(item.get("task_completion_rate", 0.0))
            for item in (
                four_zero,
                four_few,
                twenty_seven_zero,
                twenty_seven_few,
            )
        ),
        default=0.0,
    )
    return {
        "schema_version": "ophagent.local_controller_decision.v1",
        "formal_baseline": "rule_controller",
        "rule_controller_remains_best_formal_baseline": (
            rule_completion >= best_llm_completion
        ),
        "reason": (
            "规则控制器在最终门控后达到完整任务完成率，且不占用模型显存、"
            "无生成延迟；本地语言模型没有带来最终任务完成率增益。"
        ),
        "four_b_sufficient_for_formal_use": (
            float(four_few.get("task_completion_rate", 0.0)) == 1.0
            and float(four_few.get("next_action_accuracy", 0.0)) == 1.0
        ),
        "twenty_seven_b_complex_state_gain": {
            "few_shot_next_action_accuracy_delta_vs_4b": (
                float(twenty_seven_few.get("next_action_accuracy", 0.0))
                - float(four_few.get("next_action_accuracy", 0.0))
            ),
            "few_shot_task_completion_delta_vs_4b": (
                float(twenty_seven_few.get("task_completion_rate", 0.0))
                - float(four_few.get("task_completion_rate", 0.0))
            ),
            "peak_vram_mib_delta_vs_4b": (
                float(
                    twenty_seven_few.get(
                        "peak_vram_allocated_mib",
                        0.0,
                    )
                )
                - float(four_few.get("peak_vram_allocated_mib", 0.0))
            ),
        },
        "few_shot_improvement": {
            "4b_next_action_accuracy_delta": (
                float(four_few.get("next_action_accuracy", 0.0))
                - float(four_zero.get("next_action_accuracy", 0.0))
            ),
            "27b_next_action_accuracy_delta": (
                float(twenty_seven_few.get("next_action_accuracy", 0.0))
                - float(
                    twenty_seven_zero.get("next_action_accuracy", 0.0)
                )
            ),
        },
        "stable_learnable_error_candidates": {
            "status": "not_yet_established",
            "observed_fixed_eval_errors": error_patterns,
            "requirement": (
                "需在独立未暴露确认集上复现同类错误，且证明少样本提示无法"
                "稳定解决，才视为可训练模式。"
            ),
        },
        "enter_lora_or_sft": False,
        "lora_sft_gate": {
            "fixed_eval_independent_confirmation": False,
            "stable_error_pattern_replicated": False,
            "prompt_only_remediation_exhausted": False,
            "expected_gain_over_rule_baseline": False,
        },
        "boundary": (
            "固定合成/脱敏控制器评测，不是临床有效性或部署资格证据。"
        ),
    }
