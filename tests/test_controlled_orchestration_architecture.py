from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pytest

from app.model_hub_agent_v2 import (
    AgentExpertResult,
    AgentToolBundle,
    AgentToolStepResult,
    CaseStateStore,
    ControlledAgentRuntimeV2,
    ControlledCaseRequest,
    IdempotencyConflict,
    LocalLLMController,
    LocalLLMControllerConfig,
    PermissionDenied,
    RuleController,
    authorize,
    state_view_model,
)
from app.orchestration_contracts import (
    CallableModelRuntimeAdapter,
    ConfiguredTaskAdapter,
    ModelCapability,
    RouteQualification,
    TaskSpec,
    redact_free_text,
    redact_structured_value,
)
from app.route_qualification import (
    RouteQualificationRequest,
    evaluate_route_qualification,
    load_route_qualification_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _qualification() -> RouteQualification:
    return RouteQualification(
        execution_level="research_case_simulation",
        evidence_label="beneficial",
        allow_cached_replay=True,
        allow_case_simulation=True,
        allow_new_case_route=False,
        clinical_route_eligible=False,
        human_confirmation_required=True,
        error_codes=("RQ_CLINICAL_ROUTE_NEVER_AUTO_GRANTED",),
        contract_sha256="a" * 64,
        evidence_fingerprint="b" * 64,
        evidence={"pairing_id": "synthetic"},
    )


def _request(*, case_id: str = "CASE-001") -> ControlledCaseRequest:
    return ControlledCaseRequest(
        case_id=case_id,
        task_id="synthetic_task",
        idempotency_key=f"idem-{case_id}",
        case_scope="cached_prediction_replay",
        case_metadata={
            "modality": "CFP",
            "image_count": 1,
            "patient_name": "must-not-persist",
            "private_path": "/private/case.png",
        },
        remaining_budget=3.0,
        expected_expert_cost=1.0,
        qualification_policy_version="v1.1",
        route_protocol_version="frozen-v1",
    )


def _tools(*, expert: bool = False, failed: bool = False):
    response = {
        "ok": not failed,
        "code": "TOOL_EXECUTION_FAILED" if failed else "OK",
        "data": {},
    }
    return {
        "input": dict(response),
        "registry": dict(response),
        "predictions": (
            []
            if failed
            else [{"artifact_id": "scout", "probabilities": [0.8, 0.2]}]
        ),
        "audit": {
            **response,
            "data": {
                "model_disagreement": expert,
            },
        },
        "route": {
            **response,
            "data": {"expert_invoked": expert},
        },
    }


def test_new_task_profile_does_not_change_state_machine() -> None:
    spec = TaskSpec(
        task_id="synthetic_task",
        dataset_id="synthetic_public",
        modality="OCT",
        label_space=("negative", "positive"),
        primary_metric="macro_f1",
        risk_semantics="positive_probability_research_proxy",
        report_label="合成任务",
        risk_positive_class_ids=(1,),
    )
    adapter = ConfiguredTaskAdapter(spec)

    assert adapter.validate_metadata({"modality": "OCT", "image_count": 2}) == (
        True,
        "TASK_METADATA_OK",
    )
    assert adapter.risk_summary((0.25, 0.75))["value"] == 0.75


def test_fake_model_adapter_does_not_change_agent_core() -> None:
    capability = ModelCapability(
        task_id="synthetic_task",
        artifact_id="fake-model",
        adapter_type="fake",
        prediction_asset_available=True,
        offline_batch_inference_ready=True,
        online_case_inference_ready=False,
        cost_protocol_id="test-cost-v1",
        cost_ms_per_image=0.1,
        qualification_status="test_only",
    )
    adapter = CallableModelRuntimeAdapter(
        capability,
        lambda _: {"probabilities": [0.7, 0.3]},
    )

    assert adapter.infer(object())["probabilities"] == [0.7, 0.3]
    assert adapter.capability.online_case_inference_ready is False


def test_case_state_store_normalizes_numpy_scalars(tmp_path: Path) -> None:
    store = CaseStateStore(tmp_path)
    state, _ = ControlledAgentRuntimeV2(store).run(
        _request(case_id="NUMPY-SCALAR-001"),
        qualification=_qualification(),
        tool_payload=_tools(),
        controller=RuleController(),
    )
    state.runtime_payload["numpy_scalars"] = {
        "flag": np.bool_(True),
        "count": np.int64(2),
        "score": np.float64(0.75),
    }

    store.save(state)
    restored = store.load(state.case_id)

    assert restored is not None
    assert restored.runtime_payload["numpy_scalars"] == {
        "flag": True,
        "count": 2,
        "score": 0.75,
    }


def test_redaction_preserves_contract_names_but_removes_identity_names() -> None:
    redacted = redact_structured_value(
        {
            "tool_name": "model_registry.inspect",
            "model_name": "public-model",
            "task_name": "public-task",
            "name": "sensitive",
            "patient_name": "sensitive",
            "姓名": "sensitive",
        }
    )

    assert redacted == {
        "tool_name": "model_registry.inspect",
        "model_name": "public-model",
        "task_name": "public-task",
    }


def test_rule_and_mock_local_llm_share_controller_interface(
    tmp_path: Path,
) -> None:
    store = CaseStateStore(tmp_path)
    rule_state, _ = ControlledAgentRuntimeV2(store).run(
        _request(case_id="RULE-001"),
        qualification=_qualification(),
        tool_payload=_tools(),
        controller=RuleController(),
    )
    mock = LocalLLMController(
        LocalLLMControllerConfig(model_id="mock-4b"),
        inference_callable=lambda _prompt, _config: {
            "action": "KEEP_SCOUT",
            "reason_code": "LOW_RISK_KEEP_SCOUT",
            "parameters": {},
            "schema_version": "ophagent.controller_proposal.v1",
        },
    )
    llm_state, _ = ControlledAgentRuntimeV2(store).run(
        _request(case_id="LLM-001"),
        qualification=_qualification(),
        tool_payload=_tools(),
        controller=mock,
    )

    assert rule_state.final_action == "KEEP_SCOUT"
    assert llm_state.final_action == "KEEP_SCOUT"
    assert rule_state.controller_type == "rule_controller"
    assert llm_state.controller_type == "local_llm_controller"


def test_illegal_controller_proposal_cannot_bypass_gate(
    tmp_path: Path,
) -> None:
    controller = LocalLLMController(
        LocalLLMControllerConfig(model_id="mock-illegal"),
        inference_callable=lambda _prompt, _config: {
            "action": "DELETE_PATIENT",
            "reason_code": "FREE_PLAN",
            "parameters": {"tool": "model_inference.run"},
            "schema_version": "unknown",
        },
    )
    state, _ = ControlledAgentRuntimeV2(CaseStateStore(tmp_path)).run(
        _request(),
        qualification=_qualification(),
        tool_payload=_tools(expert=True),
        controller=controller,
    )

    assert state.final_action == "REFER_TO_HUMAN"
    assert state.gate_decision["gate_intercepted"] is True
    assert state.gate_decision["code"] == "GATE_CONTROLLER_PROPOSAL_INVALID"


def test_view_model_and_report_share_one_qualification(
    tmp_path: Path,
) -> None:
    store = CaseStateStore(tmp_path)
    state, _ = ControlledAgentRuntimeV2(store).run(
        _request(),
        qualification=_qualification(),
        tool_payload=_tools(),
        controller=RuleController(),
    )
    view = state_view_model(state)

    assert view["qualification"] == state.report["qualification"]
    assert view["qualification"]["contract_sha256"] == "a" * 64
    assert view["final_action"] == state.report["final_action"]


def test_shared_v1_1_gate_decision_flows_into_agent_runtime(
    tmp_path: Path,
) -> None:
    contract, contract_sha = load_route_qualification_contract(ROOT)
    decision = evaluate_route_qualification(
        RouteQualificationRequest(
            task_id="aptos_dr_5class",
            pairing_id="synthetic-gate-agent",
            scout_artifact_ids=("synthetic-scout",),
            expert_artifact_id="synthetic-expert",
            request_scope="cached_prediction_replay",
            protocol_frozen=True,
            selection_split="validation",
            validation_main_metric_delta=0.1,
            validation_delta_vs_best_single=0.05,
            validation_corrected=1.0,
            validation_introduced=0.0,
            validation_net=1.0,
            stability_ci_lower=0.01,
            requested_budget=0.1,
            expected_cost_ms_per_image=5.0,
            protocol_sha256="c" * 64,
            input_asset_fingerprint="d" * 64,
            cost_protocol_complete=True,
            cost_protocol_id=(
                "h100_fp32_forward_only_batch1_batch16_w10_r30_v1"
            ),
            cost_protocol_comparable=True,
            expert_budget=0.1,
            domain_shift_status="in_domain",
            adaptation_type="task_native",
            task_adapter_compatible=True,
            unique_protocol_identity=True,
            validation_result_sha256="e" * 64,
            frozen_result_sha256="f" * 64,
            risk_evidence_available=True,
            historical_replay_eligible=True,
        ),
        contract=contract,
        contract_sha256=contract_sha,
    )
    qualification = RouteQualification.from_decision(
        decision,
        evidence={"pairing_id": "synthetic-gate-agent"},
    )
    state, _ = ControlledAgentRuntimeV2(
        CaseStateStore(tmp_path)
    ).run(
        _request(case_id="SHARED-GATE-001"),
        qualification=qualification,
        tool_payload=_tools(),
        controller=RuleController(),
    )

    assert decision.execution_level == "research_case_simulation"
    assert state.qualification["contract_sha256"] == contract_sha
    assert state.final_action == "KEEP_SCOUT"


def test_persistent_idempotency_refresh_restart_and_rbac(
    tmp_path: Path,
) -> None:
    first_runtime = ControlledAgentRuntimeV2(CaseStateStore(tmp_path))
    first, replayed = first_runtime.run(
        _request(case_id="RECOVER-001"),
        qualification=_qualification(),
        tool_payload=_tools(expert=True),
        controller=RuleController(),
    )
    second_runtime = ControlledAgentRuntimeV2(CaseStateStore(tmp_path))
    restored, second_replayed = second_runtime.run(
        _request(case_id="RECOVER-001"),
        qualification=_qualification(),
        tool_payload=_tools(expert=True),
        controller=RuleController(),
    )

    assert replayed is False
    assert second_replayed is True
    assert restored.trace == first.trace
    assert restored.current_state == "EXPERT_PENDING_APPROVAL"
    expert_calls: list[str] = []
    approved = second_runtime.decide_expert(
        "RECOVER-001",
        approved=True,
        actor_role="reviewer",
        expert_executor=lambda state: (
            expert_calls.append(state.case_id)
            or AgentExpertResult(
                tool_payload={
                    "expert": {
                        "ok": True,
                        "code": "OK",
                        "data": {"probabilities": [0.1, 0.9]},
                    }
                },
                tool_trace={"events": []},
            )
        ),
    )
    assert approved.current_state == "REVIEW_PENDING"
    assert expert_calls == ["RECOVER-001"]
    assert approved.runtime_payload["expert"]["ok"] is True
    with pytest.raises(PermissionDenied):
        authorize("operator", "protocol.modify")

    serialized = (tmp_path / "RECOVER-001.json").read_text(encoding="utf-8")
    assert "must-not-persist" not in serialized
    assert "/private/case.png" not in serialized
    assert json.loads(serialized)["schema_version"] == "ophagent.case_state.v2"


def test_retry_resumes_after_last_completed_step_without_repetition(
    tmp_path: Path,
) -> None:
    runtime = ControlledAgentRuntimeV2(CaseStateStore(tmp_path))
    request = _request(case_id="RETRY-001")
    failed_payload = {
        "input": {"ok": True, "code": "OK", "data": {}},
        "registry": {"ok": True, "code": "OK", "data": {}},
        "predictions": [],
        "inference": {
            "ok": False,
            "code": "TOOL_EXECUTION_FAILED",
            "data": {},
        },
        "audit": {
            "ok": False,
            "code": "UPSTREAM_FAILED",
            "data": {},
        },
    }
    failed, _ = runtime.run(
        request,
        qualification=_qualification(),
        tool_payload=failed_payload,
        controller=RuleController(),
    )

    assert failed.current_state == "FAILED"
    assert failed.completed_steps == (
        "input_check",
        "task_recognition",
        "model_qualification",
    )
    retrying = runtime.retry_failed(
        request.case_id,
        actor_role="operator",
    )
    assert retrying.current_state == "CASE_VALIDATED"

    recovered, replayed = runtime.run(
        request,
        qualification=_qualification(),
        tool_payload=_tools(),
        controller=RuleController(),
    )

    assert replayed is False
    assert recovered.current_state == "REVIEW_PENDING"
    assert [event["event"] for event in recovered.trace].count(
        "validate_case"
    ) == 1
    assert recovered.completed_steps.count("scout") == 1


def test_completed_case_rejects_changed_qualification_evidence(
    tmp_path: Path,
) -> None:
    runtime = ControlledAgentRuntimeV2(CaseStateStore(tmp_path))
    request = _request(case_id="EVIDENCE-001")
    runtime.run(
        request,
        qualification=_qualification(),
        tool_payload=_tools(),
        controller=RuleController(),
    )

    with pytest.raises(
        IdempotencyConflict,
        match="AGENT_QUALIFICATION_EVIDENCE_CHANGED",
    ):
        runtime.run(
            request,
            qualification=replace(
                _qualification(),
                evidence_fingerprint="c" * 64,
            ),
            tool_payload=_tools(),
            controller=RuleController(),
        )


def test_concurrent_duplicate_executes_tools_once(tmp_path: Path) -> None:
    request = _request(case_id="CONCURRENT-001")
    calls: list[str] = []

    def execute_tools(_state: object) -> AgentToolBundle:
        calls.append("dispatch")
        time.sleep(0.15)
        return AgentToolBundle(
            qualification=_qualification(),
            tool_payload=_tools(),
            tool_trace={"events": []},
        )

    def submit() -> tuple[str, bool]:
        state, replayed = ControlledAgentRuntimeV2(
            CaseStateStore(tmp_path)
        ).execute(
            request,
            controller=RuleController(),
            tool_executor=execute_tools,
        )
        return state.current_state, replayed

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: submit(), range(2)))

    assert calls == ["dispatch"]
    assert {state for state, _ in results} == {"REVIEW_PENDING"}
    assert sorted(replayed for _, replayed in results) == [False, True]


def test_controller_crash_resumes_from_risk_audited(
    tmp_path: Path,
) -> None:
    request = _request(case_id="AUDIT-RESUME-001")

    class CrashController:
        controller_type = "rule_controller"

        def propose(self, _context: dict[str, object]) -> object:
            raise KeyboardInterrupt

    runtime = ControlledAgentRuntimeV2(CaseStateStore(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        runtime.run(
            request,
            qualification=_qualification(),
            tool_payload=_tools(),
            controller=CrashController(),
        )
    persisted = CaseStateStore(tmp_path).load(request.case_id)
    assert persisted is not None
    assert persisted.current_state == "RISK_AUDITED"

    def must_not_run(_state: object) -> AgentToolBundle:
        raise AssertionError("completed tools must not run again")

    restored, replayed = ControlledAgentRuntimeV2(
        CaseStateStore(tmp_path)
    ).execute(
        request,
        controller=RuleController(),
        tool_executor=must_not_run,
    )
    assert replayed is True
    assert restored.current_state == "REVIEW_PENDING"
    assert [
        item["event"] for item in restored.trace
    ].count("audit_risk_and_qualification") == 1


def test_stepwise_restart_resumes_at_first_unfinished_contract(
    tmp_path: Path,
) -> None:
    request = _request(case_id="STEP-RESUME-001")
    calls: list[str] = []

    class StepExecutor:
        def __init__(self, *, crash_audit: bool):
            self.crash_audit = crash_audit

        def execute_step(
            self,
            state: object,
            step: str,
        ) -> AgentToolStepResult:
            calls.append(step)
            if step == "input":
                return AgentToolStepResult(
                    {"input": {"ok": True, "code": "OK", "data": {}}},
                    {"events": []},
                )
            if step == "registry":
                return AgentToolStepResult(
                    {
                        "registry": {
                            "ok": True,
                            "code": "OK",
                            "data": {},
                        }
                    },
                    {"events": []},
                )
            if step == "route_metadata":
                return AgentToolStepResult(
                    {
                        "route": {
                            "ok": True,
                            "code": "OK",
                            "data": {
                                "protocol_requests_expert": False
                            },
                        }
                    },
                    {"events": []},
                )
            if step == "scout":
                return AgentToolStepResult(
                    {
                        "predictions": [
                            {
                                "artifact_id": "scout",
                                "probabilities": [0.8, 0.2],
                            }
                        ]
                    },
                    {"events": []},
                )
            if self.crash_audit:
                raise KeyboardInterrupt
            return AgentToolStepResult(
                {
                    "audit": {
                        "ok": True,
                        "code": "OK",
                        "data": {"model_disagreement": False},
                    }
                },
                {"events": []},
                _qualification(),
            )

    with pytest.raises(KeyboardInterrupt):
        ControlledAgentRuntimeV2(CaseStateStore(tmp_path)).execute(
            request,
            controller=RuleController(),
            tool_executor=StepExecutor(crash_audit=True),
        )
    persisted = CaseStateStore(tmp_path).load(request.case_id)
    assert persisted is not None
    assert persisted.current_state == "SCOUT_COMPLETED"

    restored, replayed = ControlledAgentRuntimeV2(
        CaseStateStore(tmp_path)
    ).execute(
        request,
        controller=RuleController(),
        tool_executor=StepExecutor(crash_audit=False),
    )
    assert replayed is True
    assert restored.current_state == "REVIEW_PENDING"
    assert calls == [
        "input",
        "registry",
        "route_metadata",
        "scout",
        "audit_and_qualification",
        "audit_and_qualification",
    ]


def test_expert_rejection_never_executes_or_persists_expert(
    tmp_path: Path,
) -> None:
    runtime = ControlledAgentRuntimeV2(CaseStateStore(tmp_path))
    request = _request(case_id="EXPERT-REJECT-001")
    pending, _ = runtime.run(
        request,
        qualification=_qualification(),
        tool_payload=_tools(expert=True),
        controller=RuleController(),
    )
    assert pending.current_state == "EXPERT_PENDING_APPROVAL"
    calls: list[str] = []
    rejected = runtime.decide_expert(
        request.case_id,
        approved=False,
        actor_role="reviewer",
        expert_executor=lambda _state: calls.append("called"),
    )
    repeated = runtime.decide_expert(
        request.case_id,
        approved=False,
        actor_role="reviewer",
    )

    assert calls == []
    assert rejected.current_state == "REVIEW_PENDING"
    assert repeated.to_dict() == rejected.to_dict()
    assert "expert" not in rejected.runtime_payload


def test_expert_tool_failure_retries_only_expert_step(
    tmp_path: Path,
) -> None:
    runtime = ControlledAgentRuntimeV2(CaseStateStore(tmp_path))
    request = _request(case_id="EXPERT-RETRY-001")
    pending, _ = runtime.run(
        request,
        qualification=_qualification(),
        tool_payload=_tools(expert=True),
        controller=RuleController(),
    )
    assert pending.current_state == "EXPERT_PENDING_APPROVAL"
    calls: list[str] = []

    failed = runtime.decide_expert(
        request.case_id,
        approved=True,
        actor_role="reviewer",
        expert_executor=lambda _state: (
            calls.append("failed")
            or AgentExpertResult(
                tool_payload={
                    "expert": {
                        "ok": False,
                        "code": "EXPERT_TOOL_EXECUTION_FAILED",
                        "data": {},
                    }
                },
                tool_trace={"events": []},
            )
        ),
    )
    assert failed.current_state == "FAILED"

    retrying = runtime.retry_failed(
        request.case_id,
        actor_role="operator",
    )
    assert retrying.current_state == "EXPERT_PENDING_APPROVAL"
    approved = runtime.decide_expert(
        request.case_id,
        approved=True,
        actor_role="reviewer",
        expert_executor=lambda _state: (
            calls.append("succeeded")
            or AgentExpertResult(
                tool_payload={
                    "expert": {
                        "ok": True,
                        "code": "OK",
                        "data": {"probabilities": [0.1, 0.9]},
                    }
                },
                tool_trace={"events": []},
            )
        ),
    )

    assert approved.current_state == "REVIEW_PENDING"
    assert calls == ["failed", "succeeded"]
    assert approved.remaining_budget == 2.0
    assert [
        item["event"] for item in approved.trace
    ].count("run_scout") == 1
    assert [
        item["event"] for item in approved.trace
    ].count("audit_risk_and_qualification") == 1


def test_free_text_redaction_removes_common_identifiers() -> None:
    value = (
        "MRN: A12345 邮箱 eye@example.org 电话 138-0013-8000 "
        "姓名：张三 路径 /private/case/image.png"
    )
    redacted = redact_free_text(value)

    for secret in (
        "A12345",
        "eye@example.org",
        "138-0013-8000",
        "张三",
        "/private/case/image.png",
    ):
        assert secret not in redacted
