from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.local_controller_benchmark import (
    CURRENT_CONTEXT_MARKER,
    LocalModelRuntimeError,
    LocalTransformersInference,
    LocalTransformersRuntimeConfig,
    ScriptedLocalInference,
    SensitiveControllerContextError,
    build_controller_decision_report,
    build_local_controller,
    ensure_controller_context_safe,
    load_evaluation_suite,
    load_prompt_suite,
    run_controller_benchmark,
    sha256_file,
    validate_raw_proposal_payload,
    verify_prompt_eval_isolation,
)
from app.model_hub_agent_v2 import LocalLLMController
from scripts.routing.run_local_controller_benchmark import (
    EXPECTED_REUSED_RESULTS,
    _load_reusable_results,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_CONFIG_ROOT = (
    ROOT
    / "experiments"
    / "opening_risk_routing_closure"
    / "configs"
    / "controllers"
)
PROMPT_PATH = CONTROLLER_CONFIG_ROOT / "local_controller_prompt_v1.json"
EVAL_PATH = CONTROLLER_CONFIG_ROOT / "local_controller_eval_v1.json"


def _proposal(
    action: str,
    reason_code: str,
) -> dict[str, object]:
    return {
        "action": action,
        "reason_code": reason_code,
        "parameters": {},
        "schema_version": "ophagent.controller_proposal.v1",
    }


def _scripted_decision(
    prompt: str,
    _config: object,
) -> dict[str, object]:
    context = json.loads(prompt.rsplit(CURRENT_CONTEXT_MARKER, 1)[-1])
    tool_codes = set(context["tool_return_codes"])
    qualification = context["model_qualification"]
    risk = context["risk_result"]
    allowed_actions = set(context["allowed_actions"])
    if "TOOL_EXECUTION_FAILED" in tool_codes:
        return _proposal("REFER_TO_HUMAN", "TOOL_FAILURE")
    if qualification["execution_level"] in {
        "blocked",
        "research_replay_only",
    }:
        return _proposal("REFER_TO_HUMAN", "QUALIFICATION_RESTRICTED")
    if allowed_actions == {"REFER_TO_HUMAN"}:
        return _proposal("REFER_TO_HUMAN", "STATE_REQUIRES_HUMAN")
    if risk["protocol_requests_expert"]:
        return _proposal(
            "REQUEST_EXPERT",
            "HIGH_RISK_REQUEST_EXPERT",
        )
    if risk["model_disagreement"]:
        return _proposal("REQUEST_EXPERT", "MODEL_DISAGREEMENT")
    return _proposal("KEEP_SCOUT", "LOW_RISK_KEEP_SCOUT")


def test_fixed_prompt_and_eval_are_isolated() -> None:
    prompt = load_prompt_suite(PROMPT_PATH)
    evaluation = load_evaluation_suite(EVAL_PATH)

    audit = verify_prompt_eval_isolation(prompt, evaluation)

    assert audit["isolation_passed"] is True
    assert audit["id_overlap_count"] == 0
    assert audit["exact_context_overlap_count"] == 0
    assert len(evaluation.cases) >= 12


def test_same_local_llm_controller_implementation_handles_4b_and_27b() -> None:
    prompt = load_prompt_suite(PROMPT_PATH)
    runtime = ScriptedLocalInference(_scripted_decision)

    controller_4b = build_local_controller(
        model_id="Qwen/Qwen3.5-4B",
        prompt_suite=prompt,
        prompt_mode="zero_shot",
        inference_callable=runtime,
        max_new_tokens=128,
    )
    controller_27b = build_local_controller(
        model_id="Qwen/Qwen3.5-27B",
        prompt_suite=prompt,
        prompt_mode="few_shot",
        inference_callable=runtime,
        max_new_tokens=128,
    )

    assert type(controller_4b) is LocalLLMController
    assert type(controller_27b) is LocalLLMController
    assert controller_4b.config.model_id.endswith("4B")
    assert controller_27b.config.model_id.endswith("27B")


def test_nested_sensitive_fields_are_rejected_before_prompting() -> None:
    context = {
        "current_state": "RISK_AUDITED",
        "task_id": "synthetic",
        "allowed_actions": ["REFER_TO_HUMAN"],
        "model_qualification": {
            "execution_level": "blocked",
            "evidence": {
                "patient_name": "should-never-leak",
                "private_path": "/private/case.png",
            },
        },
        "risk_result": {},
        "remaining_budget": 1.0,
        "tool_return_codes": ["OK"],
        "case_metadata": {},
    }

    with pytest.raises(SensitiveControllerContextError):
        ensure_controller_context_safe(context)


@pytest.mark.parametrize(
    "payload,expected_error",
    [
        (
            {
                **_proposal("KEEP_SCOUT", "LOW_RISK_KEEP_SCOUT"),
                "tool": "model.run",
            },
            "PROPOSAL_SCHEMA_FIELDS_EXTRA",
        ),
        (
            {
                **_proposal("KEEP_SCOUT", "LOW_RISK_KEEP_SCOUT"),
                "parameters": [],
            },
            "PROPOSAL_PARAMETERS_NOT_OBJECT",
        ),
        (
            _proposal("DELETE_PATIENT", "INVALID_PROPOSAL"),
            "PROPOSAL_ACTION_INVALID",
        ),
    ],
)
def test_raw_schema_validation_is_strict(
    payload: dict[str, object],
    expected_error: str,
) -> None:
    valid, errors = validate_raw_proposal_payload(payload)

    assert valid is False
    assert expected_error in errors


def test_illegal_local_model_output_is_intercepted() -> None:
    prompt = load_prompt_suite(PROMPT_PATH)
    evaluation = load_evaluation_suite(EVAL_PATH)
    runtime = ScriptedLocalInference(
        lambda _prompt, _config: _proposal(
            "DELETE_PATIENT",
            "INVALID_PROPOSAL",
        )
    )
    controller = build_local_controller(
        model_id="mock-local",
        prompt_suite=prompt,
        prompt_mode="zero_shot",
        inference_callable=runtime,
        max_new_tokens=128,
    )

    result = run_controller_benchmark(
        controller=controller,
        evaluation_suite=type(evaluation)(
            benchmark_id=evaluation.benchmark_id,
            split=evaluation.split,
            cases=(evaluation.cases[0],),
        ),
        controller_label="mock_illegal",
        prompt_mode="zero_shot",
        inference_callable=runtime,
    )

    row = result["cases"][0]
    assert row["final_action"] == "REFER_TO_HUMAN"
    assert row["gate_intercepted"] is True
    assert row["raw_schema_valid"] is False
    assert row["runtime_error"]


def test_mock_local_controller_runs_fixed_normal_and_abnormal_eval() -> None:
    prompt = load_prompt_suite(PROMPT_PATH)
    evaluation = load_evaluation_suite(EVAL_PATH)
    runtime = ScriptedLocalInference(_scripted_decision)
    controller = build_local_controller(
        model_id="mock-local",
        prompt_suite=prompt,
        prompt_mode="zero_shot",
        inference_callable=runtime,
        max_new_tokens=128,
    )

    result = run_controller_benchmark(
        controller=controller,
        evaluation_suite=evaluation,
        controller_label="mock_local_zero_shot",
        prompt_mode="zero_shot",
        inference_callable=runtime,
    )

    assert result["status"] == "completed"
    assert result["metrics"]["schema_valid_rate"] == 1.0
    assert result["metrics"]["legal_action_rate"] == 1.0
    assert result["metrics"]["task_completion_rate"] == 1.0
    assert result["metrics"]["gate_intercept_count"] >= 2


def test_local_transformers_contract_fails_closed_before_network(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "not-downloaded"

    with pytest.raises(LocalModelRuntimeError, match="LOCAL_MODEL_PATH_MISSING"):
        LocalTransformersInference(
            LocalTransformersRuntimeConfig(
                model_id="missing-model",
                model_path=missing,
            )
        )


def test_configs_contain_no_real_case_or_test_assets() -> None:
    prompt = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(
        {"prompt": prompt, "evaluation": evaluation},
        ensure_ascii=False,
    ).lower()

    assert evaluation["synthetic_or_deidentified_only"] is True
    assert evaluation["frozen_test_assets_used"] is False
    assert evaluation["raw_images_used"] is False
    assert "aptos" not in serialized
    assert "deepdrid" not in serialized
    assert "rim-one" not in serialized
    assert "trhd59" not in serialized
    assert "patient_name" not in serialized
    assert "private_path" not in serialized


def test_decision_report_keeps_rule_baseline_and_blocks_lora() -> None:
    def result(
        label: str,
        *,
        accuracy: float,
        completion: float,
        vram: float,
    ) -> dict[str, object]:
        return {
            "controller_label": label,
            "metrics": {
                "next_action_accuracy": accuracy,
                "task_completion_rate": completion,
                "peak_vram_allocated_mib": vram,
            },
            "cases": [],
        }

    report = build_controller_decision_report(
        [
            result(
                "rule_controller",
                accuracy=0.83,
                completion=1.0,
                vram=0.0,
            ),
            result(
                "qwen3_5_4b_zero_shot",
                accuracy=0.33,
                completion=0.5,
                vram=9000.0,
            ),
            result(
                "qwen3_5_4b_few_shot",
                accuracy=0.92,
                completion=0.92,
                vram=9300.0,
            ),
            result(
                "qwen3_5_27b_zero_shot",
                accuracy=0.67,
                completion=0.83,
                vram=52500.0,
            ),
            result(
                "qwen3_5_27b_few_shot",
                accuracy=1.0,
                completion=1.0,
                vram=53000.0,
            ),
        ]
    )

    assert report["formal_baseline"] == "rule_controller"
    assert report["rule_controller_remains_best_formal_baseline"] is True
    assert report["four_b_sufficient_for_formal_use"] is False
    assert report["enter_lora_or_sft"] is False


def _write_reuse_fixture(output_dir: Path) -> object:
    evaluation = load_evaluation_suite(EVAL_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, (prompt_mode, controller_type) in (
        EXPECTED_REUSED_RESULTS.items()
    ):
        payload = {
            "schema_version": "ophagent.local_controller_benchmark.v1",
            "benchmark_id": evaluation.benchmark_id,
            "controller_label": label,
            "controller_type": controller_type,
            "prompt_mode": prompt_mode,
            "status": "completed",
            "blocking_reason": "",
            "metrics": {"case_count": len(evaluation.cases)},
            "cases": [
                {"case_id": item.case_id} for item in evaluation.cases
            ],
        }
        (output_dir / f"{label}_results.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "evaluation_config_sha256": sha256_file(EVAL_PATH),
                "prompt_config_sha256": sha256_file(PROMPT_PATH),
            }
        ),
        encoding="utf-8",
    )
    return evaluation


def test_reuse_results_requires_exact_completed_bound_set(
    tmp_path: Path,
) -> None:
    evaluation = _write_reuse_fixture(tmp_path)

    results, audit = _load_reusable_results(
        output_dir=tmp_path,
        evaluation_suite=evaluation,
        eval_path=EVAL_PATH,
        prompt_path=PROMPT_PATH,
    )

    assert len(results) == 5
    assert audit["validated"] is True
    assert audit["model_inference_rerun"] is False


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing", "REUSE_RESULT_FILE_SET_MISMATCH"),
        ("extra", "REUSE_RESULT_FILE_SET_MISMATCH"),
        ("blocked", "REUSE_RESULT_VALIDATION_FAILED"),
        ("duplicate", "REUSE_RESULT_LABEL_MISMATCH"),
        ("config", "REUSE_EVALUATION_CONFIG_SHA_MISMATCH"),
    ],
)
def test_reuse_results_fails_closed_on_provenance_mismatch(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    evaluation = _write_reuse_fixture(tmp_path)
    target = tmp_path / "qwen3_5_27b_few_shot_results.json"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (tmp_path / "stale_results.json").write_text(
            "{}",
            encoding="utf-8",
        )
    elif mutation in {"blocked", "duplicate"}:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if mutation == "blocked":
            payload["status"] = "blocked"
        else:
            payload["controller_label"] = "qwen3_5_4b_few_shot"
        target.write_text(json.dumps(payload), encoding="utf-8")
    else:
        manifest_path = tmp_path / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation_config_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        _load_reusable_results(
            output_dir=tmp_path,
            evaluation_suite=evaluation,
            eval_path=EVAL_PATH,
            prompt_path=PROMPT_PATH,
        )
