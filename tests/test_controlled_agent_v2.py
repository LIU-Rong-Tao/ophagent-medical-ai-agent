from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.controlled_agent_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    EVALUATION_SET_RELATIVE_PATH,
    METRIC_SEMANTICS_VERSION,
    REQUIRED_SCENARIO_IDS,
    load_controller_evaluation_set,
    run_controlled_agent_benchmark,
    write_controlled_agent_benchmark_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = ROOT / EVALUATION_SET_RELATIVE_PATH


@pytest.fixture
def benchmark(tmp_path: Path) -> dict[str, object]:
    return run_controlled_agent_benchmark(
        EVALUATION_PATH,
        state_root=tmp_path / "state",
    )


def _row(
    benchmark: dict[str, object],
    version: str,
    scenario_id: str,
) -> dict[str, object]:
    rows = benchmark["rows"]
    assert isinstance(rows, list)
    return next(
        item
        for item in rows
        if item["controller_version"] == version
        and item["scenario_id"] == scenario_id
    )


def _summary(
    benchmark: dict[str, object],
    version: str,
) -> dict[str, object]:
    summaries = benchmark["summary"]
    assert isinstance(summaries, list)
    return next(
        item
        for item in summaries
        if item["controller_version"] == version
    )


def _execution_log(row: dict[str, object]) -> dict[str, object]:
    prefix = "synthetic_execution_log="
    notes = str(row["notes"])
    assert notes.startswith(prefix)
    payload = json.loads(notes[len(prefix) :])
    assert isinstance(payload, dict)
    return payload


def test_fixed_evaluation_set_has_twelve_deidentified_scenarios() -> None:
    evaluation = load_controller_evaluation_set(EVALUATION_PATH)

    assert len(evaluation["scenarios"]) == 12
    assert {
        item["scenario_id"] for item in evaluation["scenarios"]
    } == REQUIRED_SCENARIO_IDS
    assert evaluation["data_scope"] == (
        "synthetic_deidentified_no_locked_split"
    )
    serialized = EVALUATION_PATH.read_text(encoding="utf-8").lower()
    for forbidden in (
        "patient_name",
        "hospital_id",
        "image_path",
        "private_path",
        "locked_test",
    ):
        assert forbidden not in serialized


def test_v2_covers_all_expected_actions_states_and_gate_codes(
    benchmark: dict[str, object],
) -> None:
    expected = {
        "low_risk_keep_scout": (
            "KEEP_SCOUT",
            "REVIEW_PENDING",
            "GATE_APPROVED",
        ),
        "high_risk_request_expert": (
            "REQUEST_EXPERT",
            "REVIEW_PENDING",
            "GATE_APPROVED",
        ),
        "qualification_insufficient": (
            "REFER_TO_HUMAN",
            "REVIEW_PENDING",
            "GATE_CASE_SIMULATION_NOT_ELIGIBLE",
        ),
        "same_scout_and_expert": (
            "REFER_TO_HUMAN",
            "REVIEW_PENDING",
            "GATE_ROUTE_QUALIFICATION_BLOCKED",
        ),
        "offline_only_raw_image": (
            "REFER_TO_HUMAN",
            "FAILED",
            "QUALIFICATION_BLOCKED",
        ),
        "cost_over_budget": (
            "REFER_TO_HUMAN",
            "REVIEW_PENDING",
            "GATE_EXPERT_BUDGET_EXCEEDED",
        ),
        "frozen_result_reversal": (
            "REFER_TO_HUMAN",
            "REVIEW_PENDING",
            "GATE_CASE_SIMULATION_NOT_ELIGIBLE",
        ),
        "tool_failure_stops_downstream": (
            "REFER_TO_HUMAN",
            "FAILED",
            "TOOL_EXECUTION_FAILED",
        ),
    }
    for scenario_id, values in expected.items():
        row = _row(
            benchmark,
            "v2_rule_state_machine",
            scenario_id,
        )
        assert (
            row["actual_action"],
            row["actual_state"],
            row["actual_code"],
        ) == values
        assert row["scenario_passed"] is True
        assert row["legal_invocation"] is True


def test_v2_recovers_refresh_restart_and_idempotent_duplicate(
    benchmark: dict[str, object],
) -> None:
    recovery_expectations = {
        "page_refresh_restore": (
            "scout",
            "CASE_VALIDATED",
            {"input", "registry", "route_metadata"},
        ),
        "service_restart_restore": (
            "audit_and_qualification",
            "SCOUT_COMPLETED",
            {"input", "registry", "route_metadata", "scout"},
        ),
    }
    for scenario_id, expected in recovery_expectations.items():
        row = _row(
            benchmark,
            "v2_rule_state_machine",
            scenario_id,
        )
        assert row["recovery_applicable"] is True
        assert row["recovery_success"] is True
        assert row["duplicate_call_count"] == 0
        assert row["replayed"] is True
        execution = _execution_log(row)
        assert execution["execution_mode"] == "v2_stepwise"
        assert execution["recovery_checkpoint"] == expected[0]
        assert execution["recovery_checkpoint_state"] == expected[1]
        events = execution["events"]
        interrupted = [
            event
            for event in events
            if event["status"] == "interrupted"
        ]
        assert [event["step"] for event in interrupted] == [
            expected[0]
        ]
        for completed_step in expected[2]:
            assert sum(
                event["step"] == completed_step
                and event["status"] == "succeeded"
                for event in events
            ) == 1
        assert not any(
            event["duplicate_after_success"] for event in events
        )
    duplicate = _row(
        benchmark,
        "v2_rule_state_machine",
        "idempotent_duplicate_submission",
    )
    assert duplicate["idempotency_applicable"] is True
    assert duplicate["idempotency_success"] is True
    assert duplicate["duplicate_call_count"] == 0
    duplicate_events = _execution_log(duplicate)["events"]
    assert [
        event["step"]
        for event in duplicate_events
        if event["status"] == "succeeded"
    ] == [
        "input",
        "registry",
        "route_metadata",
        "scout",
        "audit_and_qualification",
    ]
    assert not any(
        event["duplicate_after_success"] for event in duplicate_events
    )


def test_v2_permission_and_failure_boundaries_are_enforced(
    benchmark: dict[str, object],
) -> None:
    permission = _row(
        benchmark,
        "v2_rule_state_machine",
        "unauthorized_protocol_modification",
    )
    assert permission["permission_probe_blocked"] is True
    assert permission["unauthorized_call_count"] == 0

    for scenario_id in (
        "offline_only_raw_image",
        "tool_failure_stops_downstream",
    ):
        failure = _row(
            benchmark,
            "v2_rule_state_machine",
            scenario_id,
        )
        assert failure["downstream_stopped_after_failure"] is True
        assert failure["expert_requested"] is False
        assert failure["trace_complete"] is True


def test_v1_v2_summary_reports_requested_metrics(
    benchmark: dict[str, object],
) -> None:
    assert benchmark["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert (
        benchmark["metric_semantics_version"]
        == METRIC_SEMANTICS_VERSION
    )
    v1 = _summary(benchmark, "v1_fixed_workflow")
    v2 = _summary(benchmark, "v2_rule_state_machine")
    expected_metrics = {
        "task_completion_rate",
        "legal_invocation_rate",
        "unauthorized_call_count",
        "expert_call_rate",
        "unnecessary_expert_call_count",
        "recovery_rate",
        "duplicate_call_count",
        "total_estimated_compute_cost_ms",
        "mean_latency_ms",
        "trace_completeness_rate",
    }
    assert expected_metrics.issubset(v2)
    assert v2["scenario_count"] == v1["scenario_count"] == 12
    assert v2["scenario_success_rate"] == 1.0
    assert v2["task_completion_rate"] < 1.0
    assert v1["task_completion_rate"] < 1.0
    assert v2["legal_invocation_rate"] == 1.0
    assert v1["unauthorized_call_count"] == 1
    assert v2["unauthorized_call_count"] == 0
    assert v1["recovery_rate"] == 0.0
    assert v2["recovery_rate"] == 1.0
    assert v1["duplicate_call_count"] == 3
    assert v2["duplicate_call_count"] == 0
    assert v1["total_estimated_compute_cost_ms"] > (
        v2["total_estimated_compute_cost_ms"]
    )
    assert v1["expert_call_rate"] == v2["expert_call_rate"]
    assert v2["unnecessary_expert_call_count"] == 0
    assert v2["trace_completeness_rate"] == 1.0


def test_expert_request_approval_execution_and_cost_are_distinct(
    benchmark: dict[str, object],
) -> None:
    high = _row(
        benchmark,
        "v2_rule_state_machine",
        "high_risk_request_expert",
    )
    low = _row(
        benchmark,
        "v2_rule_state_machine",
        "low_risk_keep_scout",
    )

    assert high["expert_requested"] is True
    assert high["expert_approved"] is True
    assert high["expert_executed"] is True
    assert high["estimated_compute_cost_ms"] > (
        low["estimated_compute_cost_ms"]
    )
    assert low["expert_requested"] is False
    assert low["expert_executed"] is False

    v1_high = _row(
        benchmark,
        "v1_fixed_workflow",
        "high_risk_request_expert",
    )
    assert v1_high["expert_requested"] is True
    assert v1_high["expert_approved"] is False
    assert v1_high["expert_executed"] is True
    v1_log = _execution_log(v1_high)
    assert v1_log["expert_lifecycle"] == {
        "requested": True,
        "approved": False,
        "executed": True,
    }
    expert_events = {
        event["step"]: event for event in v1_log["events"]
    }
    assert expert_events["request_expert"]["status"] == "succeeded"
    assert expert_events["expert_approval"]["status"] == (
        "not_available"
    )
    assert expert_events["expert"]["status"] == "succeeded"
    assert expert_events["expert"]["real_model_inference"] is False

    v2_log = _execution_log(high)
    assert v2_log["expert_lifecycle"] == {
        "requested": True,
        "approved": True,
        "executed": True,
    }
    assert any(
        event["step"] == "expert"
        and event["status"] == "succeeded"
        for event in v2_log["events"]
    )


def test_extension_interfaces_and_illegal_proposal_gate(
    benchmark: dict[str, object],
) -> None:
    extensibility = benchmark["extensibility"]
    assert isinstance(extensibility, dict)
    assert extensibility["all_passed"] is True
    checks = {
        item["check_id"]: item
        for item in extensibility["checks"]
    }
    assert set(checks) == {
        "new_task_spec_without_state_machine_change",
        "fake_model_adapter_without_agent_change",
        "controller_adapter_substitution",
        "illegal_proposal_cannot_bypass_gate",
        "view_and_report_share_qualification",
    }
    assert all(item["passed"] for item in checks.values())
    assert checks["illegal_proposal_cannot_bypass_gate"]["detail"] == (
        "GATE_CONTROLLER_PROPOSAL_INVALID"
    )


def test_benchmark_writer_records_hashes_and_read_only_boundaries(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "benchmark"
    paths = write_controlled_agent_benchmark_artifacts(
        ROOT,
        output_dir=output_dir,
    )

    assert set(paths) == {
        "scenario_results",
        "summary",
        "extensibility",
        "benchmark",
        "manifest",
    }
    assert all(path.is_file() for path in paths.values())
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["scenario_count"] == 12
    assert manifest["locked_split_content_used"] is False
    assert manifest["real_model_inference_used"] is False
    assert manifest["qualification_contract"]["path"].endswith(
        "route_qualification_contract_v1_1.json"
    )
    assert len(manifest["qualification_contract"]["sha256"]) == 64
    assert manifest["schema_versions"]["case_state"] == (
        "ophagent.case_state.v2"
    )
    assert set(manifest["files"]) == {
        paths["scenario_results"].name,
        paths["summary"].name,
        paths["extensibility"].name,
        paths["benchmark"].name,
    }
    with paths["scenario_results"].open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 24
    assert b"\r\n" not in paths["scenario_results"].read_bytes()
    assert b"\r\n" not in paths["summary"].read_bytes()
