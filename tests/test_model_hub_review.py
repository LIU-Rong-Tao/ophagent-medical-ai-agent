from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import app.model_hub_review as review_module
from app.model_hub_agent_v2 import PermissionDenied
from app.model_hub_demo import _model_flag
from app.model_hub_review import (
    FAULT_SCENARIO,
    NORMAL_SCENARIOS,
    ReviewCase,
    _persist_closed_review_artifacts,
)
from app.model_hub_tools import ToolContext
from app.orchestration_contracts import CaseState


ROOT = Path(__file__).resolve().parents[1]


def test_workstation_has_two_read_only_views_and_one_fault_scenario() -> None:
    assert set(NORMAL_SCENARIOS) == {
        "APTOS · 冻结 validation 资产",
        "APTOS 高风险 · 冻结 validation 资产",
    }
    assert FAULT_SCENARIO == "故障门禁 · 离线资产请求原图推理"
    assert all(
        "test" not in str(configuration).lower()
        for configuration in NORMAL_SCENARIOS.values()
    )


def test_partial_persisted_payload_degrades_without_render_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[str] = []
    payload: dict[str, object] = {
        "input": {"ok": True},
        "registry": {"ok": True},
    }
    before = json.loads(json.dumps(payload))
    monkeypatch.setattr(
        review_module.st,
        "markdown",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        review_module.st,
        "info",
        lambda message, *_args, **_kwargs: notices.append(str(message)),
    )

    review_module._render_audit_and_route("aptos_dr_5class", payload)

    assert any("风险审计结果尚未生成" in message for message in notices)
    assert any("路由证据尚未生成" in message for message in notices)
    assert payload == before


def test_model_hub_reuses_existing_clinical_workspace_entry() -> None:
    clinical_source = (ROOT / "app/model_hub_clinical.py").read_text(
        encoding="utf-8"
    )
    demo_source = (ROOT / "app/model_hub_demo.py").read_text(encoding="utf-8")

    assert "render_offline_case_review_workstation" in clinical_source
    assert "离线病例审阅工作台" in clinical_source
    assert "历史路由回放" in clinical_source
    assert "render_clinical_workspace(data)" in demo_source


def test_workstation_exposes_review_actions_and_research_boundaries() -> None:
    source = (ROOT / "app/model_hub_review.py").read_text(encoding="utf-8")

    for text in (
        "病例队列",
        "眼底图像",
        "输入与任务检查",
        "可用模型与资格",
        "单模型输出",
        "模型输出错误风险审计",
        "Scout / Expert 研究模拟",
        "受控 Agent 决策",
        "资格判定",
        "风险判定",
        "最终动作",
        "接受模型输出",
        "修改输出",
        "标记不确定",
        "加入复核队列",
        "保存并进入下一例",
        "结构化报告",
        "完整工具调用轨迹",
        "从 CLOSED 状态恢复缺失报告 / Trace（不重复确认）",
    ):
        assert text in source
    assert "route_eligible=false" in source
    assert "确定性规则基线" in source
    assert "不提供诊断、治疗或患者分流建议" in source
    assert "http://" not in source
    assert "https://" not in source


def test_read_only_pipeline_checks_route_before_prediction_assets() -> None:
    source = (ROOT / "app/model_hub_review.py").read_text(encoding="utf-8")
    pipeline = source[
        source.index("def _run_read_only_pipeline") :
        source.index("def _run_fault_pipeline")
    ]

    assert pipeline.index('"routing_protocol.evaluate"') < pipeline.index(
        '"prediction_asset.validate"'
    )


def test_report_schema_excludes_source_case_key_and_test_content() -> None:
    source = (ROOT / "app/model_hub_review.py").read_text(encoding="utf-8")

    report_body = source[source.index("def _review_report") :]
    assert '"case_alias": case.alias' in report_body
    assert '"source_case_key"' not in report_body
    assert '"test_content_used": False' in report_body
    assert '"external_network_used": False' in report_body


def test_review_permission_precedes_all_report_writes() -> None:
    source = (ROOT / "app/model_hub_review.py").read_text(encoding="utf-8")
    persistence = source[
        source.index("def _persist_closed_review_artifacts") :
        source.index("def _render_review_actions")
    ]
    actions = source[
        source.index("def _render_review_actions") :
        source.index("def _render_trace")
    ]

    assert persistence.index('authorize(actor_role, "review.confirm")') < (
        persistence.index("_save_review_state")
    )
    assert persistence.index('authorize(actor_role, "review.confirm")') < (
        persistence.index("_review_report")
    )
    assert actions.index('case_state.current_state != "REVIEW_PENDING"') < (
        actions.index(").confirm_review(")
    )
    assert actions.index(").confirm_review(") < actions.index(
        "_persist_closed_review_artifacts"
    )
    assert ").confirm_review(" not in persistence
    assert '"agent": state_view_model(case_state)' in persistence
    assert '"free_text_in_downloadable_report": False' in source


def test_role_sensitive_controls_are_disabled_before_backend_calls() -> None:
    source = (ROOT / "app/model_hub_review.py").read_text(encoding="utf-8")
    controls = source[
        source.index("def _render_v2_controls") :
        source.index("def _review_report")
    ]
    actions = source[
        source.index("def _render_review_actions") :
        source.index("def _render_trace")
    ]

    assert 'can_decide_expert = actor_role in {"reviewer", "admin"}' in controls
    assert controls.count("disabled=not can_decide_expert") == 2
    assert "所需角色：reviewer 或 admin" in controls
    assert 'can_retry = actor_role in {"operator", "admin"}' in controls
    assert "disabled=not can_retry" in controls
    assert "所需角色：operator 或 admin" in controls
    assert 'can_confirm_review = actor_role in {"reviewer", "admin"}' in actions
    assert "review_controls_enabled = review_ready and can_confirm_review" in actions
    assert actions.count("disabled=not review_controls_enabled") >= 6
    assert "disabled=not can_confirm_review" in actions
    assert "所需角色：reviewer 或 admin" in actions


def _closed_case_state() -> CaseState:
    return CaseState(
        case_id="controlled-review-case",
        task_id="synthetic_task",
        current_state="CLOSED",
        human_decision="标记不确定",
        runtime_payload={
            "predictions": [],
            "audit": {"data": {"risk_level": "low"}},
            "route": {"data": {"routing_policy": "frozen"}},
        },
        tool_trace={
            "schema_version": "ophagent.model_hub_trace.v1",
            "trace_id": "trace-fixed",
            "offline_mode": True,
            "git_commit": "a" * 40,
            "events": [],
        },
        report={"current_state": "CLOSED"},
        created_at="2026-07-28T00:00:00+00:00",
        updated_at="2026-07-28T00:01:00+00:00",
    )


def _review_case() -> ReviewCase:
    return ReviewCase(
        alias="PUBLIC-CASE-001",
        source_case_key="isolated-source-key",
        task_id="synthetic_task",
        image_paths=(),
        structured_info={},
    )


def _review_state_with_free_text() -> dict[str, object]:
    return {
        "schema_version": "ophagent.case_review_state.v1",
        "cases": {
            "PUBLIC-CASE-001": {
                "decision": "标记不确定",
                "modified_output": "SECRET MODIFIED OUTPUT",
                "note": "SECRET REVIEW NOTE",
                "review_queue": True,
                "saved_at": "2026-07-28T00:01:00+00:00",
            }
        },
    }


def test_closed_artifact_recovery_is_idempotent_and_hides_free_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_module, "REVIEW_RUNTIME_ROOT", tmp_path / "review")
    case_state = _closed_case_state()
    before = case_state.to_dict()
    artifacts = _persist_closed_review_artifacts(
        case=_review_case(),
        scenario="脱敏固定场景",
        session_id="session-a",
        case_state=case_state,
        actor_role="reviewer",
        review_state=_review_state_with_free_text(),
    )

    report_path = Path(artifacts["report_path"])
    trace_path = Path(artifacts["trace_path"])
    report_bytes = report_path.read_bytes()
    trace_bytes = trace_path.read_bytes()
    report = json.loads(report_bytes)
    assert report["controlled_agent"]["current_state"] == "CLOSED"
    assert report["human_review"]["free_text_in_downloadable_report"] is False
    assert "modified_output" not in report["human_review"]
    assert "note" not in report["human_review"]
    assert b"SECRET MODIFIED OUTPUT" not in report_bytes
    assert b"SECRET REVIEW NOTE" not in report_bytes
    assert b"SECRET MODIFIED OUTPUT" not in trace_bytes
    assert b"SECRET REVIEW NOTE" not in trace_bytes
    assert json.loads(trace_bytes) == case_state.tool_trace
    assert case_state.to_dict() == before

    def fail_if_rewritten(_path: Path, _payload: dict[str, object]) -> Path:
        raise AssertionError("complete artifacts must not be rewritten")

    monkeypatch.setattr(review_module, "_atomic_json", fail_if_rewritten)
    replay = _persist_closed_review_artifacts(
        case=_review_case(),
        scenario="脱敏固定场景",
        session_id="session-a",
        case_state=case_state,
        actor_role="admin",
        review_state=_review_state_with_free_text(),
    )
    assert replay["recovered"] is False
    assert report_path.read_bytes() == report_bytes
    assert trace_path.read_bytes() == trace_bytes
    assert case_state.to_dict() == before


def test_trace_write_failure_recovers_without_rewriting_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_module, "REVIEW_RUNTIME_ROOT", tmp_path / "review")
    original_atomic_json = review_module._atomic_json

    def fail_trace(path: Path, payload: dict[str, object]) -> Path:
        if path.parent.name == "traces":
            raise OSError("simulated trace failure")
        return original_atomic_json(path, payload)

    monkeypatch.setattr(review_module, "_atomic_json", fail_trace)
    case_state = _closed_case_state()
    before = case_state.to_dict()
    with pytest.raises(OSError, match="simulated trace failure"):
        _persist_closed_review_artifacts(
            case=_review_case(),
            scenario="脱敏固定场景",
            session_id="session-b",
            case_state=case_state,
            actor_role="reviewer",
            review_state=_review_state_with_free_text(),
        )

    report_path, trace_path = review_module._review_artifact_paths(
        "session-b",
        "PUBLIC-CASE-001",
    )
    report_bytes = report_path.read_bytes()
    assert not trace_path.exists()
    assert case_state.to_dict() == before

    monkeypatch.setattr(review_module, "_atomic_json", original_atomic_json)
    artifacts = _persist_closed_review_artifacts(
        case=_review_case(),
        scenario="脱敏固定场景",
        session_id="session-b",
        case_state=case_state,
        actor_role="admin",
        review_state=_review_state_with_free_text(),
    )
    assert artifacts["recovered"] is True
    assert report_path.read_bytes() == report_bytes
    assert json.loads(trace_path.read_bytes()) == case_state.tool_trace
    assert case_state.to_dict() == before


def test_closed_artifact_recovery_rejects_operator_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "review"
    monkeypatch.setattr(review_module, "REVIEW_RUNTIME_ROOT", runtime_root)
    case_state = _closed_case_state()
    before = case_state.to_dict()

    with pytest.raises(PermissionDenied, match="RBAC_DENIED:review.confirm"):
        _persist_closed_review_artifacts(
            case=_review_case(),
            scenario="脱敏固定场景",
            session_id="session-c",
            case_state=case_state,
            actor_role="operator",
            review_state=_review_state_with_free_text(),
        )

    assert not runtime_root.exists()
    assert case_state.to_dict() == before


def test_legacy_trace_missing_tool_name_degrades_without_crashing(
    tmp_path: Path,
) -> None:
    context = ToolContext(
        project_root=tmp_path,
        asset_registry=pd.DataFrame(),
        online_artifacts={},
    )

    runtime = review_module._runtime_from_trace(
        context,
        {
            "trace_id": "legacy-redacted-trace",
            "events": [
                {
                    "sequence": 1,
                    "status": "succeeded",
                    "code": "OK",
                }
            ],
        },
    )

    assert runtime.trace_id == "legacy-redacted-trace"
    assert runtime.events == []
    assert runtime.halted is False


def test_overview_tolerates_legacy_models_without_online_flag() -> None:
    models = pd.DataFrame({"model_id": ["a", "b"]})

    result = _model_flag(models, "task_inference_ready")

    assert result.tolist() == [False, False]
