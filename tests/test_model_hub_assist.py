from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

import app.model_hub_assist as assist
from app.model_hub_assist import (
    AssistRun,
    assist_result_view_model,
    persist_uploaded_cfp,
    progress_view_model,
    record_research_feedback,
    restore_uploaded_cfp,
    run_frozen_assist,
    run_live_assist,
    select_live_scout_artifact,
)
from app.model_hub_review import ReviewCase
from app.model_hub_agent_v2 import StateTransitionError
from app.model_hub_tools import (
    ToolContext,
    ToolRuntime,
    build_default_tool_context,
)
from app.orchestration_contracts import CaseState


ROOT = Path(__file__).resolve().parents[1]


def _image_bytes(
    *,
    color: tuple[int, int, int] = (80, 120, 160),
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (96, 80), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _tool_context(rows: list[dict[str, object]]) -> ToolContext:
    return ToolContext(
        project_root=ROOT,
        asset_registry=pd.DataFrame(rows),
        online_artifacts={},
    )


def _run(
    state: CaseState,
    *,
    source_mode: str,
    replayed: bool = False,
) -> AssistRun:
    case = ReviewCase(
        alias="D1-CASE",
        source_case_key="D1-CASE",
        task_id="aptos_dr_5class",
        image_paths=(Path("deidentified.png"),),
        structured_info={},
    )
    context = _tool_context([])
    return AssistRun(
        source_mode=source_mode,
        case=case,
        payload=dict(state.runtime_payload),
        runtime=ToolRuntime(context),
        state=state,
        replayed=replayed,
    )


def test_uploaded_cfp_is_metadata_free_content_addressed_and_restorable(
    tmp_path: Path,
) -> None:
    first = persist_uploaded_cfp(
        _image_bytes(),
        upload_root=tmp_path,
    )
    repeated = persist_uploaded_cfp(
        _image_bytes(),
        upload_root=tmp_path,
    )
    different = persist_uploaded_cfp(
        _image_bytes(color=(20, 30, 40)),
        upload_root=tmp_path,
    )

    assert first == repeated
    assert first.content_sha256 != different.content_sha256
    assert first.path.name == f"{first.content_sha256}.png"
    assert first.path.suffix == ".png"
    assert restore_uploaded_cfp(
        first.content_sha256,
        upload_root=tmp_path,
    ) == first
    assert restore_uploaded_cfp("../private", upload_root=tmp_path) is None
    with Image.open(first.path) as image:
        assert image.info == {}


def test_uploaded_cfp_rejects_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="无法解码"):
        persist_uploaded_cfp(b"not-an-image", upload_root=tmp_path)


def test_live_scout_is_chosen_from_unified_capability_index() -> None:
    context = _tool_context(
        [
            {
                "task_id": "aptos_dr_5class",
                "artifact_id": "offline::only",
                "qualification": "analytical_asset_only",
                "checkpoint_sha256": "a",
            },
            {
                "task_id": "aptos_dr_5class",
                "artifact_id": "online::z",
                "qualification": "online_case_inference_ready",
                "checkpoint_sha256": "z",
            },
            {
                "task_id": "aptos_dr_5class",
                "artifact_id": "online::a",
                "qualification": "online_case_inference_ready",
                "checkpoint_sha256": "b",
            },
        ]
    )

    assert select_live_scout_artifact(context) == "online::a"


def test_live_result_states_real_scout_and_rule_baseline_boundary() -> None:
    state = CaseState(
        case_id="D1-LIVE",
        task_id="aptos_dr_5class",
        current_state="REVIEW_PENDING",
        final_action="REFER_TO_HUMAN",
        runtime_payload={
            "inference": {
                "ok": True,
                "code": "OK",
                "data": {"pred_label": 2, "probabilities": [0, 0, 1, 0, 0]},
            }
        },
    )

    view = assist_result_view_model(
        _run(state, source_mode=assist.SOURCE_LIVE)
    )

    assert view["source_mode"] == "LIVE_INFERENCE"
    assert view["scout_real_run"] is True
    assert view["frozen_second_opinion_used"] is False
    assert view["case_handling"] == "转人工复核"
    assert "现有规则基线" in view["reason"]
    assert "Expert 路由" in view["reason"]


@pytest.mark.parametrize(
    ("action", "expert_used", "expected_handling", "expected_source"),
    [
        ("KEEP_SCOUT", False, "保留 Scout 输出", "冻结 Scout 输出"),
        (
            "REQUEST_EXPERT",
            True,
            "请求第二意见",
            "冻结 Expert 第二意见",
        ),
        ("REFER_TO_HUMAN", False, "转人工复核", "冻结 Scout 输出"),
    ],
)
def test_frozen_result_card_covers_all_three_actions(
    action: str,
    expert_used: bool,
    expected_handling: str,
    expected_source: str,
) -> None:
    payload: dict[str, object] = {
        "predictions": [{"pred_label": 1, "probabilities": [0, 1, 0, 0, 0]}]
    }
    if expert_used:
        payload["expert"] = {
            "ok": True,
            "data": {"pred_label": 2, "probabilities": [0, 0, 1, 0, 0]},
        }
    state = CaseState(
        case_id=f"D1-{action}",
        task_id="aptos_dr_5class",
        current_state="REVIEW_PENDING",
        final_action=action,
        runtime_payload=payload,
    )

    view = assist_result_view_model(
        _run(state, source_mode=assist.SOURCE_FROZEN)
    )

    assert view["scout_real_run"] is False
    assert view["frozen_second_opinion_used"] is expert_used
    assert view["case_handling"] == expected_handling
    assert view["adopted_source"] == expected_source


def test_progress_maps_persisted_steps_and_readable_failure() -> None:
    completed = CaseState(
        case_id="D1-DONE",
        task_id="aptos_dr_5class",
        completed_steps=(
            "model_qualification",
            "scout",
            "qualification_gate",
            "report_generation",
        ),
    )
    assert [row["status"] for row in progress_view_model(completed)] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]

    failed = CaseState(
        case_id="D1-FAILED",
        task_id="aptos_dr_5class",
        current_state="FAILED",
        completed_steps=("model_qualification",),
        report={"failure_stage": "run_scout"},
    )
    assert [row["status"] for row in progress_view_model(failed)] == [
        "completed",
        "failed",
        "pending",
        "pending",
    ]


def test_feedback_persists_once_without_online_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "controlled_agent_v2"
    monkeypatch.setattr(assist, "CONTROLLED_RUNTIME_ROOT", runtime_root)
    store = assist._state_store()
    state = CaseState(
        case_id="D1-FEEDBACK",
        task_id="aptos_dr_5class",
        current_state="REVIEW_PENDING",
        final_action="KEEP_SCOUT",
        report={"clinical_diagnosis": False},
    )
    store.save(state)

    saved = record_research_feedback(state.case_id, "接受")
    restored = assist._state_store().load(state.case_id)

    assert saved.current_state == "CLOSED"
    assert restored is not None
    assert restored.report["research_feedback"]["label"] == "接受"
    assert restored.report["research_feedback_only"] is True
    assert restored.report["no_online_update"] is True
    assert restored.report["model_updated"] is False
    assert restored.report["threshold_updated"] is False
    assert restored.report["route_protocol_updated"] is False

    replayed = record_research_feedback(state.case_id, "接受")
    assert replayed.report["research_feedback"]["label"] == "接受"
    with pytest.raises(StateTransitionError):
        record_research_feedback(state.case_id, "结果有误")


def test_doctor_surface_has_one_start_and_no_tabs_or_role_switch() -> None:
    source = (ROOT / "app/model_hub_assist.py").read_text(encoding="utf-8")
    ui_source = (ROOT / "app/model_hub_ui.py").read_text(encoding="utf-8")

    assert source.count('"开始辅助分析"') == 1
    assert "st.tabs(" not in source
    assert "角色模拟" not in source
    assert "手工模型" not in source
    for feedback in ("接受", "不确定", "结果有误"):
        assert feedback in source
    assert "高级研究与系统管理" in ui_source
    assert "FROZEN_REPLAY" in source
    assert "LIVE_INFERENCE" in source


@pytest.mark.skipif(
    os.environ.get("OPHAGENT_RUN_D1_REAL_SMOKE") != "1",
    reason="requires H100 model checkpoints and public validation assets",
)
def test_h100_real_scout_and_frozen_replay_smoke() -> None:
    context = build_default_tool_context(ROOT)
    options = assist.frozen_case_options(ROOT)
    by_scenario = {option.scenario_label: option for option in options}
    public_cfp = by_scenario["低风险 · 保持 Scout"].case.image_paths[0]
    stored = persist_uploaded_cfp(public_cfp.read_bytes())

    live = run_live_assist(context, stored)
    live_view = assist_result_view_model(live)
    assert live.state.runtime_payload["inference"]["ok"] is True
    assert live_view["scout_real_run"] is True
    assert live.state.final_action == "REFER_TO_HUMAN"
    live_tool_events = len(live.state.tool_trace.get("events", []))
    restored_live = run_live_assist(context, stored)
    assert restored_live.replayed is True
    assert (
        len(restored_live.state.tool_trace.get("events", []))
        == live_tool_events
    )

    expected_actions = {
        "低风险 · 保持 Scout": "KEEP_SCOUT",
        "高风险 · 请求 Expert": "REQUEST_EXPERT",
        "资格不足 · 转人工": "REFER_TO_HUMAN",
    }
    for scenario, expected_action in expected_actions.items():
        frozen = run_frozen_assist(context, by_scenario[scenario])
        assert frozen.state.final_action == expected_action
        assert (
            assist_result_view_model(frozen)["source_mode"]
            == "FROZEN_REPLAY"
        )
        tool_events = len(frozen.state.tool_trace.get("events", []))
        restored = run_frozen_assist(context, by_scenario[scenario])
        assert restored.replayed is True
        assert len(restored.state.tool_trace.get("events", [])) == tool_events
    failed = run_frozen_assist(
        context,
        by_scenario["故障门禁 · 离线资产请求原图推理"],
    )
    assert failed.state.current_state == "FAILED"
    assert assist_result_view_model(failed)["failed"] is True
