"""OphAgent 医生端一次点击辅助分析。

该页面是既有 Model Hub、Tool Contract、资格门控和受控状态机之上的薄
应用层。它不重新计算资格，不读取封存 Test，也不提供最终临床诊断。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import os
from pathlib import Path
import re
import secrets
from typing import Any, Callable

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from app.model_hub_agent_v2 import (
    CaseStateStore,
    ControlledAgentRuntimeV2,
    StateTransitionError,
    state_view_model,
)
from app.model_hub_review import (
    CONTROLLED_RUNTIME_ROOT,
    DEMO_SCENARIO_PATH,
    PROJECT_ROOT,
    ReviewCase,
    _ReviewExpertReplayExecutor,
    _execute_controlled_pipeline,
    _load_workstation_scenarios,
    _runtime_from_trace,
    build_review_cases,
)
from app.model_hub_tools import (
    ToolContext,
    ToolRuntime,
    build_default_tool_context,
)
from app.model_hub_task_adapters import task_adapter_for
from app.model_hub_ui import grade_label
from app.orchestration_contracts import CaseState


ASSIST_SCHEMA_VERSION = "ophagent.one_click_assist.d1"
ASSIST_TASK_ID = "aptos_dr_5class"
ASSIST_RUNTIME_ROOT = (
    PROJECT_ROOT / "experiments/model_hub/runtime/one_click_assist_d1"
)
UPLOAD_ROOT = ASSIST_RUNTIME_ROOT / "uploads"
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
SOURCE_LIVE = "LIVE_INFERENCE"
SOURCE_FROZEN = "FROZEN_REPLAY"
FEEDBACK_OPTIONS = ("接受", "不确定", "结果有误")
FEEDBACK_CODES = {
    "接受": "ACCEPT",
    "不确定": "UNCERTAIN",
    "结果有误": "INCORRECT",
}
DOCTOR_SCENARIOS = (
    "低风险 · 保持 Scout",
    "高风险 · 请求 Expert",
    "资格不足 · 转人工",
    "故障门禁 · 离线资产请求原图推理",
)
SCENARIO_LABELS = {
    "低风险 · 保持 Scout": "脱敏病例 01 · 常规基线",
    "高风险 · 请求 Expert": "脱敏病例 02 · 需冻结第二意见",
    "资格不足 · 转人工": "脱敏病例 03 · 资格不足",
    "故障门禁 · 离线资产请求原图推理": (
        "脱敏病例 04 · 工具故障演示"
    ),
}
ACTION_LABELS = {
    "KEEP_SCOUT": "当前结果交由医生复核",
    "REQUEST_EXPERT": "历史流程采用第二模型复核",
    "REFER_TO_HUMAN": "现有证据不足，需人工复核",
}
PROGRESS_LABELS = (
    "图像与任务检查",
    "初筛模型分析",
    "现有规则基线评估",
    "生成辅助结果",
)
SESSION_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,64}")


@dataclass(frozen=True)
class StoredCFP:
    content_sha256: str
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class FrozenCaseOption:
    key: str
    display_label: str
    scenario_label: str
    scenario: dict[str, Any]
    case: ReviewCase


@dataclass(frozen=True)
class AssistRun:
    source_mode: str
    case: ReviewCase
    payload: dict[str, Any]
    runtime: ToolRuntime
    state: CaseState
    replayed: bool
    scenario_label: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scenario_key(label: str) -> str:
    return hashlib.sha256(
        f"{ASSIST_SCHEMA_VERSION}:{label}".encode("utf-8")
    ).hexdigest()[:16]


def normalize_assist_session(value: str = "") -> str:
    """Return a non-sensitive browser namespace for durable case state."""

    token = str(value).strip()
    if SESSION_TOKEN_PATTERN.fullmatch(token):
        return token
    return secrets.token_hex(8)


def assist_scenario_scope(
    source_mode: str,
    session_id: str,
    *,
    scenario_label: str = "",
) -> str:
    """Build the idempotency scope without sharing feedback across browsers."""

    if source_mode not in {SOURCE_LIVE, SOURCE_FROZEN}:
        raise ValueError("ASSIST_SOURCE_MODE_INVALID")
    if not SESSION_TOKEN_PATTERN.fullmatch(str(session_id)):
        raise ValueError("ASSIST_SESSION_INVALID")
    parts = [
        ASSIST_SCHEMA_VERSION,
        "live" if source_mode == SOURCE_LIVE else "frozen",
        str(session_id),
    ]
    if scenario_label:
        parts.append(scenario_label)
    return ":".join(parts)


def source_matches_selection(selection: str, source_mode: str) -> bool:
    return (
        selection == "上传 CFP" and source_mode == SOURCE_LIVE
    ) or (
        selection == "选择冻结脱敏病例"
        and source_mode == SOURCE_FROZEN
    )


def persist_uploaded_cfp(
    content: bytes,
    *,
    upload_root: Path = UPLOAD_ROOT,
) -> StoredCFP:
    """Validate, strip metadata and persist one CFP by normalized content hash."""

    if not content:
        raise ValueError("请先选择一张 CFP 图像。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("图像文件过大，请上传 40 MB 以内的 JPG 或 PNG。")
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            if width < 64 or height < 64:
                raise ValueError("图像尺寸过小，无法进行可靠的输入检查。")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("图像像素数过大，请先导出常规分辨率 CFP。")
            normalized = BytesIO()
            image.save(normalized, format="PNG", compress_level=6)
    except ValueError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError("图像无法解码，请上传有效的 JPG 或 PNG。") from exc

    normalized_bytes = normalized.getvalue()
    digest = hashlib.sha256(normalized_bytes).hexdigest()
    target_root = Path(upload_root)
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"{digest}.png"
    if not target.is_file():
        temporary = target.with_suffix(".png.tmp")
        temporary.write_bytes(normalized_bytes)
        os.replace(temporary, target)
    return StoredCFP(
        content_sha256=digest,
        path=target,
        width=width,
        height=height,
    )


def restore_uploaded_cfp(
    content_sha256: str,
    *,
    upload_root: Path = UPLOAD_ROOT,
) -> StoredCFP | None:
    digest = str(content_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    path = Path(upload_root) / f"{digest}.png"
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return None
    return StoredCFP(digest, path, int(width), int(height))


def _live_case(stored: StoredCFP) -> ReviewCase:
    alias = f"D1LIVE-{stored.content_sha256[:20].upper()}"
    return ReviewCase(
        alias=alias,
        source_case_key=alias,
        task_id=ASSIST_TASK_ID,
        image_paths=(stored.path,),
        structured_info={
            "数据范围": "新上传病例",
            "模态": "彩色眼底照相（CFP）",
            "图像数量": "1",
            "身份字段": "未采集",
        },
    )


def _filter_scenario_cases(
    cases: list[ReviewCase],
    case_filter: str,
) -> list[ReviewCase]:
    if case_filter == "keep_scout":
        return [
            item
            for item in cases
            if item.structured_info.get("冻结策略请求 Expert") == "否"
        ]
    if case_filter == "request_expert":
        return [
            item
            for item in cases
            if item.structured_info.get("冻结策略请求 Expert") == "是"
        ]
    return cases


def frozen_case_options(
    project_root: Path = PROJECT_ROOT,
) -> tuple[FrozenCaseOption, ...]:
    scenarios = _load_workstation_scenarios(str(DEMO_SCENARIO_PATH))
    options: list[FrozenCaseOption] = []
    for label in DOCTOR_SCENARIOS:
        scenario = scenarios.get(label)
        if not isinstance(scenario, dict):
            continue
        cases = build_review_cases(
            project_root,
            str(scenario["task_id"]),
        )
        cases = _filter_scenario_cases(
            cases,
            str(scenario.get("case_filter", "")),
        )
        if not cases:
            continue
        options.append(
            FrozenCaseOption(
                key=_scenario_key(label),
                display_label=SCENARIO_LABELS[label],
                scenario_label=label,
                scenario=scenario,
                case=cases[0],
            )
        )
    return tuple(options)


def select_live_scout_artifact(context: ToolContext) -> str:
    """Choose the stable system default from the unified capability index."""

    registry = context.asset_registry
    eligible = registry.loc[
        registry["task_id"].astype(str).eq(ASSIST_TASK_ID)
        & registry["qualification"]
        .astype(str)
        .eq("online_case_inference_ready")
    ].copy()
    if eligible.empty:
        raise RuntimeError("当前没有已登记的真实单病例 Scout 入口。")
    eligible = eligible.sort_values(
        ["artifact_id", "checkpoint_sha256"],
        kind="stable",
    )
    return str(eligible.iloc[0]["artifact_id"])


def _state_store() -> CaseStateStore:
    return CaseStateStore(CONTROLLED_RUNTIME_ROOT / "cases")


def _annotate_report(
    case_id: str,
    updates: dict[str, Any],
) -> CaseState:
    store = _state_store()
    with store.case_lock(case_id):
        state = store.load(case_id)
        if state is None:
            raise StateTransitionError("CASE_NOT_FOUND")
        state.report = {**state.report, **updates}
        store.save(state)
        return state


def _payload_from_state(state: CaseState) -> dict[str, Any]:
    payload = dict(state.runtime_payload)
    payload["agent"] = state_view_model(state)
    return payload


def run_live_assist(
    context: ToolContext,
    stored: StoredCFP,
    *,
    session_id: str = "direct-call",
    progress_callback: Callable[[str, str], None] | None = None,
) -> AssistRun:
    case = _live_case(stored)
    artifact_id = select_live_scout_artifact(context)
    _payload, _runtime, state, replayed = _execute_controlled_pipeline(
        context,
        case,
        scenario=assist_scenario_scope(SOURCE_LIVE, session_id),
        artifact_ids=[artifact_id],
        mode="live",
        actor_role="operator",
        progress_callback=progress_callback,
    )
    state = _annotate_report(
        state.case_id,
        {
            "assist_schema_version": ASSIST_SCHEMA_VERSION,
            "source_mode": SOURCE_LIVE,
            "scout_execution": "live_model_inference",
            "help_or_harm_prediction_used": False,
            "baseline_label": "现有规则基线",
            "frozen_expert_replay": False,
            "clinical_diagnosis": False,
        },
    )
    return AssistRun(
        source_mode=SOURCE_LIVE,
        case=case,
        payload=_payload_from_state(state),
        runtime=_runtime_from_trace(context, state.tool_trace),
        state=state,
        replayed=replayed,
    )


def run_frozen_assist(
    context: ToolContext,
    option: FrozenCaseOption,
    *,
    session_id: str = "direct-call",
    progress_callback: Callable[[str, str], None] | None = None,
) -> AssistRun:
    _payload, _runtime, state, replayed = _execute_controlled_pipeline(
        context,
        option.case,
        scenario=assist_scenario_scope(
            SOURCE_FROZEN,
            session_id,
            scenario_label=option.scenario_label,
        ),
        artifact_ids=[
            str(value) for value in option.scenario["artifact_ids"]
        ],
        mode=str(option.scenario["mode"]),
        actor_role="operator",
        progress_callback=progress_callback,
    )
    if (
        state.current_state == "EXPERT_PENDING_APPROVAL"
        and state.final_action == "REQUEST_EXPERT"
    ):
        agent_runtime = ControlledAgentRuntimeV2(_state_store())
        state = agent_runtime.decide_expert(
            state.case_id,
            approved=True,
            actor_role="admin",
            expert_executor=_ReviewExpertReplayExecutor(
                context=context,
                case=option.case,
            ),
            idempotency_key=f"d1-frozen-expert-replay:{state.case_id}",
        )
    state = _annotate_report(
        state.case_id,
        {
            "assist_schema_version": ASSIST_SCHEMA_VERSION,
            "source_mode": SOURCE_FROZEN,
            "scout_execution": "frozen_prediction_replay",
            "help_or_harm_prediction_used": False,
            "baseline_label": "现有规则基线",
            "frozen_expert_replay": bool(
                state.runtime_payload.get("expert", {}).get("ok", False)
            ),
            "frozen_expert_is_live_consultation": False,
            "clinical_diagnosis": False,
        },
    )
    return AssistRun(
        source_mode=SOURCE_FROZEN,
        case=option.case,
        payload=_payload_from_state(state),
        runtime=_runtime_from_trace(context, state.tool_trace),
        state=state,
        replayed=replayed,
        scenario_label=option.scenario_label,
    )


def record_research_feedback(
    case_id: str,
    feedback: str,
) -> CaseState:
    """Persist one doctor feedback label without updating any model policy."""

    if feedback not in FEEDBACK_CODES:
        raise ValueError("RESEARCH_FEEDBACK_INVALID")
    decision = f"RESEARCH_FEEDBACK_{FEEDBACK_CODES[feedback]}"
    runtime = ControlledAgentRuntimeV2(_state_store())
    state = runtime.confirm_review(
        case_id,
        decision=decision,
        actor_role="reviewer",
        idempotency_key=f"d1-feedback:{case_id}:{FEEDBACK_CODES[feedback]}",
    )
    return _annotate_report(
        case_id,
        {
            "research_feedback": {
                "label": feedback,
                "code": FEEDBACK_CODES[feedback],
                "recorded_at": state.updated_at or _utc_now(),
            },
            "research_feedback_only": True,
            "no_online_update": True,
            "model_updated": False,
            "threshold_updated": False,
            "route_protocol_updated": False,
        },
    )


def progress_view_model(
    state: CaseState,
    *,
    running_stage: int | None = None,
) -> tuple[dict[str, str], ...]:
    completed = set(state.completed_steps)
    done = (
        "model_qualification" in completed,
        "scout" in completed,
        "qualification_gate" in completed,
        "report_generation" in completed,
    )
    failure_stage = str(state.report.get("failure_stage", ""))
    failed_index = {
        "validate_case_input": 0,
        "validate_model_registry": 0,
        "query_route_metadata": 2,
        "run_scout": 1,
        "audit_risk": 2,
        "execute_tool_contract": 1,
    }.get(failure_stage)
    rows = []
    for index, label in enumerate(PROGRESS_LABELS):
        if done[index]:
            status = "completed"
            note = "已完成"
        elif state.current_state == "FAILED" and index == failed_index:
            status = "failed"
            note = "未完成"
        elif running_stage == index:
            status = "running"
            note = "执行中"
        else:
            status = "pending"
            note = "等待执行"
        rows.append({"label": label, "status": status, "note": note})
    return tuple(rows)


def _failure_message(state: CaseState) -> str:
    code = str(state.report.get("failure_code", "TOOL_EXECUTION_FAILED"))
    response = dict(state.runtime_payload.get("inference", {}))
    detail = str(response.get("message", "")).strip()
    messages = {
        "QUALIFICATION_BLOCKED": (
            "所选资产不具备单病例原图调用资格，系统已停止后续步骤。"
        ),
        "INPUT_INVALID": "图像输入未通过检查，请确认文件可读取且格式正确。",
        "ASSET_NOT_FOUND": "所需模型资产当前不可用，系统未继续执行。",
        "TOOL_EXECUTION_FAILED": "初筛工具执行失败，系统未继续生成结果。",
        "UPSTREAM_FAILED": "上游工具失败，后续步骤已自动停止。",
    }
    return detail or messages.get(
        code,
        "辅助分析未完成，系统已安全停止后续调用。",
    )


def assist_result_view_model(run: AssistRun) -> dict[str, Any]:
    state = run.state
    if state.current_state == "FAILED":
        return {
            "failed": True,
            "failure_message": _failure_message(state),
            "source_mode": run.source_mode,
            "execution_label": "状态恢复" if run.replayed else "新执行",
        }

    payload = state.runtime_payload
    inference = dict(payload.get("inference", {}))
    inference_data = dict(inference.get("data", {}))
    predictions = list(payload.get("predictions", []))
    scout_data = (
        inference_data
        if run.source_mode == SOURCE_LIVE
        else (dict(predictions[0]) if predictions else {})
    )
    expert_response = dict(payload.get("expert", {}))
    expert_used = bool(
        run.source_mode == SOURCE_FROZEN
        and expert_response.get("ok", False)
    )
    expert_data = dict(expert_response.get("data", {}))
    adopted = expert_data if expert_used else scout_data
    adopted_label = (
        grade_label(run.case.task_id, adopted.get("pred_label"))
        if adopted
        else "未生成"
    )
    clinical_context = task_adapter_for(
        run.case.task_id
    ).clinical_assist.view(
        run.case.structured_info,
        image_count=len(run.case.image_paths),
    )
    action = state.final_action or "REFER_TO_HUMAN"
    if run.source_mode == SOURCE_LIVE:
        reason = (
            "真实 Scout 已运行；当前新病例没有已获准的 Expert 路由，"
            "现有规则基线将结果交由医生复核。"
        )
        adopted_source = "真实 Scout 输出"
        source_boundary = (
            "本次仅对上传的 CFP 运行初筛模型，未调用第二模型。"
        )
    elif action == "REQUEST_EXPERT" and expert_used:
        reason = (
            "历史冻结策略请求第二意见；系统只读载入当时的冻结 Expert "
            "结果，不代表实时专家会诊。"
        )
        adopted_source = "冻结 Expert 第二意见"
        source_boundary = (
            "这是历史脱敏病例回放，展示的是当时保存的第二模型结果；"
            "本次没有重新运行模型，也不是在线专家会诊。"
        )
    elif action == "KEEP_SCOUT":
        reason = "历史冻结策略未请求第二意见，沿用冻结 Scout 输出。"
        adopted_source = "冻结 Scout 输出"
        source_boundary = (
            "这是历史脱敏病例回放，展示的是当时保存的模型结果；"
            "本次没有重新运行模型。"
        )
    else:
        reason = (
            "现有规则基线未通过资格或预算限制，未调用 Expert，"
            "结果转人工复核。"
        )
        adopted_source = "冻结 Scout 输出"
        source_boundary = (
            "这是历史脱敏病例回放。现有证据不足，系统没有追加"
            "第二模型，本次也没有重新运行模型。"
        )
    feedback = dict(state.report.get("research_feedback", {}))
    return {
        "failed": False,
        "source_mode": run.source_mode,
        "source_label": (
            "真实 Scout 在线推理"
            if run.source_mode == SOURCE_LIVE
            else "冻结历史回放"
        ),
        "auxiliary_result": adopted_label,
        "case_handling": ACTION_LABELS.get(action, "转人工复核"),
        "scout_real_run": run.source_mode == SOURCE_LIVE
        and inference.get("ok", False),
        "frozen_second_opinion_used": expert_used,
        "adopted_source": adopted_source,
        "reason": reason,
        "execution_label": "状态恢复" if run.replayed else "新执行",
        "feedback": feedback,
        "clinical_context": clinical_context,
        "source_boundary": source_boundary,
    }


def _progress_html(rows: tuple[dict[str, str], ...]) -> str:
    status_icons = {
        "completed": "✓",
        "running": "●",
        "failed": "!",
        "pending": "·",
    }
    cards = "".join(
        '<div class="d1-progress-step '
        f'{html.escape(row["status"])}">'
        f'<span>{status_icons[row["status"]]}</span>'
        f'<div><b>{html.escape(row["label"])}</b>'
        f'<small>{html.escape(row["note"])}</small></div></div>'
        for row in rows
    )
    return f'<div class="d1-progress-grid">{cards}</div>'


def _result_html(view: dict[str, Any]) -> str:
    source_class = (
        "live" if view["source_mode"] == SOURCE_LIVE else "frozen"
    )
    source_badge = (
        "本次实时初筛 · LIVE_INFERENCE"
        if view["source_mode"] == SOURCE_LIVE
        else "历史结果回放 · FROZEN_REPLAY"
    )
    clinical = dict(view["clinical_context"])
    current_evidence = {
        "label": "当前影像证据",
        "value": str(clinical["current_evidence"]),
        "status": "provided",
        "status_label": "已提供",
        "note": str(clinical["quality_boundary"]),
    }
    clinical_fields = list(clinical["fields"])
    provided_items = [
        current_evidence,
        *[
            item
            for item in clinical_fields
            if item["status"] == "provided"
        ],
    ]
    pending_items = [
        item
        for item in clinical_fields
        if item["status"] != "provided"
    ]

    def evidence_html(items: list[dict[str, Any]]) -> str:
        return "".join(
            '<div class="d1-clinical-item '
            f'{html.escape(str(item["status"]))}">'
            '<div class="d1-clinical-item-head">'
            f'<b>{html.escape(str(item["label"]))}</b>'
            f'<span>{html.escape(str(item["status_label"]))}</span></div>'
            f'<p>{html.escape(str(item["value"]))}</p>'
            f'<small>{html.escape(str(item["note"]))}</small></div>'
            for item in items
        )

    pending_section = ""
    if pending_items:
        pending_section = (
            '<details class="d1-optional-evidence"><summary><div>'
            "<b>可选补充证据（暂未接入）</b>"
            "<span>这些资料未参与本次分析，也不代表已完成多模态分析。</span>"
            "</div>"
            f'<small>{len(pending_items)} 项待补充</small></summary>'
            '<div class="d1-optional-evidence-body">'
            f'<div class="d1-clinical-grid pending">'
            f'{evidence_html(pending_items)}</div></div></details>'
        )
    result_prefix = (
        "CFP 模型提示"
        if view["source_mode"] == SOURCE_LIVE
        else "历史病例回放"
    )
    return (
        '<section class="d1-result-card">'
        '<div class="d1-result-head"><div>'
        f'<span class="d1-result-kicker">'
        f'{html.escape(str(clinical["result_heading"]))}</span>'
        f'<h2>{html.escape(result_prefix)}：'
        f'{html.escape(view["auxiliary_result"])}</h2>'
        '<p class="d1-result-subtitle">'
        "本次结果仅基于当前展示的 CFP 证据，需由医生结合临床资料复核。"
        "</p></div>"
        '<div class="d1-result-badges">'
        f'<span class="d1-source-badge {source_class}">{source_badge}</span>'
        f'<span class="d1-run-badge">{html.escape(view["execution_label"])}</span>'
        '</div></div>'
        '<div class="d1-clinical-summary">'
        '<div><small>本例处理</small>'
        f'<b>{html.escape(view["case_handling"])}</b>'
        f'<p>{html.escape(str(clinical["review_prompt"]))}</p></div>'
        '<div><small>本次分析能回答什么</small>'
        f'<p>{html.escape(str(clinical["model_scope"]))}</p></div>'
        '</div>'
        '<div class="d1-source-note">'
        f'{html.escape(str(view["source_boundary"]))}</div>'
        '<div class="d1-clinical-block">'
        '<div class="d1-clinical-title"><b>本次实际使用的资料</b>'
        "<span>只显示系统实际收到并用于本次分析的资料。</span></div>"
        '<div class="d1-clinical-grid primary">'
        f'{evidence_html(provided_items)}</div></div>'
        f"{pending_section}"
        '<div class="d1-safety-note">'
        "本结果仅用于研究型辅助分析，不是最终临床诊断；"
        "当前规则不判断增加第二模型后本病例一定获益或受害。"
        "</div></section>"
    )


@st.cache_resource(show_spinner=False)
def _cached_tool_context() -> ToolContext:
    return build_default_tool_context(PROJECT_ROOT)


def _render_feedback(run: AssistRun, view: dict[str, Any]) -> None:
    feedback = dict(view.get("feedback", {}))
    selected = str(feedback.get("label", ""))
    st.markdown(
        '<div class="d1-feedback-title"><b>这次辅助结果是否有帮助？</b>'
        "<span>仅记录研究反馈，不在线更新模型、阈值、路由或协议。</span></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="small")
    clicked = ""
    for column, label in zip(columns, FEEDBACK_OPTIONS, strict=True):
        with column:
            if st.button(
                label,
                width="stretch",
                type="primary" if selected == label else "secondary",
                disabled=bool(selected),
                key=f"d1-feedback::{run.state.case_id}::{label}",
            ):
                clicked = label
    if clicked:
        try:
            record_research_feedback(run.state.case_id, clicked)
        except (StateTransitionError, ValueError):
            st.error("反馈未保存：当前病例状态不允许重复提交不同反馈。")
        else:
            st.toast("研究反馈已保存", icon="✅")
            st.rerun()
    if selected:
        st.success(
            f"本次会话的研究反馈已保存：{selected}。"
            "不会触发任何在线更新。"
        )


def _render_run(
    run: AssistRun,
    progress_box: Any,
) -> None:
    progress_box.markdown(
        _progress_html(progress_view_model(run.state)),
        unsafe_allow_html=True,
    )
    view = assist_result_view_model(run)
    if view["failed"]:
        st.markdown(
            '<div class="d1-failure-card"><span>辅助分析未完成</span>'
            f'<h3>{html.escape(view["failure_message"])}</h3>'
            "<p>系统已停止下游调用；不会用不完整结果继续生成结论。</p>"
            f'<small>{html.escape(view["source_mode"])} · '
            f'{html.escape(view["execution_label"])}</small></div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(_result_html(view), unsafe_allow_html=True)
    _render_feedback(run, view)


def render_one_click_assist() -> None:
    query_source = str(st.query_params.get("assist_source", ""))
    query_token = str(st.query_params.get("assist_case", ""))
    query_session = str(st.query_params.get("assist_session", ""))
    session_id = normalize_assist_session(query_session)
    if session_id != query_session:
        st.query_params["assist_session"] = session_id
    default_mode = (
        "上传 CFP"
        if query_source != SOURCE_FROZEN
        else "选择冻结脱敏病例"
    )
    options = frozen_case_options()
    options_by_key = {option.key: option for option in options}

    with st.container(key="d1-intake", border=True):
        mode = st.segmented_control(
            "病例来源",
            ["上传 CFP", "选择冻结脱敏病例"],
            default=default_mode,
            key="d1-source-mode",
        )
        uploaded = None
        selected_option: FrozenCaseOption | None = None
        if mode == "上传 CFP":
            left, right = st.columns([0.68, 0.32], gap="large")
            with left:
                uploaded = st.file_uploader(
                    "上传一张彩色眼底照片（CFP）",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False,
                    max_upload_size=40,
                    help="上传内容会去除元数据并按内容哈希保存；不记录原文件名。",
                )
                st.caption(
                    "LIVE_INFERENCE：本次真实运行 Scout；"
                    "当前不提供新病例 Expert 自动路由。"
                )
            with right:
                if uploaded is not None:
                    st.image(uploaded, width=220)
                elif query_source == SOURCE_LIVE:
                    restored = restore_uploaded_cfp(query_token)
                    if restored is not None:
                        st.image(str(restored.path), width=220)
        else:
            if not options:
                st.error("当前没有可用的冻结脱敏病例。")
            else:
                default_index = next(
                    (
                        index
                        for index, option in enumerate(options)
                        if option.key == query_token
                    ),
                    0,
                )
                selected_key = st.selectbox(
                    "选择冻结脱敏病例",
                    [option.key for option in options],
                    index=default_index,
                    format_func=lambda value: options_by_key[value].display_label,
                )
                selected_option = options_by_key[selected_key]
                preview, note = st.columns([0.32, 0.68], gap="large")
                with preview:
                    st.image(
                        str(selected_option.case.image_paths[0]),
                        width=220,
                    )
                with note:
                    st.markdown(
                        '<div class="d1-replay-note"><b>FROZEN_REPLAY</b>'
                        "<span>使用公开 validation 的冻结概率与历史路由，"
                        "不是本次实时模型会诊。页面不会读取封存 Test。</span></div>",
                        unsafe_allow_html=True,
                    )

        start = st.button(
            "开始辅助分析",
            type="primary",
            icon=":material/play_arrow:",
            width="stretch",
            disabled=mode == "选择冻结脱敏病例" and selected_option is None,
            key="d1-start-analysis",
        )

    st.markdown(
        '<div class="d1-section-title"><b>自动执行进度</b>'
        "<span>完成的步骤会持久化；刷新或重启后不会重复调用。</span></div>",
        unsafe_allow_html=True,
    )
    progress_box = st.empty()
    empty_state = CaseState(case_id="preview", task_id=ASSIST_TASK_ID)
    progress_box.markdown(
        _progress_html(progress_view_model(empty_state)),
        unsafe_allow_html=True,
    )

    run: AssistRun | None = None
    if start:
        context = _cached_tool_context()
        live_progress = [dict(row) for row in progress_view_model(empty_state)]

        def update_progress(tool_step: str, status: str) -> None:
            stage_map = {
                "input": 0,
                "registry": 0,
                "scout": 1,
                "audit_and_qualification": 2,
            }
            stage = stage_map.get(tool_step)
            if stage is None:
                return
            live_progress[stage]["status"] = status
            live_progress[stage]["note"] = {
                "running": "执行中",
                "completed": "已完成",
                "failed": "未完成",
            }.get(status, "等待执行")
            if tool_step == "registry" and status == "completed":
                live_progress[0]["status"] = "completed"
                live_progress[0]["note"] = "已完成"
            progress_box.markdown(
                _progress_html(tuple(live_progress)),
                unsafe_allow_html=True,
            )

        try:
            if mode == "上传 CFP":
                if uploaded is None:
                    raise ValueError("请先选择一张 CFP 图像。")
                stored = persist_uploaded_cfp(uploaded.getvalue())
                st.query_params["assist_source"] = SOURCE_LIVE
                st.query_params["assist_case"] = stored.content_sha256
                with st.spinner("正在运行真实 Scout，请稍候…"):
                    run = run_live_assist(
                        context,
                        stored,
                        session_id=session_id,
                        progress_callback=update_progress,
                    )
            elif selected_option is not None:
                st.query_params["assist_source"] = SOURCE_FROZEN
                st.query_params["assist_case"] = selected_option.key
                with st.spinner("正在读取冻结证据并执行现有规则基线…"):
                    run = run_frozen_assist(
                        context,
                        selected_option,
                        session_id=session_id,
                        progress_callback=update_progress,
                    )
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
    elif source_matches_selection(mode, query_source) and (
        query_source == SOURCE_LIVE
    ):
        stored = restore_uploaded_cfp(query_token)
        if stored is None:
            st.warning("上一例上传图像缓存不可用，请重新上传后开始分析。")
        else:
            try:
                run = run_live_assist(
                    _cached_tool_context(),
                    stored,
                    session_id=session_id,
                )
            except (RuntimeError, ValueError) as exc:
                st.error(f"状态恢复失败：{exc}")
    elif source_matches_selection(mode, query_source) and (
        query_source == SOURCE_FROZEN
    ):
        restored_option = options_by_key.get(query_token)
        if restored_option is None:
            st.warning("上一例冻结病例不可用，请重新选择后开始分析。")
        else:
            try:
                run = run_frozen_assist(
                    _cached_tool_context(),
                    restored_option,
                    session_id=session_id,
                )
            except (RuntimeError, ValueError) as exc:
                st.error(f"状态恢复失败：{exc}")

    if run is not None:
        _render_run(run, progress_box)
    else:
        st.caption("选择病例并点击“开始辅助分析”后，四步进度会在这里更新。")
