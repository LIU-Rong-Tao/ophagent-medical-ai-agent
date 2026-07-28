"""Model Hub V1 离线病例审阅工作台。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

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
    StateTransitionError,
    authorize,
    state_view_model,
)
from app.model_hub_tools import (
    TASK_CONTRACTS,
    TraceEvent,
    ToolContext,
    ToolRequest,
    ToolRuntime,
    build_default_tool_context,
    capability_matrix,
    trace_frame,
)
from app.model_hub_ui import grade_label, human_model
from app.route_qualification import (
    RouteQualificationRequest,
    evaluate_route_qualification,
    load_route_qualification_contract,
    route_qualification_decision_from_dict,
)
from app.orchestration_contracts import (
    CaseState,
    RouteQualification,
    redact_free_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_RUNTIME_ROOT = (
    PROJECT_ROOT / "experiments/model_hub/runtime/case_review_v1"
)
CONTROLLED_RUNTIME_ROOT = (
    PROJECT_ROOT / "experiments/model_hub/runtime/controlled_agent_v2"
)
DEMO_SCENARIO_PATH = (
    PROJECT_ROOT
    / "experiments/opening_risk_routing_closure/configs/protocols"
    / "controlled_agent_demo_scenarios_v2.json"
)
REVIEW_DECISIONS = ("接受模型输出", "修改输出", "标记不确定")
DEMO_STATE_ID_VERSION = "ophagent.controlled_agent_demo_state.v2_1"


@dataclass(frozen=True)
class ReviewCase:
    alias: str
    source_case_key: str
    task_id: str
    image_paths: tuple[Path, ...]
    structured_info: dict[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


@st.cache_data(show_spinner=False)
def _load_workstation_scenarios(path_text: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", [])
    return {
        str(row["label"]): dict(row)
        for row in scenarios
        if isinstance(row, dict) and row.get("label")
    }


_COMPATIBLE_SCENARIOS = _load_workstation_scenarios(
    str(DEMO_SCENARIO_PATH)
)
NORMAL_SCENARIOS = {
    str(row["legacy_label"]): row
    for row in _COMPATIBLE_SCENARIOS.values()
    if row.get("legacy_label")
}
FAULT_SCENARIO = next(
    (
        label
        for label, row in _COMPATIBLE_SCENARIOS.items()
        if row.get("mode") == "tool_failure"
    ),
    "故障门禁 · 离线资产请求原图推理",
)


def _read_review_state(session_id: str) -> dict[str, Any]:
    path = REVIEW_RUNTIME_ROOT / session_id / "review_state.json"
    if not path.is_file():
        return {"schema_version": "ophagent.case_review_state.v1", "cases": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": "ophagent.case_review_state.v1", "cases": {}}
    return payload if isinstance(payload, dict) else {"cases": {}}


def _save_review_state(session_id: str, state: dict[str, Any]) -> Path:
    state["updated_at"] = _utc_now()
    return _atomic_json(
        REVIEW_RUNTIME_ROOT / session_id / "review_state.json",
        state,
    )


@st.cache_data(show_spinner=False)
def _load_case_rows(project_root_text: str, task_id: str) -> list[dict[str, Any]]:
    root = Path(project_root_text)
    contract = TASK_CONTRACTS[task_id]
    trace_path = root / str(contract["route_trace"])
    trace = pd.read_csv(
        trace_path,
        low_memory=False,
        usecols=[
            "pairing_id",
            "image_key",
            "image_path",
            "routing_policy",
            "requested_budget",
            "scout_disagreement",
            "routing_score",
            "is_reviewed_by_expert",
        ],
    )
    trace = trace.loc[
        trace["pairing_id"].astype(str).eq(str(contract["route_pairing_id"]))
        & trace["routing_policy"].astype(str).eq(str(contract["route_policy"]))
        & trace["requested_budget"].astype(float).round(6).eq(
            round(float(contract["route_budget"]), 6)
        )
    ].copy()
    trace = trace.sort_values(
        ["scout_disagreement", "routing_score"],
        ascending=[False, False],
    ).drop_duplicates("image_key")
    high_risk = trace.head(10)
    keep_scout = (
        trace.loc[~trace["is_reviewed_by_expert"].astype(bool)]
        .sort_values("routing_score", ascending=True)
        .head(2)
    )
    trace = pd.concat([high_risk, keep_scout], ignore_index=True).drop_duplicates(
        "image_key"
    )

    relative_paths: dict[str, str] = {}
    manifest_path = root / str(contract.get("manifest", ""))
    if str(contract.get("manifest", "")) and manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        relative_paths = dict(
            zip(
                manifest["image_key"].astype(str),
                manifest["relative_path"].astype(str),
                strict=True,
            )
        )

    rows = []
    for _, row in trace.iterrows():
        source_key = str(row["image_key"])
        image_text = str(row.get("image_path", "") or "")
        image_path = Path(image_text) if image_text and image_text != "nan" else None
        if image_path is None or not image_path.is_file():
            relative = relative_paths.get(source_key)
            image_path = (
                Path(str(contract["data_root"])) / relative
                if relative
                else None
            )
        if image_path is None or not image_path.is_file():
            continue
        rows.append(
            {
                "source_case_key": source_key,
                "image_path": str(image_path),
                "scout_disagreement": bool(row["scout_disagreement"]),
                "routing_score": float(row["routing_score"]),
                "expert_invoked": bool(row["is_reviewed_by_expert"]),
            }
        )
    return rows


def build_review_cases(project_root: Path, task_id: str) -> list[ReviewCase]:
    rows = _load_case_rows(str(project_root), task_id)
    prefix = "CASE-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:4].upper()
    cases = []
    for index, row in enumerate(rows):
        image_paths = (Path(row["image_path"]),)
        structured = {
            "数据范围": "公开 validation",
            "模态": "彩色眼底照相（CFP）",
            "眼别": "未提供",
            "设备信息": "未提供",
            "模型分歧信号": "是" if row["scout_disagreement"] else "否",
            "冻结策略请求 Expert": "是" if row["expert_invoked"] else "否",
        }
        cases.append(
            ReviewCase(
                alias=f"{prefix}-{index + 1:04d}",
                source_case_key=str(row["source_case_key"]),
                task_id=task_id,
                image_paths=image_paths,
                structured_info=structured,
            )
        )
    if len(cases) >= 2:
        second = cases[1]
        cases[1] = ReviewCase(
            alias=second.alias,
            source_case_key=second.source_case_key,
            task_id=second.task_id,
            image_paths=(second.image_paths[0], cases[0].image_paths[0]),
            structured_info={
                **second.structured_info,
                "多图说明": "功能演示：两张公开 validation 图像，不声明来自同一患者",
            },
        )
    return cases


def _qualification_for_pipeline(
    context: ToolContext,
    case: ReviewCase,
    *,
    tool_payload: dict[str, Any],
    case_scope: str,
    mode: str,
) -> RouteQualification:
    route = tool_payload.get("route", {})
    route_data = route.get("data", {}) if isinstance(route, dict) else {}
    qualification_payload = route_data.get("route_qualification")
    if (
        mode != "qualification_blocked"
        and isinstance(qualification_payload, dict)
    ):
        qualification = route_qualification_decision_from_dict(
            qualification_payload
        )
    else:
        contract, contract_sha = load_route_qualification_contract(
            context.project_root
        )
        qualification = evaluate_route_qualification(
            RouteQualificationRequest(
                task_id=case.task_id,
                pairing_id=str(route_data.get("pairing_id", "unavailable")),
                scout_artifact_ids=tuple(
                    str(value)
                    for value in route_data.get("scout_artifact_ids", [])
                ),
                expert_artifact_id=str(
                    route_data.get("expert_artifact_id", "")
                ),
                request_scope=case_scope,
                prediction_assets_valid=mode != "qualification_blocked",
                unique_protocol_identity=mode != "qualification_blocked",
            ),
            contract=contract,
            contract_sha256=contract_sha,
        )
    return RouteQualification.from_decision(
        qualification,
        evidence={
            "pairing_id": str(route_data.get("pairing_id", "unavailable")),
            "gate_evidence": route_data.get("gate_evidence", {}),
        },
    )


def _controller_for_mode(mode: str):
    if mode != "llm_illegal":
        return RuleController()
    return LocalLLMController(
        LocalLLMControllerConfig(model_id="mock-local-llm-illegal"),
        inference_callable=lambda _prompt, _config: {
            "action": "CALL_UNAUTHORIZED_TOOL",
            "reason_code": "FREE_PLAN",
            "parameters": {"tool": "model_inference.run"},
            "schema_version": "ophagent.controller_proposal.v1",
        },
    )


def _case_state_id(
    case: ReviewCase,
    *,
    scenario: str,
    controller_type: str,
) -> str:
    scenario_hash = hashlib.sha256(
        (
            f"{DEMO_STATE_ID_VERSION}:{scenario}:{controller_type}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    return f"{case.alias}-{scenario_hash}"


def _runtime_from_trace(
    context: ToolContext,
    trace_payload: dict[str, Any],
) -> ToolRuntime:
    runtime = ToolRuntime(
        context,
        trace_id=str(trace_payload.get("trace_id", "restored-trace")),
    )
    fields = set(TraceEvent.__dataclass_fields__)
    runtime.events = []
    for event in trace_payload.get("events", []):
        if not isinstance(event, dict) or not fields.issubset(event):
            continue
        runtime.events.append(
            TraceEvent(**{field: event[field] for field in fields})
        )
    runtime.halted = any(event.status == "failed" for event in runtime.events)
    return runtime


def _run_read_only_pipeline(
    context: ToolContext,
    case: ReviewCase,
    artifact_ids: list[str],
    *,
    resume_state: CaseState | None = None,
) -> tuple[dict[str, Any], ToolRuntime]:
    runtime = ToolRuntime(context)
    common = {
        "task_id": case.task_id,
        "case_id": case.alias,
    }
    completed_steps = set(
        resume_state.completed_steps if resume_state is not None else ()
    )
    cached = dict(
        resume_state.runtime_payload if resume_state is not None else {}
    )
    if "model_qualification" in completed_steps:
        if not cached.get("input") or not cached.get("registry"):
            raise ValueError("PERSISTED_VALIDATION_PAYLOAD_MISSING")
        input_payload = dict(cached["input"])
        registry_payload = dict(cached["registry"])
    else:
        input_payload = runtime.run(
            ToolRequest(
                "case_input.validate",
                payload={
                    "split": "validation",
                    "image_paths": [str(path) for path in case.image_paths],
                    "structured_info": case.structured_info,
                },
                **common,
            )
        ).to_dict()
        registry_payload = runtime.run(
            ToolRequest("model_registry.inspect", payload={}, **common)
        ).to_dict()

    if "scout" in completed_steps:
        if not cached.get("route") or not cached.get("predictions"):
            raise ValueError("PERSISTED_SCOUT_PAYLOAD_MISSING")
        route_payload = dict(cached["route"])
        predictions = list(cached["predictions"])
    else:
        route_payload = runtime.run(
            ToolRequest(
                "routing_protocol.evaluate",
                payload={
                    "split": "validation",
                    "source_case_key": case.source_case_key,
                },
                **common,
            )
        ).to_dict()
        predictions = []
    if (
        "scout" not in completed_steps
        and input_payload.get("ok")
        and registry_payload.get("ok")
        and route_payload.get("ok")
    ):
        route_scouts = [
            str(value)
            for value in route_payload.get("data", {}).get(
                "scout_artifact_ids",
                [],
            )
        ]
        allowed_artifacts = set(artifact_ids)
        for artifact_id in route_scouts:
            if artifact_id not in allowed_artifacts:
                raise ValueError("ROUTE_SCOUT_NOT_IN_SCENARIO_CAPABILITY")
            response = runtime.run(
                ToolRequest(
                    "prediction_asset.validate",
                    payload={
                        "artifact_id": artifact_id,
                        "split": "validation",
                        "source_case_key": case.source_case_key,
                    },
                    **common,
                )
            )
            if not response.ok:
                break
            predictions.append(response.data)
    audit_response = runtime.run(
        ToolRequest(
            "result_risk_audit.run",
            payload={"predictions": predictions},
            **common,
        )
    )
    tool_payload = {
        "input": input_payload,
        "registry": registry_payload,
        "predictions": predictions,
        "audit": audit_response.to_dict(),
        "route": route_payload,
    }
    return (
        tool_payload,
        runtime,
    )


def _run_fault_pipeline(
    context: ToolContext,
    case: ReviewCase,
    artifact_id: str,
) -> tuple[dict[str, Any], ToolRuntime]:
    runtime = ToolRuntime(context)
    common = {"task_id": case.task_id, "case_id": case.alias}
    input_response = runtime.run(
        ToolRequest(
            "case_input.validate",
            payload={
                "split": "validation",
                "image_paths": [str(path) for path in case.image_paths],
            },
            **common,
        )
    )
    registry_response = runtime.run(
        ToolRequest("model_registry.inspect", payload={}, **common)
    )
    inference_response = runtime.run(
        ToolRequest(
            "model_inference.run",
            payload={
                "artifact_id": artifact_id,
                "image_paths": [str(case.image_paths[0])],
            },
            **common,
        )
    )
    downstream_response = runtime.run(
        ToolRequest(
            "result_risk_audit.run",
            payload={"predictions": []},
            **common,
        )
    )
    tool_payload = {
        "input": input_response.to_dict(),
        "registry": registry_response.to_dict(),
        "inference": inference_response.to_dict(),
        "downstream": downstream_response.to_dict(),
    }
    return (
        tool_payload,
        runtime,
    )


@dataclass(frozen=True)
class _ReviewStepToolExecutor:
    context: ToolContext
    case: ReviewCase
    artifact_ids: tuple[str, ...]
    case_scope: str
    mode: str

    def execute_step(
        self,
        state: CaseState,
        step: str,
    ) -> AgentToolStepResult:
        runtime = ToolRuntime(self.context)
        common = {
            "task_id": self.case.task_id,
            "case_id": self.case.alias,
        }
        payload: dict[str, Any]
        qualification: RouteQualification | None = None
        if step == "input":
            response = runtime.run(
                ToolRequest(
                    "case_input.validate",
                    payload={
                        "split": "validation",
                        "image_paths": [
                            str(path) for path in self.case.image_paths
                        ],
                        "structured_info": self.case.structured_info,
                    },
                    **common,
                )
            )
            payload = {"input": response.to_dict()}
        elif step == "registry":
            response = runtime.run(
                ToolRequest(
                    "model_registry.inspect",
                    payload={},
                    **common,
                )
            )
            payload = {"registry": response.to_dict()}
        elif step == "route_metadata":
            if self.mode == "tool_failure":
                payload = {
                    "route": {
                        "ok": True,
                        "code": "SKIPPED_NEW_CASE",
                        "message": "新病例不读取冻结路由结果",
                        "data": {
                            "protocol_requests_expert": False,
                            "expert_result_released": False,
                        },
                    }
                }
            else:
                response = runtime.run(
                    ToolRequest(
                        "routing_protocol.evaluate",
                        payload={
                            "split": "validation",
                            "source_case_key": (
                                self.case.source_case_key
                            ),
                        },
                        **common,
                    )
                )
                payload = {"route": response.to_dict()}
        elif step == "scout":
            if self.mode == "tool_failure":
                response = runtime.run(
                    ToolRequest(
                        "model_inference.run",
                        payload={
                            "artifact_id": self.artifact_ids[0],
                            "image_paths": [
                                str(self.case.image_paths[0])
                            ],
                        },
                        **common,
                    )
                )
                payload = {"inference": response.to_dict()}
            else:
                route_data = state.runtime_payload.get(
                    "route",
                    {},
                ).get("data", {})
                scout_ids = tuple(
                    str(value)
                    for value in route_data.get(
                        "scout_artifact_ids",
                        (),
                    )
                )
                if not scout_ids:
                    raise ValueError("ROUTE_SCOUT_IDENTITY_MISSING")
                if not set(scout_ids).issubset(self.artifact_ids):
                    raise ValueError(
                        "ROUTE_SCOUT_NOT_IN_SCENARIO_CAPABILITY"
                    )
                predictions: list[dict[str, Any]] = []
                failed_response: dict[str, Any] | None = None
                for artifact_id in scout_ids:
                    response = runtime.run(
                        ToolRequest(
                            "prediction_asset.validate",
                            payload={
                                "artifact_id": artifact_id,
                                "split": "validation",
                                "source_case_key": (
                                    self.case.source_case_key
                                ),
                            },
                            **common,
                        )
                    )
                    if not response.ok:
                        failed_response = response.to_dict()
                        break
                    predictions.append(response.data)
                payload = (
                    {"predictions": predictions}
                    if failed_response is None
                    else {
                        "predictions": [],
                        "inference": failed_response,
                    }
                )
        elif step == "audit_and_qualification":
            predictions = list(
                state.runtime_payload.get("predictions", [])
            )
            response = runtime.run(
                ToolRequest(
                    "result_risk_audit.run",
                    payload={"predictions": predictions},
                    **common,
                )
            )
            payload = {"audit": response.to_dict()}
            if response.ok:
                merged_payload = {
                    **state.runtime_payload,
                    **payload,
                }
                qualification = _qualification_for_pipeline(
                    self.context,
                    self.case,
                    tool_payload=merged_payload,
                    case_scope=self.case_scope,
                    mode=self.mode,
                )
        else:
            raise ValueError(f"UNKNOWN_AGENT_TOOL_STEP:{step}")
        return AgentToolStepResult(
            tool_payload=payload,
            tool_trace=runtime.to_dict(),
            qualification=qualification,
        )


@dataclass(frozen=True)
class _ReviewExpertReplayExecutor:
    context: ToolContext
    case: ReviewCase

    def __call__(self, state: CaseState) -> AgentExpertResult:
        route_data = state.runtime_payload.get(
            "route",
            {},
        ).get("data", {})
        expert_artifact_id = str(
            route_data.get("expert_artifact_id", "")
        )
        if not expert_artifact_id:
            raise ValueError("EXPERT_ARTIFACT_ID_MISSING")
        runtime = ToolRuntime(self.context)
        response = runtime.run(
            ToolRequest(
                "prediction_asset.validate",
                payload={
                    "artifact_id": expert_artifact_id,
                    "split": "validation",
                    "source_case_key": self.case.source_case_key,
                },
                task_id=self.case.task_id,
                case_id=self.case.alias,
            )
        )
        return AgentExpertResult(
            tool_payload={"expert": response.to_dict()},
            tool_trace=runtime.to_dict(),
        )


def _execute_controlled_pipeline(
    context: ToolContext,
    case: ReviewCase,
    *,
    scenario: str,
    artifact_ids: list[str],
    mode: str,
    actor_role: str,
) -> tuple[dict[str, Any], ToolRuntime, CaseState, bool]:
    controller = _controller_for_mode(mode)
    case_scope = "new_case" if mode == "tool_failure" else "cached_prediction_replay"
    _contract, contract_sha = load_route_qualification_contract(
        context.project_root
    )
    expected_expert_cost = 2.0 if mode == "cost_blocked" else 0.2
    remaining_budget = 0.5 if mode == "cost_blocked" else 1.0
    case_id = _case_state_id(
        case,
        scenario=scenario,
        controller_type=controller.controller_type,
    )
    request = ControlledCaseRequest(
        case_id=case_id,
        task_id=case.task_id,
        idempotency_key=f"controlled-v2:{case_id}",
        case_scope=case_scope,
        case_metadata={
            "modality": "CFP",
            "image_count": len(case.image_paths),
            "quality_flag": "public_validation_demo",
        },
        remaining_budget=remaining_budget,
        expected_expert_cost=expected_expert_cost,
        controller_type=controller.controller_type,
        qualification_policy_version=contract_sha,
        route_protocol_version="frozen-validation-route-v1",
    )
    agent_runtime = ControlledAgentRuntimeV2(
        CaseStateStore(CONTROLLED_RUNTIME_ROOT / "cases")
    )
    case_state, replayed = agent_runtime.execute(
        request,
        controller=controller,
        tool_executor=_ReviewStepToolExecutor(
            context=context,
            case=case,
            artifact_ids=tuple(artifact_ids),
            case_scope=case_scope,
            mode=mode,
        ),
        actor_role=actor_role,
    )
    payload = dict(case_state.runtime_payload)
    payload["agent"] = state_view_model(case_state)
    runtime = _runtime_from_trace(
        context,
        case_state.tool_trace,
    )
    return payload, runtime, case_state, replayed


def _model_rows(registry_payload: dict[str, Any]) -> pd.DataFrame:
    records = registry_payload.get("data", {}).get("models", [])
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    qualification_labels = {
        "analytical_asset_only": "仅分析资产",
        "offline_batch_inference_ready": "离线批量推理",
        "online_case_inference_ready": "单病例原图可调用",
    }
    frame["资格"] = frame["qualification"].map(qualification_labels)
    frame["成本"] = frame["cost_ms_per_image"].map(
        lambda value: "未完整测量"
        if value is None or pd.isna(value)
        else f"{float(value):.3f} ms/图"
    )
    frame["路由资格"] = frame["route_eligible"].map(
        {True: "可进入路由池", False: "不可进入路由池"}
    )
    return frame


def _render_case_queue(
    cases: list[ReviewCase],
    *,
    selected_index: int,
    state: dict[str, Any],
) -> None:
    st.caption("仅显示脱敏审阅编号；原始病例键不在页面展示。")
    decisions = state.get("cases", {})
    status_labels = {
        "接受模型输出": "已接受",
        "修改输出": "已修改",
        "标记不确定": "不确定",
    }
    columns = st.columns(min(4, max(1, len(cases))), gap="small")
    for index, case in enumerate(cases):
        decision = decisions.get(case.alias, {})
        status = str(decision.get("decision", "待审阅"))
        active = index == selected_index
        label = f"{case.alias} · {status_labels.get(status, status)}"
        with columns[index % len(columns)]:
            if st.button(
                label,
                key=f"review_case::{case.task_id}::{case.alias}",
                type="primary" if active else "secondary",
                width="stretch",
            ):
                st.session_state["review_case_index"] = index
                st.rerun()


def _render_images(case: ReviewCase) -> None:
    st.markdown("#### 眼底图像")
    zoom = st.slider(
        "图像显示宽度",
        min_value=360,
        max_value=900,
        value=560,
        step=20,
        key=f"review_zoom::{case.alias}",
    )
    if len(case.image_paths) == 1:
        st.image(str(case.image_paths[0]), width=zoom)
        st.caption("公开 validation 图像 · 页面不显示原始文件名或路径")
        return
    tabs = st.tabs([f"图像 {index + 1}" for index in range(len(case.image_paths))])
    for tab, path in zip(tabs, case.image_paths, strict=True):
        with tab:
            st.image(str(path), width=zoom)
    st.caption(str(case.structured_info.get("多图说明", "多图病例")))


def _render_input_status(case: ReviewCase, payload: dict[str, Any]) -> None:
    st.markdown("#### 输入与任务检查")
    response = payload["input"]
    if response["ok"]:
        data = response["data"]
        st.success(
            f"输入完整 · {data['image_count']} 张本地图像 · "
            f"{html.escape(str(data['task_name']))} · validation"
        )
    else:
        st.error(f"{response['code']} · {response['message']}")
    with st.expander("结构化病例信息", expanded=False):
        rows = [
            {"字段": key, "内容": value}
            for key, value in case.structured_info.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_model_table(payload: dict[str, Any]) -> None:
    st.markdown("#### 可用模型与资格")
    registry = _model_rows(payload["registry"])
    if registry.empty:
        st.warning("当前任务没有可展示模型。")
        return
    st.dataframe(
        registry[
            [
                "display_name",
                "资格",
                "成本",
                "qualification_reason",
            ]
        ].rename(
            columns={
                "display_name": "模型",
                "qualification_reason": "限制或阻塞原因",
            }
        ),
        hide_index=True,
        width="stretch",
        height=265,
    )
    st.caption(
        "只有“单病例原图可调用”的模型可运行新病例；"
        "冻结概率资产只允许只读审计。"
    )


def _render_predictions(task_id: str, payload: dict[str, Any]) -> None:
    st.markdown("#### 单模型输出")
    predictions = payload.get("predictions", [])
    if not predictions:
        st.info("当前流程没有可展示的只读模型输出。")
        return
    columns = st.columns(len(predictions), gap="small")
    for column, prediction in zip(columns, predictions, strict=True):
        with column:
            st.markdown(
                '<div class="review-result-card">'
                f'<span>{html.escape(human_model(prediction["artifact_id"]))}</span>'
                f'<b>{html.escape(grade_label(task_id, prediction["pred_label"]))}</b>'
                f'<small>置信度 {prediction["confidence"]:.1%}<br>'
                f'Margin {prediction["margin"]:.3f}</small></div>',
                unsafe_allow_html=True,
            )


def _render_controlled_agent_trace(payload: dict[str, Any]) -> None:
    agent = payload.get("agent", {})
    if not agent:
        return
    qualification = agent.get("qualification", {})
    level_labels = {
        "blocked": "资格阻止",
        "research_replay_only": "仅历史回放",
        "research_case_simulation": "研究病例模拟",
        "deployment_candidate": "部署候选",
        "clinical_route_eligible": "临床路由资格",
    }
    evidence_labels = {
        "beneficial": "Validation 有益",
        "risk_tradeoff": "存在代理事件权衡",
        "ineffective": "Validation 未见增益",
        "unstable": "证据不稳定",
    }
    risk_labels = {
        "protocol_requests_expert": "冻结协议触发专家",
        "model_disagreement_observed": "观察到模型分歧",
        "protocol_keeps_scout": "冻结协议保持 Scout",
        "qualification_restricted": "资格限制",
        "cost_restricted": "成本限制",
        "not_evaluated": "上游失败，未继续评估",
    }
    action_labels = {
        "KEEP_SCOUT": "保持 Scout",
        "REQUEST_EXPERT": "请求 Expert",
        "REFER_TO_HUMAN": "转人工处理",
    }
    level = str(qualification.get("execution_level", "blocked"))
    evidence = str(qualification.get("evidence_label", "unstable"))
    proposal = dict(agent.get("controller_proposal", {}))
    gate = dict(agent.get("gate_decision", {}))
    risk_state = str(proposal.get("reason_code", "not_evaluated"))
    action = str(agent.get("final_action", "REFER_TO_HUMAN"))
    risk_labels.update(
        {
            "LOW_RISK_KEEP_SCOUT": "低风险，建议保持 Scout",
            "HIGH_RISK_REQUEST_EXPERT": "高风险，建议请求 Expert",
            "MODEL_DISAGREEMENT": "模型分歧，建议请求 Expert",
            "QUALIFICATION_RESTRICTED": "资格不足，建议转人工",
            "TOOL_FAILURE": "工具失败，停止下游",
            "INVALID_PROPOSAL": "控制器提议不合法",
        }
    )
    level_class = (
        "ok"
        if level in {"research_case_simulation", "deployment_candidate"}
        else "restricted"
    )
    action_class = {
        "KEEP_SCOUT": "ok",
        "REQUEST_EXPERT": "attention",
        "REFER_TO_HUMAN": "restricted",
    }.get(action, "restricted")
    codes = " · ".join(
        str(value) for value in qualification.get("error_codes", [])[:2]
    )
    cards = [
        (
            "01 · 资格判定",
            level_labels.get(level, level),
            evidence_labels.get(evidence, evidence),
            level_class,
        ),
        (
            "02 · 风险判定",
            risk_labels.get(risk_state, risk_state),
            f"控制器提议：{proposal.get('action', '未生成')}",
            "attention" if "expert" in risk_state else "neutral",
        ),
        (
            "03 · 门控裁决",
            str(gate.get("code", "未裁决")),
            (
                "非法提议已拦截"
                if gate.get("gate_intercepted")
                else "提议通过确定性复核"
            ),
            "restricted" if gate.get("gate_intercepted") else "neutral",
        ),
        (
            "04 · 最终动作",
            action_labels.get(action, action),
            "等待人工确认后才写入审阅结论",
            action_class,
        ),
    ]
    content = "".join(
        '<div class="agent-decision-step '
        f'{css_class}"><span>{html.escape(label)}</span>'
        f"<b>{html.escape(value)}</b>"
        f"<small>{html.escape(note)}</small></div>"
        for label, value, note, css_class in cards
    )
    st.markdown(
        '<div class="agent-decision-head">'
        "<div><b>受控 Agent 决策 · V2</b>"
        "<span>Controller 只提议；规则控制器仍是确定性规则基线</span></div>"
        f'<span class="hub-chip hub-chip-blue">{html.escape(str(agent.get("controller_type", "未记录")))}</span></div>'
        f'<div class="agent-decision-strip">{content}</div>',
        unsafe_allow_html=True,
    )
    if codes:
        st.caption(f"当前限制码：{codes}")


def _render_audit_and_route(task_id: str, payload: dict[str, Any]) -> None:
    audit = payload["audit"]
    route = payload["route"]
    st.markdown("#### 模型输出错误风险审计")
    if audit["ok"]:
        data = audit["data"]
        models = data["models"]
        mean_entropy = sum(float(row["entropy"]) for row in models) / len(models)
        mean_margin = sum(float(row["margin"]) for row in models) / len(models)
        proxy = data["task_proxy"]
        columns = st.columns(3, gap="small")
        columns[0].metric("平均归一化熵", f"{mean_entropy:.3f}")
        columns[1].metric("平均 Top1-Top2 Margin", f"{mean_margin:.3f}")
        columns[2].metric("模型分歧", "存在" if data["model_disagreement"] else "未见")
        st.caption(
            f"{proxy['name']}：{float(proxy['value']):.1%}。{proxy['definition']}。"
            "该值是模型输出代理，不代表临床后果。"
        )
    else:
        st.error(f"{audit['code']} · {audit['message']}")

    st.markdown("#### Scout / Expert 研究模拟")
    if route["ok"]:
        data = route["data"]
        predictions = list(payload.get("predictions", []))
        scout_label = (
            grade_label(task_id, predictions[0]["pred_label"])
            if predictions
            else "未完成"
        )
        expert_response = dict(payload.get("expert", {}))
        expert_released = bool(expert_response.get("ok", False))
        protocol_requests_expert = bool(
            data.get(
                "protocol_requests_expert",
                data.get("expert_invoked", False),
            )
        )
        expert_label = (
            grade_label(
                task_id,
                expert_response.get("data", {}).get("pred_label"),
            )
            if expert_released
            else (
                "等待人工批准"
                if protocol_requests_expert
                else "未调用"
            )
        )
        adopted_label = expert_label if expert_released else scout_label
        result_source = (
            "批准后的冻结 Expert 回放"
            if expert_released
            else (
                "等待 Expert 批准"
                if protocol_requests_expert
                else "冻结 Scout 回放"
            )
        )
        st.warning(
            "当前仅回放冻结 validation 概率，route_eligible=false；"
            "不是在线路由，也不提供诊断或分流建议。"
        )
        route_values = [
            ("Scout 输出", scout_label),
            ("Expert 输出", expert_label),
            ("系统采用输出", adopted_label),
            ("冻结预算", f"{float(data['requested_budget']):.0%}"),
        ]
        cards = "".join(
            '<div class="review-route-card">'
            f"<span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>"
            for label, value in route_values
        )
        st.markdown(
            f'<div class="review-route-grid">{cards}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"策略：{data['routing_policy']} · "
            f"分歧：{'是' if data['scout_disagreement'] else '否'} · "
            f"结果来源：{result_source}"
        )
    else:
        st.error(f"{route['code']} · {route['message']}")
    _render_controlled_agent_trace(payload)


def _render_state_timeline(case_state: CaseState, *, replayed: bool) -> None:
    view = state_view_model(case_state)
    st.markdown("#### 状态时间线")
    status_columns = st.columns(4, gap="small")
    status_columns[0].metric("当前状态", view["current_state_label"])
    status_columns[1].metric("最终三动作", view["final_action"] or "待裁决")
    status_columns[2].metric(
        "恢复来源",
        "持久状态恢复" if replayed else "本次执行",
    )
    status_columns[3].metric(
        "Trace 完整度",
        f"{len(view['timeline'])} 个状态事件",
    )
    timeline = pd.DataFrame(view["timeline"])
    if not timeline.empty:
        st.dataframe(
            timeline.rename(
                columns={
                    "sequence": "序号",
                    "state": "状态码",
                    "label": "中文状态",
                    "code": "返回码",
                    "at": "时间",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "页面刷新与服务重启均先读取 CaseStateStore；"
        "已完成步骤不会重新调用工具。"
    )
    if replayed:
        st.success(
            "幂等恢复成功：复用持久状态，重复工具调用 0 次。"
        )


def _render_v2_controls(
    case_state: CaseState,
    *,
    actor_role: str,
    context: ToolContext,
    case: ReviewCase,
) -> None:
    runtime = ControlledAgentRuntimeV2(
        CaseStateStore(CONTROLLED_RUNTIME_ROOT / "cases")
    )
    st.markdown("#### 人工批准与权限")
    if case_state.current_state == "EXPERT_PENDING_APPROVAL":
        can_decide_expert = actor_role in {"reviewer", "admin"}
        approve_column, reject_column = st.columns(2)
        if approve_column.button(
            "批准冻结 Expert 回放",
            type="primary",
            disabled=not can_decide_expert,
            width="stretch",
            key=f"expert-approve::{case_state.case_id}",
        ):
            try:
                runtime.decide_expert(
                    case_state.case_id,
                    approved=True,
                    actor_role=actor_role,
                    expert_executor=_ReviewExpertReplayExecutor(
                        context=context,
                        case=case,
                    ),
                )
            except PermissionDenied as exc:
                st.error(f"权限阻塞：{exc}")
            else:
                st.rerun()
        if reject_column.button(
            "拒绝 Expert，转人工",
            disabled=not can_decide_expert,
            width="stretch",
            key=f"expert-reject::{case_state.case_id}",
        ):
            try:
                runtime.decide_expert(
                    case_state.case_id,
                    approved=False,
                    actor_role=actor_role,
                )
            except PermissionDenied as exc:
                st.error(f"权限阻塞：{exc}")
            else:
                st.rerun()
        st.caption(
            "批准只释放已经存在的合规冻结 Expert 结果；"
            "本页不会触发新训练或重新推理。"
        )
        if not can_decide_expert:
            st.info("所需角色：reviewer 或 admin。当前角色不可批准或拒绝 Expert。")
    elif case_state.current_state == "FAILED":
        can_retry = actor_role in {"operator", "admin"}
        if st.button(
            "重试允许的失败步骤",
            disabled=not can_retry,
            width="stretch",
            key=f"retry::{case_state.case_id}",
        ):
            try:
                runtime.retry_failed(
                    case_state.case_id,
                    actor_role=actor_role,
                )
            except (PermissionDenied, StateTransitionError) as exc:
                st.error(f"重试阻塞：{exc}")
            else:
                st.rerun()
        if not can_retry:
            st.info("所需角色：operator 或 admin。当前角色不可重试失败步骤。")
    elif case_state.current_state == "REVIEW_PENDING":
        st.info("Agent 已停止在人工确认点；保存审阅结论后可关闭病例。")
    else:
        st.caption(f"当前状态无需 Expert 审批：{case_state.current_state}")

    if st.button(
        "验证协议修改权限（不执行修改）",
        width="stretch",
        key=f"permission-check::{case_state.case_id}",
    ):
        try:
            authorize(actor_role, "protocol.modify")
        except PermissionDenied as exc:
            st.warning(f"权限阻塞成功：{exc}")
        else:
            st.error("权限配置异常：协议修改不应由病例角色直接执行。")


def _review_report(
    *,
    case: ReviewCase,
    scenario: str,
    payload: dict[str, Any],
    trace_payload: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    predictions = [
        {
            "artifact_id": item["artifact_id"],
            "pred_label": item["pred_label"],
            "probabilities": item["probabilities"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "prediction_asset_sha256": item["prediction_asset_sha256"],
            "preprocessing_id": item["preprocessing_id"],
        }
        for item in payload.get("predictions", [])
    ]
    return {
        "schema_version": "ophagent.offline_case_review_report.v1",
        "created_at": str(decision["saved_at"]),
        "case_alias": case.alias,
        "scenario": scenario,
        "task_id": case.task_id,
        "data_scope": "public_validation_read_only",
        "input_image_count": len(case.image_paths),
        "model_outputs": predictions,
        "risk_audit": payload.get("audit", {}).get("data", {}),
        "routing_simulation": payload.get("route", {}).get("data", {}),
        "controlled_agent": payload.get("agent", {}),
        "human_review": decision,
        "trace": trace_payload,
        "boundaries": {
            "clinical_diagnosis": False,
            "online_routing": False,
            "route_eligible": False,
            "test_content_used": False,
            "external_network_used": False,
        },
    }


def _review_artifact_paths(
    session_id: str,
    case_alias: str,
) -> tuple[Path, Path]:
    session_root = REVIEW_RUNTIME_ROOT / session_id
    return (
        session_root / "reports" / f"{case_alias}.json",
        session_root / "traces" / f"{case_alias}.json",
    )


def _closed_review_records(
    case_state: CaseState,
    stored: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if case_state.current_state != "CLOSED":
        raise StateTransitionError("REVIEW_ARTIFACT_RECOVERY_REQUIRES_CLOSED")
    decision = str(case_state.human_decision)
    if decision not in REVIEW_DECISIONS:
        raise StateTransitionError("CLOSED_REVIEW_DECISION_INVALID")
    stored = dict(stored or {})
    stored_decision = str(stored.get("decision", ""))
    if stored_decision and stored_decision != decision:
        raise StateTransitionError("CLOSED_REVIEW_DECISION_MISMATCH")
    saved_at = str(
        stored.get("saved_at")
        or case_state.updated_at
        or case_state.created_at
    )
    if not saved_at:
        raise StateTransitionError("CLOSED_REVIEW_TIMESTAMP_MISSING")
    isolated = {
        "decision": decision,
        "modified_output": redact_free_text(
            str(stored.get("modified_output", ""))
        ),
        "note": redact_free_text(str(stored.get("note", ""))),
        "review_queue": bool(stored.get("review_queue", False)),
        "saved_at": saved_at,
    }
    downloadable = {
        "decision": decision,
        "review_queue": isolated["review_queue"],
        "saved_at": saved_at,
        "modified_output_recorded_in_isolated_store": bool(
            isolated["modified_output"]
        ),
        "note_recorded_in_isolated_store": bool(isolated["note"]),
        "free_text_in_downloadable_report": False,
    }
    return isolated, downloadable


def _persist_closed_review_artifacts(
    *,
    case: ReviewCase,
    scenario: str,
    session_id: str,
    case_state: CaseState,
    actor_role: str,
    review_state: dict[str, Any],
) -> dict[str, Any]:
    """Write only missing evidence from a durable CLOSED CaseState."""

    authorize(actor_role, "review.confirm")
    stored = review_state.get("cases", {}).get(case.alias)
    isolated, downloadable = _closed_review_records(case_state, stored)
    report_path, trace_path = _review_artifact_paths(
        session_id,
        case.alias,
    )
    if report_path.is_file() and trace_path.is_file():
        return {
            "report_path": report_path,
            "trace_path": trace_path,
            "recovered": False,
        }
    trace_payload = dict(case_state.tool_trace)
    if not trace_payload:
        raise StateTransitionError("CLOSED_TOOL_TRACE_MISSING")
    persisted = _read_review_state(session_id).get("cases", {}).get(case.alias)
    if persisted != isolated:
        review_state.setdefault("cases", {})[case.alias] = isolated
        _save_review_state(session_id, review_state)
    report_payload = {
        **case_state.runtime_payload,
        "agent": state_view_model(case_state),
    }
    report = _review_report(
        case=case,
        scenario=scenario,
        payload=report_payload,
        trace_payload=trace_payload,
        decision=downloadable,
    )
    recovered = False
    if not report_path.is_file():
        _atomic_json(report_path, report)
        recovered = True
    if not trace_path.is_file():
        _atomic_json(trace_path, trace_payload)
        recovered = True
    return {
        "report_path": report_path,
        "trace_path": trace_path,
        "recovered": recovered,
    }


def _render_review_actions(
    *,
    case: ReviewCase,
    cases: list[ReviewCase],
    scenario: str,
    selected_index: int,
    state: dict[str, Any],
    session_id: str,
    case_state: CaseState,
    actor_role: str,
) -> None:
    st.markdown("#### 人工审阅")
    review_ready = case_state.current_state == "REVIEW_PENDING"
    review_closed = case_state.current_state == "CLOSED"
    can_confirm_review = actor_role in {"reviewer", "admin"}
    review_controls_enabled = review_ready and can_confirm_review
    report_path, trace_path = _review_artifact_paths(
        session_id,
        case.alias,
    )
    if review_closed:
        st.success("人工确认已持久化，病例保持 CLOSED。")
        st.caption(
            f"已确认结论：{case_state.human_decision}。"
            "自由文本仅保留在隔离审阅状态中，"
            "本页不会回显，也不会写入下载报告或 Trace。"
        )
    elif not review_ready:
        st.info(
            "仅“等待人工确认（REVIEW_PENDING）”状态可提交正式审阅；"
            "请先完成 Expert 批准或允许的失败恢复。"
        )
    elif not can_confirm_review:
        st.info("所需角色：reviewer 或 admin。当前角色的审阅控件已禁用。")
    previous = state.get("cases", {}).get(case.alias, {})
    default = str(previous.get("decision", "标记不确定"))
    if default not in REVIEW_DECISIONS:
        default = "标记不确定"
    decision: str | None = case_state.human_decision or default
    modified_output = ""
    uncertain_reason = ""
    add_to_queue = bool(previous.get("review_queue", False))
    if not review_closed:
        decision = st.segmented_control(
            "审阅结论",
            REVIEW_DECISIONS,
            default=default,
            disabled=not review_controls_enabled,
            key=f"review_decision::{case.alias}::{actor_role}",
        )
        modified_output = st.text_input(
            "人工修改内容（选择“修改输出”时填写）",
            value=(
                str(previous.get("modified_output", ""))
                if review_controls_enabled
                else ""
            ),
            disabled=not review_controls_enabled,
            key=f"review_modified::{case.alias}::{actor_role}",
        )
        uncertain_reason = st.text_area(
            "审阅备注",
            value=(
                str(previous.get("note", ""))
                if review_controls_enabled
                else ""
            ),
            placeholder="可记录图像质量、模型分歧或需补充的信息；不要填写患者身份信息。",
            disabled=not review_controls_enabled,
            key=f"review_note::{case.alias}::{actor_role}",
        )
        add_to_queue = st.checkbox(
            "加入复核队列",
            value=(
                bool(previous.get("review_queue", False))
                if review_controls_enabled
                else False
            ),
            disabled=not review_controls_enabled,
            key=f"review_queue::{case.alias}::{actor_role}",
        )
    button_columns = st.columns([1, 1.35, 1])
    if button_columns[0].button(
        "上一例",
        disabled=selected_index <= 0,
        width="stretch",
    ):
        st.session_state["review_case_index"] = selected_index - 1
        st.rerun()
    save_current = button_columns[1].button(
        "保存并进入下一例",
        type="primary",
        disabled=not review_controls_enabled,
        width="stretch",
    )
    if button_columns[2].button(
        "仅保存",
        disabled=not review_controls_enabled,
        width="stretch",
    ):
        save_current = True
        move_next = False
    else:
        move_next = True
    if save_current:
        if case_state.current_state != "REVIEW_PENDING":
            st.warning("状态阻塞：当前病例尚未进入 REVIEW_PENDING，未写入任何资产。")
            return
        try:
            authorize(actor_role, "review.confirm")
        except PermissionDenied as exc:
            st.warning(
                f"权限阻塞：{exc}。未写入审阅状态、报告或 Trace。"
            )
            return
        case_decision = {
            "decision": decision or "标记不确定",
            "modified_output": redact_free_text(
                modified_output.strip()
            ),
            "note": redact_free_text(uncertain_reason.strip()),
            "review_queue": bool(add_to_queue),
            "saved_at": "",
        }
        try:
            closed_state = ControlledAgentRuntimeV2(
                CaseStateStore(CONTROLLED_RUNTIME_ROOT / "cases")
            ).confirm_review(
                case_state.case_id,
                decision=str(case_decision["decision"]),
                actor_role=actor_role,
            )
        except (PermissionDenied, StateTransitionError):
            st.warning("人工确认被权限或状态门禁阻塞，未关闭病例或写入报告。")
            return
        case_decision["saved_at"] = (
            closed_state.updated_at or closed_state.created_at
        )
        state.setdefault("cases", {})[case.alias] = case_decision
        try:
            artifacts = _persist_closed_review_artifacts(
                case=case,
                scenario=scenario,
                session_id=session_id,
                case_state=closed_state,
                actor_role=actor_role,
                review_state=state,
            )
        except (OSError, TypeError, ValueError, StateTransitionError):
            st.error(
                "人工确认已持久化，病例保持 CLOSED；报告或 Trace sidecar "
                "写入失败。刷新后使用恢复按钮重试，不会重复确认。"
            )
            return
        report_path = Path(artifacts["report_path"])
        st.session_state[f"review_saved::{case.alias}"] = str(report_path)
        if move_next and selected_index < len(cases) - 1:
            st.session_state["review_case_index"] = selected_index + 1
            st.rerun()
        st.success("审阅状态、结构化报告和工具轨迹已保存。")

    if review_closed and (not report_path.is_file() or not trace_path.is_file()):
        st.warning(
            "CLOSED 证据不完整："
            f"结构化报告{'完整' if report_path.is_file() else '缺失'}；"
            f"Trace sidecar {'完整' if trace_path.is_file() else '缺失'}。"
        )
        if not can_confirm_review:
            st.info("所需角色：reviewer 或 admin。当前角色不可执行恢复写入。")
        if st.button(
            "从 CLOSED 状态恢复缺失报告 / Trace（不重复确认）",
            disabled=not can_confirm_review,
            width="stretch",
            key=f"recover-review-artifacts::{case.alias}",
        ):
            try:
                artifacts = _persist_closed_review_artifacts(
                    case=case,
                    scenario=scenario,
                    session_id=session_id,
                    case_state=case_state,
                    actor_role=actor_role,
                    review_state=state,
                )
            except PermissionDenied:
                st.warning("权限阻塞：仅 reviewer 或 admin 可恢复审阅证据。")
            except (OSError, TypeError, ValueError, StateTransitionError):
                st.error(
                    "恢复写入仍未完成；病例保持 CLOSED，"
                    "未重复人工确认，也未改变状态。"
                )
            else:
                report_path = Path(artifacts["report_path"])
                st.session_state[f"review_saved::{case.alias}"] = str(
                    report_path
                )
                st.success(
                    "已从 CLOSED 统一状态幂等恢复缺失证据；"
                    "未重复人工确认，病例状态未改变。"
                )
    elif review_closed:
        st.success("CLOSED 结构化报告与 Trace sidecar 均完整。")

    if report_path.is_file():
        st.download_button(
            "下载结构化报告",
            data=report_path.read_bytes(),
            file_name=f"{case.alias}_review_report.json",
            mime="application/json",
            width="stretch",
        )


def _render_trace(runtime: ToolRuntime, payload: dict[str, Any]) -> None:
    with st.expander("完整工具调用轨迹与来源证据", expanded=False):
        frame = trace_frame(runtime)
        display = frame[
            [
                "sequence",
                "tool_name",
                "status",
                "code",
                "duration_ms",
                "qualification",
                "message",
            ]
        ].rename(
            columns={
                "sequence": "序号",
                "tool_name": "工具",
                "status": "状态",
                "code": "错误码",
                "duration_ms": "耗时（ms）",
                "qualification": "资格",
                "message": "说明",
            }
        )
        st.dataframe(display, hide_index=True, width="stretch")
        route = payload.get("route", {}).get("data", {})
        if route:
            st.caption(
                f"Protocol SHA256：{route.get('protocol_sha256', '未记录')} · "
                f"Commit：{route.get('git_commit', '未记录')} · "
                "成本口径来自模型 registry 的 H100 forward-only 记录。"
            )
        agent = payload.get("agent", {})
        if agent:
            agent_trace = pd.DataFrame(agent.get("trace", []))
            if not agent_trace.empty:
                st.markdown("##### 受控 Agent 状态轨迹")
                st.dataframe(
                    agent_trace,
                    hide_index=True,
                    width="stretch",
                )
            qualification = agent.get("qualification", {})
            st.caption(
                "Route Qualification Contract SHA256："
                f"{qualification.get('contract_sha256', '未记录')} · "
                "Evidence fingerprint："
                f"{qualification.get('evidence_fingerprint', '未记录')}"
            )


def _render_fault(payload: dict[str, Any], runtime: ToolRuntime) -> None:
    st.markdown("#### 结构化故障门禁")
    failure = payload["inference"]
    downstream = payload.get(
        "downstream",
        {
            "code": "NOT_DISPATCHED_AFTER_UPSTREAM_FAILURE",
        },
    )
    st.error(
        f"{failure['code']} · {failure['message']}\n\n"
        f"当前资格：{failure.get('qualification') or failure['data'].get('qualification', '未记录')}；"
        "要求：online_case_inference_ready。"
    )
    st.info(
        f"后续风险审计状态：{downstream['code']}。"
        "上游失败后没有继续执行越权调用，重复工具调用 0 次。"
    )
    _render_controlled_agent_trace(payload)
    _render_trace(runtime, payload)


def render_offline_case_review_workstation() -> None:
    context = build_default_tool_context(PROJECT_ROOT)
    bind_mode = os.environ.get("OPHAGENT_BIND_MODE", "localhost").strip().lower()
    bind_label = "院内局域网模式" if bind_mode == "lan" else "localhost 单机模式"
    session_id = "public-demo-review"
    state = _read_review_state(session_id)
    st.markdown(
        '<div class="review-offline-banner">'
        '<div><b>离线病例审阅工作台</b><span>本地资产 · 无公网调用 · validation 只读场景</span></div>'
        '<div><span class="hub-chip hub-chip-teal">离线模式</span>'
        f'<span class="hub-chip hub-chip-blue">{html.escape(bind_label)}</span>'
        '<span class="hub-chip hub-chip-amber">研究用途</span></div></div>',
        unsafe_allow_html=True,
    )
    workstation_scenarios = _load_workstation_scenarios(
        str(DEMO_SCENARIO_PATH)
    )
    control_column, role_column = st.columns([1.45, 0.55], gap="small")
    with control_column:
        scenario = st.selectbox(
            "脱敏验收场景",
            list(workstation_scenarios),
            help=(
                "正常场景只读冻结 validation 概率；"
                "异常场景验证资格、成本、Schema 与工具失败门禁。"
            ),
        )
    role_labels = {
        "operator": "operator · 病例操作员",
        "reviewer": "reviewer · 人工复核员",
        "admin": "admin · 管理员",
    }
    with role_column:
        actor_role = st.selectbox(
            "脱敏 Demo 角色模拟（非身份认证）",
            list(role_labels),
            format_func=lambda value: role_labels[value],
            index=0,
        )
        st.caption("正式部署时角色必须来自可信会话或院内反向代理。")
    scenario_config = workstation_scenarios[scenario]
    task_id = str(scenario_config["task_id"])
    cases = build_review_cases(PROJECT_ROOT, task_id)
    case_filter = str(scenario_config.get("case_filter", ""))
    if case_filter == "keep_scout":
        cases = [
            item
            for item in cases
            if item.structured_info.get("冻结策略请求 Expert") == "否"
        ]
    elif case_filter == "request_expert":
        cases = [
            item
            for item in cases
            if item.structured_info.get("冻结策略请求 Expert") == "是"
        ]
    if not cases:
        st.error("当前任务没有可用的公开 validation 图像与冻结 route trace。")
        return
    selected_index = min(
        max(0, int(st.session_state.get("review_case_index", 0))),
        len(cases) - 1,
    )
    case = cases[selected_index]

    with st.expander(
        f"病例队列 · {case.alias} · {len(cases)} 例",
        expanded=False,
    ):
        _render_case_queue(
            cases,
            selected_index=selected_index,
            state=state,
        )

    st.markdown(
        f"### {html.escape(case.alias)}"
        f" <span class='hub-chip hub-chip-blue'>{html.escape(TASK_CONTRACTS[task_id]['task_name'])}</span>",
        unsafe_allow_html=True,
    )
    try:
        payload, runtime, case_state, replayed = _execute_controlled_pipeline(
            context,
            case,
            scenario=scenario,
            artifact_ids=list(scenario_config["artifact_ids"]),
            mode=str(scenario_config["mode"]),
            actor_role=actor_role,
        )
    except PermissionDenied as exc:
        st.error(
            f"权限阻塞：{exc}。新病例须由 operator 或 admin 提交；"
            "reviewer 可恢复已有病例并执行人工批准。"
        )
        return

    image_column, context_column = st.columns([0.62, 0.38], gap="large")
    with image_column:
        _render_images(case)
    with context_column:
        _render_input_status(case, payload)

        registry = _model_rows(payload.get("registry", {}))
        st.markdown("#### 当前审阅范围")
        st.metric("已登记模型", len(registry))
        st.caption(
            "当前病例只读调用冻结 validation prediction；"
            "离线资产不会被当作新病例原图模型。"
        )

    if str(scenario_config["mode"]) == "tool_failure":
        _render_fault(payload, runtime)
        _render_state_timeline(case_state, replayed=replayed)
        _render_v2_controls(
            case_state,
            actor_role=actor_role,
            context=context,
            case=case,
        )
        return

    result_tab, route_tab, review_tab, trace_tab = st.tabs(
        ["模型结果", "门控与状态机", "人工审阅", "完整 Trace"]
    )
    with result_tab:
        _render_predictions(task_id, payload)
        _render_model_table(payload)
    with route_tab:
        _render_audit_and_route(task_id, payload)
        _render_state_timeline(case_state, replayed=replayed)
    with review_tab:
        _render_v2_controls(
            case_state,
            actor_role=actor_role,
            context=context,
            case=case,
        )
        _render_review_actions(
            case=case,
            cases=cases,
            scenario=scenario,
            selected_index=selected_index,
            state=state,
            session_id=session_id,
            case_state=case_state,
            actor_role=actor_role,
        )
    with trace_tab:
        st.markdown("#### 工具能力")
        status = capability_matrix(context)
        st.dataframe(
            status.rename(
                columns={
                    "tool_name": "工具",
                    "status": "实现状态",
                    "boundary": "边界",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        _render_trace(runtime, payload)
    st.caption("本工作台只支持离线研究审阅，不提供诊断、治疗或患者分流建议。")
