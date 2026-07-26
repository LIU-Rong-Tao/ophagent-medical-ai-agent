"""Model Hub V1 离线病例审阅工作台。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.model_hub_tools import (
    TASK_CONTRACTS,
    ToolContext,
    ToolRequest,
    ToolRuntime,
    build_default_tool_context,
    capability_matrix,
    trace_frame,
)
from app.model_hub_ui import grade_label, human_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_RUNTIME_ROOT = (
    PROJECT_ROOT / "experiments/model_hub/runtime/case_review_v1"
)

NORMAL_SCENARIOS = {
    "APTOS · 冻结 validation 资产": {
        "task_id": "aptos_dr_5class",
        "artifact_ids": ["flair", "ret_clip", "retfound_cfp"],
    },
    "青光眼 · 冻结 validation 资产": {
        "task_id": "glaucoma_3class",
        "artifact_ids": [
            "glaucoma_retfound_dinov2",
            "glaucoma_vit_b",
            "glaucoma_swin_tiny",
        ],
    },
}
FAULT_SCENARIO = "故障门禁 · 离线资产请求原图推理"


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
            }
        )
        if len(rows) >= 12:
            break
    return rows


def build_review_cases(project_root: Path, task_id: str) -> list[ReviewCase]:
    rows = _load_case_rows(str(project_root), task_id)
    prefix = "DR-V" if task_id == "aptos_dr_5class" else "GLA-V"
    cases = []
    for index, row in enumerate(rows):
        image_paths = (Path(row["image_path"]),)
        structured = {
            "数据范围": "公开 validation",
            "模态": "彩色眼底照相（CFP）",
            "眼别": "未提供",
            "设备信息": "未提供",
            "模型分歧信号": "是" if row["scout_disagreement"] else "否",
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


def _run_read_only_pipeline(
    context: ToolContext,
    case: ReviewCase,
    artifact_ids: list[str],
) -> tuple[dict[str, Any], ToolRuntime]:
    runtime = ToolRuntime(context)
    common = {
        "task_id": case.task_id,
        "case_id": case.alias,
    }
    input_response = runtime.run(
        ToolRequest(
            "case_input.validate",
            payload={
                "split": "validation",
                "image_paths": [str(path) for path in case.image_paths],
                "structured_info": case.structured_info,
            },
            **common,
        )
    )
    registry_response = runtime.run(
        ToolRequest("model_registry.inspect", payload={}, **common)
    )
    predictions = []
    if input_response.ok and registry_response.ok:
        for artifact_id in artifact_ids:
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
    route_response = runtime.run(
        ToolRequest(
            "routing_protocol.evaluate",
            payload={
                "split": "validation",
                "source_case_key": case.source_case_key,
            },
            **common,
        )
    )
    return (
        {
            "input": input_response.to_dict(),
            "registry": registry_response.to_dict(),
            "predictions": predictions,
            "audit": audit_response.to_dict(),
            "route": route_response.to_dict(),
        },
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
    return (
        {
            "input": input_response.to_dict(),
            "registry": registry_response.to_dict(),
            "inference": inference_response.to_dict(),
            "downstream": downstream_response.to_dict(),
        },
        runtime,
    )


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
        st.warning(
            "当前仅回放冻结 validation 概率，route_eligible=false；"
            "不是在线路由，也不提供诊断或分流建议。"
        )
        route_values = [
            (
                "Scout 输出",
                grade_label(task_id, payload["predictions"][0]["pred_label"]),
            ),
            (
                "Expert 输出",
                (
                    grade_label(task_id, data["expert_pred_label"])
                    if data["expert_invoked"]
                    else "未调用"
                ),
            ),
            (
                "系统采用输出",
                grade_label(task_id, data["final_pred_label"]),
            ),
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
            f"结果来源：{data['final_source']}"
        )
    else:
        st.error(f"{route['code']} · {route['message']}")


def _review_report(
    *,
    case: ReviewCase,
    scenario: str,
    payload: dict[str, Any],
    runtime: ToolRuntime,
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
        "created_at": _utc_now(),
        "case_alias": case.alias,
        "scenario": scenario,
        "task_id": case.task_id,
        "data_scope": "public_validation_read_only",
        "input_image_count": len(case.image_paths),
        "model_outputs": predictions,
        "risk_audit": payload.get("audit", {}).get("data", {}),
        "routing_simulation": payload.get("route", {}).get("data", {}),
        "human_review": decision,
        "trace": runtime.to_dict(),
        "boundaries": {
            "clinical_diagnosis": False,
            "online_routing": False,
            "route_eligible": False,
            "test_content_used": False,
            "external_network_used": False,
        },
    }


def _render_review_actions(
    *,
    case: ReviewCase,
    cases: list[ReviewCase],
    scenario: str,
    selected_index: int,
    state: dict[str, Any],
    payload: dict[str, Any],
    runtime: ToolRuntime,
    session_id: str,
) -> None:
    st.markdown("#### 人工审阅")
    previous = state.get("cases", {}).get(case.alias, {})
    options = ["接受模型输出", "修改输出", "标记不确定"]
    default = str(previous.get("decision", "标记不确定"))
    if default not in options:
        default = "标记不确定"
    decision = st.segmented_control(
        "审阅结论",
        options,
        default=default,
        key=f"review_decision::{case.alias}",
    )
    modified_output = st.text_input(
        "人工修改内容（选择“修改输出”时填写）",
        value=str(previous.get("modified_output", "")),
        key=f"review_modified::{case.alias}",
    )
    uncertain_reason = st.text_area(
        "审阅备注",
        value=str(previous.get("note", "")),
        placeholder="可记录图像质量、模型分歧或需补充的信息；不要填写患者身份信息。",
        key=f"review_note::{case.alias}",
    )
    add_to_queue = st.checkbox(
        "加入复核队列",
        value=bool(previous.get("review_queue", False)),
        key=f"review_queue::{case.alias}",
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
        width="stretch",
    )
    if button_columns[2].button("仅保存", width="stretch"):
        save_current = True
        move_next = False
    else:
        move_next = True
    if save_current:
        case_decision = {
            "decision": decision or "标记不确定",
            "modified_output": modified_output.strip(),
            "note": uncertain_reason.strip(),
            "review_queue": bool(add_to_queue),
            "saved_at": _utc_now(),
        }
        state.setdefault("cases", {})[case.alias] = case_decision
        _save_review_state(session_id, state)
        report = _review_report(
            case=case,
            scenario=scenario,
            payload=payload,
            runtime=runtime,
            decision=case_decision,
        )
        report_path = _atomic_json(
            REVIEW_RUNTIME_ROOT / session_id / "reports" / f"{case.alias}.json",
            report,
        )
        runtime.save(
            REVIEW_RUNTIME_ROOT / session_id / "traces" / f"{case.alias}.json"
        )
        st.session_state[f"review_saved::{case.alias}"] = str(report_path)
        if move_next and selected_index < len(cases) - 1:
            st.session_state["review_case_index"] = selected_index + 1
            st.rerun()
        st.success("审阅状态、结构化报告和工具轨迹已保存。")
    report_path_text = st.session_state.get(f"review_saved::{case.alias}")
    if report_path_text and Path(str(report_path_text)).is_file():
        report_path = Path(str(report_path_text))
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


def _render_fault(payload: dict[str, Any], runtime: ToolRuntime) -> None:
    st.markdown("#### 结构化故障门禁")
    failure = payload["inference"]
    downstream = payload["downstream"]
    st.error(
        f"{failure['code']} · {failure['message']}\n\n"
        f"当前资格：{failure.get('qualification') or failure['data'].get('qualification', '未记录')}；"
        "要求：online_case_inference_ready。"
    )
    st.info(
        f"后续风险审计状态：{downstream['code']}。"
        "上游失败后没有继续执行越权调用。"
    )
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
    scenario = st.selectbox(
        "审阅场景",
        [*NORMAL_SCENARIOS, FAULT_SCENARIO],
        help="APTOS 与青光眼场景只读取冻结 validation 概率；故障场景验证资格门禁。",
    )
    if scenario == FAULT_SCENARIO:
        scenario_config = NORMAL_SCENARIOS["APTOS · 冻结 validation 资产"]
    else:
        scenario_config = NORMAL_SCENARIOS[scenario]
    task_id = str(scenario_config["task_id"])
    cases = build_review_cases(PROJECT_ROOT, task_id)
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
    pipeline_key = f"review_pipeline::{scenario}::{case.alias}"
    if pipeline_key not in st.session_state:
        if scenario == FAULT_SCENARIO:
            payload, runtime = _run_fault_pipeline(
                context,
                case,
                artifact_id="flair",
            )
        else:
            payload, runtime = _run_read_only_pipeline(
                context,
                case,
                list(scenario_config["artifact_ids"]),
            )
        st.session_state[pipeline_key] = (payload, runtime)
    payload, runtime = st.session_state[pipeline_key]

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

    if scenario == FAULT_SCENARIO:
        _render_fault(payload, runtime)
        return

    result_tab, route_tab, review_tab, trace_tab = st.tabs(
        ["模型结果", "路由与错误风险", "人工审阅", "资产与调用轨迹"]
    )
    with result_tab:
        _render_predictions(task_id, payload)
        _render_model_table(payload)
    with route_tab:
        _render_audit_and_route(task_id, payload)
    with review_tab:
        _render_review_actions(
            case=case,
            cases=cases,
            scenario=scenario,
            selected_index=selected_index,
            state=state,
            payload=payload,
            runtime=runtime,
            session_id=session_id,
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
