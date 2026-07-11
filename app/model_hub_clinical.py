"""病例回放与路由解释：隔离模型输出视图和标签依赖研究审计。"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.model_hub_ui import grade_label, human_model


def paginate_cases(frame: pd.DataFrame, page: int, page_size: int) -> tuple[pd.DataFrame, int]:
    if page_size <= 0:
        raise ValueError("page_size 必须大于 0")
    total_pages = max(1, math.ceil(len(frame) / page_size))
    page = min(max(1, int(page)), total_pages)
    start = (page - 1) * page_size
    return frame.iloc[start : start + page_size].copy(), total_pages


def filter_case_view(
    frame: pd.DataFrame,
    filters: list[str],
    *,
    research_mode: bool,
) -> pd.DataFrame:
    filtered = frame.copy()
    if "已调用专家" in filters and "is_reviewed_by_expert" in filtered.columns:
        filtered = filtered.loc[filtered["is_reviewed_by_expert"].astype(bool)]
    if "模型分歧" in filters and "scout_disagreement" in filtered.columns:
        filtered = filtered.loc[filtered["scout_disagreement"].astype(bool)]
    if (
        research_mode
        and "与参考标签不一致" in filters
        and "was_final_correct" in filtered.columns
    ):
        filtered = filtered.loc[filtered["was_final_correct"].eq(False)]
    return filtered.copy()


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _plain_reason(row: pd.Series) -> str:
    if not bool(row.get("is_reviewed_by_expert", False)):
        return "该病例未进入当前策略的专家调用额度，结果保留为路由模型输出。"
    if bool(row.get("scout_disagreement", False)):
        return "路由模型之间存在分歧，本次策略将它交给专家模型接管。"
    return "路由模型的不确定性较高，本次策略将它交给专家模型接管。"


def _case_summary_html(cases: pd.DataFrame, *, research_mode: bool) -> str:
    total = len(cases)
    reviewed = int(cases.get("is_reviewed_by_expert", pd.Series(dtype=bool)).astype(bool).sum())
    disagreement = int(cases.get("scout_disagreement", pd.Series(dtype=bool)).astype(bool).sum())
    values = [
        ("当前回放病例", f"{total:,}", "当前组合输出"),
        ("送专家调用", f"{reviewed:,}", f"{reviewed / total:.0%}" if total else "0%"),
        ("路由模型分歧", f"{disagreement:,}", f"{disagreement / total:.0%}" if total else "0%"),
    ]
    if research_mode and "was_final_correct" in cases.columns:
        mismatches = int(cases["was_final_correct"].eq(False).sum())
        values.append(("研究审计不一致", f"{mismatches:,}", f"{mismatches / total:.0%}" if total else "0%"))
    else:
        adopted_by_expert = int(
            cases.get("final_source", pd.Series(dtype=object)).astype(str).eq("expert").sum()
            if "final_source" in cases.columns
            else reviewed
        )
        values.append(("系统采用专家输出", f"{adopted_by_expert:,}", "模型轨迹"))
    cards = "".join(
        '<div class="hub-mini-stat">'
        f'<span>{html.escape(label)}</span><b>{html.escape(value)}</b><small>{html.escape(note)}</small>'
        "</div>"
        for label, value, note in values
    )
    return f'<div class="hub-mini-strip">{cards}</div>'


@st.dialog("病例路由解释", width="large")
def _case_dialog(
    row: pd.Series,
    task_id: str,
    label: str,
    *,
    research_mode: bool = False,
) -> None:
    st.markdown(f"### 病例 {html.escape(str(row.get('image_key')))}")
    st.caption(label)
    image_path = Path(str(row.get("image_path", "")))
    image_col, result_col = st.columns([0.95, 1.05], gap="large")
    with image_col:
        if image_path.is_file():
            st.image(str(image_path), width="stretch")
        else:
            st.info("服务器图像未找到，当前仅展示模型输出。")
    with result_col:
        st.markdown("#### 模型结果对照")
        route_result = grade_label(task_id, row.get("primary_scout_pred_label"))
        expert_result = grade_label(task_id, row.get("expert_pred_label")) if bool(row.get("is_reviewed_by_expert", False)) else "未调用"
        final_result = grade_label(task_id, row.get("final_pred_label"))
        agreement = (
            "结果一致"
            if bool(row.get("is_reviewed_by_expert", False))
            and row.get("primary_scout_pred_label") == row.get("expert_pred_label")
            else "结果不一致" if bool(row.get("is_reviewed_by_expert", False)) else "未调用专家"
        )
        st.markdown(
            '<div class="case-result-grid">'
            f'<div class="case-result-card"><small>路由模型结果</small><b>{html.escape(route_result)}</b><span>默认模型输出</span></div>'
            f'<div class="case-result-card"><small>专家模型结果</small><b>{html.escape(expert_result)}</b><span>{"已调用专家" if bool(row.get("is_reviewed_by_expert", False)) else "未进入专家调用额度"}</span></div>'
            f'<div class="case-result-card"><small>系统采用输出</small><b>{html.escape(final_result)}</b><span>{html.escape(agreement)}</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### 路由依据")
        st.write(_plain_reason(row))
        st.warning("本页只展示模型调用轨迹；结果是否可靠需要独立标注或人工复核确认。")
    with st.expander("查看模型概率与路由依据"):
        score = row.get("routing_score")
        cutoff = row.get("routing_cutoff")
        if pd.notna(score):
            st.write(f"路由分数：{float(score):.3f}")
        if pd.notna(cutoff):
            st.write(f"本次专家调用分界值：{float(cutoff):.3f}")
        route_probabilities = _json_dict(row.get("scout_probabilities"))
        expert_probabilities = _json_dict(row.get("expert_probabilities"))
        probability_rows = []
        for model_id, values in {**route_probabilities, **expert_probabilities}.items():
            for class_index, probability in enumerate(values):
                probability_rows.append(
                    {
                        "模型": human_model(model_id),
                        "类别": grade_label(task_id, class_index),
                        "概率": float(probability),
                    }
                )
        if probability_rows:
            st.dataframe(
                pd.DataFrame(probability_rows).style.format({"概率": "{:.1%}"}),
                hide_index=True,
                width="stretch",
            )
    if research_mode and "true_label" in row.index:
        st.markdown("#### 回顾性研究审计")
        st.caption("以下信息依赖公开测试标签或冻结事件定义，不参与在线路由，不提供诊断或患者分流决定。")
        audit_columns = st.columns(2)
        audit_columns[0].metric("参考标签", grade_label(task_id, row.get("true_label")))
        audit_columns[1].metric(
            "与参考标签一致",
            "是" if bool(row.get("was_final_correct", False)) else "否",
        )
        event_labels = {
            "dr_large_undergrading_final_residual": "大跨度低估研究审计代理事件",
            "dr_referable_miss_final_residual": "可转诊漏检研究审计代理事件",
            "dr_severe_pdr_miss_final_residual": "重症漏检研究审计代理事件",
        }
        residual_events = [event for column, event in event_labels.items() if bool(row.get(column, False))]
        if residual_events:
            st.warning("研究审计代理事件：" + "、".join(residual_events))
        else:
            st.info("当前病例未命中已登记的冻结风险代理事件。")


def render_clinical_workspace(data: dict[str, object]) -> None:
    st.subheader("病例回放与路由解释")
    display_mode = st.segmented_control(
        "展示层",
        ["模型输出回放", "研究审计"],
        default="模型输出回放",
        help="模型输出回放不读取真实标签；研究审计用于公开测试集回顾性分析。",
    )
    research_mode = display_mode == "研究审计"
    st.caption(
        "研究审计会显示公开测试标签和研究审计代理事件，不参与在线路由，不提供诊断或患者分流决定。"
        if research_mode
        else "仅展示推理时可获得的图像、模型结果、模型分歧和专家调用原因。"
    )
    cases = st.session_state.get(
        "model_hub_last_research_cases" if research_mode else "model_hub_last_cases"
    )
    metrics = st.session_state.get("model_hub_last_metrics")
    label = st.session_state.get("model_hub_last_label", "当前组合")
    if cases is None or not isinstance(cases, pd.DataFrame) or cases.empty:
        st.info("请先在“模型工程 → 研究评测”中运行一个组合。")
        return
    task_id = str(metrics.get("task_id", ""))
    st.markdown(_case_summary_html(cases, research_mode=research_mode), unsafe_allow_html=True)
    st.markdown(
        '<div class="case-list-note">'
        f'<strong>当前组合：</strong>{html.escape(str(label))}。'
        + (
            "本视图显示公开测试标签与研究审计代理事件，仅用于回顾性分析。"
            if research_mode
            else "本视图只展示推理时可获得的模型输出、路由依据和专家调用轨迹。"
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    view = cases.copy()
    filter_options = ["已调用专家", "模型分歧"]
    if research_mode and "was_final_correct" in view.columns:
        filter_options.append("与参考标签不一致")
    filters = st.multiselect(
        "病例筛选",
        filter_options,
        default=[],
    )
    view = filter_case_view(view, filters, research_mode=research_mode)
    search = st.text_input("病例编号搜索", placeholder="输入完整或部分病例编号")
    if search.strip():
        view = view.loc[view["image_key"].astype(str).str.contains(search.strip(), case=False, regex=False)]
    page_size = int(st.selectbox("每页病例数", [25, 50, 100], index=1, key="clinical_page_size"))
    page_key = "clinical_page"
    current_page = int(st.session_state.get(page_key, 1))
    table, total_pages = paginate_cases(view, current_page, page_size)
    current_page = min(max(1, current_page), total_pages)
    st.session_state[page_key] = current_page
    previous_col, status_col, next_col = st.columns([1, 2, 1])
    with previous_col:
        if st.button("上一页", disabled=current_page <= 1, width="stretch"):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with status_col:
        st.markdown(
            f"<div style='text-align:center;padding:.45rem'>第 {current_page}/{total_pages} 页 · 共 {len(view)} 例</div>",
            unsafe_allow_html=True,
        )
    with next_col:
        if st.button("下一页", disabled=current_page >= total_pages, width="stretch"):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    table["路由模型结果"] = table["primary_scout_pred_label"].map(lambda value: grade_label(task_id, value))
    table["调用专家"] = table["is_reviewed_by_expert"].map({True: "是", False: "否"})
    table["专家结果"] = table.apply(
        lambda row: grade_label(task_id, row["expert_pred_label"]) if bool(row["is_reviewed_by_expert"]) else "—",
        axis=1,
    )
    table["系统采用输出"] = table["final_pred_label"].map(lambda value: grade_label(task_id, value))
    table["调用原因"] = table.apply(_plain_reason, axis=1)
    display_columns = ["image_key", "routing_score", "路由模型结果", "调用专家", "专家结果", "系统采用输出", "调用原因"]
    rename_columns = {"image_key": "病例编号", "routing_score": "路由分数"}
    if research_mode and "true_label" in table.columns:
        table["参考标签"] = table["true_label"].map(lambda value: grade_label(task_id, value))
        table["与参考标签一致"] = table["was_final_correct"].map({True: "是", False: "否"})
        display_columns.extend(["参考标签", "与参考标签一致"])
    display_table = table[display_columns].rename(columns=rename_columns)
    if research_mode and "与参考标签一致" in display_table.columns:
        def highlight_reference_mismatch(row: pd.Series) -> list[str]:
            if row.get("与参考标签一致") != "否":
                return [""] * len(row)
            return [
                "background-color:#fee4e2;color:#9f1c16;font-weight:700"
                if column == "与参考标签一致"
                else "background-color:#fff4f2"
                for column in row.index
            ]

        displayed: pd.DataFrame | pd.io.formats.style.Styler = display_table.style.apply(
            highlight_reference_mismatch,
            axis=1,
        )
    else:
        displayed = display_table
    st.dataframe(
        displayed,
        hide_index=True,
        width="stretch",
    )
    if table.empty:
        return
    selected = st.selectbox("选择病例", table["image_key"].astype(str).tolist())
    if st.button("查看病例路由解释", icon=":material/open_in_new:"):
        _case_dialog(
            table.loc[table["image_key"].astype(str).eq(selected)].iloc[0],
            task_id,
            label,
            research_mode=research_mode,
        )
