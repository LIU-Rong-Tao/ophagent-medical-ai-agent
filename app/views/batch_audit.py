"""批量复核排序页。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.audit_core import translate_risk_reasons
from app.checkpoints import (
    ModelArtifact,
    discover_model_artifacts,
    select_preferred_artifacts,
)
from app.plots import plot_review_budget_curve
from app.views.case_detail import (
    build_pre_review_case,
    filter_case_queue,
    initialize_review_capacity,
    index_prediction_records,
    normalize_image_key,
    paginate_case_queue,
    render_case_detail_dialog,
    select_review_capacity,
)
from app.ui import (
    metric_card,
    page_header,
    render_boundary,
    render_case_card,
    render_empty_state,
    render_source_caption,
    section_header,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRADEOFF_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "summary"
    / "v0_6_7c"
    / "unified_ranking_method_tradeoff.csv"
)
EXACT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "summary"
    / "v0_6_7"
    / "v067_showcase_exact_numbers.csv"
)
RISK_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "summary"
    / "v0_6_6"
    / "full_test_backbones"
)
CLINICAL_CASES_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "summary"
    / "v0_6_7"
    / "clinical_event_cases.csv"
)

EVENT_NAMES = {
    "general_error": "通用错分",
    "large_undergrading": "跨两级低估",
    "referable_dr_miss": "可转诊等级漏检代理",
    "vision_threatening_dr_miss": "VTDR miss 等级代理",
    "high_confidence_vtdr_miss": "高置信 VTDR miss 代理",
}

METHOD_NAMES = {
    "gated_severe_prob_mass_only": "门控重症概率质量",
    "expected_gap_only": "期望等级差",
    "ophagent_combined": "早期 combined 规则",
    "confidence_only": "1-MSP",
    "entropy_only": "归一化熵",
    "margin_only": "Top1-Top2 间隔",
    "uncertainty_rank_fusion": "不确定性排序融合",
}

DR_GRADE_NAMES = {
    "No DR": "未见 DR",
    "Mild DR": "轻度 DR",
    "Moderate DR": "中度 DR",
    "Severe DR": "重度 DR",
    "Proliferative DR": "增殖期 DR",
}


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_artifact_registry() -> dict[str, ModelArtifact]:
    return select_preferred_artifacts(discover_model_artifacts(PROJECT_ROOT))


def risk_table_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if RISK_ROOT.exists():
        for path in RISK_ROOT.glob("*/pre_review_risk_table.csv"):
            paths[path.parent.name] = path
    return paths


def _posthoc_for_case(
    *,
    backbone: str,
    normalized_image_key: str,
) -> tuple[dict | None, pd.DataFrame | None]:
    if (
        st.session_state.get("display_mode") != "研究审计"
        or not CLINICAL_CASES_PATH.exists()
    ):
        return None, None
    frame = load_csv(str(CLINICAL_CASES_PATH)).copy()
    frame["_normalized_image_key"] = frame["image_key"].map(normalize_image_key)
    match = frame[
        (frame["backbone"].astype(str) == backbone)
        & (frame["_normalized_image_key"] == normalized_image_key)
    ]
    if len(match) > 1:
        raise ValueError(
            f"后验病例表存在重复连接键：({backbone}, {normalized_image_key})"
        )
    same_image = frame[frame["_normalized_image_key"] == normalized_image_key]
    peer_columns = [
        "backbone",
        "pred_label",
        "top2_label",
        "confidence",
        "margin",
        "severe_prob_mass",
    ]
    peer_columns = [column for column in peer_columns if column in same_image.columns]
    posthoc = match.iloc[0].to_dict() if len(match) == 1 else None
    return posthoc, same_image[peer_columns].copy()


def render_pre_review_queue() -> None:
    section_header(
        "预审队列：此处不读取真实标签",
        "队列来自 v0.6.6 的模型输出后风险表，用于演示医生在标签未知时能看到什么。",
    )
    paths = risk_table_paths()
    if not paths:
        render_empty_state("没有病例级风险表", "改为在“后验验证”页查看聚合 tradeoff 结果。")
        return

    registry = load_artifact_registry()
    model_keys = [key for key in registry if key in paths]
    if not model_keys:
        render_empty_state(
            "模型产物与预审风险表未对齐",
            "不会回退到 ConvNeXt，也不会使用其他模型的队列。",
        )
        return
    if st.session_state.get("selected_backbone") not in model_keys:
        st.session_state["selected_backbone"] = model_keys[0]
    backbone = st.selectbox(
        "当前模型",
        model_keys,
        key="selected_backbone",
        format_func=lambda key: registry[key].display_name,
    )
    artifact = registry[backbone]
    st.caption(
        f"当前队列：{artifact.display_name} · {artifact.experiment_dir.name}。"
        "所有排序仅在该 backbone 内部进行。"
    )
    frame = load_csv(str(paths[backbone]))
    display_columns = [
        "case_id",
        "image_path",
        "pred_grade",
        "pred_label",
        "confidence",
        "top2_grade",
        "top2_label",
        "top2_confidence",
        "margin",
        "entropy_norm",
        "severe_prob_mass",
        "risk_reasons",
        "pre_review_risk_level",
        "review_priority_rank",
    ]
    display_columns = [column for column in display_columns if column in frame.columns]
    queue = frame.sort_values("review_priority_rank")[display_columns].copy()
    prediction_index: dict[tuple[str, str], dict] = {}
    if artifact.test_predictions_path is not None:
        try:
            prediction_index = index_prediction_records(
                load_csv(str(artifact.test_predictions_path)),
                backbone=backbone,
            )
        except ValueError as exc:
            st.error(f"当前模型 prediction records 无法建立唯一病例索引：{exc}")
    level_counts = queue["pre_review_risk_level"].value_counts()
    count_cols = st.columns(3, gap="small")
    with count_cols[0]:
        metric_card(
            "高优先级",
            f"{int(level_counts.get('high', 0))}",
            "进入首批模型结果复核",
            accent="red",
        )
    with count_cols[1]:
        metric_card(
            "中优先级",
            f"{int(level_counts.get('medium', 0))}",
            "建议在常规队列中提前查看",
            accent="amber",
        )
    with count_cols[2]:
        metric_card(
            "常规队列",
            f"{int(level_counts.get('low', 0))}",
            "排序较后，不表示安全放行",
            accent="teal",
        )

    section_header(
        "病例复核队列",
        "先筛选候选池，再模拟科室当日复核容量；点击病例卡下方按钮打开详情。",
    )
    filter_col, search_col = st.columns([0.35, 0.65], gap="large")
    with filter_col:
        priority_filter = st.selectbox(
            "优先级筛选",
            ["全部", "优先", "关注", "常规"],
            key=f"priority_filter_{backbone}",
        )
    with search_col:
        case_search = st.text_input(
            "病例编号搜索",
            key=f"case_search_{backbone}",
            placeholder="输入完整或部分病例编号",
        )
    filtered = filter_case_queue(
        queue,
        priority=priority_filter,
        search=case_search,
    )
    if filtered.empty:
        render_empty_state("没有匹配病例", "请调整优先级筛选或病例编号。")
        return

    capacity_key = f"review_capacity_{backbone}"
    initialize_review_capacity(
        st.session_state,
        capacity_key,
        pool_size=len(filtered),
    )

    with st.container(border=True):
        st.markdown("#### 科室复核容量模拟")
        st.caption(
            "模拟当前科室本轮最多能复核多少例。默认按风险顺序取 Top N；"
            "随机抽样仅作为同等工作量下的展示对照。"
        )
        capacity_col, method_col, seed_col = st.columns(
            [0.28, 0.46, 0.26],
            gap="large",
        )
        with capacity_col:
            review_capacity = int(
                st.number_input(
                    "复核容量 N",
                    min_value=1,
                    max_value=len(filtered),
                    step=1,
                    key=capacity_key,
                    help="容量作用于当前优先级筛选和病例搜索得到的候选池。",
                )
            )
        with method_col:
            selection_method = st.radio(
                "病例选取方式",
                ["风险 Top N", "随机抽 N"],
                horizontal=True,
                key=f"review_selection_method_{backbone}",
                help="风险 Top N 使用当前模型内部排序；随机抽 N 不改变模型推理。",
            )
        with seed_col:
            random_seed = int(
                st.number_input(
                    "随机种子",
                    min_value=0,
                    max_value=2_147_483_647,
                    value=42,
                    step=1,
                    disabled=selection_method != "随机抽 N",
                    key=f"review_random_seed_{backbone}",
                    help="固定随机种子可保证演示时重复得到同一批对照病例。",
                )
            )

        selected_pool = select_review_capacity(
            filtered,
            capacity=review_capacity,
            method=selection_method,
            random_seed=random_seed,
        )
        coverage = len(selected_pool) / len(filtered)
        method_explanation = (
            "按当前模型风险排序优先取前 N 例"
            if selection_method == "风险 Top N"
            else f"从候选池随机抽取 N 例（seed={random_seed}）"
        )
        st.markdown(
            "<div style='margin-top:.35rem;padding:.72rem .9rem;"
            "border-left:4px solid #0F8A83;background:#F3F8F8;"
            "color:#243447;border-radius:4px'>"
            f"<strong>当前模拟复核 {len(selected_pool)} / {len(filtered)} 例"
            f"（{coverage:.1%}）</strong><br>"
            f"<span style='color:#617080'>{method_explanation}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    page_key = f"queue_page_{backbone}"
    selection_signature = (
        priority_filter,
        case_search,
        review_capacity,
        selection_method,
        random_seed if selection_method == "随机抽 N" else None,
    )
    signature_key = f"queue_selection_signature_{backbone}"
    if st.session_state.get(signature_key) != selection_signature:
        st.session_state[page_key] = 1
        st.session_state[signature_key] = selection_signature
    page_number = int(st.session_state.get(page_key, 1))
    cards, total_pages, page_number = paginate_case_queue(
        selected_pool,
        page_number=page_number,
        page_size=12,
    )
    st.session_state[page_key] = page_number
    nav_left, nav_center, nav_right = st.columns([0.2, 0.6, 0.2])
    with nav_left:
        if st.button(
            "上一页",
            disabled=page_number <= 1,
            key=f"queue_prev_{backbone}",
            use_container_width=True,
        ):
            st.session_state[page_key] = page_number - 1
            st.rerun()
    with nav_center:
        st.markdown(
            f"<div style='text-align:center;color:#5B6878;padding:.55rem'>"
            f"第 {page_number} / {total_pages} 页 · "
            f"本次模拟复核 {len(selected_pool)} 条"
            "</div>",
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button(
            "下一页",
            disabled=page_number >= total_pages,
            key=f"queue_next_{backbone}",
            use_container_width=True,
        ):
            st.session_state[page_key] = page_number + 1
            st.rerun()
    card_columns = st.columns(2, gap="large")
    for index, (_, row) in enumerate(cards.iterrows()):
        raw_level = str(row.get("pre_review_risk_level", "medium"))
        priority_level = {
            "high": "high",
            "medium": "medium",
            "low": "routine",
        }.get(raw_level, "medium")
        priority_label = {
            "high": "优先复核",
            "medium": "建议关注",
            "low": "常规队列",
        }.get(raw_level, "建议关注")
        reasons = translate_risk_reasons(row.get("risk_reasons"))
        prediction_raw = str(row.get("pred_label", "未记录"))
        top2_raw = str(row.get("top2_label", "未记录"))
        prediction = DR_GRADE_NAMES.get(prediction_raw, prediction_raw)
        top2 = DR_GRADE_NAMES.get(top2_raw, top2_raw)
        top2_probability = float(row.get("top2_confidence", 0.0))
        severe_mass = float(row.get("severe_prob_mass", 0.0))
        summary = (
            f"模型预测为 {prediction}；第二候选为 {top2}（{top2_probability:.1%}）。"
            f"重症类别概率质量为 {severe_mass:.1%}。"
        )
        action = {
            "high": "建议进入首批模型结果复核队列",
            "medium": "建议在常规队列中提前查看",
            "low": "按常规流程复核，不代表无需查看",
        }.get(raw_level, "建议结合图像质量进一步复核")
        normalized_key = normalize_image_key(
            row.get("image_path") or row.get("case_id")
        )
        prediction_record = prediction_index.get((backbone, normalized_key), {})
        merged = {**row.to_dict(), **prediction_record}
        case = build_pre_review_case(merged, backbone=backbone)
        with card_columns[index % 2]:
            render_case_card(
                case_id=str(row.get("case_id", "")),
                priority_level=priority_level,
                priority_label=priority_label,
                prediction=f"模型结果：{prediction}",
                summary=summary,
                action=action,
                reasons=reasons,
            )
            if st.button(
                "查看复核详情",
                key=(
                    f"case_detail_{backbone}_"
                    f"{case['normalized_image_key']}_{page_number}_{index}"
                ),
                use_container_width=True,
            ):
                try:
                    posthoc, peers = _posthoc_for_case(
                        backbone=backbone,
                        normalized_image_key=case["normalized_image_key"],
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    render_case_detail_dialog(
                        case,
                        display_mode=st.session_state.get(
                            "display_mode",
                            "临床展示",
                        ),
                        posthoc=posthoc,
                        peer_predictions=peers,
                    )

    rename = {
        "case_id": "病例记录",
        "pred_label": "预测等级",
        "confidence": "置信度",
        "top2_label": "第二候选",
        "top2_confidence": "第二候选概率",
        "margin": "Top1-Top2 间隔",
        "entropy_norm": "归一化熵",
        "severe_prob_mass": "重症概率质量",
        "risk_reasons": "排序原因",
        "pre_review_risk_level": "风险层级",
        "review_priority_rank": "复核顺序",
    }
    render_boundary(
        "这里没有 true label、correct 或 event 字段。病例卡只表达模型结果的复核优先级，"
        "不是临床风险分级，也不能替代医生判断。"
    )

    if st.session_state.get("display_mode") == "研究审计":
        with st.expander("查看完整技术字段", expanded=False):
            technical = queue.head(50).copy()
            technical["risk_reasons"] = technical["risk_reasons"].map(
                lambda value: "；".join(translate_risk_reasons(value))
            )
            technical["pre_review_risk_level"] = technical[
                "pre_review_risk_level"
            ].map({"high": "高", "medium": "中", "low": "常规"})
            st.dataframe(
                technical.rename(columns=rename),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "置信度": st.column_config.NumberColumn(format="%.3f"),
                    "第二候选概率": st.column_config.NumberColumn(format="%.3f"),
                    "Top1-Top2 间隔": st.column_config.NumberColumn(format="%.3f"),
                    "归一化熵": st.column_config.NumberColumn(format="%.3f"),
                    "重症概率质量": st.column_config.NumberColumn(format="%.3f"),
                },
            )
        render_source_caption(paths[backbone].relative_to(PROJECT_ROOT))
        st.download_button(
            "下载当前完整预审风险表",
            frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{backbone}_pre_review_risk_table.csv",
            mime="text/csv",
        )


def render_posthoc_validation() -> None:
    section_header(
        "后验验证：真实标签只在这里用于检查排序收益",
        "选择危险事件和排序方法，查看不同复核预算下的捕获、富集与残余事件。",
    )
    if not TRADEOFF_PATH.exists():
        render_empty_state("缺少 v0.6.7c 结果", str(TRADEOFF_PATH))
        return

    frame = load_csv(str(TRADEOFF_PATH))
    registry = load_artifact_registry()
    current_backbone = st.session_state.get("selected_backbone")
    if current_backbone not in set(frame["backbone"].astype(str)):
        current_backbone = next(iter(registry), None)
    scope_options = ["当前模型"]
    if st.session_state.get("display_mode") == "研究审计":
        scope_options.append("六模型汇总")
    evidence_scope = st.selectbox(
        "结果范围",
        scope_options,
        key="posthoc_evidence_scope",
    )
    if evidence_scope == "当前模型":
        if current_backbone is None:
            render_empty_state("没有当前模型", "请先在预审队列选择模型。")
            return
        frame = frame[frame["backbone"].astype(str) == current_backbone].copy()
        if frame.empty:
            render_empty_state(
                "当前模型暂无后验验证产物",
                "不会回退到 ConvNeXt 或其他 backbone。",
            )
            return
        display_name = (
            registry[current_backbone].display_name
            if current_backbone in registry
            else current_backbone
        )
        st.caption(f"当前模型：{display_name}")
    else:
        st.caption("当前范围：六模型冻结内部研究汇总，不代表某个即时 checkpoint。")
    events = frame["clinical_event"].dropna().astype(str).unique().tolist()
    default_event = (
        "vision_threatening_dr_miss"
        if "vision_threatening_dr_miss" in events
        else events[0]
    )
    event = st.selectbox(
        "危险事件",
        events,
        index=events.index(default_event),
        format_func=lambda value: EVENT_NAMES.get(value, value),
    )
    event_frame = frame[frame["clinical_event"] == event].copy()
    methods = event_frame["ranking_method"].dropna().astype(str).unique().tolist()
    preferred = (
        "gated_severe_prob_mass_only"
        if event == "vision_threatening_dr_miss"
        else "expected_gap_only"
    )
    default_methods = [
        method
        for method in [preferred, "ophagent_combined", "confidence_only", "entropy_only"]
        if method in methods
    ]
    default_limit = 2 if st.session_state.get("display_mode") == "临床展示" else 4
    selected_methods = st.multiselect(
        "比较方法",
        methods,
        default=default_methods[:default_limit] or methods[:2],
        format_func=lambda value: METHOD_NAMES.get(value, value),
    )
    if not selected_methods:
        render_empty_state("尚未选择方法", "至少选择一个排序方法。")
        return

    chart_frame = (
        event_frame[event_frame["ranking_method"].isin(selected_methods)]
        .groupby(["ranking_method", "review_budget"], as_index=False)[
            "dangerous_error_recall_at_k"
        ]
        .mean()
        .rename(
            columns={
                "ranking_method": "method",
                "review_budget": "budget",
                "dangerous_error_recall_at_k": "event_recall",
            }
        )
    )
    st.pyplot(
        plot_review_budget_curve(
            chart_frame,
            highlight_method=preferred,
            title="Review budget vs event recall",
        ),
        use_container_width=True,
    )

    method = st.selectbox(
        "核心方法摘要",
        selected_methods,
        index=selected_methods.index(preferred) if preferred in selected_methods else 0,
        format_func=lambda value: METHOD_NAMES.get(value, value),
    )
    budget = st.select_slider(
        "固定复核预算",
        options=sorted(event_frame["review_budget"].unique().tolist()),
        value=0.2 if 0.2 in set(event_frame["review_budget"]) else 0.1,
        format_func=lambda value: f"{value:.0%}",
    )
    point = event_frame[
        (event_frame["ranking_method"] == method)
        & (event_frame["review_budget"] == budget)
    ]
    if not point.empty:
        recall = point["dangerous_error_recall_at_k"].mean()
        precision = point["dangerous_error_precision_at_k"].mean()
        lift = point["dangerous_error_lift_vs_random"].mean()
        residual = point["residual_dangerous_error_count"].sum()
        total = point["dangerous_error_total"].sum()
        captured = point["dangerous_error_captured"].sum()
        cols = st.columns(4, gap="small")
        aggregation_note = (
            "当前模型记录"
            if evidence_scope == "当前模型"
            else "六个 backbone 记录汇总"
        )
        with cols[0]:
            metric_card("事件召回率", f"{recall:.1%}", f"固定复核 {budget:.0%}")
        with cols[1]:
            metric_card(
                "捕获 / 总数",
                f"{captured:.0f} / {total:.0f}",
                aggregation_note,
            )
        with cols[2]:
            metric_card("相对随机富集", f"{lift:.2f}×", f"复核队列精确率 {precision:.1%}", accent="amber")
        with cols[3]:
            metric_card("残余事件记录", f"{residual:.0f}", "未进入优先复核区", accent="red")

    detail_columns = [
        "backbone",
        "ranking_method",
        "clinical_event",
        "review_budget",
        "reviewed_n",
        "dangerous_error_total",
        "dangerous_error_captured",
        "dangerous_error_recall_at_k",
        "dangerous_error_precision_at_k",
        "dangerous_error_lift_vs_random",
        "residual_dangerous_error_count",
        "residual_dangerous_error_rate",
        "dangerous_errors_per_100_reviewed",
        "number_needed_to_review",
    ]
    detail = event_frame[
        (event_frame["ranking_method"] == method)
        & (event_frame["review_budget"] == budget)
    ][detail_columns]
    if st.session_state.get("display_mode") == "研究审计":
        with st.expander("查看各骨干模型后验结果", expanded=False):
            st.dataframe(
                detail.rename(
                    columns={
                        "backbone": "骨干模型",
                        "ranking_method": "排序方法",
                        "clinical_event": "目标事件",
                        "review_budget": "复核预算",
                        "reviewed_n": "复核记录数",
                        "dangerous_error_total": "目标事件总数",
                        "dangerous_error_captured": "捕获事件数",
                        "dangerous_error_recall_at_k": "事件召回率",
                        "dangerous_error_precision_at_k": "复核队列精确率",
                        "dangerous_error_lift_vs_random": "相对随机富集",
                        "residual_dangerous_error_count": "残余事件数",
                        "residual_dangerous_error_rate": "残余事件率",
                        "dangerous_errors_per_100_reviewed": "每百条复核捕获",
                        "number_needed_to_review": "每捕获一例需复核数",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        render_source_caption(TRADEOFF_PATH.relative_to(PROJECT_ROOT))
    render_boundary(
        "内部 APTOS 结果用于发现和解释排序信号。它不能替代独立外部验证；"
        "v0.7.1b 页面随后检查核心信号是否超过 random gate-only。"
    )


def render() -> None:
    page_header(
        "批量复核排序",
        "同一套输出信号分成两个视角：标签未知时生成队列；标签可用后再检验捕获率与残余风险。",
        "v0.6.6–v0.6.7c 内部证据",
    )
    pre_tab, post_tab = st.tabs(["预审队列", "后验验证"])
    with pre_tab:
        render_pre_review_queue()
    with post_tab:
        render_posthoc_validation()
