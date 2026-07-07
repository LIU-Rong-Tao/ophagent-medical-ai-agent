"""模型工程工作区中的研究评测层。"""

from __future__ import annotations

import altair as alt
import html
import pandas as pd
from pathlib import Path
import streamlit as st

from app.model_hub_inference_jobs import (
    checkpoint_inference_capability,
    latest_inference_job,
    submit_checkpoint_inference,
)
from app.model_hub_data import (
    DR_RISK_EVENTS,
    available_routing_policies,
    build_online_case_view,
    enrich_cost_curve,
    estimate_global_composition_count,
    evaluate_exploratory_composition,
    scan_global_composition_candidates,
    select_operating_points,
    split_task_models,
    task_metric_profile,
    task_evaluation_summary,
)
from app.model_hub_scan_jobs import (
    latest_completed_global_scan,
    load_global_scan_results,
    submit_global_scan_job,
)
from app.model_hub_ui import grade_label, human_model, task_label


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_REGISTRY = PROJECT_ROOT / "experiments/v0_8_5_model_registry_scout_expert_protocol/configs/task_registry.csv"
MAX_INTERACTIVE_GLOBAL_SCAN_EVALUATIONS = 1200


POLICY_LABELS = {
    "low_confidence": "低置信度优先",
    "low_margin": "Top1/Top2 间隔小优先",
    "high_entropy": "高熵优先",
    "disagreement_then_uncertainty": "路由模型分歧优先，其次不确定性",
    "mean_uncertainty": "路由模型平均不确定性",
    "dense_expert": "专家模型全量输出",
}

POLICY_HELP = {
    "low_confidence": "置信度就是 Top1 概率；路由分数 = 1 - Top1 概率，分数越高越优先调用专家。",
    "low_margin": "路由分数 = 1 -（Top1 概率 - Top2 概率），前两类越接近，分数越高。",
    "high_entropy": "路由分数 = 类别概率的归一化熵；概率越分散，分数越高。",
    "mean_uncertainty": "路由分数 = 多个路由模型的平均（1 - Top1 概率）。",
    "disagreement_then_uncertainty": "路由分数 = 2 × 是否分歧 + 平均不确定性；先排模型结论不一致的病例，再比较不确定性。",
    "dense_expert": "无路由模型门控，所选专家对全部病例输出。",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "qwk": "QWK",
}

HANDOFF_LABELS = {
    "fixed_expert": "固定专家接管",
    "mean_probability_pool": "专家池平均概率融合",
    "none": "未调用专家",
}

DR_PROXY_EVENT_HELP = {
    "大跨度低估": "公式：参考标签 >= 4 且默认输出 <= 2。公开测试标签为 4 级增殖期 DR，而默认输出为 0–2 级。",
    "可转诊漏检": "公式：参考标签 >= 2 且默认输出 <= 1。公开测试标签达到中度及以上，而默认输出为未见或轻度 DR。",
    "重症漏检": "公式：参考标签 >= 3 且默认输出 <= 2。公开测试标签达到重度或增殖期，而默认输出不高于中度 DR。",
}


def build_proxy_event_table_html(rows: pd.DataFrame) -> str:
    headers = ["事件", "总事件", "送专家", "纠正", "残余"]
    body = []
    cards = []
    for _, row in rows.iterrows():
        event = str(row.get("事件", ""))
        explanation = DR_PROXY_EVENT_HELP.get(event, "当前事件尚未登记解释。")
        total = int(pd.to_numeric(pd.Series([row.get("总事件", 0)]), errors="coerce").fillna(0).iloc[0])
        sent = int(pd.to_numeric(pd.Series([row.get("送专家", 0)]), errors="coerce").fillna(0).iloc[0])
        corrected = int(pd.to_numeric(pd.Series([row.get("纠正", 0)]), errors="coerce").fillna(0).iloc[0])
        residual = int(pd.to_numeric(pd.Series([row.get("残余", 0)]), errors="coerce").fillna(0).iloc[0])
        sent_rate = sent / total if total else 0
        corrected_rate = corrected / total if total else 0
        residual_rate = residual / total if total else 0
        cards.append(
            '<div class="proxy-event-card">'
            f'<div class="proxy-event-card-title">{html.escape(event)} '
            f'<span class="proxy-help-icon" tabindex="0" aria-label="{html.escape(explanation)}" '
            f'title="{html.escape(explanation)}">?</span></div>'
            '<div class="proxy-event-flow">'
            f'<span>总事件 <b>{total}</b></span>'
            f'<span>送专家 <b>{sent}</b><small>{sent_rate:.0%}</small></span>'
            f'<span>纠正 <b>{corrected}</b><small>{corrected_rate:.0%}</small></span>'
            f'<span class="proxy-residual">残余 <b>{residual}</b><small>{residual_rate:.0%}</small></span>'
            '</div></div>'
        )
        cells = [
            '<td class="proxy-event-name">'
            f'{html.escape(event)} <span class="proxy-help-icon" tabindex="0" '
            f'aria-label="{html.escape(explanation)}" title="{html.escape(explanation)}">?</span></td>'
        ]
        for column in headers[1:]:
            value = row.get(column, "")
            cells.append(f'<td class="proxy-event-value">{html.escape(str(value))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
    return (
        f'<div class="proxy-event-cards">{"".join(cards)}</div>'
        '<div class="proxy-table-wrap"><table class="proxy-table">'
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _human_model_list(value: object) -> str:
    values = [item for item in str(value or "").split("|") if item]
    return " + ".join(human_model(item) for item in values) or "无"


def build_comparison_table(runs: pd.DataFrame, display_metrics: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "组合": runs["组合"],
            "路由模型": runs["scout_ids"].map(_human_model_list),
            "默认输出模型": runs["primary_scout_id"].map(lambda value: human_model(value) if str(value or "") else "无"),
            "专家模型": runs["active_expert_ids"].map(_human_model_list),
            "专家接管方式": runs["expert_handoff_mode"].map(lambda value: HANDOFF_LABELS.get(str(value), str(value))),
            "路由机制": runs["routing_policy"].map(lambda value: POLICY_LABELS.get(str(value), str(value))),
        }
    )
    for metric in display_metrics:
        if metric in runs.columns:
            table[METRIC_LABELS.get(metric, metric)] = pd.to_numeric(runs[metric], errors="coerce")
    table["专家调用比例"] = pd.to_numeric(runs["realized_budget"], errors="coerce")
    table["入选病例数"] = pd.to_numeric(runs["selected_n"], errors="coerce").astype("Int64")
    table["估算前向成本（ms/图）"] = pd.to_numeric(
        runs["estimated_total_compute_ms_per_image"], errors="coerce"
    )
    return table


def _role_models(models: pd.DataFrame, role: str) -> pd.DataFrame:
    return models.loc[models["role_candidates"].fillna("").astype(str).str.contains(role, regex=False)]


def _render_evaluation_inputs(models: pd.DataFrame, config: dict[str, object]) -> None:
    task_id = str(config["task_id"])
    selected_ids = list(
        dict.fromkeys(
            [*config.get("route_ids", []), *config.get("expert_ids", [])]
        )
    )
    selected = models.loc[
        models["task_id"].astype(str).eq(task_id)
        & models["artifact_id"].astype(str).isin(selected_ids)
    ].copy()
    if selected.empty:
        return
    task_registry = pd.read_csv(TASK_REGISTRY)
    task_rows = task_registry.loc[task_registry["task_id"].astype(str).eq(task_id)]
    if task_rows.empty:
        return
    task = task_rows.iloc[0]
    source_labels = {
        "adapter": "真实 Adapter 已生成",
        "checkpoint_generated": "由已登记 checkpoint 生成",
        "legacy": "历史冻结 prediction",
        "missing": "缺少评测输出",
    }
    with st.expander("本次评测输入与权重来源", expanded=True):
        st.caption(
            "组合指标读取下表所列 prediction；checkpoint 只在重新生成任务中加载。"
            "同一 artifact 的 checkpoint 与 prediction 必须成对登记，不能单独替换。"
        )
        source_table = pd.DataFrame(
            {
                "模型": selected["artifact_id"].map(human_model),
                "评测输入": selected["prediction_source"].map(
                    lambda value: source_labels.get(str(value), str(value))
                ),
                "Checkpoint": selected["checkpoint_path"].fillna("未登记"),
                "Prediction": selected["prediction_path"].fillna("未登记"),
            }
        )
        st.dataframe(source_table, hide_index=True, use_container_width=True)
        if st.button("刷新生成状态", icon=":material/refresh:", key=f"refresh_inference_{task_id}"):
            st.rerun()
        for _, row in selected.iterrows():
            artifact_id = str(row["artifact_id"])
            supported, reason = checkpoint_inference_capability(row)
            latest = latest_inference_job(task_id, artifact_id)
            columns = st.columns([2.1, 2.4, 1.2])
            columns[0].write(f"**{human_model(artifact_id)}**")
            if latest:
                status_labels = {
                    "queued": "等待运行",
                    "running": "正在从 checkpoint 生成",
                    "succeeded": "生成完成，刷新后生效",
                    "failed": "生成失败",
                }
                status = str(latest.get("status", "unknown"))
                detail = status_labels.get(status, status)
                if latest.get("error_message"):
                    detail += f"：{latest['error_message']}"
                columns[1].caption(detail)
            else:
                columns[1].caption(reason)
            running = bool(latest and latest.get("status") in {"queued", "running"})
            if columns[2].button(
                "后台重新生成",
                key=f"generate_prediction_{task_id}_{artifact_id}",
                disabled=not supported or running,
                help=reason,
            ):
                try:
                    job_id = submit_checkpoint_inference(row, task, device="cuda:2")
                except Exception as exc:
                    st.error(f"提交失败：{exc}")
                else:
                    st.success(f"已提交：{job_id}。任务使用 cuda:2，完成后刷新页面。")


def _controls(models: pd.DataFrame) -> dict[str, object] | None:
    task_ids = sorted(models["task_id"].dropna().astype(str).unique())
    task_id = st.selectbox("任务", task_ids, format_func=task_label, key="research_task")
    metric_profile = task_metric_profile(pd.read_csv(TASK_REGISTRY), task_id)
    available, _ = split_task_models(models, task_id)
    if available.empty:
        st.warning("当前任务没有可用于组合评测的 prediction。")
        return None
    route_rows = _role_models(available, "scout")
    expert_rows = _role_models(available, "expert")
    route_options = route_rows["artifact_id"].astype(str).tolist()
    expert_options = expert_rows["artifact_id"].astype(str).tolist()
    default_route = route_options[:1]
    default_expert = [next((value for value in expert_options if "official_protocol" in value), expert_options[0])] if expert_options else []
    left, right = st.columns(2)
    with left:
        route_ids = st.pills(
            "选择路由模型（可多选，也可不选）",
            route_options,
            default=default_route,
            selection_mode="multi",
            format_func=human_model,
            key=f"route_models_{task_id}",
        ) or []
    with right:
        expert_ids = st.pills(
            "选择专家模型（可多选，也可不选）",
            expert_options,
            default=default_expert,
            selection_mode="multi",
            format_func=human_model,
            key=f"expert_models_{task_id}",
        ) or []
    if not route_ids and not expert_ids:
        st.warning("至少选择一个路由模型或专家模型。")
        return None
    primary_id = None
    if route_ids:
        primary_id = st.selectbox(
            "默认输出模型",
            route_ids,
            format_func=human_model,
            help=(
                "未调用专家的病例由它给出最终输出；单模型路由机制的分数也由它计算。"
                "多模型机制下，其他路由模型仅参与分歧或平均不确定性计算，不直接融合为最终分类结果。"
            ),
        )
    expert_handoff_mode = "none"
    fixed_expert_id = None
    if expert_ids:
        if len(expert_ids) == 1:
            expert_handoff_mode = "fixed_expert"
            fixed_expert_id = expert_ids[0]
            st.caption(f"专家接管：固定调用 {human_model(fixed_expert_id)}")
        else:
            expert_handoff_mode = st.selectbox(
                "专家接管方式",
                ["fixed_expert", "mean_probability_pool"],
                format_func=lambda value: {
                    "fixed_expert": "固定专家接管",
                    "mean_probability_pool": "专家池平均概率融合",
                }[value],
                help=(
                    "固定专家接管：被路由病例只运行一个指定专家，成本清楚。\n\n"
                    "专家池平均概率融合：被路由病例运行全部所选专家，并对类别概率取平均；"
                    "这不是系统自动选择最合适的专家。"
                ),
            )
            if expert_handoff_mode == "fixed_expert":
                fixed_expert_id = st.selectbox(
                    "固定接管专家",
                    expert_ids,
                    format_func=human_model,
                )
            else:
                st.caption("所选专家都会运行，最终结果为各专家类别概率的算术平均。")
    if route_ids and expert_ids:
        policies = available_routing_policies(len(route_ids), len(expert_ids))
        all_policy_help = "\n\n".join(
            f"{POLICY_LABELS[value]}：{POLICY_HELP[value]}" for value in policies
        )
        policy = st.selectbox(
            "路由机制",
            policies,
            format_func=lambda value: POLICY_LABELS[value],
            help=all_policy_help,
        )
        st.caption(POLICY_HELP[policy])
        budget = st.slider("专家调用比例", 0, 100, 30, 5) / 100
    else:
        policy = "dense_expert" if expert_ids else ("low_confidence" if len(route_ids) <= 1 else "mean_uncertainty")
        budget = 1.0 if expert_ids else 0.0
    return {
        "task_id": task_id,
        "route_ids": list(route_ids),
        "primary_id": primary_id,
        "expert_ids": list(expert_ids),
        "expert_handoff_mode": expert_handoff_mode,
        "fixed_expert_id": fixed_expert_id,
        "policy": policy,
        "budget": budget,
        **metric_profile,
    }


def _composition_label(config: dict[str, object]) -> str:
    left = " + ".join(human_model(value) for value in config["route_ids"]) or "无路由模型"
    if config.get("expert_handoff_mode") == "fixed_expert" and config.get("fixed_expert_id"):
        right = f"固定专家：{human_model(config['fixed_expert_id'])}"
    elif config.get("expert_ids"):
        right = "专家池平均：" + " + ".join(human_model(value) for value in config["expert_ids"])
    else:
        right = "无专家模型"
    return f"{left} → {right}"


def _render_summary(metrics: dict[str, object], display_metrics: list[str]) -> None:
    metric_values = []
    for metric in display_metrics:
        value = metrics.get(metric)
        metric_values.append((METRIC_LABELS.get(metric, metric), f"{float(value):.3f}" if pd.notna(value) else "不适用"))
    cost = metrics.get("estimated_total_compute_ms_per_image")
    operating_values = [
        ("调用专家", f"{int(metrics['selected_n'])} 例"),
        ("调用比例", f"{float(metrics['realized_budget']):.0%}"),
        ("估算前向成本", f"{float(cost):.3f} ms/图" if pd.notna(cost) else "未测"),
    ]
    for values in (metric_values, operating_values):
        columns = st.columns(len(values))
        for column, (label, value) in zip(columns, values):
            column.metric(label, value)


def _render_task_evaluation(task_id: str, metrics: dict[str, object], cases: pd.DataFrame) -> None:
    summary = task_evaluation_summary(task_id, metrics, cases)
    if summary["profile"] == "unavailable":
        return
    if summary["profile"] == "disease_proxy":
        st.markdown("#### 标签依赖的任务安全代理评测")
        st.caption("仅在公开测试标签可用时计算，不进入在线路由，也不是临床金标准。")
        st.markdown(build_proxy_event_table_html(summary["rows"]), unsafe_allow_html=True)
        st.caption(
            "总事件按默认输出计算；送专家是其中进入专家调用的数量；纠正表示最终输出不再命中该代理事件；"
            "残余表示最终输出仍命中。纠正与残余之和等于总事件。"
        )
        return
    st.markdown("#### 通用分类表现")
    st.caption("当前任务没有冻结专病风险协议，仅显示数据集定义类别的召回率。")
    rows = summary["rows"].copy()
    rows["类别"] = rows["类别"].map(lambda value: grade_label(task_id, value))
    st.dataframe(
        rows.style.format({"召回率": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )


def _evaluate_curve(models: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    budgets = [index / 10 for index in range(11)] if config["route_ids"] and config["expert_ids"] else [float(config["budget"])]
    rows = []
    for budget in budgets:
        metrics, _ = evaluate_exploratory_composition(
            models,
            task_id=str(config["task_id"]),
            scout_ids=list(config["route_ids"]),
            primary_scout_id=config["primary_id"],
            expert_ids=list(config["expert_ids"]),
            expert_handoff_mode=str(config["expert_handoff_mode"]),
            fixed_expert_id=config["fixed_expert_id"],
            policy=str(config["policy"]),
            requested_budget=budget,
        )
        rows.append(metrics)
    return enrich_cost_curve(pd.DataFrame(rows))


def _render_tradeoff(curve: pd.DataFrame) -> None:
    points = select_operating_points(curve)
    if not points:
        st.info("当前组合缺少完整的 forward-only 成本，无法计算相对成本操作点。")
        return
    labels = {"efficient": "省算力", "balanced": "推荐折中", "performance": "最高性能"}
    columns = st.columns(3)
    for column, name in zip(columns, ("efficient", "balanced", "performance")):
        point = points[name]
        column.metric(
            labels[name],
            f"{float(point['realized_budget']):.0%} 调用",
            delta=(
                f"Accuracy {float(point['accuracy']):.3f} · "
                f"成本 {float(point['estimated_total_compute_ms_per_image']):.3f} ms/图"
            ),
            delta_color="off",
        )
    plot = curve.dropna(subset=["estimated_total_compute_ms_per_image", "accuracy"]).copy()
    plot["调用比例"] = plot["realized_budget"].map(lambda value: f"{float(value):.0%}")
    plot["专家接管方式"] = plot["expert_handoff_mode"].map(lambda value: HANDOFF_LABELS.get(str(value), str(value)))
    base = alt.Chart(plot).encode(
        x=alt.X("estimated_total_compute_ms_per_image:Q", title="估算 forward-only 成本（ms/图）", scale=alt.Scale(zero=False)),
        y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "realized_budget:Q",
            title="专家调用比例",
            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
            legend=alt.Legend(format=".0%"),
        ),
        shape=alt.Shape("专家接管方式:N"),
        tooltip=[
            alt.Tooltip("调用比例:N"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".4f"),
            alt.Tooltip("relative_cost:Q", title="相对成本", format=".2f"),
            alt.Tooltip("estimated_total_compute_ms_per_image:Q", title="forward-only ms/图", format=".3f"),
        ],
    )
    chart = base.mark_point(size=150, filled=True) + base.mark_text(dy=-13, fontSize=11).encode(text="调用比例:N")
    st.markdown("#### 成本-效果折中图　<span style='color:#0f766e;font-size:.85rem'>越左上越优</span>", unsafe_allow_html=True)
    st.altair_chart(chart.properties(height=330), width="stretch")
    st.caption("横轴为估算 forward-only 成本；不包含解码、预处理、I/O、排队和服务开销。相对成本仅保留在悬浮详情中。")


def _global_scan_label(row: pd.Series) -> str:
    left = _human_model_list(row.get("scout_ids", ""))
    if str(row.get("expert_handoff_mode", "")) == "fixed_expert":
        right = f"固定专家：{human_model(row.get('fixed_expert_id') or row.get('active_expert_ids'))}"
    else:
        right = "专家池平均：" + _human_model_list(row.get("configured_expert_ids", ""))
    return f"{left} → {right}"


def _global_scan_table(scan: pd.DataFrame, display_metrics: list[str]) -> pd.DataFrame:
    completed = scan.loc[scan["scan_status"].eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame()
    table = pd.DataFrame(
        {
            "全局排名": pd.to_numeric(completed["global_rank_primary"], errors="coerce").astype("Int64"),
            "组合": completed.apply(_global_scan_label, axis=1),
            "路由模型": completed["scout_ids"].map(_human_model_list),
            "专家模型": completed["active_expert_ids"].map(_human_model_list),
            "专家接管方式": completed["expert_handoff_mode"].map(lambda value: HANDOFF_LABELS.get(str(value), str(value))),
            "路由机制": completed["routing_policy"].map(lambda value: POLICY_LABELS.get(str(value), str(value))),
        }
    )
    for metric in display_metrics:
        if metric in completed.columns:
            table[METRIC_LABELS.get(metric, metric)] = pd.to_numeric(completed[metric], errors="coerce")
    table["专家调用比例"] = pd.to_numeric(completed["realized_budget"], errors="coerce")
    table["估算前向成本（ms/图）"] = pd.to_numeric(completed["estimated_total_compute_ms_per_image"], errors="coerce")
    table["相对成本"] = pd.to_numeric(completed["relative_cost"], errors="coerce")
    table["Pareto 前沿"] = completed["is_pareto"].map(lambda value: "是" if bool(value) else "否")
    return table.sort_values(["全局排名", "估算前向成本（ms/图）"], na_position="last")


def _render_global_scan(models: pd.DataFrame, config: dict[str, object]) -> None:
    task_id = str(config["task_id"])
    task_models, _ = split_task_models(models, task_id)
    route_pool = _role_models(task_models, "scout")
    expert_pool = _role_models(task_models, "expert")
    if route_pool.empty or expert_pool.empty:
        return
    route_ids = route_pool["artifact_id"].astype(str).tolist()
    expert_ids = expert_pool["artifact_id"].astype(str).tolist()
    session_key = f"model_hub_global_scan_{task_id}"

    st.markdown("#### 全局候选扫描")
    st.caption(
        "在当前已登记模型池和预算网格内扫描候选组合，用于快速发现性能、成本和组合规模的折中点；"
        "这是探索性视图，不等同于最终冻结协议或外部验证结论。"
    )
    with st.expander("扫描设置", expanded=False):
        columns = st.columns(4)
        max_routes = columns[0].slider(
            "最多路由模型数",
            1,
            min(3, len(route_pool)),
            1,
            key=f"global_scan_max_routes_{task_id}",
        )
        max_experts = columns[1].slider(
            "最多专家模型数",
            1,
            min(3, len(expert_pool)),
            1,
            key=f"global_scan_max_experts_{task_id}",
        )
        budget_step = columns[2].selectbox(
            "预算步长",
            [5, 10, 20],
            index=1,
            format_func=lambda value: f"{value}%",
            key=f"global_scan_budget_step_{task_id}",
        )
        top_n = columns[3].selectbox(
            "显示 Top-N",
            [10, 20, 50],
            index=1,
            key=f"global_scan_top_n_{task_id}",
        )
        budgets = [value / 100 for value in range(0, 101, int(budget_step))]
        estimated_points = estimate_global_composition_count(
            n_scouts=len(route_pool),
            n_experts=len(expert_pool),
            max_scouts=int(max_routes),
            max_experts=int(max_experts),
            n_budgets=len(budgets),
        )
        scan_too_large = estimated_points > MAX_INTERACTIVE_GLOBAL_SCAN_EVALUATIONS
        if scan_too_large:
            st.warning(
                f"预计需要评估 {estimated_points:,} 个候选点，超过交互上限 "
                f"{MAX_INTERACTIVE_GLOBAL_SCAN_EVALUATIONS:,}。请降低最多路由/专家模型数，"
                "或调大预算步长；大规模全组合扫描后续应走离线 runner。"
            )
        else:
            st.caption(f"预计评估 {estimated_points:,} 个候选点。")
        if st.button(
            "运行交互扫描",
            icon=":material/travel_explore:",
            key=f"run_global_scan_{task_id}",
            disabled=scan_too_large,
            help="小规模扫描可直接在当前页面运行；大规模扫描请提交后台任务。",
        ):
            st.session_state[session_key] = scan_global_composition_candidates(
                models,
                task_id=task_id,
                scout_ids=route_ids,
                expert_ids=expert_ids,
                budgets=budgets,
                max_scouts=int(max_routes),
                max_experts=int(max_experts),
                primary_metric=str(config["primary_metric"]),
            )
        if st.button(
            "提交后台扫描任务",
            icon=":material/task_alt:",
            key=f"submit_global_scan_{task_id}",
            help="适合大规模组合扫描；任务会进入“任务运行记录”，完成后可自动载入研究评测区。",
        ):
            job_id = submit_global_scan_job(
                task_id=task_id,
                scout_ids=route_ids,
                expert_ids=expert_ids,
                budgets=budgets,
                max_scouts=int(max_routes),
                max_experts=int(max_experts),
                primary_metric=str(config["primary_metric"]),
                top_n=int(top_n),
                display_metrics=list(config["display_metrics"]),
            )
            st.success(f"已提交后台全局扫描任务：{job_id}。可到“任务运行记录”查看进度。")

    latest_scan = latest_completed_global_scan(task_id)
    if latest_scan:
        if session_key not in st.session_state:
            loaded = load_global_scan_results(latest_scan)
            if not loaded.empty:
                st.session_state[session_key] = loaded
        st.caption(f"最近完成后台扫描：{latest_scan.get('job_id')} · {latest_scan.get('estimated_points', '未知')} 个候选点")
        if st.button("载入最近完成后台扫描", key=f"load_latest_global_scan_{task_id}"):
            loaded = load_global_scan_results(latest_scan)
            if loaded.empty:
                st.warning("最近后台扫描没有可读取的结果文件。")
            else:
                st.session_state[session_key] = loaded
                st.rerun()

    scan = st.session_state.get(session_key)
    if not isinstance(scan, pd.DataFrame) or scan.empty:
        st.info("还没有运行全局候选扫描。先点击“运行全局候选扫描”，再查看最高性能、推荐折中和 Pareto 前沿。")
        return

    completed = scan.loc[scan["scan_status"].eq("completed")].copy()
    failed = scan.loc[scan["scan_status"].ne("completed")]
    if not failed.empty:
        st.warning(f"有 {len(failed)} 个候选组合因为输入不兼容或文件问题被跳过，可在扫描表中查看失败原因。")
    if completed.empty:
        st.error("当前扫描没有可用候选组合。")
        return

    primary_metric = str(config["primary_metric"])
    completed[primary_metric] = pd.to_numeric(completed[primary_metric], errors="coerce")
    completed["estimated_total_compute_ms_per_image"] = pd.to_numeric(
        completed["estimated_total_compute_ms_per_image"], errors="coerce"
    )
    primary_label = METRIC_LABELS.get(primary_metric, primary_metric)
    st.caption(
        f"当前主指标：{primary_label}（来自任务注册协议）。"
        "Pareto 前沿表示：在本次扫描结果中，不存在另一个组合同时做到更低成本且更高主指标。"
    )
    performance = completed.sort_values(
        [primary_metric, "estimated_total_compute_ms_per_image"],
        ascending=[False, True],
        na_position="last",
    ).iloc[0]
    valid_utility = completed.dropna(subset=["global_utility"]).copy()
    balanced = (
        valid_utility.sort_values(["global_utility", "estimated_total_compute_ms_per_image"], ascending=[False, True]).iloc[0]
        if not valid_utility.empty
        else performance
    )
    pareto = completed.loc[completed["is_pareto"].astype(bool)].copy()
    efficient = (
        pareto.sort_values("estimated_total_compute_ms_per_image", na_position="last").iloc[0]
        if not pareto.empty
        else completed.sort_values("estimated_total_compute_ms_per_image", na_position="last").iloc[0]
    )

    cards = [
        ("最高性能", performance),
        ("推荐折中", balanced),
        ("最低成本前沿", efficient),
    ]
    columns = st.columns(3)
    for column, (title, row) in zip(columns, cards):
        with column.container(border=True):
            st.metric(
                title,
                f"{float(row[primary_metric]):.3f}",
                delta=f"{float(row['realized_budget']):.0%} 调用 · {float(row['estimated_total_compute_ms_per_image']):.3f} ms/图",
                delta_color="off",
            )
            st.caption(_global_scan_label(row))
            with st.expander("查看组合细节"):
                detail_rows = [
                    ("主指标", primary_label),
                    ("路由模型", _human_model_list(row.get("scout_ids", ""))),
                    ("专家模型", _human_model_list(row.get("active_expert_ids", ""))),
                    ("专家接管方式", HANDOFF_LABELS.get(str(row.get("expert_handoff_mode", "")), str(row.get("expert_handoff_mode", "")))),
                    ("路由机制", POLICY_LABELS.get(str(row.get("routing_policy", "")), str(row.get("routing_policy", "")))),
                    ("专家调用比例", f"{float(row['realized_budget']):.0%}"),
                    ("估算前向成本", f"{float(row['estimated_total_compute_ms_per_image']):.3f} ms/图"),
                    ("Pareto 前沿", "是" if bool(row.get("is_pareto")) else "否"),
                ]
                st.dataframe(pd.DataFrame(detail_rows, columns=["字段", "内容"]), hide_index=True, width="stretch")

    table = _global_scan_table(completed, list(config["display_metrics"])).head(int(top_n))
    metric_formats = {METRIC_LABELS[metric]: "{:.4f}" for metric in config["display_metrics"] if metric in METRIC_LABELS}
    metric_formats.update({"专家调用比例": "{:.0%}", "估算前向成本（ms/图）": "{:.3f}", "相对成本": "{:.2f}"})
    st.dataframe(table.style.format(metric_formats), hide_index=True, width="stretch")

    plot = completed.dropna(subset=[primary_metric, "estimated_total_compute_ms_per_image"]).copy()
    if not plot.empty:
        rank = pd.to_numeric(plot.get("global_rank_primary"), errors="coerce")
        plot_view = plot.loc[plot["is_pareto"].astype(bool) | rank.le(int(top_n))].copy()
        if plot_view.empty:
            plot_view = plot.sort_values([primary_metric, "estimated_total_compute_ms_per_image"], ascending=[False, True]).head(int(top_n)).copy()
        st.caption("成本-效果图仅展示 Pareto 前沿和 Top-N 候选，避免全量扫描点过密；完整结果请看上方排序表或后台输出 CSV。")
        plot_view["组合"] = plot_view.apply(_global_scan_label, axis=1).astype(str)
        plot_view["专家调用比例"] = pd.to_numeric(plot_view["realized_budget"], errors="coerce")
        plot_view["Pareto 前沿"] = plot_view["is_pareto"].map(lambda value: "Pareto" if bool(value) else "候选")
        chart = (
            alt.Chart(plot_view)
            .mark_circle(size=130, opacity=0.86)
            .encode(
                x=alt.X("estimated_total_compute_ms_per_image:Q", title="估算 forward-only 成本（ms/图）", scale=alt.Scale(zero=False)),
                y=alt.Y(f"{primary_metric}:Q", title=METRIC_LABELS.get(primary_metric, primary_metric), scale=alt.Scale(zero=False)),
                color=alt.Color("专家调用比例:Q", title="专家调用比例", scale=alt.Scale(scheme="viridis", domain=[0, 1]), legend=alt.Legend(format=".0%")),
                shape=alt.Shape("Pareto 前沿:N"),
                tooltip=[
                    "组合:N",
                    alt.Tooltip(f"{primary_metric}:Q", title=METRIC_LABELS.get(primary_metric, primary_metric), format=".4f"),
                    alt.Tooltip("专家调用比例:Q", format=".0%"),
                    alt.Tooltip("estimated_total_compute_ms_per_image:Q", title="成本 ms/图", format=".3f"),
                ],
            )
        )
        st.altair_chart(chart.properties(height=340), width="stretch")


def _add_comparison(label: str, metrics: dict[str, object]) -> None:
    runs = st.session_state.setdefault("model_hub_comparison", [])
    key = (
        metrics.get("scout_ids"),
        metrics.get("configured_expert_ids"),
        metrics.get("expert_handoff_mode"),
        metrics.get("active_expert_ids"),
        metrics.get("routing_policy"),
        metrics.get("realized_budget"),
    )
    runs[:] = [item for item in runs if item["key"] != key]
    runs.append({"key": key, "组合": label, **metrics})
    del runs[:-4]


def _comparison_plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "组合",
        "expert_handoff_mode",
        "realized_budget",
        "accuracy",
        "estimated_total_compute_ms_per_image",
    ]
    plot = frame.dropna(subset=["estimated_total_compute_ms_per_image", "accuracy"]).copy()
    plot = plot[[column for column in columns if column in plot.columns]].copy()
    text_columns = ["组合", "expert_handoff_mode"]
    for column in text_columns:
        if column in plot.columns:
            plot[column] = plot[column].fillna("").astype(str)
    numeric_columns = ["realized_budget", "accuracy", "estimated_total_compute_ms_per_image"]
    for column in numeric_columns:
        if column in plot.columns:
            plot[column] = pd.to_numeric(plot[column], errors="coerce")
    plot = plot.dropna(subset=["estimated_total_compute_ms_per_image", "accuracy"]).copy()
    if "realized_budget" in plot.columns:
        plot["专家调用比例"] = plot["realized_budget"].map(lambda value: f"{float(value):.0%}")
    if "expert_handoff_mode" in plot.columns:
        plot["专家接管方式"] = plot["expert_handoff_mode"].map(lambda value: HANDOFF_LABELS.get(str(value), str(value)))
    return plot


def _render_comparison(task_id: str, display_metrics: list[str]) -> None:
    runs = st.session_state.get("model_hub_comparison", [])
    if not runs:
        return
    st.markdown("#### 组合对比")
    frame = pd.DataFrame(runs)
    if "task_id" in frame.columns:
        frame = frame.loc[frame["task_id"].astype(str).eq(str(task_id))]
    if frame.empty:
        st.info("当前任务尚未加入对比组合。")
        return
    table = build_comparison_table(frame, display_metrics)
    metric_formats = {METRIC_LABELS[metric]: "{:.4f}" for metric in display_metrics if metric in METRIC_LABELS}
    metric_formats.update({"专家调用比例": "{:.0%}", "估算前向成本（ms/图）": "{:.3f}"})
    st.dataframe(table.style.format(metric_formats), hide_index=True, width="stretch")
    plot = _comparison_plot_frame(frame)
    if len(plot) > 1:
        chart = (
            alt.Chart(plot)
            .mark_point(size=180, filled=True)
            .encode(
                x=alt.X("estimated_total_compute_ms_per_image:Q", title="估算 forward-only 成本（ms/图）", scale=alt.Scale(zero=False)),
                y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "realized_budget:Q",
                    title="专家调用比例",
                    scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                    legend=alt.Legend(format=".0%"),
                ),
                shape=alt.Shape("专家接管方式:N"),
                tooltip=["组合:N", "专家调用比例:N", alt.Tooltip("accuracy:Q", format=".4f")],
            )
        )
        st.altair_chart(chart.properties(height=300), width="stretch")
    if st.button("清空组合对比", icon=":material/delete_sweep:"):
        st.session_state["model_hub_comparison"] = []
        st.rerun()


def render_research_workspace(models: pd.DataFrame) -> None:
    st.subheader("研究评测")
    st.caption("组合计算基于同任务、同数据集、同标签空间的冻结 prediction；结果不会覆盖正式科研产物。")
    config = _controls(models)
    if config is None:
        return
    _render_evaluation_inputs(models, config)
    try:
        metrics, cases = evaluate_exploratory_composition(
            models,
            task_id=str(config["task_id"]),
            scout_ids=list(config["route_ids"]),
            primary_scout_id=config["primary_id"],
            expert_ids=list(config["expert_ids"]),
            expert_handoff_mode=str(config["expert_handoff_mode"]),
            fixed_expert_id=config["fixed_expert_id"],
            policy=str(config["policy"]),
            requested_budget=float(config["budget"]),
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    label = _composition_label(config)
    curve = _evaluate_curve(models, config)
    st.markdown(f'<div class="hub-band"><strong>当前组合：</strong>{label}</div>', unsafe_allow_html=True)
    _render_summary(metrics, list(config["display_metrics"]))
    if st.button("加入组合对比", icon=":material/add_chart:", width="stretch"):
        _add_comparison(label, metrics)
    _render_tradeoff(curve)
    _render_global_scan(models, config)
    _render_task_evaluation(str(config["task_id"]), metrics, cases)
    _render_comparison(str(config["task_id"]), list(config["display_metrics"]))
    st.session_state["model_hub_last_cases"] = build_online_case_view(cases)
    st.session_state["model_hub_last_research_cases"] = cases
    st.session_state["model_hub_last_metrics"] = metrics
    st.session_state["model_hub_last_label"] = label
