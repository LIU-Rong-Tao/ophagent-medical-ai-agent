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
from app.model_hub_result_audit import render_result_table_risk_audit
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

DR_PROXY_EVENT_LABELS = {
    "large_undergrading": "大跨度低估",
    "referable_miss": "可转诊漏检",
    "severe_pdr_miss": "重症漏检",
}

DR_PROXY_EVENT_WEIGHTS = {
    "severe_pdr_miss": 0.45,
    "referable_miss": 0.35,
    "large_undergrading": 0.20,
}

AUDIT_PROXY_PENALTY_WEIGHT = 0.03


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
        st.markdown("#### 标签依赖安全代理事件评测")
        st.caption("仅在公开测试标签可用时计算，不进入在线路由，也不是临床金标准，不提供诊断或患者分流决定。")
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
    return enrich_cost_curve(pd.DataFrame(rows), metric_column=str(config["primary_metric"]))


def _render_tradeoff(curve: pd.DataFrame, primary_metric: str) -> None:
    points = select_operating_points(curve, metric_column=primary_metric)
    if not points:
        st.info("当前组合缺少完整的 forward-only 成本，无法计算相对成本操作点。")
        return
    primary_label = METRIC_LABELS.get(primary_metric, primary_metric)
    labels = {"efficient": "省算力", "balanced": "推荐折中", "performance": "最高性能"}
    columns = st.columns(3)
    for column, name in zip(columns, ("efficient", "balanced", "performance")):
        point = points[name]
        column.metric(
            labels[name],
            f"{float(point['realized_budget']):.0%} 调用",
            delta=(
                f"{primary_label} {float(point[primary_metric]):.3f} · "
                f"成本 {float(point['estimated_total_compute_ms_per_image']):.3f} ms/图"
            ),
            delta_color="off",
        )
    plot = curve.dropna(subset=["estimated_total_compute_ms_per_image", primary_metric]).copy()
    plot["调用比例"] = plot["realized_budget"].map(lambda value: f"{float(value):.0%}")
    plot["专家接管方式"] = plot["expert_handoff_mode"].map(lambda value: HANDOFF_LABELS.get(str(value), str(value)))
    base = alt.Chart(plot).encode(
        x=alt.X("estimated_total_compute_ms_per_image:Q", title="估算 forward-only 成本（ms/图）", scale=alt.Scale(zero=False)),
        y=alt.Y(f"{primary_metric}:Q", title=primary_label, scale=alt.Scale(zero=False)),
        color=alt.Color(
            "realized_budget:Q",
            title="专家调用比例",
            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
            legend=alt.Legend(format=".0%"),
        ),
        shape=alt.Shape("专家接管方式:N"),
        tooltip=[
            alt.Tooltip("调用比例:N"),
            alt.Tooltip(f"{primary_metric}:Q", title=primary_label, format=".4f"),
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
    if "audit_proxy_score" in completed.columns:
        table["审计代理事件评分"] = pd.to_numeric(completed["audit_proxy_score"], errors="coerce")
    table["Pareto 前沿"] = completed["is_pareto"].map(lambda value: "是" if bool(value) else "否")
    return table.sort_values(["全局排名", "估算前向成本（ms/图）"], na_position="last")


def _metric_value(row: pd.Series, column: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def _metric_text(row: pd.Series, column: str, *, percent: bool = False, unit: str = "") -> str:
    value = _metric_value(row, column)
    if pd.isna(value):
        return "不适用"
    if percent:
        return f"{value:.0%}"
    if unit:
        return f"{value:.3f} {unit}"
    return f"{value:.3f}"


def _primary_metric_explanation(task_id: str, primary_label: str) -> str:
    if task_id == "aptos_dr_5class" and primary_label == "QWK":
        return (
            "当前任务为 DR 五级有序分级，主指标采用 QWK；"
            "Accuracy 与 Macro-F1 作为辅助观察指标。"
        )
    return f"当前主指标为 {primary_label}，来自任务注册协议；其他指标仅作辅助观察。"


def _same_operating_context(left: pd.Series, right: pd.Series) -> bool:
    text_columns = ["scout_ids", "active_expert_ids", "expert_handoff_mode"]
    same_text = all(str(left.get(column, "")) == str(right.get(column, "")) for column in text_columns)
    same_budget = abs(_metric_value(left, "realized_budget") - _metric_value(right, "realized_budget")) < 1e-9
    same_cost = (
        abs(
            _metric_value(left, "estimated_total_compute_ms_per_image")
            - _metric_value(right, "estimated_total_compute_ms_per_image")
        )
        < 1e-9
    )
    return same_text and same_budget and same_cost


def _near_tie_note(
    title: str,
    row: pd.Series,
    *,
    primary_metric: str,
    performance: pd.Series,
    balanced: pd.Series,
) -> str:
    if title not in {"最高性能", "推荐折中"}:
        return ""
    if row.name == performance.name == balanced.name:
        return "同一组合：本次扫描中同一个候选同时满足最高性能和推荐折中公式。"
    delta = abs(_metric_value(performance, primary_metric) - _metric_value(balanced, primary_metric))
    if delta > 0.002 or not _same_operating_context(performance, balanced):
        return ""
    other = "推荐折中" if title == "最高性能" else "最高性能"
    return (
        f"近似并列：与{other}主指标仅差 {delta:.3f}，模型、调用比例和成本相同；"
        "主要差别来自路由机制。"
    )


def _selection_overlap_note(
    title: str,
    row: pd.Series,
    *,
    primary_metric: str,
    performance: pd.Series,
    balanced: pd.Series,
    audit_priority: pd.Series | None,
) -> str:
    peers = {
        "最高性能": performance,
        "推荐折中": balanced,
    }
    if audit_priority is not None:
        peers["代理事件优先"] = audit_priority
    exact_matches = [name for name, peer in peers.items() if name != title and peer.name == row.name]
    if exact_matches:
        return f"同一组合：本次扫描中该候选同时满足{'、'.join(exact_matches)}公式。"
    return _near_tie_note(
        title,
        row,
        primary_metric=primary_metric,
        performance=performance,
        balanced=balanced,
    )


def _global_scan_reason(
    title: str,
    row: pd.Series,
    *,
    primary_label: str,
) -> tuple[str, str]:
    if title == "最高性能":
        pareto_note = "Pareto 前沿" if bool(row.get("is_pareto")) else "非 Pareto"
        return (
            "QWK 最优" if primary_label == "QWK" else f"{primary_label} 最优",
            f"主指标最大，用作性能上界参考；当前为{pareto_note}，代理事件残余不一定更优。",
        )
    if title == "推荐折中":
        return (
            "综合推荐",
            "综合主指标、相对成本和标签依赖审计代理事件评分后推荐。",
        )
    if title == "代理事件优先":
        return (
            "代理事件最低",
            "按标签依赖审计代理事件评分从低到高选择，用于观察公开标签研究审计口径下的残余事件。",
        )
    return (
        "最低成本",
        "作为无专家或低专家调用基线进入候选视图，用来锚定最低成本能达到的表现。",
    )


def _risk_proxy_rows_from_metrics(metrics: dict[str, object] | pd.Series) -> pd.DataFrame:
    rows = []
    metric_series = pd.Series(metrics)

    def metric_int(column: str) -> int:
        value = _metric_value(metric_series, column)
        return int(value) if pd.notna(value) else 0

    for name, label in DR_PROXY_EVENT_LABELS.items():
        rows.append(
            {
                "事件": label,
                "总事件": metric_int(f"dr_{name}_event_total"),
                "送专家": metric_int(f"dr_{name}_selected_n"),
                "纠正": metric_int(f"dr_{name}_resolved_n"),
                "残余": metric_int(f"dr_{name}_residual_n"),
            }
        )
    return pd.DataFrame(rows)


def _audit_proxy_score(metrics: dict[str, object] | pd.Series) -> float:
    metric_series = pd.Series(metrics)
    if str(metric_series.get("risk_semantics", "")) != "label_based_safety_proxy_not_clinical_gold_standard":
        return float("nan")
    weighted_sum = 0.0
    observed_weight = 0.0
    for event, weight in DR_PROXY_EVENT_WEIGHTS.items():
        total = _metric_value(metric_series, f"dr_{event}_event_total")
        residual = _metric_value(metric_series, f"dr_{event}_residual_n")
        if pd.isna(total) or total <= 0 or pd.isna(residual):
            continue
        weighted_sum += weight * max(float(residual), 0.0) / float(total)
        observed_weight += weight
    if observed_weight <= 0:
        return float("nan")
    return weighted_sum / observed_weight


def _audit_proxy_score_text(row: pd.Series) -> str:
    score = _metric_value(row, "audit_proxy_score")
    if pd.isna(score):
        return "不适用"
    return f"{score:.1%}"


def _refresh_global_scan_derived(completed: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    refreshed = completed.copy()
    if refreshed.empty or primary_metric not in refreshed.columns:
        return refreshed
    refreshed[primary_metric] = pd.to_numeric(refreshed[primary_metric], errors="coerce")
    refreshed["estimated_total_compute_ms_per_image"] = pd.to_numeric(
        refreshed["estimated_total_compute_ms_per_image"], errors="coerce"
    )
    try:
        curve = enrich_cost_curve(refreshed, metric_column=primary_metric)
    except ValueError:
        curve = refreshed.copy()
        curve["relative_cost"] = pd.to_numeric(curve.get("relative_cost"), errors="coerce")
        curve["is_pareto"] = curve.get("is_pareto", False)
    for column in ["relative_cost", "is_pareto"]:
        if column in curve.columns:
            refreshed.loc[curve.index, column] = curve[column]
    rank_source = refreshed.dropna(subset=[primary_metric]).sort_values(
        [primary_metric, "estimated_total_compute_ms_per_image"],
        ascending=[False, True],
        na_position="last",
    )
    refreshed["global_rank_primary"] = pd.NA
    refreshed.loc[rank_source.index, "global_rank_primary"] = range(1, len(rank_source) + 1)
    refreshed["global_utility"] = pd.to_numeric(refreshed[primary_metric], errors="coerce") - 0.01 * pd.to_numeric(
        refreshed["relative_cost"], errors="coerce"
    )
    refreshed["audit_proxy_score"] = refreshed.apply(_audit_proxy_score, axis=1)
    refreshed["audit_adjusted_utility"] = refreshed["global_utility"] - AUDIT_PROXY_PENALTY_WEIGHT * pd.to_numeric(
        refreshed["audit_proxy_score"], errors="coerce"
    )
    return refreshed


def _render_selection_note(text: str, *, muted: bool = False, label: str = "近似并列") -> None:
    if not text:
        text = "当前组合与相邻候选没有达到近似并列阈值，主要按本卡片公式入选。"
        muted = True
        label = "选择说明"
    elif text.startswith("同一组合："):
        label = "同一组合"
    css_class = "hub-selection-note hub-selection-note-muted" if muted else "hub-selection-note"
    content = text.removeprefix("近似并列：").removeprefix("同一组合：")
    st.markdown(
        f'<div class="{css_class}"><b>{html.escape(label)}</b><span>{html.escape(content)}</span></div>',
        unsafe_allow_html=True,
    )


def _formula_help_text(title: str, *, primary_label: str) -> str:
    if title == "最高性能":
        return f"按 {primary_label} 从高到低排序；并列时选择估算前向成本更低的组合。"
    if title == "推荐折中":
        return (
            f"综合效用 = {primary_label} - 0.01 × 相对成本 - "
            f"{AUDIT_PROXY_PENALTY_WEIGHT:.2f} × 标签依赖审计代理事件评分；"
            "若没有代理事件字段，则退回为主指标 - 0.01 × 相对成本。"
        )
    if title == "代理事件优先":
        return (
            "代理事件评分 = 0.45 × 重症漏检残余率 + 0.35 × 可转诊漏检残余率 + "
            "0.20 × 大跨度低估残余率；数值越低越好；并列时优先主指标更高、成本更低。"
        )
    return "先取 Pareto 前沿中的最低估算前向成本；若没有 Pareto 标记，则取全量最低成本候选。"


def _formula_pill_html(title: str, *, primary_label: str) -> str:
    help_text = (
        _formula_help_text(title, primary_label=primary_label)
        + " 代理事件评分仅用于公开标签研究审计，不进入在线路由，也不是临床金标准。"
    )
    return (
        '<span class="hub-formula-pill">'
        f'{html.escape(title)}'
        f'<span class="hub-help-icon" tabindex="0" title="{html.escape(help_text)}" '
        f'aria-label="{html.escape(help_text)}">?</span>'
        "</span>"
    )


def _render_formula_hint(titles: list[str], *, primary_label: str) -> None:
    pills = "".join(_formula_pill_html(title, primary_label=primary_label) for title in titles)
    st.markdown(
        f'<div class="hub-formula-line"><span class="hub-formula-label">命中公式</span>{pills}</div>',
        unsafe_allow_html=True,
    )


def _audit_proxy_score_help(score_text: str) -> str:
    return (
        "标签依赖审计代理事件评分 = 0.45 × 重症漏检残余率 + 0.35 × 可转诊漏检残余率 + "
        f"0.20 × 大跨度低估残余率。这里的 {score_text} 表示按上述权重加权后的残余比例，"
        "越低越好；它不是模型置信度，也不是临床风险概率。"
    )


def _selection_groups(cards: list[tuple[str, pd.Series]]) -> list[tuple[list[str], pd.Series]]:
    groups: list[tuple[list[str], pd.Series]] = []
    index_to_group: dict[object, int] = {}
    for title, row in cards:
        key = row.name
        if key in index_to_group:
            groups[index_to_group[key]][0].append(title)
        else:
            index_to_group[key] = len(groups)
            groups.append(([title], row))
    return groups


def _selection_group_title(titles: list[str]) -> str:
    if len(titles) == 1:
        return titles[0]
    return "多公式命中推荐"


def _selection_group_reason(titles: list[str], row: pd.Series, *, primary_label: str) -> str:
    if len(titles) == 1:
        return _global_scan_reason(titles[0], row, primary_label=primary_label)[1]
    return (
        f"同一候选同时命中{'、'.join(titles)}，页面已合并展示，避免把同一组合重复成多张卡。"
        "可打开审计卡片查看该组合下的标签依赖代理事件。"
    )


def _selection_group_badges(titles: list[str], row: pd.Series, *, primary_label: str) -> list[str]:
    if len(titles) == 1:
        return [_global_scan_reason(titles[0], row, primary_label=primary_label)[0]]
    return [_global_scan_reason(title, row, primary_label=primary_label)[0] for title in titles]


@st.dialog("组合审计卡片", width="large")
def _global_scan_audit_dialog(
    title: str,
    row: pd.Series,
    *,
    task_id: str,
    primary_metric: str,
    display_metrics: list[str],
    performance: pd.Series,
    balanced: pd.Series,
    formula_titles: list[str] | None = None,
) -> None:
    primary_label = METRIC_LABELS.get(primary_metric, primary_metric)
    formula_titles = formula_titles or [title]
    reason = _selection_group_reason(formula_titles, row, primary_label=primary_label)
    tie_note = (
        f"同一组合：本次扫描中该候选同时满足{'、'.join(formula_titles)}公式。"
        if len(formula_titles) > 1
        else _near_tie_note(
            title,
            row,
            primary_metric=primary_metric,
            performance=performance,
            balanced=balanced,
        )
    )
    suffix = "（近似并列）" if tie_note and title in {"最高性能", "推荐折中"} else ""
    st.markdown(f"### {title}{suffix}")
    st.caption(_global_scan_label(row))
    _render_formula_hint(formula_titles, primary_label=primary_label)
    st.markdown(
        f'<div class="hub-band"><strong>为什么入选：</strong>{html.escape(reason)}</div>',
        unsafe_allow_html=True,
    )
    _render_selection_note(tie_note)
    metric_columns = st.columns(min(6, max(1, len(display_metrics) + 3)))
    values: list[tuple[str, str]] = []
    for metric in display_metrics:
        if metric in row.index:
            values.append((METRIC_LABELS.get(metric, metric), _metric_text(row, metric)))
    values.extend(
        [
            ("专家调用", _metric_text(row, "realized_budget", percent=True)),
            ("估算成本（ms/图）", _metric_text(row, "estimated_total_compute_ms_per_image")),
            ("代理事件评分", _audit_proxy_score_text(row)),
        ]
    )
    for column, (label, value) in zip(metric_columns, values):
        column.metric(label, value)
    detail_rows = [
        ("主指标", primary_label),
        ("路由模型", _human_model_list(row.get("scout_ids", ""))),
        ("专家模型", _human_model_list(row.get("active_expert_ids", ""))),
        ("专家接管方式", HANDOFF_LABELS.get(str(row.get("expert_handoff_mode", "")), str(row.get("expert_handoff_mode", "")))),
        ("路由机制", POLICY_LABELS.get(str(row.get("routing_policy", "")), str(row.get("routing_policy", "")))),
        ("Pareto 前沿", "是" if bool(row.get("is_pareto")) else "否"),
        ("综合效用", _metric_text(row, "global_utility")),
        ("代理事件调整效用", _metric_text(row, "audit_adjusted_utility")),
    ]
    st.markdown("#### 组合策略")
    st.dataframe(pd.DataFrame(detail_rows, columns=["字段", "内容"]), hide_index=True, width="stretch")
    if str(row.get("risk_semantics", "")) == "label_based_safety_proxy_not_clinical_gold_standard":
        st.markdown("#### 标签依赖安全代理事件")
        st.caption("仅公开测试标签研究审计，不进入在线路由，也不是临床金标准，不提供诊断或患者分流决定。")
        st.markdown(build_proxy_event_table_html(_risk_proxy_rows_from_metrics(row)), unsafe_allow_html=True)
    else:
        st.info("当前组合没有可展示的标签依赖安全代理事件。")


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
    completed = _refresh_global_scan_derived(completed, primary_metric)
    primary_label = METRIC_LABELS.get(primary_metric, primary_metric)
    st.caption(
        _primary_metric_explanation(task_id, primary_label)
        + " Pareto 前沿表示：在本次扫描结果中，不存在另一个组合同时做到更低成本且更高主指标。"
        "页面会按当前任务主指标即时重算扫描派生列，避免旧结果文件中的派生字段错位。"
    )
    performance = completed.sort_values(
        [primary_metric, "estimated_total_compute_ms_per_image"],
        ascending=[False, True],
        na_position="last",
    ).iloc[0]
    utility_column = (
        "audit_adjusted_utility"
        if "audit_adjusted_utility" in completed.columns and completed["audit_adjusted_utility"].notna().any()
        else "global_utility"
    )
    valid_utility = completed.dropna(subset=[utility_column]).copy()
    balanced = (
        valid_utility.sort_values([utility_column, "estimated_total_compute_ms_per_image"], ascending=[False, True]).iloc[0]
        if not valid_utility.empty
        else performance
    )
    valid_audit = completed.dropna(subset=["audit_proxy_score"]).copy()
    audit_priority = (
        valid_audit.sort_values(
            ["audit_proxy_score", primary_metric, "estimated_total_compute_ms_per_image"],
            ascending=[True, False, True],
        ).iloc[0]
        if not valid_audit.empty
        else None
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
    ]
    if audit_priority is not None:
        cards.append(("代理事件优先", audit_priority))
    cards.append(("最低成本前沿", efficient))
    selection_groups = _selection_groups(cards)
    columns = st.columns(len(selection_groups))
    for column, (titles, row) in zip(columns, selection_groups):
        with column.container(border=True):
            title = _selection_group_title(titles)
            badges = _selection_group_badges(titles, row, primary_label=primary_label)
            reason = _selection_group_reason(titles, row, primary_label=primary_label)
            st.metric(
                title,
                f"{float(row[primary_metric]):.3f}",
                delta=f"{float(row['realized_budget']):.0%} 调用 · {float(row['estimated_total_compute_ms_per_image']):.3f} ms/图",
                delta_color="off",
            )
            badge_html = " ".join(f'<span class="hub-chip hub-chip-blue">{html.escape(badge)}</span>' for badge in badges)
            st.markdown(
                f'<span class="hub-chip hub-chip-{"teal" if bool(row.get("is_pareto")) else "amber"}">'
                f'{"Pareto" if bool(row.get("is_pareto")) else "非 Pareto"}</span> '
                f"{badge_html}",
                unsafe_allow_html=True,
            )
            st.caption(_global_scan_label(row))
            _render_formula_hint(titles, primary_label=primary_label)
            st.markdown(
                f'<div class="hub-card-reason">{html.escape(reason)}</div>',
                unsafe_allow_html=True,
            )
            score_text = _audit_proxy_score_text(row)
            score_help = html.escape(_audit_proxy_score_help(score_text))
            st.markdown(
                '<div class="hub-audit-score"><span>审计代理事件评分'
                f'<span class="hub-help-icon" tabindex="0" title="{score_help}" aria-label="{score_help}">?</span>'
                f'</span><b>{score_text}</b></div>',
                unsafe_allow_html=True,
            )
            tie_note = (
                f"同一组合：本次扫描中该候选同时满足{'、'.join(titles)}公式。"
                if len(titles) > 1
                else _selection_overlap_note(
                    titles[0],
                    row,
                    primary_metric=primary_metric,
                    performance=performance,
                    balanced=balanced,
                    audit_priority=audit_priority,
                )
            )
            _render_selection_note(tie_note)
            if st.button(
                "打开审计卡片",
                key=f"global_scan_audit_{task_id}_{'_'.join(titles)}_{row.name}",
                icon=":material/open_in_new:",
                width="stretch",
                help="查看组合细节已升级为审计卡片弹窗，重点展示标签依赖安全代理事件。",
            ):
                _global_scan_audit_dialog(
                    title,
                    row,
                    task_id=task_id,
                    primary_metric=primary_metric,
                    display_metrics=list(config["display_metrics"]),
                    performance=performance,
                    balanced=balanced,
                    formula_titles=titles,
                )

    table = _global_scan_table(completed, list(config["display_metrics"])).head(int(top_n))
    metric_formats = {METRIC_LABELS[metric]: "{:.4f}" for metric in config["display_metrics"] if metric in METRIC_LABELS}
    metric_formats.update(
        {
            "专家调用比例": "{:.0%}",
            "估算前向成本（ms/图）": "{:.3f}",
            "相对成本": "{:.2f}",
            "审计代理事件评分": "{:.1%}",
        }
    )
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


def _render_routing_composition_workspace(models: pd.DataFrame) -> None:
    mode = st.segmented_control(
        "评测模式",
        ["性能回放", "成本—性能评测"],
        default="性能回放",
        key="research_evaluation_mode",
    )
    if mode == "性能回放":
        st.caption("使用同任务、同数据集、同标签空间的冻结 prediction；不要求成本数据。")
        selectable = models
    else:
        st.caption("只显示已完成统一 forward-only 测量的模型。")
        measured = models.get("cost_status", pd.Series("", index=models.index)).astype(str).eq(
            "measured"
        )
        selectable = models.loc[measured].copy()
        st.success(f"当前有 {len(selectable)} 个已测成本模型可进入曲线比较。")
        active = models.get("lifecycle_status", pd.Series("active", index=models.index)).fillna(
            "active"
        ).astype(str).ne("superseded")
        excluded = models.loc[
            ~measured & active & models["prediction_source"].astype(str).ne("missing")
        ]
        if not excluded.empty:
            names = ", ".join(excluded["artifact_id"].map(human_model).astype(str).unique())
            st.info(f"以下模型尚未完成统一 forward-only 测量，本模式不可选：{names}")
    config = _controls(selectable)
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
    if mode == "成本—性能评测":
        _render_tradeoff(curve, str(config["primary_metric"]))
    _render_global_scan(models, config)
    _render_task_evaluation(str(config["task_id"]), metrics, cases)
    _render_comparison(str(config["task_id"]), list(config["display_metrics"]))
    st.session_state["model_hub_last_cases"] = build_online_case_view(cases)
    st.session_state["model_hub_last_research_cases"] = cases
    st.session_state["model_hub_last_metrics"] = metrics
    st.session_state["model_hub_last_label"] = label


def render_research_workspace(models: pd.DataFrame) -> None:
    st.markdown("#### 选择研究评测方式")
    st.caption("路由组合评测使用已登记任务模型；结果表风险审计消费外部预测产物，两者不会互相授予模型资格。")
    workspace = st.segmented_control(
        "研究评测功能",
        ["路由组合评测", "结果表风险审计"],
        default="路由组合评测",
        key="research_workspace_layer",
    )
    if workspace == "结果表风险审计":
        render_result_table_risk_audit()
    else:
        st.markdown(
            '<div class="hub-band"><strong>评测边界：</strong>'
            '当前页用于比较受控路由/专家组合、专家调用预算与估算成本；'
            '公开标签研究审计代理事件不进入在线路由。</div>',
            unsafe_allow_html=True,
        )
        _render_routing_composition_workspace(models)
