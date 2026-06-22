"""新任务离线审计沙盒。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.audit_core import (
    class_specific_miss_ranking,
    compute_confidence,
    compute_entropy,
    compute_margin,
    compute_top2,
    evaluate_topk_capture,
    infer_prob_columns,
    rank_by_score,
    validate_probability_columns,
)
from app.ui import (
    metric_card,
    page_header,
    render_boundary,
    render_empty_state,
    section_header,
)


ID_CANDIDATES = ["image_id", "image_key", "filename", "case_id"]


def sample_csv() -> bytes:
    frame = pd.DataFrame(
        {
            "image_id": ["case_001", "case_002"],
            "pred_class": ["CSC", "ERM"],
            "prob_CSC": [0.62, 0.18],
            "prob_ERM": [0.20, 0.55],
            "prob_MH": [0.10, 0.15],
            "prob_RD": [0.08, 0.12],
        }
    )
    return frame.to_csv(index=False).encode("utf-8-sig")


def prepare_queue(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    prob_cols = infer_prob_columns(frame)
    queue = frame.copy()
    queue["confidence"] = compute_confidence(queue, prob_cols)
    queue["margin"] = compute_margin(queue, prob_cols)
    queue["entropy_norm"] = compute_entropy(queue, prob_cols)
    queue = pd.concat([queue, compute_top2(queue, prob_cols)], axis=1)

    if method == "1-MSP":
        queue["review_score"] = 1.0 - queue["confidence"]
        queue["risk_reason"] = "最大类别概率较低"
    elif method == "Top1-Top2 间隔":
        queue["review_score"] = -queue["margin"]
        queue["risk_reason"] = "前两类概率接近"
    else:
        queue["review_score"] = queue["entropy_norm"]
        queue["risk_reason"] = "概率分布更分散"
    return rank_by_score(queue, "review_score")


def render_queue(queue: pd.DataFrame, id_column: str) -> None:
    columns = [
        id_column,
        "pred_class",
        "confidence",
        "margin",
        "entropy_norm",
        "top2_class",
        "top2_probability",
        "risk_reason",
        "review_rank",
    ]
    columns = [column for column in columns if column in queue.columns]
    limit = 15 if st.session_state.get("display_mode", "临床展示") == "临床展示" else 50
    display = queue.head(limit)[columns].rename(
        columns={
            id_column: "样本 ID",
            "pred_class": "预测类别",
            "confidence": "置信度",
            "margin": "Top1-Top2 间隔",
            "entropy_norm": "归一化熵",
            "top2_class": "第二候选",
            "top2_probability": "第二候选概率",
            "risk_reason": "排序原因",
            "review_rank": "复核顺序",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "置信度": st.column_config.NumberColumn(format="%.3f"),
            "Top1-Top2 间隔": st.column_config.NumberColumn(format="%.3f"),
            "归一化熵": st.column_config.NumberColumn(format="%.3f"),
            "第二候选概率": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def render_general_validation(queue: pd.DataFrame) -> None:
    queue = queue.copy()
    queue["general_error"] = (
        queue["pred_class"].astype(str) != queue["true_class"].astype(str)
    )
    metrics = evaluate_topk_capture(queue, "general_error", budgets=(0.1, 0.2, 0.3))
    cols = st.columns(3, gap="small")
    for column, (_, row) in zip(cols, metrics.iterrows()):
        with column:
            metric_card(
                f"Top{row['review_budget']:.0%}",
                f"{row['event_recall']:.1%}",
                f"捕获 {int(row['captured_event'])}/{int(row['total_event'])}；"
                f"残余 {int(row['residual_event_count'])}",
            )
    st.dataframe(metrics, hide_index=True, use_container_width=True)


def render_class_specific_validation(frame: pd.DataFrame, classes: list[str]) -> None:
    target = st.selectbox("目标漏检类别", classes)
    ranked = class_specific_miss_ranking(frame, target)
    metrics = evaluate_topk_capture(ranked, "target_event", budgets=(0.1, 0.2, 0.3))
    cols = st.columns(3, gap="small")
    for column, (_, row) in zip(cols, metrics.iterrows()):
        with column:
            metric_card(
                f"Top{row['review_budget']:.0%}",
                f"{row['event_recall']:.1%}",
                f"目标类别 {target}；残余 {int(row['residual_event_count'])}",
                accent="amber",
            )
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    with st.expander("查看目标类别高风险样本"):
        columns = [
            column
            for column in [
                "image_id",
                "image_key",
                "filename",
                "case_id",
                "pred_class",
                "true_class",
                "target_probability",
                "target_event",
                "review_rank",
            ]
            if column in ranked.columns
        ]
        st.dataframe(ranked.head(30)[columns], hide_index=True, use_container_width=True)


def render() -> None:
    page_header(
        "新任务审计沙盒",
        "上传任意多分类模型的 prediction CSV，先检查概率输出是否能形成复核队列；有参考标签时再做回顾性收益预检。",
        "协议迁移入口，不代表迁移成功",
    )
    render_boundary(
        "当前任务协议：generic_multiclass_v1。"
        "跨疾病时不能直接沿用 DR 的期望等级差或门控重症概率质量。"
        "新任务先使用通用不确定性和类别特异漏检排序；临床危险事件需要专科医生或可靠指南重新定义。"
    )

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        upload = st.file_uploader("上传 prediction CSV", type=["csv"])
    with right:
        st.download_button(
            "下载最小 CSV 模板",
            sample_csv(),
            file_name="ophagent_audit_template.csv",
            mime="text/csv",
        )
        st.caption("必需：pred_class、至少两个 prob_* 列；true_class 可选。")

    if upload is None:
        render_empty_state(
            "等待 prediction CSV",
            "系统不会上传或训练模型，只读取预测类别与完整概率分布。",
        )
        return

    try:
        frame = pd.read_csv(upload)
        prob_cols = validate_probability_columns(frame)
        id_column = next((column for column in ID_CANDIDATES if column in frame.columns), None)
        if id_column is None:
            frame = frame.copy()
            frame["case_id"] = [f"row_{index:05d}" for index in range(len(frame))]
            id_column = "case_id"
        frame["pred_class"] = frame["pred_class"].astype(str)
        if "true_class" in frame.columns:
            frame["true_class"] = frame["true_class"].astype(str)
    except Exception as exc:
        st.error(f"数据检查未通过：{exc}")
        return

    classes = [column[len("prob_") :] for column in prob_cols]
    section_header("数据检查结果")
    check_cols = st.columns(4, gap="small")
    with check_cols[0]:
        metric_card("记录数", f"{len(frame):,}", "prediction records")
    with check_cols[1]:
        metric_card("类别数", str(len(classes)), "、".join(classes[:4]))
    with check_cols[2]:
        metric_card("样本标识列", id_column, "自动识别")
    with check_cols[3]:
        metric_card(
            "参考标签",
            "已提供" if "true_class" in frame.columns else "未提供",
            "决定是否显示后验评价",
            accent="teal" if "true_class" in frame.columns else "amber",
        )

    mode = st.radio(
        "审计模式",
        ["无标签预审", "回顾性验证"] if "true_class" in frame.columns else ["无标签预审"],
        horizontal=True,
    )
    method = st.selectbox("通用排序信号", ["1-MSP", "Top1-Top2 间隔", "归一化熵"])
    queue = prepare_queue(frame, method)

    if mode == "无标签预审":
        section_header("复核优先队列")
        render_queue(queue, id_column)
        render_boundary(
            "当前没有使用 true_class，因此这里只能给出排序和风险原因。"
            "页面不会显示捕获率、召回率或残余事件。"
        )
    else:
        validation_type = st.radio(
            "后验评价目标",
            ["通用错分", "类别特异漏检"],
            horizontal=True,
        )
        if validation_type == "通用错分":
            render_general_validation(queue)
        else:
            render_class_specific_validation(frame, classes)
        render_boundary(
            "这里的结果是小规模离线收益预检。若迁移到医院新病种，"
            "还需要确认标签映射、患者级重复、样本量、设备域移和医生认可的危险事件。"
            "疾病专属完整审计协议当前仅支持 DR 五级任务。"
        )

    if st.session_state.get("display_mode") == "研究审计":
        st.download_button(
            "下载当前排序结果",
            queue.to_csv(index=False).encode("utf-8-sig"),
            file_name="ophagent_review_queue.csv",
            mime="text/csv",
        )
