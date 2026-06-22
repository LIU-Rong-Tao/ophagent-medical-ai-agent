"""病例复核详情的数据隔离与 Streamlit 弹窗。"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.audit_core import translate_risk_reasons
from app.clinical_semantics import summarize_clinical_display
from app.ui import (
    metric_card,
    render_boundary,
    render_case_card,
    render_empty_state,
    render_probability_profile,
    section_header,
)


PRE_REVIEW_FIELDS = {
    "case_id",
    "image_path",
    "image_key",
    "pred_grade",
    "pred_label",
    "pred_label_raw",
    "top2_grade",
    "top2_label",
    "confidence",
    "top2_confidence",
    "margin",
    "entropy",
    "entropy_norm",
    "severe_prob_mass",
    "expected_grade",
    "expected_gap",
    "pre_review_risk_score",
    "risk_reasons",
    "pre_review_risk_level",
    "review_priority_rank",
}

POSTHOC_FIELDS = {
    "true_grade",
    "true_label",
    "general_error",
    "any_undergrading",
    "large_undergrading",
    "referable_dr_miss",
    "vision_threatening_dr_miss",
    "high_confidence_vision_threatening_miss",
    "captured",
    "residual",
}

GRADE_LABELS = [
    "0级 · 未见 DR",
    "1级 · 轻度",
    "2级 · 中度",
    "3级 · 重度",
    "4级 · 增殖期",
]


def _as_dict(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    return row.to_dict() if isinstance(row, pd.Series) else dict(row)


def normalize_image_key(value: Any) -> str:
    """跨 Windows/Linux 路径分隔符提取稳定的小写文件名。"""

    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1].lower()


def build_pre_review_case(
    row: Mapping[str, Any] | pd.Series,
    *,
    backbone: str,
) -> dict[str, Any]:
    """使用白名单构造无标签病例，任何后验字段都会被丢弃。"""

    source = _as_dict(row)
    result = {
        key: source[key]
        for key in PRE_REVIEW_FIELDS
        if key in source and not pd.isna(source[key])
    }
    for key, value in source.items():
        if str(key).startswith("prob_") and not pd.isna(value):
            result[str(key)] = value
    raw_key = source.get("image_key") or source.get("image_path") or source.get("case_id")
    normalized = normalize_image_key(raw_key)
    result["backbone"] = str(backbone)
    result["normalized_image_key"] = normalized
    result["connection_key"] = (str(backbone), normalized)
    return result


def attach_posthoc_evidence(
    case: Mapping[str, Any],
    evidence: Mapping[str, Any] | pd.Series,
) -> dict[str, Any]:
    """显式附加后验字段；调用方必须位于研究或后验视图。"""

    result = dict(case)
    source = _as_dict(evidence)
    for key in POSTHOC_FIELDS:
        if key in source and not pd.isna(source[key]):
            result[key] = source[key]
    return result


def index_prediction_records(
    frame: pd.DataFrame,
    *,
    backbone: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """按 ``(backbone, normalized_image_key)`` 建立严格唯一索引。"""

    if "image_path" not in frame.columns:
        raise ValueError("预测记录缺少 image_path，无法建立病例连接键。")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        normalized = normalize_image_key(row["image_path"])
        key = (str(backbone), normalized)
        if key in index:
            raise ValueError(f"预测记录存在重复连接键：{key}")
        record = row.to_dict()
        record["backbone"] = str(backbone)
        record["normalized_image_key"] = normalized
        index[key] = record
    return index


def clinical_context_placeholders() -> dict[str, str]:
    return {
        "视力": "未接入",
        "病史": "未接入",
        "OCT": "未接入",
        "治疗记录": "未接入",
        "随访记录": "未接入",
    }


def resolve_case_image(case: Mapping[str, Any]) -> Path | None:
    """只返回病例自身的真实文件；缺失时不使用演示图替代。"""

    raw = case.get("image_path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_file() else None


def filter_case_queue(
    frame: pd.DataFrame,
    *,
    priority: str = "全部",
    search: str = "",
) -> pd.DataFrame:
    """按复核优先级与病例编号过滤队列，保持原排序。"""

    result = frame.copy()
    level = {
        "优先": "high",
        "关注": "medium",
        "常规": "low",
    }.get(priority)
    if level is not None and "pre_review_risk_level" in result.columns:
        result = result[result["pre_review_risk_level"].astype(str) == level]
    query = str(search).strip().lower()
    if query and "case_id" in result.columns:
        result = result[
            result["case_id"].astype(str).str.lower().str.contains(
                query,
                regex=False,
            )
        ]
    return result


def select_review_capacity(
    frame: pd.DataFrame,
    *,
    capacity: int,
    method: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """从已筛选候选池取风险 Top N 或固定种子的随机 N。"""

    if int(capacity) <= 0:
        raise ValueError("capacity 必须大于 0。")
    if method not in {"风险 Top N", "随机抽 N"}:
        raise ValueError("method 必须是“风险 Top N”或“随机抽 N”。")
    selected_n = min(int(capacity), len(frame))
    if selected_n == 0:
        return frame.copy()
    if method == "风险 Top N":
        return frame.head(selected_n).copy()
    return frame.sample(
        n=selected_n,
        replace=False,
        random_state=int(random_seed),
    ).copy()


def initialize_review_capacity(
    state: MutableMapping[str, Any],
    key: str,
    *,
    pool_size: int,
    default_capacity: int = 50,
) -> int:
    """初始化或修正 Streamlit 会话中的复核容量。"""

    if int(pool_size) <= 0:
        raise ValueError("pool_size 必须大于 0。")
    default_value = min(int(default_capacity), int(pool_size))
    current = int(state.get(key, default_value))
    if key not in state or current < 1 or current > int(pool_size):
        state[key] = default_value
        return default_value
    return current


def paginate_case_queue(
    frame: pd.DataFrame,
    *,
    page_number: int,
    page_size: int = 12,
) -> tuple[pd.DataFrame, int, int]:
    """返回当前页、总页数和修正后的 1-based 页码。"""

    if page_size <= 0:
        raise ValueError("page_size 必须大于 0。")
    total_pages = max(1, math.ceil(len(frame) / page_size))
    current = min(max(1, int(page_number)), total_pages)
    start = (current - 1) * page_size
    return frame.iloc[start : start + page_size].copy(), total_pages, current


def _audit_priority_level(case: Mapping[str, Any]) -> str:
    raw = str(case.get("pre_review_risk_level", "medium"))
    return {
        "high": "high",
        "medium": "medium",
        "low": "routine",
        "routine": "routine",
    }.get(raw, "medium")


def probability_values_for_case(case: Mapping[str, Any]) -> list[float] | None:
    numeric = [f"prob_{grade}" for grade in range(5)]
    named = [
        "prob_No DR",
        "prob_Mild DR",
        "prob_Moderate DR",
        "prob_Severe DR",
        "prob_Proliferative DR",
    ]
    columns = numeric if all(column in case for column in numeric) else named
    if not all(column in case for column in columns):
        return None
    return [float(case[column]) for column in columns]


def _derived_metrics(case: Mapping[str, Any]) -> dict[str, float | None]:
    values = probability_values_for_case(case)
    if values is None:
        return {
            "confidence": _float_or_none(case.get("confidence")),
            "margin": _float_or_none(case.get("margin")),
            "entropy_norm": _float_or_none(case.get("entropy_norm")),
            "severe_mass": _float_or_none(case.get("severe_prob_mass")),
            "expected_grade": _float_or_none(case.get("expected_grade")),
            "expected_gap": _float_or_none(case.get("expected_gap")),
        }
    order = sorted(values, reverse=True)
    pred_grade = int(case.get("pred_grade", max(range(5), key=values.__getitem__)))
    safe = [max(value, 1e-12) for value in values]
    expected_grade = sum(index * value for index, value in enumerate(values))
    return {
        "confidence": order[0],
        "margin": order[0] - order[1],
        "entropy_norm": -sum(
            value * math.log(safe_value)
            for value, safe_value in zip(values, safe)
        )
        / math.log(5),
        "severe_mass": values[3] + values[4],
        "expected_grade": expected_grade,
        "expected_gap": expected_grade - pred_grade,
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, pattern: str = ".3f") -> str:
    return "未记录" if value is None else format(value, pattern)


@st.dialog("病例复核详情", width="large")
def render_case_detail_dialog(
    case: Mapping[str, Any],
    *,
    display_mode: str = "临床展示",
    posthoc: Mapping[str, Any] | None = None,
    peer_predictions: pd.DataFrame | None = None,
) -> None:
    """展示当前模型记录；默认不接入后验标签。"""

    payload = dict(case)
    if posthoc is not None and display_mode == "研究审计":
        payload = attach_posthoc_evidence(payload, posthoc)
    audit_priority = _audit_priority_level(payload)
    reasons = translate_risk_reasons(payload.get("risk_reasons"))
    image = resolve_case_image(payload)
    probabilities = probability_values_for_case(payload)
    clinical_summary = summarize_clinical_display(
        pred_grade=int(payload.get("pred_grade", 0)),
        probabilities=probabilities,
        severe_probability_mass=_float_or_none(payload.get("severe_prob_mass")),
        review_priority=audit_priority,
    )

    if image is not None:
        image_col, summary_col = st.columns([0.9, 1.1], gap="large")
        with image_col:
            st.image(str(image), use_container_width=True)
            st.caption(f"图像：{image.name}")
    else:
        image_col, summary_col = st.columns([0.9, 1.1], gap="large")
        with image_col:
            render_empty_state(
                "服务器图像未找到",
                "当前详情只展示对应模型记录，不会替换为其他演示图像。",
            )
    with summary_col:
        render_case_card(
            case_id=str(payload.get("case_id", payload.get("normalized_image_key", ""))),
            clinical_summary=clinical_summary,
            model_context=str(
                payload.get(
                    "backbone_display_name",
                    payload.get("backbone", "未记录"),
                )
            ),
            reasons=reasons,
        )

    with st.expander("查看模型输出依据", expanded=False):
        if probabilities is not None:
            pred_grade = int(
                payload.get("pred_grade", max(range(5), key=probabilities.__getitem__))
            )
            render_probability_profile(
                GRADE_LABELS,
                probabilities,
                pred_grade=pred_grade,
            )
        metrics = _derived_metrics(payload)
        columns = st.columns(3, gap="small")
        with columns[0]:
            metric_card("置信度", _fmt(metrics["confidence"]))
        with columns[1]:
            metric_card("Top1-Top2 间隔", _fmt(metrics["margin"]))
        with columns[2]:
            metric_card("归一化熵", _fmt(metrics["entropy_norm"]))
        columns = st.columns(3, gap="small")
        with columns[0]:
            metric_card(
                "重症类别概率和",
                _fmt(metrics["severe_mass"], ".1%"),
                "3级与4级概率之和",
                accent="amber",
            )
        with columns[1]:
            metric_card("期望等级", _fmt(metrics["expected_grade"], ".2f"))
        with columns[2]:
            metric_card("期望等级差", _fmt(metrics["expected_gap"], "+.2f"))

    section_header(
        "医生复核关注点",
        "以下为人工核对清单，系统没有自动识别这些临床征象。",
    )
    st.markdown(
        "- 图像质量是否足以判读，是否需要补拍。\n"
        "- 是否存在出血、渗出、新生血管等需要关注的重症线索。\n"
        "- 黄斑区改变是否需要结合 OCT 或进一步检查。\n"
        "- 是否需要二读、调整复核优先级或按科室流程转诊。"
    )

    section_header("尚未接入的信息")
    context_columns = st.columns(5, gap="small")
    for column, (name, status) in zip(
        context_columns,
        clinical_context_placeholders().items(),
    ):
        with column:
            metric_card(name, status, "当前公共数据未提供", accent="baseline")

    render_boundary(
        "当前为公共数据集上的模型输出审计记录。图像级结果不能替代患者级病史、"
        "视力、OCT、医生检查和真实工作流判断。"
    )

    if display_mode == "研究审计":
        with st.expander("研究审计字段与后验信息", expanded=False):
            st.json(payload)
        if peer_predictions is not None and not peer_predictions.empty:
            with st.expander("同一图像的六模型预测对照", expanded=False):
                st.dataframe(
                    peer_predictions,
                    hide_index=True,
                    use_container_width=True,
                )
