"""把 DR 模型预测等级与输出复核优先级拆成独立展示语义。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


SEVERE_MASS_SIGNAL_THRESHOLD = 0.15

GRADE_LABELS = (
    "No DR（0级）",
    "Mild DR（1级）",
    "Moderate DR（2级）",
    "Severe DR（3级）",
    "PDR（4级）",
)

AUDIT_PRIORITY_LABELS = {
    "high": "优先输出复核",
    "medium": "关注输出复核",
    "routine": "常规输出复核",
}


@dataclass(frozen=True)
class ClinicalDisplaySummary:
    """临床首屏需要的稳定展示字段。"""

    predicted_grade_label: str
    predicted_severity_band: str
    clinical_message: str
    clinical_severity_level: str
    audit_priority_label: str
    audit_priority_message: str
    audit_priority_level: str
    has_predicted_severe_grade: bool
    has_possible_underestimation_signal: bool
    disclaimer: str


def summarize_clinical_display(
    pred_grade: int,
    probabilities: Sequence[float] | None,
    review_priority: str,
    *,
    severe_probability_mass: float | None = None,
) -> ClinicalDisplaySummary:
    """生成病情等级与模型输出可疑度相互独立的展示摘要。"""

    grade = int(pred_grade)
    if grade < 0 or grade >= len(GRADE_LABELS):
        raise ValueError("pred_grade 必须位于 0 到 4。")

    if probabilities is not None:
        values = [float(value) for value in probabilities]
        if len(values) != 5:
            raise ValueError("DR 临床展示需要五级概率。")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("五级概率必须是有限的非负数。")
        severe_mass = values[3] + values[4]
    elif severe_probability_mass is not None:
        severe_mass = float(severe_probability_mass)
        if not math.isfinite(severe_mass) or not 0.0 <= severe_mass <= 1.0:
            raise ValueError("重症概率和必须位于 0 到 1。")
    else:
        raise ValueError("需要五级概率或已有重症概率和。")

    priority = str(review_priority).strip().lower()
    if priority not in AUDIT_PRIORITY_LABELS:
        raise ValueError("review_priority 必须是 high、medium 或 routine。")

    has_predicted_severe_grade = grade >= 3
    has_possible_underestimation_signal = (
        not has_predicted_severe_grade
        and severe_mass >= SEVERE_MASS_SIGNAL_THRESHOLD
    )

    if has_predicted_severe_grade:
        severity_band = "模型预测涉及重症等级"
        clinical_message = "建议人工重点确认，并结合科室既有流程处理。"
        clinical_level = "severe"
    elif grade == 2:
        severity_band = "模型预测未达重症等级"
        clinical_message = "仍需结合眼底图像及临床信息人工判断。"
        clinical_level = "moderate"
    else:
        severity_band = "模型预测未达重症等级"
        clinical_message = "仍需结合眼底图像及临床信息人工审核。"
        clinical_level = "nonsevere"

    if priority == "high" and has_possible_underestimation_signal:
        audit_message = (
            "模型最终预测较轻，但输出分布仍保留较高重症概率，"
            "存在疑似低估信号，建议人工优先复核。"
        )
    elif priority == "high":
        audit_message = "模型输出触发主要复核信号，建议人工优先检查。"
    elif priority == "medium":
        audit_message = "模型输出存在边界接近或概率分散，建议提前复核。"
    elif has_predicted_severe_grade:
        audit_message = (
            "模型对当前重症等级判断较明确，未触发主要输出异常信号；"
            "仍需人工确认。"
        )
    else:
        audit_message = "当前输出未触发主要风险信号，仍需人工审核。"

    return ClinicalDisplaySummary(
        predicted_grade_label=GRADE_LABELS[grade],
        predicted_severity_band=severity_band,
        clinical_message=clinical_message,
        clinical_severity_level=clinical_level,
        audit_priority_label=AUDIT_PRIORITY_LABELS[priority],
        audit_priority_message=audit_message,
        audit_priority_level=priority,
        has_predicted_severe_grade=has_predicted_severe_grade,
        has_possible_underestimation_signal=has_possible_underestimation_signal,
        disclaimer=(
            "输出复核优先级表示模型输出是否可疑，"
            "不等同于患者病情轻重，也不等同于真实临床转诊优先级。"
        ),
    )
