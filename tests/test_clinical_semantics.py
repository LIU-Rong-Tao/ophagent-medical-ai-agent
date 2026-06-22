import pytest

from app.clinical_semantics import summarize_clinical_display
from app.ui import build_case_card_html


def test_severe_prediction_remains_clinically_prominent_when_audit_is_routine():
    summary = summarize_clinical_display(
        pred_grade=4,
        probabilities=[0.01, 0.01, 0.03, 0.10, 0.85],
        review_priority="routine",
    )

    assert summary.predicted_grade_label == "PDR（4级）"
    assert summary.predicted_severity_band == "模型预测涉及重症等级"
    assert "人工重点确认" in summary.clinical_message
    assert summary.audit_priority_label == "常规输出复核"
    assert "输出异常信号" in summary.audit_priority_message
    assert summary.has_predicted_severe_grade is True

    combined_text = " ".join(
        [
            summary.predicted_severity_band,
            summary.clinical_message,
            summary.audit_priority_label,
            summary.audit_priority_message,
        ]
    )
    assert "普通病例" not in combined_text
    assert "无需重点查看" not in combined_text


def test_nonsevere_prediction_with_severe_mass_uses_possible_underestimation_wording():
    summary = summarize_clinical_display(
        pred_grade=2,
        probabilities=[0.05, 0.10, 0.50, 0.20, 0.15],
        review_priority="high",
    )

    assert summary.predicted_grade_label == "Moderate DR（2级）"
    assert summary.predicted_severity_band == "模型预测未达重症等级"
    assert summary.audit_priority_label == "优先输出复核"
    assert summary.has_possible_underestimation_signal is True
    assert "疑似低估信号" in summary.audit_priority_message
    assert "发生低估" not in summary.audit_priority_message


def test_stable_mild_prediction_keeps_disease_severity_and_audit_priority_separate():
    summary = summarize_clinical_display(
        pred_grade=1,
        probabilities=[0.10, 0.80, 0.07, 0.02, 0.01],
        review_priority="routine",
    )

    assert summary.predicted_grade_label == "Mild DR（1级）"
    assert summary.predicted_severity_band == "模型预测未达重症等级"
    assert summary.audit_priority_label == "常规输出复核"
    assert summary.has_predicted_severe_grade is False
    assert summary.has_possible_underestimation_signal is False
    assert "不等同于患者病情轻重" in summary.disclaimer
    assert "不等同于真实临床转诊优先级" in summary.disclaimer


def test_clinical_semantics_rejects_invalid_probability_shape_and_priority():
    with pytest.raises(ValueError, match="五级概率"):
        summarize_clinical_display(
            pred_grade=1,
            probabilities=[0.2, 0.8],
            review_priority="routine",
        )

    with pytest.raises(ValueError, match="review_priority"):
        summarize_clinical_display(
            pred_grade=1,
            probabilities=[0.10, 0.80, 0.07, 0.02, 0.01],
            review_priority="unknown",
        )


def test_clinical_semantics_can_use_existing_severe_mass_without_faking_probabilities():
    summary = summarize_clinical_display(
        pred_grade=2,
        probabilities=None,
        severe_probability_mass=0.30,
        review_priority="high",
    )

    assert summary.has_possible_underestimation_signal is True
    assert "疑似低估信号" in summary.audit_priority_message


def test_case_card_html_displays_severity_and_audit_priority_as_two_dimensions():
    summary = summarize_clinical_display(
        pred_grade=4,
        probabilities=[0.01, 0.01, 0.03, 0.10, 0.85],
        review_priority="routine",
    )

    card_html = build_case_card_html(
        case_id="case-pdr",
        clinical_summary=summary,
        model_context="RETFound",
        reasons=["模型判断较明确"],
    )

    assert "模型预测等级" in card_html
    assert "模型输出复核优先级" in card_html
    assert "PDR（4级）" in card_html
    assert "模型预测涉及重症等级" in card_html
    assert "常规输出复核" in card_html
    assert "常规队列" not in card_html
    assert "不等同于患者病情轻重" in card_html
