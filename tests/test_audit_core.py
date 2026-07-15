
import pandas as pd
import pytest

from app.audit_core import (
    class_specific_miss_ranking,
    compute_confidence,
    compute_entropy,
    compute_expected_gap_for_dr,
    compute_gated_severe_mass_for_dr,
    compute_margin,
    compute_top2,
    evaluate_topk_capture,
    infer_prob_columns,
    summarize_dr_review_priority,
    translate_risk_reasons,
    validate_probability_columns,
)


def probability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "pred_class": ["No DR", "Mild DR", "Severe DR", "No DR"],
            "true_class": ["No DR", "Severe DR", "Severe DR", "PDR"],
            "prob_No DR": [0.70, 0.10, 0.05, 0.35],
            "prob_Mild DR": [0.10, 0.45, 0.05, 0.15],
            "prob_Moderate DR": [0.10, 0.20, 0.10, 0.30],
            "prob_Severe DR": [0.05, 0.20, 0.70, 0.15],
            "prob_PDR": [0.05, 0.05, 0.10, 0.05],
        }
    )


def test_infer_probability_columns_preserves_dataframe_order():
    df = probability_frame()
    assert infer_prob_columns(df) == [
        "prob_No DR",
        "prob_Mild DR",
        "prob_Moderate DR",
        "prob_Severe DR",
        "prob_PDR",
    ]


def test_probability_validation_rejects_range_sum_and_unknown_prediction():
    df = probability_frame()
    validate_probability_columns(df, infer_prob_columns(df))

    out_of_range = df.copy()
    out_of_range.loc[0, "prob_No DR"] = 1.2
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        validate_probability_columns(out_of_range)

    bad_sum = df.copy()
    bad_sum.loc[0, "prob_No DR"] = 0.5
    with pytest.raises(ValueError, match="概率和"):
        validate_probability_columns(bad_sum)

    unknown = df.copy()
    unknown.loc[0, "pred_class"] = "Unknown"
    with pytest.raises(ValueError, match="pred_class"):
        validate_probability_columns(unknown)


def test_confidence_margin_entropy_and_top2_are_computed_from_probabilities():
    df = probability_frame()
    confidence = compute_confidence(df)
    margin = compute_margin(df)
    entropy = compute_entropy(df)
    top2 = compute_top2(df)

    assert confidence.iloc[0] == pytest.approx(0.70)
    assert margin.iloc[0] == pytest.approx(0.60)
    assert 0.0 <= entropy.iloc[0] <= 1.0
    assert top2.loc[0, "top2_class"] == "Mild DR"
    assert top2.loc[0, "top2_probability"] == pytest.approx(0.10)


def test_dr_specific_scores_match_their_operational_definitions():
    df = pd.DataFrame(
        {
            "prob_0": [0.35, 0.05],
            "prob_1": [0.15, 0.05],
            "prob_2": [0.30, 0.10],
            "prob_3": [0.15, 0.70],
            "prob_4": [0.05, 0.10],
            "pred_grade": [0, 3],
        }
    )

    expected = compute_expected_gap_for_dr(df)
    gated = compute_gated_severe_mass_for_dr(df)

    assert expected.loc[0, "expected_grade"] == pytest.approx(1.40)
    assert expected.loc[0, "expected_gap"] == pytest.approx(1.40)
    assert gated.iloc[0] == pytest.approx(0.20)
    assert gated.iloc[1] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="prob_4"):
        compute_expected_gap_for_dr(df.drop(columns=["prob_4"]))


def test_topk_capture_uses_ceil_budget_and_reports_residual_events():
    df = pd.DataFrame(
        {
            "score": [0.9, 0.8, 0.7, 0.1],
            "event": [True, False, True, False],
        }
    )
    result = evaluate_topk_capture(df, "event", budgets=(0.2, 0.5))

    first = result.iloc[0]
    assert first["top_k"] == 1
    assert first["captured_event"] == 1
    assert first["event_recall"] == pytest.approx(0.5)
    assert first["residual_event_count"] == 1
    assert first["random_recall"] == pytest.approx(0.25)


def test_unlabeled_frame_cannot_produce_capture_metrics():
    unlabeled = probability_frame().drop(columns=["true_class"])
    with pytest.raises(KeyError, match="event"):
        evaluate_topk_capture(unlabeled, "event")


def test_class_specific_miss_ranking_uses_target_probability_and_optional_labels():
    df = probability_frame()
    ranked = class_specific_miss_ranking(df, "Severe DR")

    assert (ranked["pred_class"] != "Severe DR").all()
    assert ranked.iloc[0]["image_id"] == "b"
    assert ranked.iloc[0]["target_probability"] == pytest.approx(0.20)
    assert ranked.iloc[0]["target_event"]

    unlabeled = df.drop(columns=["true_class"])
    ranked_unlabeled = class_specific_miss_ranking(unlabeled, "Severe DR")
    assert "target_event" not in ranked_unlabeled.columns


def test_review_priority_summary_is_operational_not_a_clinical_diagnosis():
    high = summarize_dr_review_priority(
        pred_grade=2,
        probabilities=[0.10, 0.10, 0.45, 0.25, 0.10],
    )
    routine = summarize_dr_review_priority(
        pred_grade=0,
        probabilities=[0.95, 0.03, 0.01, 0.005, 0.005],
    )

    assert high["level"] == "high"
    assert high["label"] == "优先复核"
    assert "重症类别" in high["summary"]
    assert routine["level"] == "routine"
    assert routine["label"] == "常规队列"
    assert "不代表病例安全" in routine["summary"]


def test_risk_reason_codes_are_translated_for_clinical_display():
    translated = translate_risk_reasons(
        "low_margin_boundary;high_entropy;second_choice_more_severe"
    )
    assert translated == [
        "前两类概率接近",
        "多个类别概率分散",
        "第二候选等级更重",
    ]
