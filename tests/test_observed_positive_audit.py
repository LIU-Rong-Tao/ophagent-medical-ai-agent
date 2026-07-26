import pandas as pd
import pytest

from app.generic_result_audit import run_observed_positive_audit


def test_observed_positive_audit_does_not_infer_negative_labels():
    predictions = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "observed_label_ids": ["0", "1|2"],
            "prob_0": [0.7, 0.8],
            "prob_1": [0.2, 0.1],
            "prob_2": [0.1, 0.1],
        }
    )

    audit = run_observed_positive_audit(
        predictions,
        probability_columns=["prob_0", "prob_1", "prob_2"],
        high_confidence_threshold=0.75,
    )

    assert audit.summary["unobserved_classes_treated_as_negative"] is False
    assert audit.summary["observed_positive_hit_at_1"] == pytest.approx(0.5)
    assert audit.summary["high_confidence_observed_label_inconsistency_count"] == 1
    assert audit.case_scores.loc[1, "observed_positive_probability_mass"] == pytest.approx(
        0.2
    )


def test_observed_positive_audit_rejects_invalid_probabilities():
    predictions = pd.DataFrame(
        {
            "case_id": ["a"],
            "observed_label_ids": ["0"],
            "prob_0": [0.8],
            "prob_1": [0.8],
        }
    )

    with pytest.raises(ValueError, match="非法概率矩阵"):
        run_observed_positive_audit(
            predictions,
            probability_columns=["prob_0", "prob_1"],
        )
