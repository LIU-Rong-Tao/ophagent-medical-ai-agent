from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.disagreement_review_prioritization import (
    SCORE_COLUMN,
    disagreement_only,
    fit_full_development_priority_and_predict,
    harmful_conflict_target,
    nested_group_oof_priority_scores,
)


def _fixture(n_cases: int = 100) -> tuple[pd.DataFrame, np.ndarray]:
    truth = np.arange(n_cases) % 5
    harmful = np.arange(n_cases) % 2 == 0
    scout = np.where(harmful, truth, (truth + 1) % 5)
    review = np.where(harmful, (truth + 1) % 5, truth)
    both_wrong = np.arange(n_cases) % 10 == 0
    scout[both_wrong] = (truth[both_wrong] + 1) % 5
    review[both_wrong] = (truth[both_wrong] + 2) % 5
    introduced = harmful & (~both_wrong)
    corrected = (~harmful) & (~both_wrong)

    def probabilities(prediction: np.ndarray) -> np.ndarray:
        values = np.full((n_cases, 5), 0.075)
        values[np.arange(n_cases), prediction] = 0.7
        return values

    scout_probabilities = probabilities(scout)
    review_probabilities = probabilities(review)
    frame = pd.DataFrame(
        {
            "case_id": [f"case-{index:03d}" for index in range(n_cases)],
            "resampling_group_id": [
                f"group-{index:03d}" for index in range(n_cases)
            ],
            "scout_pred": scout,
            "expert_pred": review,
            "both_correct": False,
            "corrected": corrected,
            "introduced": introduced,
            "both_wrong": both_wrong,
            "dangerous_introduced": introduced & (truth >= 3),
        }
    )
    for index in range(5):
        frame[f"scout_prob_{index}"] = scout_probabilities[:, index]
        frame[f"expert_prob_{index}"] = review_probabilities[:, index]
    rng = np.random.default_rng(19)
    embeddings = rng.normal(size=(n_cases, 12)).astype(np.float32)
    embeddings[:, 0] += harmful.astype(float)
    return frame, embeddings


def test_disagreement_filter_and_target_contract() -> None:
    cases, _ = _fixture()
    assert len(disagreement_only(cases)) == len(cases)
    target = harmful_conflict_target(cases)
    np.testing.assert_array_equal(
        target,
        (cases["introduced"] | cases["both_wrong"]).astype(int),
    )
    agreeing = cases.copy()
    agreeing.loc[0, "expert_pred"] = agreeing.loc[0, "scout_pred"]
    assert len(disagreement_only(agreeing)) == len(cases) - 1
    agreeing.loc[1, "both_correct"] = True
    with pytest.raises(ValueError, match="cannot have both models correct"):
        disagreement_only(agreeing)


def test_priority_scores_are_finite_and_development_only() -> None:
    cases, embeddings = _fixture()
    oof = nested_group_oof_priority_scores(
        cases,
        embeddings,
        n_folds=5,
        pca_components=6,
        minimum_events=10,
        salt="unit-priority",
    )
    assert oof[SCORE_COLUMN].between(0, 1).all()
    assert not oof["retrospective_outcome_used_for_fit"].any()
    _, target = fit_full_development_priority_and_predict(
        cases.iloc[:80].reset_index(drop=True),
        embeddings[:80],
        cases.iloc[80:].reset_index(drop=True),
        embeddings[80:],
        n_folds=4,
        pca_components=6,
        minimum_events=10,
        salt="unit-priority-full",
    )
    assert len(target) == 20
    assert target[SCORE_COLUMN].between(0, 1).all()
