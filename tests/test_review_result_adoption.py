from __future__ import annotations

import numpy as np
import pandas as pd

from app.review_result_adoption import (
    MODEL_FEATURE_COLUMNS,
    baseline_actions,
    build_cross_fitted_features,
    build_static_features,
    fit_full_development_adoption_and_predict,
    learned_adoption_actions,
    nested_group_oof_adoption_predictions,
)


def _fixture(n_cases: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    outcomes = np.arange(n_cases) % 4
    truth = np.arange(n_cases) % 5
    scout = truth.copy()
    review = truth.copy()
    scout[outcomes == 1] = (truth[outcomes == 1] + 1) % 5
    review[outcomes == 2] = (truth[outcomes == 2] + 1) % 5
    scout[outcomes == 3] = (truth[outcomes == 3] + 1) % 5
    review[outcomes == 3] = (truth[outcomes == 3] + 2) % 5

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
            "y_true": truth,
            "scout_pred": scout,
            "expert_pred": review,
            "both_correct": outcomes == 0,
            "corrected": outcomes == 1,
            "introduced": outcomes == 2,
            "both_wrong": outcomes == 3,
        }
    )
    for index in range(5):
        frame[f"scout_prob_{index}"] = scout_probabilities[:, index]
        frame[f"expert_prob_{index}"] = review_probabilities[:, index]
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(n_cases, 16)).astype(np.float32)
    embeddings[:, :4] += np.eye(4)[outcomes]
    return frame, embeddings


def test_static_and_cross_fitted_features_are_finite() -> None:
    cases, _ = _fixture()
    static = build_static_features(cases)
    assert "y_true" not in static
    assert "case_id" not in static
    features = build_cross_fitted_features(
        cases,
        n_folds=4,
        salt="unit-profile",
    )
    assert tuple(features.columns) == MODEL_FEATURE_COLUMNS
    assert np.isfinite(features.to_numpy()).all()


def test_nested_adoption_model_is_development_only() -> None:
    cases, embeddings = _fixture()
    oof = nested_group_oof_adoption_predictions(
        cases,
        embeddings,
        n_folds=4,
        pca_components=8,
        minimum_events=10,
        salt="unit-oof",
    )
    probability_columns = [
        "probability_both_correct",
        "probability_corrected",
        "probability_introduced",
        "probability_both_wrong",
    ]
    np.testing.assert_allclose(
        oof[probability_columns].sum(axis=1),
        1.0,
        atol=1e-6,
    )
    assert not oof["retrospective_outcome_used_for_fit"].any()

    _, target = fit_full_development_adoption_and_predict(
        cases.iloc[:60].reset_index(drop=True),
        embeddings[:60],
        cases.iloc[60:].reset_index(drop=True),
        embeddings[60:],
        n_folds=3,
        pca_components=8,
        minimum_events=10,
        salt="unit-full",
    )
    assert len(target) == 20
    assert not target["retrospective_outcome_used_for_fit"].any()


def test_actions_respect_exact_human_review_budget() -> None:
    cases, embeddings = _fixture()
    predictions = nested_group_oof_adoption_predictions(
        cases,
        embeddings,
        n_folds=4,
        pca_components=8,
        minimum_events=10,
        salt="unit-actions",
    )
    actions = learned_adoption_actions(
        cases,
        predictions,
        human_review_fraction=0.2,
    )
    assert set(actions) <= {
        "KEEP_SCOUT",
        "ADOPT_REVIEW_RESULT",
        "HUMAN_REVIEW",
    }
    assert int((actions == "HUMAN_REVIEW").sum()) == 16
    for policy in (
        "keep_scout",
        "always_adopt_review",
        "higher_confidence",
        "soft_vote",
    ):
        baseline, final = baseline_actions(
            cases,
            policy=policy,
            human_review_fraction=0.2,
        )
        assert int((baseline == "HUMAN_REVIEW").sum()) == 16
        assert len(final) == len(cases)
