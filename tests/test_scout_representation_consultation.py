from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.scout_representation_consultation import (
    align_embeddings,
    fit_full_development_representation_and_predict,
    nested_group_oof_representation_predictions,
)


def _cases(n_cases: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    probabilities = rng.dirichlet(np.ones(5), size=n_cases)
    frame = pd.DataFrame(
        {
            "case_id": [f"case-{index:03d}" for index in range(n_cases)],
            "resampling_group_id": [
                f"group-{index:03d}" for index in range(n_cases)
            ],
            "scout_pred": probabilities.argmax(axis=1),
            "scout_confidence": probabilities.max(axis=1),
            "scout_entropy": -(
                probabilities * np.log(probabilities + 1e-12)
            ).sum(axis=1),
            "scout_margin": np.sort(probabilities, axis=1)[:, -1]
            - np.sort(probabilities, axis=1)[:, -2],
            "scout_severe_probability_mass": probabilities[:, 3:].sum(axis=1),
            "corrected": [int(index % 5 == 0) for index in range(n_cases)],
            "introduced": [
                int(index % 7 == 1) for index in range(n_cases)
            ],
        }
    )
    for index in range(5):
        frame[f"scout_prob_{index}"] = probabilities[:, index]
    embeddings = rng.normal(size=(n_cases, 12)).astype(np.float32)
    embeddings[:, 0] += frame["corrected"].to_numpy() * 0.8
    embeddings[:, 1] += frame["introduced"].to_numpy() * 0.8
    return frame, embeddings


def test_align_embeddings_is_case_id_exact() -> None:
    cases, embeddings = _cases(12)
    reversed_ids = cases["case_id"].iloc[::-1].tolist()
    reversed_embeddings = embeddings[::-1]
    aligned = align_embeddings(
        cases,
        embedding_case_ids=reversed_ids,
        embeddings=reversed_embeddings,
    )
    np.testing.assert_allclose(aligned, embeddings)


def test_align_embeddings_rejects_missing_or_duplicate_ids() -> None:
    cases, embeddings = _cases(12)
    with pytest.raises(ValueError, match="duplicate"):
        align_embeddings(
            cases,
            embedding_case_ids=["same"] * len(cases),
            embeddings=embeddings,
        )
    with pytest.raises(ValueError, match="missing"):
        align_embeddings(
            cases,
            embedding_case_ids=cases["case_id"].iloc[:-1],
            embeddings=embeddings[:-1],
        )


def test_representation_predictions_are_finite_and_expert_free() -> None:
    cases, embeddings = _cases()
    oof = nested_group_oof_representation_predictions(
        cases,
        embeddings,
        n_folds=3,
        pca_components=6,
        minimum_route_events=5,
        salt="unit-test",
    )
    assert len(oof) == len(cases)
    assert np.isfinite(
        oof[
            [
                "predicted_corrected_probability",
                "predicted_introduced_probability",
            ]
        ].to_numpy()
    ).all()
    assert not oof["current_case_expert_output_used"].any()
    assert not oof["retrospective_outcome_used_for_fit"].any()

    model, target = fit_full_development_representation_and_predict(
        cases.iloc[:45].reset_index(drop=True),
        embeddings[:45],
        cases.iloc[45:].reset_index(drop=True),
        embeddings[45:],
        n_folds=3,
        pca_components=6,
        minimum_route_events=5,
        salt="unit-test-full",
    )
    assert len(target) == 15
    assert model.may_grant_eligibility is False
    assert not target["current_case_expert_output_used"].any()
