"""Prioritize human review within frozen Scout/review-model disagreements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.help_or_harm_benchmark import stable_group_fold
from app.review_result_adoption import (
    MODEL_FEATURE_COLUMNS,
    build_cross_fitted_features,
    build_transfer_features,
)


SCHEMA_VERSION = "ophagent.disagreement_review_prioritization.v0_1"
SCORE_COLUMN = "predicted_harmful_conflict_probability"


def disagreement_only(cases: pd.DataFrame) -> pd.DataFrame:
    """Return the predeclared disagreement cohort without changing row order."""

    required = {"scout_pred", "expert_pred", "introduced", "both_wrong"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Disagreement cases are missing: {missing}")
    result = cases.loc[
        cases["scout_pred"].astype(int).ne(cases["expert_pred"].astype(int))
    ].copy()
    if result.empty:
        raise ValueError("The route has no disagreement cases.")
    if result["both_correct"].astype(bool).any():
        raise ValueError("A disagreement case cannot have both models correct.")
    return result


def harmful_conflict_target(cases: pd.DataFrame) -> np.ndarray:
    return (
        cases["introduced"].astype(bool)
        | cases["both_wrong"].astype(bool)
    ).astype(int).to_numpy()


@dataclass(frozen=True)
class DisagreementPrioritizationModel:
    embedding_scaler: StandardScaler
    embedding_pca: PCA
    classifier: Pipeline
    pca_components: int
    schema_version: str = SCHEMA_VERSION

    def predict_score(
        self,
        features: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> pd.Series:
        projected = self.embedding_pca.transform(
            self.embedding_scaler.transform(embeddings)
        )
        matrix = np.column_stack(
            [
                features.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(
                    dtype=float
                ),
                projected,
            ]
        )
        return pd.Series(
            self.classifier.predict_proba(matrix)[:, 1],
            index=features.index,
            name=SCORE_COLUMN,
        )


def fit_prioritization_model(
    features: pd.DataFrame,
    embeddings: np.ndarray,
    cases: pd.DataFrame,
    *,
    pca_components: int = 8,
    minimum_events: int = 10,
) -> DisagreementPrioritizationModel:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(features):
        raise ValueError("Disagreement embeddings and features do not align.")
    target = harmful_conflict_target(cases)
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives < minimum_events or negatives < minimum_events:
        raise ValueError(
            f"Insufficient harmful/non-harmful events: {positives}/{negatives}"
        )
    components = min(pca_components, len(features) - 1, matrix.shape[1])
    embedding_scaler = StandardScaler()
    standardized = embedding_scaler.fit_transform(matrix)
    embedding_pca = PCA(
        n_components=components,
        whiten=False,
        svd_solver="full",
        random_state=0,
    )
    projected = embedding_pca.fit_transform(standardized)
    combined = np.column_stack(
        [
            features.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float),
            projected,
        ]
    )
    classifier = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=3000,
                    penalty="l2",
                    solver="lbfgs",
                    random_state=0,
                ),
            ),
        ]
    )
    classifier.fit(combined, target)
    return DisagreementPrioritizationModel(
        embedding_scaler=embedding_scaler,
        embedding_pca=embedding_pca,
        classifier=classifier,
        pca_components=components,
    )


def nested_group_oof_priority_scores(
    development: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_folds: int = 5,
    pca_components: int = 8,
    minimum_events: int = 10,
    salt: str = "ophagent-disagreement-priority-v0.1",
) -> pd.DataFrame:
    matrix = np.asarray(embeddings, dtype=np.float32)
    folds = stable_group_fold(
        development["resampling_group_id"].astype(str),
        n_folds=n_folds,
        salt=f"{salt}:outer",
    )
    result = pd.DataFrame(index=development.index)
    result["outer_fold"] = folds
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        training = development.loc[~held_out]
        validation = development.loc[held_out]
        inner_folds = min(
            max(2, n_folds - 1),
            training["resampling_group_id"].nunique(),
        )
        training_features = build_cross_fitted_features(
            training,
            n_folds=int(inner_folds),
            salt=f"{salt}:outer-{int(fold)}:inner",
        )
        validation_features = build_transfer_features(training, validation)
        model = fit_prioritization_model(
            training_features,
            matrix[~held_out],
            training,
            pca_components=pca_components,
            minimum_events=2,
        )
        result.loc[held_out, SCORE_COLUMN] = model.predict_score(
            validation_features,
            matrix[held_out],
        ).to_numpy()
    if result[SCORE_COLUMN].isna().any():
        raise ValueError("At least one disagreement case lacks an OOF score.")
    result["prediction_source"] = "nested_group_oof_development_only"
    result["retrospective_outcome_used_for_fit"] = False
    return result


def fit_full_development_priority_and_predict(
    development: pd.DataFrame,
    development_embeddings: np.ndarray,
    target: pd.DataFrame,
    target_embeddings: np.ndarray,
    *,
    n_folds: int = 5,
    pca_components: int = 8,
    minimum_events: int = 10,
    salt: str = "ophagent-disagreement-priority-v0.1",
) -> tuple[DisagreementPrioritizationModel, pd.DataFrame]:
    training_features = build_cross_fitted_features(
        development,
        n_folds=n_folds,
        salt=f"{salt}:full",
    )
    target_features = build_transfer_features(development, target)
    model = fit_prioritization_model(
        training_features,
        development_embeddings,
        development,
        pca_components=pca_components,
        minimum_events=minimum_events,
    )
    predictions = pd.DataFrame(
        model.predict_score(target_features, target_embeddings)
    )
    predictions["prediction_source"] = "full_development_only"
    predictions["retrospective_outcome_used_for_fit"] = False
    return model, predictions
