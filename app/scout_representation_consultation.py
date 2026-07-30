"""Leakage-controlled consultation models using frozen Scout representations.

The current-case Expert output is deliberately absent.  Every fitted
transformation is learned inside the development training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.help_or_harm_benchmark import stable_group_fold
from app.selective_consultation import (
    METHOD_FEATURE_COLUMNS,
    MODEL_TARGETS,
    InsufficientOutcomeEvents,
    build_cross_fitted_method_features,
    build_transfer_method_features,
)


SCHEMA_VERSION = "ophagent.scout_representation_consultation.v0_1"
PREDICTION_COLUMNS = (
    "predicted_corrected_probability",
    "predicted_introduced_probability",
)


def align_embeddings(
    cases: pd.DataFrame,
    *,
    embedding_case_ids: Sequence[str],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Align a Scout embedding asset to a route case frame by unique case ID."""

    if "case_id" not in cases:
        raise ValueError("Case frame is missing case_id.")
    identifiers = pd.Series(embedding_case_ids, dtype=str)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(identifiers):
        raise ValueError("Embedding rows and identifiers do not align.")
    if identifiers.duplicated().any():
        raise ValueError("Embedding asset contains duplicate case identifiers.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding asset contains a non-finite value.")
    lookup = {case_id: index for index, case_id in enumerate(identifiers)}
    requested = cases["case_id"].astype(str).tolist()
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"Embedding asset is missing case IDs: {missing[:5]}")
    return matrix[np.asarray([lookup[case_id] for case_id in requested])]


def _validate_inputs(
    cases: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    require_outcomes: bool,
) -> np.ndarray:
    required = {"case_id", "resampling_group_id", *METHOD_FEATURE_COLUMNS[:3]}
    if require_outcomes:
        required.update(MODEL_TARGETS)
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Representation cases are missing: {missing}")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(cases):
        raise ValueError("Case and embedding rows must have equal length.")
    if matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Scout embedding matrix is invalid.")
    if cases["case_id"].astype(str).duplicated().any():
        raise ValueError("Route case identifiers must be unique.")
    return matrix


def _event_check(
    outcomes: pd.DataFrame,
    *,
    minimum_events: int,
) -> None:
    for outcome in MODEL_TARGETS:
        values = outcomes[outcome].astype(int)
        positives = int(values.sum())
        negatives = int(len(values) - positives)
        if positives < minimum_events or negatives < minimum_events:
            raise InsufficientOutcomeEvents(
                f"{outcome} requires {minimum_events} positive and negative "
                f"events; observed {positives} and {negatives}."
            )


@dataclass(frozen=True)
class ScoutRepresentationModel:
    """Fixed PCA plus separate L2 logistic help/harm models."""

    embedding_scaler: StandardScaler
    embedding_pca: PCA
    help_model: Pipeline
    harm_model: Pipeline
    pca_components: int
    schema_version: str = SCHEMA_VERSION

    @property
    def may_grant_eligibility(self) -> bool:
        return False

    def predict(
        self,
        method_features: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        projected = self.embedding_pca.transform(
            self.embedding_scaler.transform(embeddings)
        )
        features = np.column_stack(
            [
                method_features.loc[:, list(METHOD_FEATURE_COLUMNS)].to_numpy(
                    dtype=float
                ),
                projected,
            ]
        )
        return pd.DataFrame(
            {
                PREDICTION_COLUMNS[0]: self.help_model.predict_proba(features)[
                    :, 1
                ],
                PREDICTION_COLUMNS[1]: self.harm_model.predict_proba(features)[
                    :, 1
                ],
            },
            index=method_features.index,
        )


def fit_representation_model(
    method_features: pd.DataFrame,
    embeddings: np.ndarray,
    outcomes: pd.DataFrame,
    *,
    pca_components: int = 32,
    minimum_events: int = 2,
) -> ScoutRepresentationModel:
    """Fit the predeclared fixed representation model without search."""

    _event_check(outcomes, minimum_events=minimum_events)
    matrix = np.asarray(embeddings, dtype=np.float32)
    components = min(pca_components, matrix.shape[0] - 1, matrix.shape[1])
    if components < 2:
        raise ValueError("Too few cases or dimensions for representation PCA.")
    embedding_scaler = StandardScaler()
    standardized = embedding_scaler.fit_transform(matrix)
    embedding_pca = PCA(
        n_components=components,
        whiten=False,
        svd_solver="full",
        random_state=0,
    )
    projected = embedding_pca.fit_transform(standardized)
    features = np.column_stack(
        [
            method_features.loc[:, list(METHOD_FEATURE_COLUMNS)].to_numpy(
                dtype=float
            ),
            projected,
        ]
    )

    def fit_one(target: str) -> Pipeline:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=1.0,
                        class_weight=None,
                        max_iter=2000,
                        penalty="l2",
                        solver="lbfgs",
                        random_state=0,
                    ),
                ),
            ]
        )
        model.fit(features, outcomes[target].astype(int).to_numpy())
        return model

    return ScoutRepresentationModel(
        embedding_scaler=embedding_scaler,
        embedding_pca=embedding_pca,
        help_model=fit_one("corrected"),
        harm_model=fit_one("introduced"),
        pca_components=components,
    )


def nested_group_oof_representation_predictions(
    development: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_folds: int = 5,
    pca_components: int = 32,
    minimum_route_events: int = 10,
    salt: str = "ophagent-scout-representation-v0.1",
) -> pd.DataFrame:
    """Produce nested group OOF predictions with fold-internal PCA/profiles."""

    matrix = _validate_inputs(development, embeddings, require_outcomes=True)
    _event_check(development, minimum_events=minimum_route_events)
    groups = development["resampling_group_id"].astype(str)
    folds = stable_group_fold(groups, n_folds=n_folds, salt=f"{salt}:outer")
    result = pd.DataFrame(index=development.index)
    result["outer_fold"] = folds
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        training = development.loc[~held_out].copy()
        validation = development.loc[held_out].copy()
        training_matrix = matrix[~held_out]
        validation_matrix = matrix[held_out]
        inner_folds = min(
            max(2, n_folds - 1),
            int(training["resampling_group_id"].nunique()),
        )
        training_features = build_cross_fitted_method_features(
            training,
            n_folds=inner_folds,
            salt=f"{salt}:outer-{int(fold)}:inner",
        )
        validation_features = build_transfer_method_features(
            training,
            validation,
        )
        model = fit_representation_model(
            training_features,
            training_matrix,
            training.loc[:, list(MODEL_TARGETS)],
            pca_components=pca_components,
            minimum_events=2,
        )
        predictions = model.predict(validation_features, validation_matrix)
        result.loc[held_out, list(PREDICTION_COLUMNS)] = predictions.to_numpy()
    if result.loc[:, list(PREDICTION_COLUMNS)].isna().any().any():
        raise ValueError("At least one case lacks an OOF representation score.")
    result["prediction_source"] = (
        "nested_group_oof_development_scout_representation"
    )
    result["current_case_expert_output_used"] = False
    result["retrospective_outcome_used_for_fit"] = False
    return result


def fit_full_development_representation_and_predict(
    development: pd.DataFrame,
    development_embeddings: np.ndarray,
    target: pd.DataFrame,
    target_embeddings: np.ndarray,
    *,
    n_folds: int = 5,
    pca_components: int = 32,
    minimum_route_events: int = 10,
    salt: str = "ophagent-scout-representation-v0.1",
) -> tuple[ScoutRepresentationModel, pd.DataFrame]:
    """Fit using development-only features and score a frozen target."""

    development_matrix = _validate_inputs(
        development,
        development_embeddings,
        require_outcomes=True,
    )
    target_matrix = _validate_inputs(
        target,
        target_embeddings,
        require_outcomes=False,
    )
    _event_check(development, minimum_events=minimum_route_events)
    training_features = build_cross_fitted_method_features(
        development,
        n_folds=n_folds,
        salt=f"{salt}:full",
    )
    target_features = build_transfer_method_features(development, target)
    model = fit_representation_model(
        training_features,
        development_matrix,
        development.loc[:, list(MODEL_TARGETS)],
        pca_components=pca_components,
        minimum_events=minimum_route_events,
    )
    predictions = model.predict(target_features, target_matrix)
    predictions["prediction_source"] = (
        "full_development_scout_representation"
    )
    predictions["current_case_expert_output_used"] = False
    predictions["retrospective_outcome_used_for_fit"] = False
    return model, predictions
