"""Leakage-controlled safety adoption after a review model has run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.help_or_harm_benchmark import rank_top_budget, stable_group_fold


SCHEMA_VERSION = "ophagent.review_result_adoption.v0_1"
SCOUT_PROBABILITY_COLUMNS = tuple(f"scout_prob_{index}" for index in range(5))
REVIEW_PROBABILITY_COLUMNS = tuple(f"expert_prob_{index}" for index in range(5))
OUTCOME_NAMES = ("both_correct", "corrected", "introduced", "both_wrong")
PROFILE_COLUMNS = (
    "transition_corrected_rate",
    "transition_introduced_rate",
    "transition_both_wrong_rate",
    "transition_support_log1p",
)
STATIC_FEATURE_COLUMNS = (
    *SCOUT_PROBABILITY_COLUMNS,
    *REVIEW_PROBABILITY_COLUMNS,
    "scout_confidence",
    "review_confidence",
    "scout_entropy",
    "review_entropy",
    "scout_margin",
    "review_margin",
    "confidence_delta",
    "probability_l1_distance",
    "probability_js_divergence",
    "prediction_disagreement",
    "grade_delta",
    "absolute_grade_delta",
    "scout_expected_grade",
    "review_expected_grade",
    "expected_grade_delta",
    "scout_severe_probability_mass",
    "review_severe_probability_mass",
    "severe_probability_mass_delta",
)
MODEL_FEATURE_COLUMNS = (*STATIC_FEATURE_COLUMNS, *PROFILE_COLUMNS)


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=1)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    left_safe = np.clip(left, 1e-12, 1.0)
    right_safe = np.clip(right, 1e-12, 1.0)
    midpoint_safe = np.clip(midpoint, 1e-12, 1.0)
    return 0.5 * (
        (left_safe * np.log(left_safe / midpoint_safe)).sum(axis=1)
        + (right_safe * np.log(right_safe / midpoint_safe)).sum(axis=1)
    )


def build_static_features(cases: pd.DataFrame) -> pd.DataFrame:
    """Build legal post-review features without labels or identifiers."""

    required = {
        "scout_pred",
        "expert_pred",
        *SCOUT_PROBABILITY_COLUMNS,
        *REVIEW_PROBABILITY_COLUMNS,
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Post-review cases are missing: {missing}")
    scout = cases.loc[:, list(SCOUT_PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    review = cases.loc[:, list(REVIEW_PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(scout).all() or not np.isfinite(review).all():
        raise ValueError("Post-review probabilities must be finite.")
    if not np.allclose(scout.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Scout probabilities do not sum to one.")
    if not np.allclose(review.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Review probabilities do not sum to one.")
    scout_pred = cases["scout_pred"].to_numpy(dtype=int)
    review_pred = cases["expert_pred"].to_numpy(dtype=int)
    grade_axis = np.arange(scout.shape[1], dtype=float)
    output = pd.DataFrame(index=cases.index)
    for index, column in enumerate(SCOUT_PROBABILITY_COLUMNS):
        output[column] = scout[:, index]
    for index, column in enumerate(REVIEW_PROBABILITY_COLUMNS):
        output[column] = review[:, index]
    output["scout_confidence"] = scout.max(axis=1)
    output["review_confidence"] = review.max(axis=1)
    output["scout_entropy"] = _entropy(scout)
    output["review_entropy"] = _entropy(review)
    output["scout_margin"] = _margin(scout)
    output["review_margin"] = _margin(review)
    output["confidence_delta"] = (
        output["review_confidence"] - output["scout_confidence"]
    )
    output["probability_l1_distance"] = np.abs(scout - review).sum(axis=1)
    output["probability_js_divergence"] = _js_divergence(scout, review)
    output["prediction_disagreement"] = scout_pred != review_pred
    output["grade_delta"] = review_pred - scout_pred
    output["absolute_grade_delta"] = np.abs(output["grade_delta"])
    output["scout_expected_grade"] = scout @ grade_axis
    output["review_expected_grade"] = review @ grade_axis
    output["expected_grade_delta"] = (
        output["review_expected_grade"] - output["scout_expected_grade"]
    )
    output["scout_severe_probability_mass"] = scout[:, 3:].sum(axis=1)
    output["review_severe_probability_mass"] = review[:, 3:].sum(axis=1)
    output["severe_probability_mass_delta"] = (
        output["review_severe_probability_mass"]
        - output["scout_severe_probability_mass"]
    )
    return output.loc[:, list(STATIC_FEATURE_COLUMNS)].astype(float)


def _outcome_code(cases: pd.DataFrame) -> np.ndarray:
    required = set(OUTCOME_NAMES)
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Post-review outcomes are missing: {missing}")
    matrix = cases.loc[:, OUTCOME_NAMES].astype(bool).to_numpy()
    if not np.all(matrix.sum(axis=1) == 1):
        raise ValueError("Every case must have exactly one outcome state.")
    return matrix.argmax(axis=1).astype(int)


@dataclass(frozen=True)
class TransitionProfile:
    table: pd.DataFrame
    global_rates: np.ndarray
    alpha: float


def fit_transition_profile(
    development: pd.DataFrame,
    *,
    alpha: float = 0.5,
) -> TransitionProfile:
    """Fit a smoothed route profile for Scout→review grade transitions."""

    labels = _outcome_code(development)
    frame = pd.DataFrame(
        {
            "scout_pred": development["scout_pred"].astype(int),
            "review_pred": development["expert_pred"].astype(int),
            "outcome": labels,
        }
    )
    counts = (
        frame.groupby(["scout_pred", "review_pred", "outcome"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(len(OUTCOME_NAMES)), fill_value=0)
    )
    support = counts.sum(axis=1)
    global_counts = np.bincount(labels, minlength=len(OUTCOME_NAMES)).astype(
        float
    )
    global_rates = (global_counts + alpha) / (
        global_counts.sum() + alpha * len(OUTCOME_NAMES)
    )
    smoothed = counts.to_numpy(dtype=float) + alpha * global_rates
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    table = pd.DataFrame(
        smoothed,
        index=counts.index,
        columns=[f"outcome_rate_{index}" for index in range(len(OUTCOME_NAMES))],
    )
    table["support"] = support.to_numpy(dtype=float)
    return TransitionProfile(table=table, global_rates=global_rates, alpha=alpha)


def apply_transition_profile(
    cases: pd.DataFrame,
    profile: TransitionProfile,
) -> pd.DataFrame:
    keys = list(
        zip(
            cases["scout_pred"].astype(int),
            cases["expert_pred"].astype(int),
            strict=True,
        )
    )
    rows = []
    for key in keys:
        if key in profile.table.index:
            row = profile.table.loc[key]
            rates = row[
                [f"outcome_rate_{index}" for index in range(4)]
            ].to_numpy(dtype=float)
            support = float(row["support"])
        else:
            rates = profile.global_rates
            support = 0.0
        rows.append(
            {
                "transition_corrected_rate": rates[1],
                "transition_introduced_rate": rates[2],
                "transition_both_wrong_rate": rates[3],
                "transition_support_log1p": np.log1p(support),
            }
        )
    return pd.DataFrame(rows, index=cases.index)


def build_cross_fitted_features(
    development: pd.DataFrame,
    *,
    n_folds: int,
    salt: str,
) -> pd.DataFrame:
    groups = development["resampling_group_id"].astype(str)
    folds = stable_group_fold(groups, n_folds=n_folds, salt=salt)
    result = build_static_features(development)
    for column in PROFILE_COLUMNS:
        result[column] = np.nan
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        profile = fit_transition_profile(development.loc[~held_out])
        transferred = apply_transition_profile(
            development.loc[held_out],
            profile,
        )
        result.loc[held_out, list(PROFILE_COLUMNS)] = transferred.to_numpy()
    if result.loc[:, list(MODEL_FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("At least one cross-fitted feature is missing.")
    return result.loc[:, list(MODEL_FEATURE_COLUMNS)].astype(float)


def build_transfer_features(
    development: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    result = build_static_features(target)
    profile = fit_transition_profile(development)
    transferred = apply_transition_profile(target, profile)
    for column in PROFILE_COLUMNS:
        result[column] = transferred[column]
    return result.loc[:, list(MODEL_FEATURE_COLUMNS)].astype(float)


@dataclass(frozen=True)
class ReviewAdoptionModel:
    embedding_scaler: StandardScaler
    embedding_pca: PCA
    classifier: Pipeline
    pca_components: int
    schema_version: str = SCHEMA_VERSION

    def predict_proba(
        self,
        features: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> pd.DataFrame:
        projected = self.embedding_pca.transform(
            self.embedding_scaler.transform(embeddings)
        )
        matrix = np.column_stack(
            [
                features.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(),
                projected,
            ]
        )
        probabilities = self.classifier.predict_proba(matrix)
        output = pd.DataFrame(index=features.index)
        for index, name in enumerate(OUTCOME_NAMES):
            output[f"probability_{name}"] = probabilities[:, index]
        return output


def fit_adoption_model(
    features: pd.DataFrame,
    embeddings: np.ndarray,
    cases: pd.DataFrame,
    *,
    pca_components: int = 32,
    minimum_events: int = 10,
) -> ReviewAdoptionModel:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape[0] != len(features) or matrix.ndim != 2:
        raise ValueError("Embedding and feature rows do not align.")
    labels = _outcome_code(cases)
    counts = np.bincount(labels, minlength=len(OUTCOME_NAMES))
    if np.any(counts < minimum_events):
        raise ValueError(f"Insufficient outcome events: {counts.tolist()}")
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
            features.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(),
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
    classifier.fit(combined, labels)
    return ReviewAdoptionModel(
        embedding_scaler=embedding_scaler,
        embedding_pca=embedding_pca,
        classifier=classifier,
        pca_components=components,
    )


def nested_group_oof_adoption_predictions(
    development: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    n_folds: int = 5,
    pca_components: int = 32,
    minimum_events: int = 10,
    salt: str = "ophagent-review-adoption-v0.1",
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
        model = fit_adoption_model(
            training_features,
            matrix[~held_out],
            training,
            pca_components=pca_components,
            minimum_events=2,
        )
        predictions = model.predict_proba(
            validation_features,
            matrix[held_out],
        )
        result.loc[held_out, predictions.columns] = predictions.to_numpy()
    probability_columns = [
        f"probability_{name}" for name in OUTCOME_NAMES
    ]
    if result[probability_columns].isna().any().any():
        raise ValueError("At least one adoption OOF score is missing.")
    result["prediction_source"] = "nested_group_oof_development_only"
    result["retrospective_outcome_used_for_fit"] = False
    return result


def fit_full_development_adoption_and_predict(
    development: pd.DataFrame,
    development_embeddings: np.ndarray,
    target: pd.DataFrame,
    target_embeddings: np.ndarray,
    *,
    n_folds: int = 5,
    pca_components: int = 32,
    minimum_events: int = 10,
    salt: str = "ophagent-review-adoption-v0.1",
) -> tuple[ReviewAdoptionModel, pd.DataFrame]:
    training_features = build_cross_fitted_features(
        development,
        n_folds=n_folds,
        salt=f"{salt}:full",
    )
    target_features = build_transfer_features(development, target)
    model = fit_adoption_model(
        training_features,
        development_embeddings,
        development,
        pca_components=pca_components,
        minimum_events=minimum_events,
    )
    predictions = model.predict_proba(target_features, target_embeddings)
    predictions["prediction_source"] = "full_development_only"
    predictions["retrospective_outcome_used_for_fit"] = False
    return model, predictions


def learned_adoption_actions(
    cases: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    human_review_fraction: float,
) -> np.ndarray:
    """Minimize predicted automatic error, deferring the riskiest cases."""

    corrected = predictions["probability_corrected"].to_numpy(dtype=float)
    introduced = predictions["probability_introduced"].to_numpy(dtype=float)
    both_wrong = predictions["probability_both_wrong"].to_numpy(dtype=float)
    scout_error = corrected + both_wrong
    review_error = introduced + both_wrong
    adopt = review_error < scout_error
    automatic_error = np.minimum(scout_error, review_error)
    review = rank_top_budget(
        case_ids=cases["case_id"].astype(str).to_numpy(),
        scores=automatic_error,
        budget=human_review_fraction,
    )
    actions = np.where(adopt, "ADOPT_REVIEW_RESULT", "KEEP_SCOUT").astype(
        object
    )
    actions[review] = "HUMAN_REVIEW"
    return actions


def baseline_actions(
    cases: pd.DataFrame,
    *,
    policy: str,
    human_review_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return baseline actions and non-review predictions at the same budget."""

    scout = cases.loc[:, list(SCOUT_PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    review = cases.loc[:, list(REVIEW_PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    scout_pred = cases["scout_pred"].to_numpy(dtype=int)
    review_pred = cases["expert_pred"].to_numpy(dtype=int)
    if policy == "keep_scout":
        final_pred = scout_pred
        confidence = scout.max(axis=1)
        action = np.full(len(cases), "KEEP_SCOUT", dtype=object)
    elif policy == "always_adopt_review":
        final_pred = review_pred
        confidence = review.max(axis=1)
        action = np.full(len(cases), "ADOPT_REVIEW_RESULT", dtype=object)
    elif policy == "higher_confidence":
        adopt = review.max(axis=1) > scout.max(axis=1)
        final_pred = np.where(adopt, review_pred, scout_pred)
        confidence = np.maximum(review.max(axis=1), scout.max(axis=1))
        action = np.where(
            adopt,
            "ADOPT_REVIEW_RESULT",
            "KEEP_SCOUT",
        ).astype(object)
    elif policy == "soft_vote":
        averaged = 0.5 * (scout + review)
        final_pred = averaged.argmax(axis=1)
        confidence = averaged.max(axis=1)
        action = np.full(len(cases), "SOFT_VOTE", dtype=object)
    else:
        raise ValueError(f"Unknown adoption baseline: {policy}")
    review_selected = rank_top_budget(
        case_ids=cases["case_id"].astype(str).to_numpy(),
        scores=1.0 - confidence,
        budget=human_review_fraction,
    )
    action[review_selected] = "HUMAN_REVIEW"
    return action, final_pred
