"""Leakage-controlled pre-consultation Help-or-Harm policies.

The module contains only reusable modelling and policy primitives.  It does
not read experiment directories, grant route eligibility, execute an Expert,
or alter the frozen v1.1 qualification contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.help_or_harm_benchmark import (
    ConsultationPolicyBaselineV1_1,
    apply_expert_history_profile,
    build_cross_fitted_expert_history,
    build_cross_fitted_reference_js,
    fit_expert_history_profile,
    jensen_shannon_divergence,
    rank_top_budget,
    stable_group_fold,
)


SCHEMA_VERSION = "ophagent.selective_consultation.v0_1"
SCOUT_PROBABILITY_COLUMNS = tuple(f"scout_prob_{index}" for index in range(5))
METHOD_FEATURE_COLUMNS = (
    "scout_entropy",
    "scout_margin",
    "scout_severe_probability_mass",
    "expert_history_corrected_rate",
    "expert_history_introduced_rate",
    "scout_reference_js_divergence",
)
MODEL_TARGETS = ("corrected", "introduced")
LEARNED_POLICY_NAMES = (
    "help_only_logistic",
    "harm_only_logistic",
    "dual_logistic_harm_screened_help",
)
NON_ORACLE_POLICY_NAMES = (
    "entropy",
    "margin",
    *LEARNED_POLICY_NAMES,
)
FORBIDDEN_MODEL_INPUT_COLUMNS = (
    "dataset_id",
    "expert_pred",
    "expert_prob_0",
    "expert_prob_1",
    "expert_prob_2",
    "expert_prob_3",
    "expert_prob_4",
    "current_case_expert_prediction",
    "current_case_expert_probability",
    "expert_embedding",
    "patient_group_id",
    "case_id",
    "image_sha256",
    "image_path",
    "private_path",
)


class InsufficientOutcomeEvents(ValueError):
    """Raised when a route cannot support the predeclared binary models."""


def _validate_case_frame(
    cases: pd.DataFrame,
    *,
    require_outcomes: bool = True,
) -> None:
    required = {
        "case_id",
        "resampling_group_id",
        "scout_pred",
        "scout_confidence",
        *SCOUT_PROBABILITY_COLUMNS,
        *METHOD_FEATURE_COLUMNS[:3],
    }
    if require_outcomes:
        required.update(MODEL_TARGETS)
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Selective consultation cases are missing: {missing}")
    if cases.empty:
        raise ValueError("Selective consultation requires at least one case.")
    if cases["case_id"].astype(str).duplicated().any():
        raise ValueError("A route/split frame contains duplicate case identifiers.")
    if cases["resampling_group_id"].fillna("").astype(str).eq("").any():
        raise ValueError("Every case requires a non-empty resampling group.")


def _validate_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(METHOD_FEATURE_COLUMNS) - set(features.columns))
    if missing:
        raise ValueError(f"Method feature frame is missing: {missing}")
    result = features.loc[:, list(METHOD_FEATURE_COLUMNS)].astype(float).copy()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Method feature frame contains a non-finite value.")
    if set(FORBIDDEN_MODEL_INPUT_COLUMNS).intersection(result.columns):
        raise ValueError("A forbidden field entered the controller feature frame.")
    return result


def build_cross_fitted_method_features(
    development: pd.DataFrame,
    *,
    n_folds: int,
    salt: str,
) -> pd.DataFrame:
    """Build development features without using the current resampling group."""

    _validate_case_frame(development)
    history = build_cross_fitted_expert_history(
        development,
        group_column="resampling_group_id",
        n_folds=n_folds,
        alpha=0.5,
        salt=f"{salt}:history",
    )
    reference = build_cross_fitted_reference_js(
        development,
        group_column="resampling_group_id",
        probability_columns=SCOUT_PROBABILITY_COLUMNS,
        n_folds=n_folds,
        salt=f"{salt}:reference",
    )
    result = development.copy()
    for column in (
        "expert_history_corrected_rate",
        "expert_history_introduced_rate",
    ):
        result[column] = history[column]
    result["scout_reference_js_divergence"] = reference[
        "scout_reference_js_divergence"
    ]
    return _validate_feature_frame(result)


def build_transfer_method_features(
    development: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Apply development-only route history and reference to target cases."""

    _validate_case_frame(development)
    _validate_case_frame(target, require_outcomes=False)
    profile = fit_expert_history_profile(development, alpha=0.5)
    history = apply_expert_history_profile(target, profile)
    reference = development.loc[
        :, list(SCOUT_PROBABILITY_COLUMNS)
    ].to_numpy(dtype=float).mean(axis=0)
    result = target.copy()
    for column in (
        "expert_history_corrected_rate",
        "expert_history_introduced_rate",
    ):
        result[column] = history[column]
    result["scout_reference_js_divergence"] = jensen_shannon_divergence(
        target.loc[:, list(SCOUT_PROBABILITY_COLUMNS)].to_numpy(dtype=float),
        reference,
    )
    return _validate_feature_frame(result)


def _fit_binary_logistic(
    features: pd.DataFrame,
    target: Sequence[int] | np.ndarray,
    *,
    minimum_events: int,
) -> Pipeline:
    values = np.asarray(target, dtype=int)
    positives = int(values.sum())
    negatives = int(len(values) - positives)
    if positives < minimum_events or negatives < minimum_events:
        raise InsufficientOutcomeEvents(
            "Binary route model requires at least "
            f"{minimum_events} positive and negative cases; "
            f"observed {positives} and {negatives}."
        )
    model = Pipeline(
        steps=[
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
    model.fit(_validate_feature_frame(features), values)
    return model


@dataclass(frozen=True)
class DualLogisticConsultationModel:
    """Two independent pre-consultation outcome models."""

    help_model: Pipeline
    harm_model: Pipeline
    feature_columns: tuple[str, ...] = METHOD_FEATURE_COLUMNS
    schema_version: str = SCHEMA_VERSION

    @property
    def may_grant_eligibility(self) -> bool:
        return False

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        values = _validate_feature_frame(features)
        return pd.DataFrame(
            {
                "predicted_corrected_probability": self.help_model.predict_proba(
                    values
                )[:, 1],
                "predicted_introduced_probability": self.harm_model.predict_proba(
                    values
                )[:, 1],
            },
            index=features.index,
        )


def fit_dual_logistic_model(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    minimum_events: int = 2,
) -> DualLogisticConsultationModel:
    """Fit fixed L2 logistic models without hyperparameter search."""

    missing = sorted(set(MODEL_TARGETS) - set(outcomes.columns))
    if missing:
        raise ValueError(f"Outcome frame is missing: {missing}")
    return DualLogisticConsultationModel(
        help_model=_fit_binary_logistic(
            features,
            outcomes["corrected"].astype(int),
            minimum_events=minimum_events,
        ),
        harm_model=_fit_binary_logistic(
            features,
            outcomes["introduced"].astype(int),
            minimum_events=minimum_events,
        ),
    )


def nested_group_oof_predictions(
    development: pd.DataFrame,
    *,
    n_folds: int = 5,
    minimum_route_events: int = 10,
    salt: str = "ophagent-selective-consultation-v0.1",
) -> pd.DataFrame:
    """Generate patient/image-group OOF predictions with nested profile fitting.

    The Expert history used to construct an outer-fold validation feature is
    fitted on outer-fold training cases only.  Training features are themselves
    cross-fitted inside the outer training partition.
    """

    _validate_case_frame(development)
    groups = development["resampling_group_id"].astype(str)
    if groups.nunique() < n_folds:
        raise ValueError("Fewer resampling groups than requested outer folds.")
    for target in MODEL_TARGETS:
        positives = int(development[target].astype(bool).sum())
        negatives = int(len(development) - positives)
        if positives < minimum_route_events or negatives < minimum_route_events:
            raise InsufficientOutcomeEvents(
                f"{target} has {positives} events and {negatives} non-events; "
                f"the route-level minimum is {minimum_route_events}."
            )
    folds = stable_group_fold(groups, n_folds=n_folds, salt=f"{salt}:outer")
    result = pd.DataFrame(index=development.index)
    result["outer_fold"] = folds
    result["outer_training_case_count"] = 0
    for fold in sorted(np.unique(folds)):
        held_out = folds == fold
        training = development.loc[~held_out].copy()
        validation = development.loc[held_out].copy()
        if validation.empty:
            continue
        inner_fold_count = min(
            max(2, n_folds - 1),
            int(training["resampling_group_id"].nunique()),
        )
        training_features = build_cross_fitted_method_features(
            training,
            n_folds=inner_fold_count,
            salt=f"{salt}:outer-{int(fold)}:inner",
        )
        validation_features = build_transfer_method_features(
            training,
            validation,
        )
        model = fit_dual_logistic_model(
            training_features,
            training.loc[:, list(MODEL_TARGETS)],
            minimum_events=2,
        )
        predictions = model.predict(validation_features)
        result.loc[held_out, predictions.columns] = predictions
        result.loc[held_out, "outer_training_case_count"] = len(training)
    probability_columns = [
        "predicted_corrected_probability",
        "predicted_introduced_probability",
    ]
    if result[probability_columns].isna().any().any():
        raise ValueError("At least one case did not receive an OOF prediction.")
    result["prediction_source"] = "nested_group_oof_development_only"
    result["current_case_expert_output_used"] = False
    result["retrospective_outcome_used_for_fit"] = False
    return result


def fit_full_development_and_predict(
    development: pd.DataFrame,
    target: pd.DataFrame,
    *,
    n_folds: int = 5,
    minimum_route_events: int = 10,
    salt: str = "ophagent-selective-consultation-v0.1",
) -> tuple[DualLogisticConsultationModel, pd.DataFrame]:
    """Fit on cross-fitted development features and predict a frozen target."""

    _validate_case_frame(development)
    _validate_case_frame(target, require_outcomes=False)
    for outcome in MODEL_TARGETS:
        events = int(development[outcome].astype(bool).sum())
        if events < minimum_route_events:
            raise InsufficientOutcomeEvents(
                f"{outcome} has only {events} development events."
            )
    training_features = build_cross_fitted_method_features(
        development,
        n_folds=n_folds,
        salt=f"{salt}:full",
    )
    target_features = build_transfer_method_features(development, target)
    model = fit_dual_logistic_model(
        training_features,
        development.loc[:, list(MODEL_TARGETS)],
        minimum_events=minimum_route_events,
    )
    predictions = model.predict(target_features)
    predictions["prediction_source"] = "full_development_only"
    predictions["current_case_expert_output_used"] = False
    predictions["retrospective_outcome_used_for_fit"] = False
    return model, predictions


def _stable_selected(order: np.ndarray, *, n_cases: int, selected_n: int) -> np.ndarray:
    selected = np.zeros(n_cases, dtype=bool)
    selected[np.asarray(order[:selected_n], dtype=int)] = True
    return selected


def select_consultations(
    cases: pd.DataFrame,
    *,
    policy: str,
    budget: float,
    predictions: pd.DataFrame | None = None,
    safe_pool_multiplier: float = 2.0,
    v1_1_baseline: ConsultationPolicyBaselineV1_1 | None = None,
) -> np.ndarray:
    """Select an exact call budget using a deterministic policy.

    The dual policy first retains the lowest predicted-harm pool and only then
    ranks that pool by predicted help.  It never combines the two outcomes into
    a post-hoc weighted score.
    """

    if not 0.0 <= float(budget) <= 1.0:
        raise ValueError("budget must be within [0, 1].")
    if safe_pool_multiplier < 1.0:
        raise ValueError("safe_pool_multiplier must be at least one.")
    identifiers = cases["case_id"].astype(str).to_numpy()
    n_cases = len(cases)
    selected_n = min(n_cases, max(0, int(round(n_cases * float(budget)))))
    if policy == "scout_only":
        return np.zeros(n_cases, dtype=bool)
    if policy == "always_expert":
        return np.ones(n_cases, dtype=bool)
    if policy == "entropy":
        return rank_top_budget(
            case_ids=identifiers,
            scores=cases["scout_entropy"].to_numpy(dtype=float),
            budget=budget,
        )
    if policy == "margin":
        return rank_top_budget(
            case_ids=identifiers,
            scores=1.0 - cases["scout_margin"].to_numpy(dtype=float),
            budget=budget,
        )
    if policy == "consultation_policy_baseline_v1_1":
        if v1_1_baseline is None:
            raise ValueError("The frozen v1.1 policy definition is required.")
        return rank_top_budget(
            case_ids=identifiers,
            scores=v1_1_baseline.scores(cases),
            budget=budget,
        )
    if policy == "oracle":
        priority = np.where(
            cases["corrected"].astype(bool).to_numpy(),
            2,
            np.where(cases["introduced"].astype(bool).to_numpy(), 0, 1),
        )
        order = np.lexsort((identifiers, -priority))
        return _stable_selected(
            order,
            n_cases=n_cases,
            selected_n=selected_n,
        )
    if policy not in LEARNED_POLICY_NAMES:
        raise ValueError(f"Unknown selective consultation policy: {policy}")
    if predictions is None:
        raise ValueError(f"{policy} requires pre-consultation predictions.")
    if len(predictions) != n_cases:
        raise ValueError("Prediction and case frames must have equal length.")
    help_score = predictions[
        "predicted_corrected_probability"
    ].to_numpy(dtype=float)
    harm_score = predictions[
        "predicted_introduced_probability"
    ].to_numpy(dtype=float)
    if not np.isfinite(help_score).all() or not np.isfinite(harm_score).all():
        raise ValueError("A learned policy score is non-finite.")
    if policy == "help_only_logistic":
        order = np.lexsort((identifiers, -help_score))
    elif policy == "harm_only_logistic":
        order = np.lexsort((identifiers, harm_score))
    else:
        if selected_n == 0:
            return np.zeros(n_cases, dtype=bool)
        safe_pool_n = min(
            n_cases,
            max(
                selected_n,
                int(math.ceil(selected_n * safe_pool_multiplier)),
            ),
        )
        harm_order = np.lexsort((identifiers, harm_score))
        safe_indices = harm_order[:safe_pool_n]
        safe_help_order = np.lexsort(
            (identifiers[safe_indices], -help_score[safe_indices])
        )
        order = safe_indices[safe_help_order]
    return _stable_selected(order, n_cases=n_cases, selected_n=selected_n)


def safe_binary_metric(
    target: Sequence[int] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    *,
    metric: str,
) -> float:
    values = np.asarray(target, dtype=int)
    scores = np.asarray(score, dtype=float)
    if np.unique(values).size != 2:
        return float("nan")
    if metric == "auroc":
        return float(roc_auc_score(values, scores))
    if metric == "auprc":
        return float(average_precision_score(values, scores))
    raise ValueError(f"Unknown binary metric: {metric}")


def model_discrimination_rows(
    cases: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return operational and conditional outcome discrimination metrics."""

    scout_correct = cases["scout_pred"].astype(int).eq(
        cases["y_true"].astype(int)
    )
    specifications = (
        (
            "corrected",
            "predicted_corrected_probability",
            "all_cases",
            np.ones(len(cases), dtype=bool),
        ),
        (
            "corrected",
            "predicted_corrected_probability",
            "scout_wrong_only",
            (~scout_correct).to_numpy(),
        ),
        (
            "introduced",
            "predicted_introduced_probability",
            "all_cases",
            np.ones(len(cases), dtype=bool),
        ),
        (
            "introduced",
            "predicted_introduced_probability",
            "scout_correct_only",
            scout_correct.to_numpy(),
        ),
    )
    rows: list[dict[str, Any]] = []
    for outcome, score_column, cohort, eligible in specifications:
        target = cases.loc[eligible, outcome].astype(int).to_numpy()
        score = predictions.loc[eligible, score_column].to_numpy(dtype=float)
        rows.append(
            {
                "outcome": outcome,
                "evaluation_cohort": cohort,
                "n_cases": int(len(target)),
                "events": int(target.sum()),
                "prevalence": float(target.mean()) if len(target) else np.nan,
                "auroc": safe_binary_metric(target, score, metric="auroc"),
                "auprc": safe_binary_metric(target, score, metric="auprc"),
            }
        )
    return rows


def paired_cluster_bootstrap_difference(
    cases: pd.DataFrame,
    *,
    method_selected: Sequence[bool] | np.ndarray,
    baseline_selected: Sequence[bool] | np.ndarray,
    group_column: str = "resampling_group_id",
    replicates: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute paired cluster-bootstrap CIs for selection count differences."""

    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    if group_column not in cases:
        raise ValueError(f"Missing bootstrap grouping column: {group_column}")
    method = np.asarray(method_selected, dtype=bool)
    baseline = np.asarray(baseline_selected, dtype=bool)
    if method.shape != baseline.shape or method.shape != (len(cases),):
        raise ValueError("Bootstrap selection arrays must match the case frame.")
    corrected = cases["corrected"].astype(bool).to_numpy()
    introduced = cases["introduced"].astype(bool).to_numpy()
    contributions = pd.DataFrame(
        {
            "group": cases[group_column].astype(str),
            "corrected_difference": (
                method.astype(int) - baseline.astype(int)
            )
            * corrected.astype(int),
            "introduced_difference": (
                method.astype(int) - baseline.astype(int)
            )
            * introduced.astype(int),
        }
    )
    by_group = contributions.groupby("group", sort=True)[
        ["corrected_difference", "introduced_difference"]
    ].sum()
    by_group["net_difference"] = (
        by_group["corrected_difference"] - by_group["introduced_difference"]
    )
    values = by_group.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(values),
        size=(replicates, len(values)),
    )
    bootstrapped = values[samples].sum(axis=1)
    result: dict[str, float] = {}
    for index, column in enumerate(by_group.columns):
        result[f"{column}_ci_lower"] = float(
            np.quantile(bootstrapped[:, index], 0.025)
        )
        result[f"{column}_ci_upper"] = float(
            np.quantile(bootstrapped[:, index], 0.975)
        )
    result["bootstrap_replicates"] = float(replicates)
    result["bootstrap_group_count"] = float(len(values))
    return result
