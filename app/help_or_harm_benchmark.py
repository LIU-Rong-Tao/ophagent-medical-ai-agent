"""Case-level Help-or-Harm research contracts and deterministic baselines.

The module deliberately separates the non-replaceable safety/qualification
boundary from replaceable consultation-ranking policies.  It never reads
prediction assets or experiment directories; data loading belongs to the
benchmark build script and the existing Model Hub adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "ophagent.help_or_harm_case_contract.v0_1"
CLASS_ORDER = (0, 1, 2, 3, 4)
OUTCOME_COLUMNS = (
    "corrected",
    "introduced",
    "both_correct",
    "both_wrong",
    "dangerous_introduced",
)
LEGAL_FEATURE_COLUMNS = (
    "scout_prob_0",
    "scout_prob_1",
    "scout_prob_2",
    "scout_prob_3",
    "scout_prob_4",
    "scout_confidence",
    "scout_entropy",
    "scout_margin",
    "scout_severe_probability_mass",
    "expert_history_corrected_rate",
    "expert_history_introduced_rate",
    "expert_history_net",
    "expert_history_support",
    "scout_reference_js_divergence",
)
FORBIDDEN_FEATURE_COLUMNS = (
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
    "patient_id",
    "image_path",
    "private_path",
)


def _validated_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(CLASS_ORDER):
        raise ValueError("Help-or-Harm DR features require an n×5 probability matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Probability matrix contains a non-finite value.")
    if (values < -1e-8).any() or (values > 1.0 + 1e-8).any():
        raise ValueError("Probability values must be within [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one.")
    return values


def build_scout_feature_frame(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> pd.DataFrame:
    """Derive legal, pre-consultation features from one Scout output only."""

    values = _validated_probabilities(probabilities)
    ordered = np.sort(values, axis=1)
    clipped = np.clip(values, np.finfo(float).tiny, 1.0)
    entropy = -(values * np.log(clipped)).sum(axis=1) / math.log(len(CLASS_ORDER))
    result = pd.DataFrame(
        {
            f"scout_prob_{index}": values[:, index]
            for index in range(len(CLASS_ORDER))
        }
    )
    result["scout_confidence"] = values.max(axis=1)
    result["scout_entropy"] = entropy
    result["scout_margin"] = ordered[:, -1] - ordered[:, -2]
    result["scout_severe_probability_mass"] = values[:, 3] + values[:, 4]
    return result


def compute_case_outcomes(
    *,
    y_true: Sequence[int] | np.ndarray,
    scout_pred: Sequence[int] | np.ndarray,
    expert_pred: Sequence[int] | np.ndarray,
) -> pd.DataFrame:
    """Build mutually exclusive label-defined error outcomes.

    These outcomes are model-error proxies.  They are not clinical harm,
    diagnosis, treatment outcome, or patient-safety endpoints.
    """

    truth = np.asarray(y_true, dtype=int)
    scout = np.asarray(scout_pred, dtype=int)
    expert = np.asarray(expert_pred, dtype=int)
    if truth.shape != scout.shape or truth.shape != expert.shape:
        raise ValueError("Truth, Scout, and Expert arrays must have equal shape.")
    if not (
        np.isin(truth, CLASS_ORDER).all()
        and np.isin(scout, CLASS_ORDER).all()
        and np.isin(expert, CLASS_ORDER).all()
    ):
        raise ValueError("DR labels and predictions must use the fixed class order 0..4.")
    scout_correct = scout == truth
    expert_correct = expert == truth
    scout_dangerous_undergrade = (truth >= 3) & (scout < 3)
    expert_dangerous_undergrade = (truth >= 3) & (expert < 3)
    return pd.DataFrame(
        {
            "corrected": (~scout_correct) & expert_correct,
            "introduced": scout_correct & (~expert_correct),
            "both_correct": scout_correct & expert_correct,
            "both_wrong": (~scout_correct) & (~expert_correct),
            "dangerous_introduced": (
                (~scout_dangerous_undergrade) & expert_dangerous_undergrade
            ),
        }
    )


def stable_group_fold(
    group_ids: Sequence[str],
    *,
    n_folds: int = 5,
    salt: str = "ophagent-help-or-harm-v0.1",
) -> np.ndarray:
    """Assign every patient/image group to one deterministic development fold."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least two.")
    folds: list[int] = []
    for value in group_ids:
        digest = hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).digest()
        folds.append(int.from_bytes(digest[:8], "big") % n_folds)
    return np.asarray(folds, dtype=int)


def deterministic_random_scores(
    case_ids: Sequence[str],
    *,
    salt: str,
) -> np.ndarray:
    """Return reproducible pseudo-random ranking scores without fitting."""

    denominator = float(2**64)
    values: list[float] = []
    for case_id in case_ids:
        digest = hashlib.sha256(f"{salt}\0{case_id}".encode("utf-8")).digest()
        values.append(int.from_bytes(digest[:8], "big") / denominator)
    return np.asarray(values, dtype=float)


def _confidence_boundaries(values: np.ndarray) -> tuple[float, float, float]:
    if not len(values):
        return (0.25, 0.5, 0.75)
    quantiles = np.quantile(values.astype(float), [0.25, 0.5, 0.75])
    return tuple(float(value) for value in quantiles)


def _confidence_strata(
    values: Sequence[float] | np.ndarray,
    boundaries: Sequence[float],
) -> np.ndarray:
    return np.searchsorted(
        np.asarray(boundaries, dtype=float),
        np.asarray(values, dtype=float),
        side="right",
    ).astype(int)


def fit_expert_history_profile(
    development: pd.DataFrame,
    *,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Fit a smoothed route history using development outcomes only."""

    required = {
        "scout_pred",
        "scout_confidence",
        "corrected",
        "introduced",
    }
    missing = sorted(required - set(development.columns))
    if missing:
        raise ValueError(f"Expert history profile is missing columns: {missing}")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    frame = development.copy()
    boundaries = _confidence_boundaries(
        frame["scout_confidence"].to_numpy(dtype=float)
    )
    frame["_confidence_stratum"] = _confidence_strata(
        frame["scout_confidence"],
        boundaries,
    )
    corrected_total = int(frame["corrected"].astype(bool).sum())
    introduced_total = int(frame["introduced"].astype(bool).sum())
    support_total = int(len(frame))
    global_corrected = (corrected_total + alpha) / (support_total + 2 * alpha)
    global_introduced = (introduced_total + alpha) / (support_total + 2 * alpha)
    strata: dict[str, dict[str, float | int]] = {}
    for (grade, confidence_stratum), group in frame.groupby(
        ["scout_pred", "_confidence_stratum"],
        sort=True,
        dropna=False,
    ):
        support = int(len(group))
        corrected = int(group["corrected"].astype(bool).sum())
        introduced = int(group["introduced"].astype(bool).sum())
        strata[f"{int(grade)}:{int(confidence_stratum)}"] = {
            "support": support,
            "corrected_rate": (corrected + alpha) / (support + 2 * alpha),
            "introduced_rate": (introduced + alpha) / (support + 2 * alpha),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "alpha": float(alpha),
        "support": support_total,
        "confidence_boundaries": list(boundaries),
        "global_corrected_rate": float(global_corrected),
        "global_introduced_rate": float(global_introduced),
        "strata": strata,
    }


def apply_expert_history_profile(
    cases: pd.DataFrame,
    profile: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply a fitted history without reading current-case Expert output."""

    required = {"scout_pred", "scout_confidence"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Cases are missing profile input columns: {missing}")
    boundaries = tuple(float(value) for value in profile["confidence_boundaries"])
    strata = _confidence_strata(cases["scout_confidence"], boundaries)
    result_rows: list[dict[str, float | int]] = []
    default_corrected = float(profile["global_corrected_rate"])
    default_introduced = float(profile["global_introduced_rate"])
    profile_strata = dict(profile.get("strata", {}))
    for grade, stratum in zip(cases["scout_pred"], strata, strict=True):
        value = profile_strata.get(f"{int(grade)}:{int(stratum)}")
        if value is None:
            corrected = default_corrected
            introduced = default_introduced
            support = 0
        else:
            corrected = float(value["corrected_rate"])
            introduced = float(value["introduced_rate"])
            support = int(value["support"])
        result_rows.append(
            {
                "expert_history_corrected_rate": corrected,
                "expert_history_introduced_rate": introduced,
                "expert_history_net": corrected - introduced,
                "expert_history_support": support,
            }
        )
    return pd.DataFrame(result_rows, index=cases.index)


def build_cross_fitted_expert_history(
    development: pd.DataFrame,
    *,
    group_column: str,
    n_folds: int = 5,
    alpha: float = 0.5,
    salt: str = "ophagent-help-or-harm-v0.1",
) -> pd.DataFrame:
    """Generate development features from other deterministic folds only."""

    if group_column not in development:
        raise ValueError(f"Missing grouping column: {group_column}")
    frame = development.copy()
    folds = stable_group_fold(
        frame[group_column].fillna("").astype(str),
        n_folds=n_folds,
        salt=salt,
    )
    result = pd.DataFrame(index=frame.index)
    result["profile_fold"] = folds
    result["profile_training_case_count"] = 0
    for fold in range(n_folds):
        held_out = folds == fold
        training = frame.loc[~held_out]
        if training.empty:
            raise ValueError("Cross-fitted profile has an empty training fold.")
        profile = fit_expert_history_profile(training, alpha=alpha)
        applied = apply_expert_history_profile(frame.loc[held_out], profile)
        for column in applied:
            result.loc[held_out, column] = applied[column]
        result.loc[held_out, "profile_training_case_count"] = len(training)
    result["profile_source"] = "other_development_folds_only"
    result["profile_excludes_current_group"] = True
    return result


def jensen_shannon_divergence(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    reference_distribution: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute base-2 JS divergence from a legal unlabeled reference profile."""

    values = _validated_probabilities(probabilities)
    reference = np.asarray(reference_distribution, dtype=float)
    if reference.shape != (len(CLASS_ORDER),):
        raise ValueError("Reference distribution must contain five probabilities.")
    if not np.isfinite(reference).all() or (reference < 0).any():
        raise ValueError("Reference distribution is invalid.")
    reference_sum = float(reference.sum())
    if reference_sum <= 0:
        raise ValueError("Reference distribution must have positive mass.")
    reference = reference / reference_sum
    midpoint = 0.5 * (values + reference[None, :])
    tiny = np.finfo(float).tiny
    left = np.where(
        values > 0,
        values * np.log2(np.clip(values, tiny, None) / midpoint),
        0.0,
    ).sum(axis=1)
    right = np.where(
        reference[None, :] > 0,
        reference[None, :]
        * np.log2(np.clip(reference[None, :], tiny, None) / midpoint),
        0.0,
    ).sum(axis=1)
    return 0.5 * (left + right)


def build_cross_fitted_reference_js(
    development: pd.DataFrame,
    *,
    group_column: str,
    probability_columns: Sequence[str],
    n_folds: int = 5,
    salt: str = "ophagent-help-or-harm-v0.1",
) -> pd.DataFrame:
    """Build an unlabeled distribution-shift feature without self-fold leakage."""

    if group_column not in development:
        raise ValueError(f"Missing grouping column: {group_column}")
    folds = stable_group_fold(
        development[group_column].fillna("").astype(str),
        n_folds=n_folds,
        salt=salt,
    )
    result = pd.DataFrame(index=development.index)
    result["reference_fold"] = folds
    result["reference_training_case_count"] = 0
    for fold in range(n_folds):
        held_out = folds == fold
        training_values = development.loc[
            ~held_out, list(probability_columns)
        ].to_numpy(dtype=float)
        reference = _validated_probabilities(training_values).mean(axis=0)
        target = development.loc[
            held_out, list(probability_columns)
        ].to_numpy(dtype=float)
        result.loc[held_out, "scout_reference_js_divergence"] = (
            jensen_shannon_divergence(target, reference)
        )
        result.loc[held_out, "reference_training_case_count"] = len(training_values)
    result["reference_source"] = "other_development_folds_only"
    result["reference_excludes_current_group"] = True
    return result


def extract_legal_feature_frame(cases: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the frozen v0.1 feature contract, never metadata/outcomes."""

    missing = sorted(set(LEGAL_FEATURE_COLUMNS) - set(cases.columns))
    if missing:
        raise ValueError(f"Case table is missing legal features: {missing}")
    return cases.loc[:, list(LEGAL_FEATURE_COLUMNS)].copy()


@dataclass(frozen=True)
class ConsultationPolicyBaselineV1_1:
    """Replaceable frozen empirical case-ranking policy.

    This policy can rank cases only.  It cannot grant task/model eligibility,
    alter a budget, or execute an Expert.
    """

    route_id: str
    routing_policy: str
    budget: float
    qualification_level: str

    def __post_init__(self) -> None:
        if self.routing_policy not in {
            "low_confidence",
            "low_margin",
            "high_entropy",
        }:
            raise ValueError(
                "The case-level v1.1 baseline accepts only a frozen single-Scout "
                "routing policy."
            )
        if not 0.0 <= float(self.budget) <= 1.0:
            raise ValueError("Frozen consultation budget must be within [0, 1].")

    def scores(self, cases: pd.DataFrame) -> np.ndarray:
        if self.routing_policy == "low_confidence":
            return 1.0 - cases["scout_confidence"].to_numpy(dtype=float)
        if self.routing_policy == "low_margin":
            return 1.0 - cases["scout_margin"].to_numpy(dtype=float)
        return cases["scout_entropy"].to_numpy(dtype=float)

    @property
    def may_grant_eligibility(self) -> bool:
        return False


@dataclass(frozen=True)
class SafetyEligibilityGate:
    """Non-replaceable adapter around the shared Route Qualification service."""

    contract: Mapping[str, Any]
    contract_sha256: str | None = None

    def evaluate(self, request: Any) -> Any:
        from app.route_qualification import evaluate_route_qualification

        return evaluate_route_qualification(
            request,
            contract=dict(self.contract),
            contract_sha256=self.contract_sha256,
        )


def rank_top_budget(
    *,
    case_ids: Sequence[str],
    scores: Sequence[float] | np.ndarray,
    budget: float,
) -> np.ndarray:
    """Select the deterministic top budget using the existing round convention."""

    if not 0.0 <= float(budget) <= 1.0:
        raise ValueError("budget must be within [0, 1].")
    identifiers = np.asarray([str(value) for value in case_ids], dtype=object)
    values = np.asarray(scores, dtype=float)
    if identifiers.shape != values.shape:
        raise ValueError("case_ids and scores must have equal shape.")
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores contain a non-finite value.")
    selected_n = min(len(values), max(0, int(round(len(values) * float(budget)))))
    order = np.lexsort((identifiers, -values))
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:selected_n]] = True
    return selected
