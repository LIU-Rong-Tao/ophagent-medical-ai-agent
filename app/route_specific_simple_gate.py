"""Development-only screening for high-quality Scout-to-Expert routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app.help_or_harm_benchmark import stable_group_fold


SCHEMA_VERSION = "ophagent.route_specific_simple_gate.v0_1"


@dataclass(frozen=True)
class RouteScreeningCriteria:
    """Predeclared route-quality requirements evaluated on development only."""

    minimum_scout_accuracy: float
    minimum_expert_accuracy: float
    minimum_expert_accuracy_gain: float
    minimum_corrected_to_introduced_ratio: float
    minimum_corrected_events: int
    minimum_introduced_events: int
    minimum_net_events: int
    n_stability_folds: int
    minimum_positive_gain_folds: int
    minimum_nonnegative_net_folds: int
    maximum_fold_gain_standard_deviation: float
    fold_salt: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> RouteScreeningCriteria:
        return cls(
            minimum_scout_accuracy=float(value["minimum_scout_accuracy"]),
            minimum_expert_accuracy=float(value["minimum_expert_accuracy"]),
            minimum_expert_accuracy_gain=float(
                value["minimum_expert_accuracy_gain"]
            ),
            minimum_corrected_to_introduced_ratio=float(
                value["minimum_corrected_to_introduced_ratio"]
            ),
            minimum_corrected_events=int(value["minimum_corrected_events"]),
            minimum_introduced_events=int(value["minimum_introduced_events"]),
            minimum_net_events=int(value["minimum_net_events"]),
            n_stability_folds=int(value["n_stability_folds"]),
            minimum_positive_gain_folds=int(
                value["minimum_positive_gain_folds"]
            ),
            minimum_nonnegative_net_folds=int(
                value["minimum_nonnegative_net_folds"]
            ),
            maximum_fold_gain_standard_deviation=float(
                value["maximum_fold_gain_standard_deviation"]
            ),
            fold_salt=str(value["fold_salt"]),
        )


def _validate_development_cases(cases: pd.DataFrame) -> None:
    required = {
        "task_id",
        "route_id",
        "scout_id",
        "expert_id",
        "benchmark_split",
        "case_id",
        "resampling_group_id",
        "y_true",
        "scout_pred",
        "expert_pred",
        "corrected",
        "introduced",
    }
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"Route screening cases are missing columns: {missing}")
    if cases.empty:
        raise ValueError("Route screening requires development cases.")
    if set(cases["benchmark_split"].astype(str)) != {"development"}:
        raise ValueError("Route screening may read development cases only.")
    if cases["resampling_group_id"].fillna("").astype(str).eq("").any():
        raise ValueError("Route screening requires a grouping identifier.")


def screen_high_quality_routes(
    development_cases: pd.DataFrame,
    *,
    criteria: RouteScreeningCriteria,
) -> pd.DataFrame:
    """Return one auditable development-only quality row per candidate route."""

    _validate_development_cases(development_cases)
    frame = development_cases.copy()
    frame["_scout_correct"] = frame["scout_pred"].eq(frame["y_true"]).astype(int)
    frame["_expert_correct"] = (
        frame["expert_pred"].eq(frame["y_true"]).astype(int)
    )
    frame["_quality_fold"] = stable_group_fold(
        frame["resampling_group_id"].astype(str),
        n_folds=criteria.n_stability_folds,
        salt=criteria.fold_salt,
    )
    route_keys = ["task_id", "route_id", "scout_id", "expert_id"]
    overall = (
        frame.groupby(route_keys, sort=True)
        .agg(
            development_cases=("case_id", "size"),
            development_groups=("resampling_group_id", "nunique"),
            scout_accuracy=("_scout_correct", "mean"),
            expert_accuracy=("_expert_correct", "mean"),
            corrected_events=("corrected", "sum"),
            introduced_events=("introduced", "sum"),
        )
        .reset_index()
    )
    overall["expert_accuracy_gain"] = (
        overall["expert_accuracy"] - overall["scout_accuracy"]
    )
    overall["net_events"] = (
        overall["corrected_events"] - overall["introduced_events"]
    )
    overall["corrected_to_introduced_ratio"] = (
        overall["corrected_events"]
        / overall["introduced_events"].replace(0, np.nan)
    )

    by_fold = (
        frame.groupby(["route_id", "_quality_fold"], sort=True)
        .agg(
            scout_accuracy=("_scout_correct", "mean"),
            expert_accuracy=("_expert_correct", "mean"),
            corrected_events=("corrected", "sum"),
            introduced_events=("introduced", "sum"),
        )
        .reset_index()
    )
    by_fold["expert_accuracy_gain"] = (
        by_fold["expert_accuracy"] - by_fold["scout_accuracy"]
    )
    by_fold["net_events"] = (
        by_fold["corrected_events"] - by_fold["introduced_events"]
    )
    stability = (
        by_fold.groupby("route_id", sort=True)
        .agg(
            observed_stability_folds=("_quality_fold", "nunique"),
            positive_gain_folds=(
                "expert_accuracy_gain",
                lambda values: int((values > 0).sum()),
            ),
            nonnegative_net_folds=(
                "net_events",
                lambda values: int((values >= 0).sum()),
            ),
            fold_gain_standard_deviation=("expert_accuracy_gain", "std"),
            minimum_fold_gain=("expert_accuracy_gain", "min"),
        )
        .reset_index()
    )
    result = overall.merge(stability, on="route_id", validate="one_to_one")

    requirements = {
        "scout_accuracy_pass": result["scout_accuracy"].ge(
            criteria.minimum_scout_accuracy
        ),
        "expert_accuracy_pass": result["expert_accuracy"].ge(
            criteria.minimum_expert_accuracy
        ),
        "expert_gain_pass": result["expert_accuracy_gain"].ge(
            criteria.minimum_expert_accuracy_gain
        ),
        "complementarity_ratio_pass": result[
            "corrected_to_introduced_ratio"
        ].ge(criteria.minimum_corrected_to_introduced_ratio),
        "corrected_events_pass": result["corrected_events"].ge(
            criteria.minimum_corrected_events
        ),
        "introduced_events_pass": result["introduced_events"].ge(
            criteria.minimum_introduced_events
        ),
        "net_events_pass": result["net_events"].ge(
            criteria.minimum_net_events
        ),
        "positive_gain_folds_pass": result["positive_gain_folds"].ge(
            criteria.minimum_positive_gain_folds
        ),
        "nonnegative_net_folds_pass": result["nonnegative_net_folds"].ge(
            criteria.minimum_nonnegative_net_folds
        ),
        "fold_gain_stability_pass": result[
            "fold_gain_standard_deviation"
        ].le(criteria.maximum_fold_gain_standard_deviation),
        "all_folds_observed_pass": result["observed_stability_folds"].eq(
            criteria.n_stability_folds
        ),
    }
    for column, values in requirements.items():
        result[column] = values.astype(bool)
    pass_columns = tuple(requirements)
    result["qualified"] = result.loc[:, pass_columns].all(axis=1)
    result["failure_reasons"] = result.apply(
        lambda row: "|".join(
            column.removesuffix("_pass")
            for column in pass_columns
            if not bool(row[column])
        ),
        axis=1,
    )
    result["screening_data_scope"] = "development_only"
    result["screening_schema_version"] = SCHEMA_VERSION
    result["fold_salt"] = criteria.fold_salt
    return result.sort_values("route_id").reset_index(drop=True)
