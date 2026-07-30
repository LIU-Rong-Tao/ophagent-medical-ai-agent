from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.help_or_harm_benchmark import (
    build_scout_feature_frame,
    stable_group_fold,
)
from app.selective_consultation import (
    METHOD_FEATURE_COLUMNS,
    build_cross_fitted_method_features,
    build_transfer_method_features,
    fit_dual_logistic_model,
    nested_group_oof_predictions,
    paired_cluster_bootstrap_difference,
    select_consultations,
)
from scripts.run_selective_consultation_method_v0_1 import (
    dominant_budgets,
    route_qualifies,
)


def _case_frame(n_cases: int = 100) -> pd.DataFrame:
    rows: list[list[float]] = []
    scout_predictions: list[int] = []
    for index in range(n_cases):
        predicted = index % 5
        scout_predictions.append(predicted)
        probabilities = np.full(5, 0.05)
        probabilities[predicted] = 0.70
        probabilities[(predicted + 1) % 5] = 0.15
        rows.append(probabilities.tolist())
    scout_features = build_scout_feature_frame(np.asarray(rows))
    outcome_state = np.arange(n_cases) % 5
    corrected = outcome_state == 0
    introduced = outcome_state == 1
    both_correct = outcome_state == 2
    both_wrong = ~(corrected | introduced | both_correct)
    truth = np.asarray([(index + 2) % 5 for index in range(n_cases)])
    scout = truth.copy()
    scout[corrected | both_wrong] = (truth[corrected | both_wrong] + 1) % 5
    expert = truth.copy()
    expert[introduced | both_wrong] = (
        truth[introduced | both_wrong] + 2
    ) % 5
    frame = pd.DataFrame(
        {
            "case_id": [f"case-{index:03d}" for index in range(n_cases)],
            "resampling_group_id": [
                f"patient-{index // 2:03d}" for index in range(n_cases)
            ],
            "y_true": truth,
            "scout_pred": scout,
            "expert_pred": expert,
            "corrected": corrected,
            "introduced": introduced,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "dangerous_introduced": False,
        }
    )
    for column in scout_features:
        frame[column] = scout_features[column].to_numpy()
    return frame


def test_method_features_are_small_and_preconsultation_only() -> None:
    frame = _case_frame()

    features = build_cross_fitted_method_features(
        frame,
        n_folds=5,
        salt="unit-feature",
    )

    assert tuple(features.columns) == METHOD_FEATURE_COLUMNS
    assert not {
        "case_id",
        "resampling_group_id",
        "y_true",
        "expert_pred",
        "corrected",
        "introduced",
    }.intersection(features.columns)
    assert np.isfinite(features.to_numpy()).all()


def test_transfer_features_do_not_require_target_outcomes_or_expert_output() -> None:
    development = _case_frame(80)
    target = _case_frame(20).drop(
        columns=[
            "y_true",
            "expert_pred",
            "corrected",
            "introduced",
            "both_correct",
            "both_wrong",
            "dangerous_introduced",
        ]
    )
    left = build_transfer_method_features(development, target)
    right = build_transfer_method_features(
        development,
        target.assign(expert_pred=4),
    )

    pd.testing.assert_frame_equal(left, right)


def test_outer_fold_predictions_ignore_held_out_outcomes() -> None:
    frame = _case_frame()
    salt = "unit-nested"
    folds = stable_group_fold(
        frame["resampling_group_id"],
        n_folds=5,
        salt=f"{salt}:outer",
    )
    held_out_fold = int(folds[0])
    held_out = folds == held_out_fold
    first = nested_group_oof_predictions(
        frame,
        n_folds=5,
        minimum_route_events=5,
        salt=salt,
    )
    changed = frame.copy()
    changed.loc[held_out, "corrected"] = ~changed.loc[
        held_out, "corrected"
    ].astype(bool)
    changed.loc[held_out, "introduced"] = ~changed.loc[
        held_out, "introduced"
    ].astype(bool)
    second = nested_group_oof_predictions(
        changed,
        n_folds=5,
        minimum_route_events=5,
        salt=salt,
    )

    probability_columns = [
        "predicted_corrected_probability",
        "predicted_introduced_probability",
    ]
    assert np.allclose(
        first.loc[held_out, probability_columns],
        second.loc[held_out, probability_columns],
    )


def test_dual_policy_controls_harm_before_help_and_uses_exact_budget() -> None:
    cases = _case_frame(10)
    predictions = pd.DataFrame(
        {
            "predicted_corrected_probability": [
                0.1,
                0.2,
                0.8,
                0.9,
                1.0,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
            ],
            "predicted_introduced_probability": np.arange(10) / 10,
        }
    )

    selected = select_consultations(
        cases,
        policy="dual_logistic_harm_screened_help",
        budget=0.2,
        predictions=predictions,
        safe_pool_multiplier=2.0,
    )

    assert selected.sum() == 2
    assert np.flatnonzero(selected).tolist() == [2, 3]


def test_policy_selection_does_not_change_with_expert_prediction() -> None:
    cases = _case_frame(20)
    predictions = pd.DataFrame(
        {
            "predicted_corrected_probability": np.linspace(0, 1, 20),
            "predicted_introduced_probability": np.linspace(1, 0, 20),
        }
    )
    first = select_consultations(
        cases,
        policy="dual_logistic_harm_screened_help",
        budget=0.3,
        predictions=predictions,
    )
    second = select_consultations(
        cases.assign(expert_pred=(cases["expert_pred"] + 1) % 5),
        policy="dual_logistic_harm_screened_help",
        budget=0.3,
        predictions=predictions,
    )

    assert np.array_equal(first, second)


def test_fitted_controller_cannot_grant_route_eligibility() -> None:
    frame = _case_frame()
    features = build_cross_fitted_method_features(
        frame,
        n_folds=5,
        salt="unit-fit",
    )

    model = fit_dual_logistic_model(
        features,
        frame[["corrected", "introduced"]],
        minimum_events=5,
    )

    assert model.may_grant_eligibility is False
    assert model.feature_columns == METHOD_FEATURE_COLUMNS
    assert model.predict(features).shape == (len(frame), 2)


def test_oracle_is_explicitly_outcome_dependent_upper_bound() -> None:
    cases = _case_frame(20)

    selected = select_consultations(
        cases,
        policy="oracle",
        budget=0.2,
    )

    assert selected.sum() == 4
    assert cases.loc[selected, "corrected"].all()


def test_paired_cluster_bootstrap_is_deterministic() -> None:
    cases = _case_frame(40)
    method = np.arange(40) % 3 == 0
    baseline = np.arange(40) % 4 == 0

    first = paired_cluster_bootstrap_difference(
        cases,
        method_selected=method,
        baseline_selected=baseline,
        replicates=200,
        seed=17,
    )
    second = paired_cluster_bootstrap_difference(
        cases,
        method_selected=method,
        baseline_selected=baseline,
        replicates=200,
        seed=17,
    )

    assert first == second
    assert first["bootstrap_group_count"] == 20
    assert first["bootstrap_replicates"] == 200


def test_safe_pool_multiplier_cannot_be_less_than_one() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_consultations(
            _case_frame(10),
            policy="dual_logistic_harm_screened_help",
            budget=0.2,
            predictions=pd.DataFrame(
                {
                    "predicted_corrected_probability": np.linspace(0, 1, 10),
                    "predicted_introduced_probability": np.linspace(1, 0, 10),
                }
            ),
            safe_pool_multiplier=0.5,
        )


def test_decision_dominance_excludes_non_operating_five_percent_budget() -> None:
    core = pd.DataFrame(
        {
            "record_type": ["policy_performance"] * 4,
            "route_id": ["route-a"] * 4,
            "analysis_split": ["retrospective_evaluation"] * 4,
            "comparison_axis": ["same_budget"] * 4,
            "policy": ["dual_logistic_harm_screened_help"] * 4,
            "requested_budget": [0.05, 0.10, 0.20, 0.30],
            "delta_corrected_selected": [0, -1, -1, 0],
            "delta_introduced_selected": [-1, -1, -1, -1],
            "delta_net_selected": [1, 0, 0, 1],
        }
    )

    dominant = dominant_budgets(
        core,
        route_id="route-a",
        analysis_split="retrospective_evaluation",
        policy="dual_logistic_harm_screened_help",
    )

    assert dominant == [0.30]
    assert route_qualifies(dominant) is False
