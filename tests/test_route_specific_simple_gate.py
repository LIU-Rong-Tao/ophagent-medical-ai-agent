from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.route_specific_simple_gate import (
    RouteScreeningCriteria,
    screen_high_quality_routes,
)
from scripts.run_route_specific_simple_gate_v0_1 import route_success


def _criteria() -> RouteScreeningCriteria:
    return RouteScreeningCriteria(
        minimum_scout_accuracy=0.5,
        minimum_expert_accuracy=0.6,
        minimum_expert_accuracy_gain=0.1,
        minimum_corrected_to_introduced_ratio=1.5,
        minimum_corrected_events=4,
        minimum_introduced_events=2,
        minimum_net_events=2,
        n_stability_folds=2,
        minimum_positive_gain_folds=2,
        minimum_nonnegative_net_folds=2,
        maximum_fold_gain_standard_deviation=0.5,
        fold_salt="unit-route-quality",
    )


def _development_routes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for route_id, expert_good in (("route-good", True), ("route-bad", False)):
        for index in range(20):
            truth = index % 5
            scout_wrong = index < 8
            scout = (truth + 1) % 5 if scout_wrong else truth
            if expert_good:
                expert_wrong = 6 <= index < 10
            else:
                expert_wrong = index < 12
            expert = (truth + 2) % 5 if expert_wrong else truth
            rows.append(
                {
                    "task_id": "task",
                    "route_id": route_id,
                    "scout_id": "scout",
                    "expert_id": "expert-good" if expert_good else "expert-bad",
                    "benchmark_split": "development",
                    "case_id": f"{route_id}-{index}",
                    "resampling_group_id": f"patient-{index}",
                    "y_true": truth,
                    "scout_pred": scout,
                    "expert_pred": expert,
                    "corrected": bool(scout != truth and expert == truth),
                    "introduced": bool(scout == truth and expert != truth),
                }
            )
    return pd.DataFrame(rows)


def test_screening_uses_fixed_development_criteria() -> None:
    result = screen_high_quality_routes(
        _development_routes(),
        criteria=_criteria(),
    ).set_index("route_id")

    assert bool(result.loc["route-good", "qualified"]) is True
    assert bool(result.loc["route-bad", "qualified"]) is False
    assert result.loc["route-good", "screening_data_scope"] == "development_only"
    assert int(result.loc["route-good", "corrected_events"]) == 6
    assert int(result.loc["route-good", "introduced_events"]) == 2


def test_screening_rejects_frozen_evaluation_rows() -> None:
    cases = _development_routes()
    cases.loc[cases.index[0], "benchmark_split"] = "retrospective_frozen"

    with pytest.raises(ValueError, match="development cases only"):
        screen_high_quality_routes(cases, criteria=_criteria())


def test_route_success_requires_two_budgets_including_thirty_percent() -> None:
    rows = pd.DataFrame(
        {
            "requested_budget": [0.1, 0.2, 0.3],
            "delta_corrected_selected": [0, 1, 0],
            "delta_introduced_selected": [-1, -1, -1],
            "delta_net_selected": [1, 2, 1],
            "net_difference_ci_lower": [-1, 0, 0],
        }
    )

    success, budgets = route_success(
        rows,
        minimum_dominant_budgets=2,
        required_budget=0.3,
    )

    assert success is True
    assert budgets == [0.1, 0.2, 0.3]


def test_route_success_does_not_accept_non_operating_budget() -> None:
    rows = pd.DataFrame(
        {
            "requested_budget": [0.05, 0.1, 0.2, 0.3],
            "delta_corrected_selected": [1, -1, -1, 0],
            "delta_introduced_selected": [-1, -1, -1, -1],
            "delta_net_selected": [2, 0, 0, 1],
            "net_difference_ci_lower": [1, -1, -1, 0],
        }
    )

    success, budgets = route_success(
        rows,
        minimum_dominant_budgets=2,
        required_budget=0.3,
    )

    assert success is False
    assert budgets == [0.3]


def test_screening_output_has_no_case_or_patient_identifiers() -> None:
    result = screen_high_quality_routes(
        _development_routes(),
        criteria=_criteria(),
    )

    assert not {
        "case_id",
        "patient_group_id",
        "resampling_group_id",
        "image_path",
    }.intersection(result.columns)
    assert np.isfinite(result["fold_gain_standard_deviation"]).all()
