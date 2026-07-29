from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.help_or_harm_benchmark import (
    ConsultationPolicyBaselineV1_1,
    LEGAL_FEATURE_COLUMNS,
    SafetyEligibilityGate,
    apply_expert_history_profile,
    build_cross_fitted_expert_history,
    build_cross_fitted_reference_js,
    build_scout_feature_frame,
    compute_case_outcomes,
    deterministic_random_scores,
    extract_legal_feature_frame,
    fit_expert_history_profile,
    jensen_shannon_divergence,
    rank_top_budget,
    stable_group_fold,
)


def _probabilities() -> np.ndarray:
    return np.asarray(
        [
            [0.70, 0.10, 0.10, 0.05, 0.05],
            [0.10, 0.20, 0.30, 0.25, 0.15],
            [0.05, 0.05, 0.10, 0.30, 0.50],
        ]
    )


def test_scout_features_are_preconsultation_only() -> None:
    features = build_scout_feature_frame(_probabilities())

    assert features.columns.tolist() == [
        "scout_prob_0",
        "scout_prob_1",
        "scout_prob_2",
        "scout_prob_3",
        "scout_prob_4",
        "scout_confidence",
        "scout_entropy",
        "scout_margin",
        "scout_severe_probability_mass",
    ]
    assert features.loc[0, "scout_confidence"] == pytest.approx(0.70)
    assert features.loc[0, "scout_margin"] == pytest.approx(0.60)
    assert features.loc[2, "scout_severe_probability_mass"] == pytest.approx(0.80)


def test_scout_features_reject_malformed_probabilities() -> None:
    malformed = _probabilities().copy()
    malformed[0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        build_scout_feature_frame(malformed)


def test_case_outcomes_are_mutually_exclusive_error_proxies() -> None:
    outcomes = compute_case_outcomes(
        y_true=[2, 1, 4, 4],
        scout_pred=[1, 1, 4, 2],
        expert_pred=[2, 0, 4, 1],
    )

    assert outcomes[["corrected", "introduced", "both_correct", "both_wrong"]].sum(
        axis=1
    ).eq(1).all()
    assert outcomes["corrected"].tolist() == [True, False, False, False]
    assert outcomes["introduced"].tolist() == [False, True, False, False]
    assert outcomes["dangerous_introduced"].tolist() == [False, False, False, False]


def test_dangerous_introduced_requires_expert_to_create_undergrading() -> None:
    outcomes = compute_case_outcomes(
        y_true=[4, 4, 2],
        scout_pred=[4, 1, 2],
        expert_pred=[1, 0, 0],
    )

    assert outcomes["dangerous_introduced"].tolist() == [True, False, False]


def test_group_folds_are_stable_and_keep_groups_together() -> None:
    groups = ["patient-a", "patient-a", "patient-b", "patient-c"]
    first = stable_group_fold(groups, n_folds=3)
    second = stable_group_fold(groups, n_folds=3)

    assert np.array_equal(first, second)
    assert first[0] == first[1]


def test_cross_fitted_history_excludes_current_fold() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"case-{index}" for index in range(20)],
            "group_id": [f"patient-{index // 2}" for index in range(20)],
            "scout_pred": [index % 5 for index in range(20)],
            "scout_confidence": np.linspace(0.2, 0.95, 20),
            "corrected": [index % 4 == 0 for index in range(20)],
            "introduced": [index % 6 == 0 for index in range(20)],
        }
    )
    result = build_cross_fitted_expert_history(
        frame,
        group_column="group_id",
        n_folds=5,
    )

    for fold, rows in result.groupby("profile_fold"):
        expected = len(frame) - int((result["profile_fold"] == fold).sum())
        assert rows["profile_training_case_count"].eq(expected).all()
    assert result["profile_excludes_current_group"].all()
    assert result["expert_history_corrected_rate"].between(0, 1).all()
    assert result["expert_history_introduced_rate"].between(0, 1).all()


def test_applied_history_does_not_depend_on_current_expert_output() -> None:
    development = pd.DataFrame(
        {
            "scout_pred": [0, 0, 1, 1],
            "scout_confidence": [0.6, 0.8, 0.7, 0.9],
            "corrected": [True, False, False, True],
            "introduced": [False, True, False, False],
        }
    )
    profile = fit_expert_history_profile(development)
    left = pd.DataFrame(
        {
            "scout_pred": [0, 1],
            "scout_confidence": [0.75, 0.75],
            "expert_pred": [0, 0],
        }
    )
    right = left.assign(expert_pred=[4, 4])

    pd.testing.assert_frame_equal(
        apply_expert_history_profile(left, profile),
        apply_expert_history_profile(right, profile),
    )


def test_reference_js_is_zero_for_reference_distribution() -> None:
    values = np.asarray([[0.2] * 5, [0.2] * 5])
    divergence = jensen_shannon_divergence(values, [0.2] * 5)

    assert np.allclose(divergence, 0.0)


def test_cross_fitted_reference_uses_other_folds_only() -> None:
    probabilities = np.tile(np.asarray([[0.6, 0.1, 0.1, 0.1, 0.1]]), (12, 1))
    frame = pd.DataFrame(
        probabilities,
        columns=[f"scout_prob_{index}" for index in range(5)],
    )
    frame["group_id"] = [f"patient-{index // 2}" for index in range(12)]

    result = build_cross_fitted_reference_js(
        frame,
        group_column="group_id",
        probability_columns=[f"scout_prob_{index}" for index in range(5)],
        n_folds=3,
    )

    assert result["reference_excludes_current_group"].all()
    assert np.allclose(result["scout_reference_js_divergence"], 0.0)


def test_legal_feature_extraction_never_includes_expert_or_identity_fields() -> None:
    frame = pd.DataFrame(
        {column: [0.1] for column in LEGAL_FEATURE_COLUMNS}
    ).assign(
        dataset_id="private-dataset",
        patient_id="patient-1",
        expert_pred=4,
        expert_prob_4=0.99,
        image_path="/private/patient.png",
    )

    legal = extract_legal_feature_frame(frame)

    assert tuple(legal.columns) == LEGAL_FEATURE_COLUMNS
    assert not {
        "dataset_id",
        "patient_id",
        "expert_pred",
        "expert_prob_4",
        "image_path",
    }.intersection(legal.columns)


def test_v1_1_baseline_ranks_only_and_cannot_grant_eligibility() -> None:
    cases = build_scout_feature_frame(_probabilities())
    baseline = ConsultationPolicyBaselineV1_1(
        route_id="frozen-route",
        routing_policy="high_entropy",
        budget=0.1,
        qualification_level="research_replay_only",
    )

    assert np.array_equal(
        baseline.scores(cases),
        cases["scout_entropy"].to_numpy(),
    )
    assert baseline.may_grant_eligibility is False


def test_random_baseline_and_budget_ranking_are_deterministic() -> None:
    case_ids = ["case-c", "case-a", "case-b", "case-d"]
    first = deterministic_random_scores(case_ids, salt="route-a")
    second = deterministic_random_scores(case_ids, salt="route-a")
    selected = rank_top_budget(
        case_ids=case_ids,
        scores=[0.5, 0.5, 0.1, 0.0],
        budget=0.5,
    )

    assert np.array_equal(first, second)
    assert selected.tolist() == [True, True, False, False]


def test_safety_gate_delegates_to_shared_qualification_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict[str, object], str | None]] = []

    def fake_evaluate(
        request: object,
        *,
        contract: dict[str, object],
        contract_sha256: str | None,
    ) -> str:
        calls.append((request, contract, contract_sha256))
        return "shared-decision"

    monkeypatch.setattr(
        "app.route_qualification.evaluate_route_qualification",
        fake_evaluate,
    )
    gate = SafetyEligibilityGate(
        contract={"protocol_id": "route_qualification_gate_v1_1"},
        contract_sha256="abc",
    )

    assert gate.evaluate("request") == "shared-decision"
    assert calls == [
        (
            "request",
            {"protocol_id": "route_qualification_gate_v1_1"},
            "abc",
        )
    ]
