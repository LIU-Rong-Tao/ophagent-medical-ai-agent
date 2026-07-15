from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repository-root bootstrap above must run before these imports.
from scripts.routing.evaluate_task_agnostic_scout_expert import (  # noqa: E402
    EvaluationError,
    compute_metrics,
    merge_scout_expert,
    oracle_exact_k_curve,
    oracle_up_to_k_curve,
    random_same_budget,
    recompute_probability_signals,
    run_evaluation,
    select_for_expert,
    validate_artifact_compatibility,
    validate_prediction_frame,
)


PROBABILITY_COLUMNS = ["prob_0", "prob_1", "prob_2"]
ROOT = Path(__file__).resolve().parents[1]


def prediction_frame(model_name: str = "scout") -> pd.DataFrame:
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
            [0.55, 0.35, 0.10],
            [0.20, 0.45, 0.35],
            [0.20, 0.25, 0.55],
        ]
    )
    return pd.DataFrame(
        {
            "image_key": [f"case_{index}" for index in range(len(probabilities))],
            "true_label": [0, 1, 2, 1, 2, 0],
            "pred_label": [0, 1, 2, 0, 1, 2],
            "model_name": model_name,
            "split": "test",
            **{
                column: probabilities[:, index]
                for index, column in enumerate(PROBABILITY_COLUMNS)
            },
        }
    )


def test_probability_signals_are_recomputed_from_probabilities():
    frame = prediction_frame()
    frame.loc[0, "pred_label"] = 2
    frame["confidence"] = -1.0
    frame["margin"] = -1.0

    signals = recompute_probability_signals(frame, PROBABILITY_COLUMNS)

    assert signals.loc[0, "pred_label"] == 0
    assert signals.loc[0, "confidence"] == pytest.approx(0.7)
    assert signals.loc[0, "margin"] == pytest.approx(0.5)
    expected_entropy = -(
        0.7 * np.log(0.7) + 0.2 * np.log(0.2) + 0.1 * np.log(0.1)
    )
    assert signals.loc[0, "entropy"] == pytest.approx(expected_entropy)
    assert signals.loc[0, "normalized_entropy"] == pytest.approx(
        expected_entropy / np.log(3)
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns="prob_2"), "概率列"),
        (lambda frame: frame.assign(prob_0=0.9), "概率和"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "image_key",
        ),
    ],
)
def test_prediction_validation_rejects_invalid_inputs(mutator, message):
    with pytest.raises(EvaluationError, match=message):
        validate_prediction_frame(mutator(prediction_frame()), PROBABILITY_COLUMNS)


def test_merge_rejects_inconsistent_true_labels():
    scout = recompute_probability_signals(prediction_frame("scout"), PROBABILITY_COLUMNS)
    expert = recompute_probability_signals(prediction_frame("expert"), PROBABILITY_COLUMNS)
    expert.loc[0, "true_label"] = 2

    with pytest.raises(EvaluationError, match="真实标签"):
        merge_scout_expert(scout, expert)


@pytest.mark.parametrize("field", ["dataset_id", "modality", "label_schema_id"])
def test_artifact_compatibility_rejects_cross_protocol_mismatch(field):
    scout = pd.Series(
        {
            "artifact_id": "scout",
            "task_id": "task",
            "dataset_id": "dataset",
            "modality": "fundus",
            "label_schema_id": "schema",
            "split": "test",
        }
    )
    expert = scout.copy()
    expert["artifact_id"] = "expert"
    expert[field] = "mismatch"

    with pytest.raises(EvaluationError, match=field):
        validate_artifact_compatibility(
            scout,
            expert,
            {"task_id": "task", "evaluation_split": "test"},
        )


def test_compute_metrics_uses_small_unified_classification_set():
    frame = recompute_probability_signals(prediction_frame(), PROBABILITY_COLUMNS)
    y_true = frame["true_label"].to_numpy()
    y_pred = frame["pred_label"].to_numpy()
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy()

    metrics = compute_metrics(y_true, y_pred, probabilities)
    binary_targets = label_binarize(y_true, classes=[0, 1, 2])

    assert metrics == pytest.approx(
        {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_auroc_ovr": roc_auc_score(
                binary_targets, probabilities, average="macro", multi_class="ovr"
            ),
            "macro_aupr_ovr": average_precision_score(
                binary_targets, probabilities, average="macro"
            ),
            "n_error": int(np.sum(y_true != y_pred)),
        }
    )
    assert "qwk" not in metrics
    assert "cohen_kappa" not in metrics
    assert "weighted_f1" not in metrics


def test_compute_metrics_supports_binary_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    probabilities = np.array(
        [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]
    )

    metrics = compute_metrics(y_true, y_pred, probabilities)

    assert metrics["macro_auroc_ovr"] == pytest.approx(
        roc_auc_score(y_true, probabilities[:, 1])
    )
    assert metrics["macro_aupr_ovr"] == pytest.approx(
        average_precision_score(y_true, probabilities[:, 1])
    )


def routing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_key": ["b", "a", "d", "c"],
            "true_label": [0, 1, 0, 1],
            "scout_pred_label": [0, 0, 1, 1],
            "expert_pred_label": [1, 1, 0, 0],
            "scout_confidence": [0.5, 0.5, 0.8, 0.9],
            "scout_margin": [0.2, 0.2, 0.5, 0.7],
            "scout_normalized_entropy": [0.7, 0.7, 0.3, 0.1],
        }
    )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("low_confidence", ["a", "b"]),
        ("low_margin", ["a", "b"]),
        ("high_entropy", ["a", "b"]),
    ],
)
def test_policy_selection_is_deterministic(policy, expected):
    selected = select_for_expert(routing_frame(), policy=policy, selected_n=2)

    assert selected == expected


def test_random_baseline_is_reproducible():
    frame = routing_frame()

    first = random_same_budget(frame, budgets=[0.5], trials=200, seed=42)
    second = random_same_budget(frame, budgets=[0.5], trials=200, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_oracle_up_to_k_is_monotonic_and_does_not_force_harmful_calls():
    frame = routing_frame()
    budgets = [0.25, 0.5, 0.75, 1.0]

    exact = oracle_exact_k_curve(frame, budgets)
    up_to = oracle_up_to_k_curve(frame, budgets)

    assert exact["selected_n"].tolist() == [1, 2, 3, 4]
    assert up_to["accuracy"].is_monotonic_increasing
    assert up_to["selected_n"].tolist() == [1, 2, 2, 2]
    assert up_to.iloc[-1]["accuracy"] == pytest.approx(1.0)


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_end_to_end_writes_four_raw_tables_without_disease_specific_metrics(tmp_path: Path):
    scout_path = tmp_path / "scout.csv"
    expert_path = tmp_path / "expert.csv"
    registry_path = tmp_path / "registry.csv"
    protocol_path = tmp_path / "protocol.json"
    work_dir = tmp_path / "work"
    scout = prediction_frame("fixture_scout")
    expert = prediction_frame("fixture_expert").copy()
    expert.loc[:, PROBABILITY_COLUMNS] = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
            [0.70, 0.20, 0.10],
        ]
    )
    write_prediction(scout_path, scout)
    write_prediction(expert_path, expert)
    pd.DataFrame(
        [
            {
                "artifact_id": "fixture_scout",
                "task_id": "fixture_3class",
                "dataset_id": "fixture",
                "modality": "fundus",
                "label_schema_id": "fixture_v1",
                "model_family": "fixture",
                "prediction_csv": str(scout_path),
                "cost_csv": "",
                "checkpoint_path": "",
                "split": "test",
                "enabled": 1,
            },
            {
                "artifact_id": "fixture_expert",
                "task_id": "fixture_3class",
                "dataset_id": "fixture",
                "modality": "fundus",
                "label_schema_id": "fixture_v1",
                "model_family": "fixture",
                "prediction_csv": str(expert_path),
                "cost_csv": "",
                "checkpoint_path": "",
                "split": "test",
                "enabled": 1,
            },
        ]
    ).to_csv(registry_path, index=False)
    protocol_path.write_text(
        json.dumps(
            {
                "protocol_id": "fixture_protocol",
                "task_id": "fixture_3class",
                "task_type": "generic_multiclass",
                "mode": "exploratory",
                "selection_split": "test",
                "evaluation_split": "test",
                "metric_profile": "classification_standard_v1",
                "scouts": ["fixture_scout"],
                "experts": ["fixture_expert"],
                "budgets": [0.5],
                "policies": ["low_confidence"],
                "random_trials": 20,
                "seed": 42,
                "risk_events": [],
                "case_export_points": [
                    {"budget": 0.5, "policy": "low_confidence"}
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    run_evaluation(registry_path, protocol_path, work_dir)

    assert sorted(path.name for path in work_dir.iterdir()) == [
        "case_audit.csv",
        "model_baselines.csv",
        "risk_results.csv",
        "routing_results.csv",
    ]
    baselines = pd.read_csv(work_dir / "model_baselines.csv")
    routing = pd.read_csv(work_dir / "routing_results.csv")
    risk = pd.read_csv(work_dir / "risk_results.csv")
    cases = pd.read_csv(work_dir / "case_audit.csv")

    assert set(baselines["cost_status"]) == {"missing"}
    assert set(routing["cost_status"]) == {"missing"}
    assert len(risk) == 0
    assert len(cases) == len(scout)
    assert {"protocol_id", "budget", "policy", "scout_artifact", "expert_artifact"} <= set(
        cases.columns
    )
    forbidden = ("qwk", "vtdr", "large_undergrading", "referable_miss", "severe_pdr")
    assert not any(marker in column.lower() for column in routing.columns for marker in forbidden)


def test_glaucoma_replay_preserves_historical_key_results(tmp_path: Path):
    registry = (
        ROOT
        / "experiments/v0_8_4_task_agnostic_evaluator/configs/artifact_registry.csv"
    )
    source_protocol = (
        ROOT
        / "experiments/v0_8_4_task_agnostic_evaluator/configs/"
        "glaucoma_convnext_retfound_protocol.yaml"
    )
    import yaml

    protocol = yaml.safe_load(source_protocol.read_text(encoding="utf-8"))
    protocol["random_trials"] = 2
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    run_evaluation(registry, protocol_path, tmp_path / "work")

    baselines = pd.read_csv(tmp_path / "work/model_baselines.csv").set_index("role")
    routing = pd.read_csv(tmp_path / "work/routing_results.csv")
    uncertainty = routing[routing["method_kind"] == "uncertainty"].set_index(
        ["budget", "policy"]
    )
    oracle = routing[routing["policy"] == "oracle_up_to_k"].sort_values("budget")

    assert baselines.loc["scout", "accuracy"] == pytest.approx(0.8086021505)
    assert baselines.loc["expert", "accuracy"] == pytest.approx(0.8602150538)
    assert uncertainty.loc[(0.2, "high_entropy"), "accuracy"] == pytest.approx(
        0.8451612903
    )
    assert uncertainty.loc[(0.3, "high_entropy"), "accuracy"] == pytest.approx(
        0.8559139785
    )
    assert uncertainty.loc[(0.5, "low_margin"), "accuracy"] == pytest.approx(
        0.8602150538
    )
    assert uncertainty.loc[(0.5, "low_confidence"), "accuracy"] == pytest.approx(
        0.8602150538
    )
    assert oracle["accuracy"].is_monotonic_increasing


def test_repository_v084_runner_has_a_valid_dry_run():
    config = (
        ROOT
        / "experiments/v0_8_4_task_agnostic_evaluator/configs/"
        "glaucoma_convnext_retfound_runner.yaml"
    )
    runner = ROOT / "scripts/routing/run_controlled_protocol.py"

    result = subprocess.run(
        [sys.executable, str(runner), "--config", str(config), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[PLANNED]") == 1
    assert "[DRY-RUN COMPLETE]" in result.stdout
