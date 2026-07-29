from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def case_keys_and_outcomes() -> pd.DataFrame:
    return pd.read_csv(
        OUTPUT / "case_level_benchmark.csv.gz",
        usecols=[
            "task_id",
            "route_id",
            "benchmark_split",
            "case_id",
            "analysis_unit_id",
            "primary_cohort_eligible",
            "cross_split_exact_duplicate",
            "exact_duplicate_label_conflict",
            "corrected",
            "introduced",
            "both_correct",
            "both_wrong",
            "current_case_expert_output_used_for_feature",
            "test_outcome_used_for_feature_or_threshold",
        ],
    )


def test_help_or_harm_summary_freezes_conditional_go_scope() -> None:
    summary = json.loads(
        (OUTPUT / "benchmark_summary.json").read_text(encoding="utf-8")
    )

    assert summary["decision"] == "CONDITIONAL_GO"
    assert summary["candidate_routes"] == 210
    assert summary["route_split_rows"] == 390
    assert summary["case_route_rows"] == 214_410
    assert summary["prediction_assets"] == 46
    assert summary["prediction_assets_complete"] is True
    assert summary["route_alignment_error_rows"] == 0
    assert summary["test_policy"] == {
        "test_used_for_feature_definition": False,
        "test_used_for_threshold": False,
        "test_used_for_route_selection": False,
        "retrospective_only": True,
        "independent_confirmation_missing": True,
    }


def test_image_leakage_counts_match_independent_hash_audit() -> None:
    leakage = pd.read_csv(OUTPUT / "image_leakage_audit.csv").set_index(
        "dataset_id"
    )
    aptos = leakage.loc["APTOS2019"]
    deepdrid = leakage.loc["DeepDRiD_v1.1"]

    assert int(aptos["images"]) == 3_662
    assert int(aptos["exact_duplicate_groups"]) == 123
    assert int(aptos["cross_split_exact_groups"]) == 74
    assert int(aptos["label_conflict_exact_groups"]) == 30
    assert int(aptos["cross_split_label_conflict_exact_groups"]) == 15
    assert float(aptos["patient_id_coverage"]) == 0.0
    assert int(deepdrid["exact_duplicate_groups"]) == 0
    assert int(deepdrid["patient_cross_split_groups"]) == 0
    assert float(deepdrid["patient_id_coverage"]) == 1.0


def test_case_table_has_unique_units_and_mutually_exclusive_outcomes(
    case_keys_and_outcomes: pd.DataFrame,
) -> None:
    frame = case_keys_and_outcomes
    assert not frame.duplicated(
        ["route_id", "benchmark_split", "case_id"]
    ).any()
    primary = frame.loc[frame["primary_cohort_eligible"].astype(bool)]
    assert not primary.duplicated(
        ["route_id", "benchmark_split", "analysis_unit_id"]
    ).any()
    assert (
        frame[["corrected", "introduced", "both_correct", "both_wrong"]]
        .astype(int)
        .sum(axis=1)
        .eq(1)
        .all()
    )
    assert not frame["current_case_expert_output_used_for_feature"].astype(bool).any()
    assert not frame["test_outcome_used_for_feature_or_threshold"].astype(bool).any()


def test_exact_duplicate_and_conflict_rows_cannot_enter_primary_cohort(
    case_keys_and_outcomes: pd.DataFrame,
) -> None:
    frame = case_keys_and_outcomes
    blocked = frame.loc[
        frame["cross_split_exact_duplicate"].astype(bool)
        | frame["exact_duplicate_label_conflict"].astype(bool)
    ]

    assert len(blocked) > 0
    assert not blocked["primary_cohort_eligible"].astype(bool).any()


def test_conditional_signal_and_profile_contracts_are_present() -> None:
    signals = pd.read_csv(OUTPUT / "development_signal_results.csv")
    profiles = pd.read_csv(OUTPUT / "expert_history_profile_audit.csv")
    baselines = pd.read_csv(OUTPUT / "baseline_results.csv")

    assert {"scout_wrong_only", "scout_correct_only"}.issubset(
        set(signals["evaluation_cohort"])
    )
    assert not profiles["test_outcome_used"].astype(bool).any()
    assert not profiles["current_case_expert_output_used"].astype(bool).any()
    assert (
        baselines["policy"].eq("consultation_policy_baseline_v1_1").sum() == 16
    )
    assert not baselines[
        "test_used_for_feature_threshold_or_route_selection"
    ].astype(bool).any()


def test_artifact_manifest_hashes_every_declared_output() -> None:
    manifest = json.loads(
        (OUTPUT / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["decision"] == "CONDITIONAL_GO"
    assert manifest["model_inference_performed"] is False
    assert manifest["route_selection_performed"] is False
    for artifact in manifest["outputs"]:
        uri = str(artifact["uri"])
        assert uri.startswith("repo://")
        path = ROOT / uri.removeprefix("repo://")
        assert path.is_file()
        assert path.stat().st_size == int(artifact["size_bytes"])
        assert _sha256(path) == artifact["sha256"]
