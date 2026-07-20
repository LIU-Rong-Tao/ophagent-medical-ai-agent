from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.training.train_preti_aptos import (
    CandidateResult,
    build_registration_record,
    choose_candidate,
    dataset_manifest,
)


CLASSES = (
    "anodr",
    "bmilddr",
    "cmoderatedr",
    "dseveredr",
    "eproliferativedr",
)


def _dataset(root: Path, *, overlap: bool = False) -> None:
    for split in ("train", "val", "test"):
        for index, class_name in enumerate(CLASSES):
            directory = root / split / class_name
            directory.mkdir(parents=True)
            key = "shared" if overlap and index == 0 else f"{split}_{index}"
            (directory / f"{key}.png").write_bytes(b"not-opened-by-manifest")


def _candidate(lr: float, qwk: float, macro_f1: float) -> CandidateResult:
    return CandidateResult(
        learning_rate=lr,
        best_epoch=3,
        validation_metrics={
            "quadratic_kappa": qwk,
            "macro_f1": macro_f1,
            "accuracy": 0.5,
        },
        encoder_state={},
        classifier_state={},
    )


def test_dataset_manifest_rejects_case_overlap_between_splits(tmp_path: Path) -> None:
    _dataset(tmp_path, overlap=True)

    with pytest.raises(ValueError, match="存在重复图像键"):
        dataset_manifest(tmp_path)


def test_candidate_selection_uses_validation_qwk_then_macro_f1() -> None:
    selected = choose_candidate(
        [
            _candidate(5e-5, 0.80, 0.70),
            _candidate(1e-5, 0.82, 0.60),
            _candidate(3e-5, 0.82, 0.72),
        ]
    )

    assert selected.learning_rate == 3e-5


def test_registration_grants_task_inference_but_not_route_eligibility(
    tmp_path: Path,
) -> None:
    config = {
        "foundation": {"checkpoint_sha256": "a" * 64},
    }
    record = build_registration_record(
        output_dir=tmp_path,
        prediction_path=tmp_path / "predictions.csv",
        checkpoint_path=tmp_path / "task.pth",
        config=config,
    )

    assert record["task_adapted"] is True
    assert record["task_inference_ready"] is True
    assert record["offline_evaluation_eligible"] is True
    assert record["route_eligible"] is False
    assert record["inference_cost_measured"] is False
    assert record["test_used_for_selection"] is False
    assert record["evaluation_role"] == "project_downstream_adaptation"
    assert (
        record["research_claim_status"]
        == "official_model_project_downstream_adaptation"
    )


def test_recipe_separates_paper_disclosures_from_project_choices() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/model_hub/training_protocols"
        / "preti_aptos_paper_anchored_v1.yaml"
    )
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert recipe["training"]["epochs"] == 50
    assert recipe["training"]["batch_size"] == 16
    assert recipe["training"]["image_size"] == 224
    assert recipe["evaluation"]["selection_split"] == "val"
    assert recipe["evaluation"]["test_once_after_selection"] is True
    assert recipe["foundation"]["official_downstream_recipe_complete"] is False
    assert (
        recipe["recipe"]["recipe_kind"]
        == "official_model_project_downstream_adaptation"
    )
    provenance = "ophagent_declared_due_to_missing_official_downstream_recipe"
    assert recipe["optimizer"]["provenance"] == provenance
    assert recipe["scheduler"]["provenance"] == provenance
    assert recipe["model"]["provenance"] == provenance
    assert recipe["loss"]["provenance"] == provenance
    assert recipe["augmentation"]["provenance"] == provenance
    assert recipe["data"]["root"] is None
    assert recipe["foundation"]["checkpoint_path"] is None
