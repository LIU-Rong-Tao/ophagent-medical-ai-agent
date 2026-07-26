from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


from scripts.training.run_aptos_frozen_encoder_probe import (
    prepare_task_manifests,
)


def test_deepdrid_internal_split_is_patient_level_and_deterministic(tmp_path):
    rows = []
    for patient_index in range(25):
        label = patient_index % 5
        for image_index in range(2):
            rows.append(
                {
                    "case_id": f"train-{patient_index}-{image_index}",
                    "patient_id": f"patient-{patient_index}",
                    "source_split": "official_train",
                    "y_true": label,
                    "relative_image_path": f"train/{patient_index}-{image_index}.jpg",
                }
            )
    rows.append(
        {
            "case_id": "missing-label",
            "patient_id": "missing-patient",
            "source_split": "official_train",
            "y_true": None,
            "relative_image_path": "train/missing.jpg",
        }
    )
    for patient_index in range(5):
        rows.append(
            {
                "case_id": f"test-{patient_index}",
                "patient_id": f"test-patient-{patient_index}",
                "source_split": "official_validation",
                "y_true": patient_index,
                "relative_image_path": f"test/{patient_index}.jpg",
            }
        )
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    config = {
        "task_id": "deepdrid_test",
        "source_manifest": str(source),
        "source_train_split": "official_train",
        "frozen_test_split": "official_validation",
        "internal_validation_fraction": 0.2,
        "random_seed": 42,
        "manifest_output_dir": str(tmp_path / "manifests"),
    }

    first, first_digest = prepare_task_manifests(config)
    second, second_digest = prepare_task_manifests(config)

    assert first_digest == second_digest
    assert len(first["train"]) == 40
    assert len(first["val"]) == 10
    assert len(first["test"]) == 5
    assert set(first["train"]["patient_id"]).isdisjoint(first["val"]["patient_id"])
    assert set(first["train"]["patient_id"]).isdisjoint(first["test"]["patient_id"])
    assert set(first["val"]["patient_id"]).isdisjoint(first["test"]["patient_id"])
    assert not pd.concat(first.values())["case_id"].eq("missing-label").any()
    assert second["val"]["case_id"].tolist() == first["val"]["case_id"].tolist()


def test_deepdrid_protocol_freezes_official_validation_and_class_order():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "experiments/opening_risk_routing_closure/configs/protocols"
        / "deepdrid_native_probe_pool.yaml"
    )
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert protocol["selection_metric"] == "macro_f1"
    assert protocol["class_order"] == [0, 1, 2, 3, 4]
    assert protocol["frozen_evaluation_split"] == "official_validation"
    assert protocol["test_used_for_selection"] is False
