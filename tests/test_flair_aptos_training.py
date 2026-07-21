import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from app.flair_task_adapter import preprocess_flair_image
from scripts.training.train_flair_aptos import task_probabilities, torch_classifier
from scripts.routing.qualify_aptos_task_artifact import EXPECTED_MANIFEST_SHA256, qualify


def test_flair_preprocessing_returns_official_canvas_shape() -> None:
    array = np.zeros((300, 500, 3), dtype=np.uint8)
    array[40:260, 80:420] = 127
    image = Image.fromarray(array)
    tensor = preprocess_flair_image(image)
    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == torch.float32


def test_flair_protocol_is_project_aptos_adaptation_with_official_lp() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/model_hub/training_protocols"
        / "flair_aptos_official_lp_project_v1.yaml"
    )
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert protocol["classifier"]["c"] == 0.316
    assert protocol["classifier"]["class_weight"] == "balanced"
    assert protocol["evaluation"]["selection_split"] == "none_fixed_official_c"
    assert protocol["evaluation"]["test_once_after_fit"] is True
    assert protocol["claim"]["official_aptos_recipe_available"] is False
    assert protocol["claim"]["status"] == "official_framework_project_downstream_adaptation"


def test_torch_classifier_matches_sklearn_coefficients() -> None:
    class Probe:
        coef_ = np.arange(15, dtype=np.float64).reshape(5, 3)
        intercept_ = np.arange(5, dtype=np.float64)

    classifier = torch_classifier(Probe())
    assert torch.equal(classifier.weight, torch.from_numpy(Probe.coef_).float())
    assert torch.equal(classifier.bias, torch.from_numpy(Probe.intercept_).float())
    probabilities = task_probabilities(classifier, np.ones((2, 3), dtype=np.float32))
    assert probabilities.shape == (2, 5)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_aptos_qualification_requires_evidence_and_preserves_boundary(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "task.pth"
    checkpoint.write_bytes(b"task checkpoint")
    rows = {
        "image_key": [f"case-{index:04d}" for index in range(1100)],
        "true_label": [index % 5 for index in range(1100)],
        "pred_label": [index % 5 for index in range(1100)],
    }
    for index in range(5):
        rows[f"prob_{index}"] = [1.0 if value == index else 0.0 for value in rows["pred_label"]]
    predictions = pd.DataFrame(rows)
    prediction_path = run_dir / "test_predictions.csv"
    reference_path = tmp_path / "reference.csv"
    predictions.to_csv(prediction_path, index=False)
    predictions.to_csv(reference_path, index=False)
    pd.DataFrame(
        [
            {
                "artifact_id": "flair-test",
                "task_id": "aptos_dr_5class",
                "n_classes": 5,
                "prediction_path": str(prediction_path),
                "checkpoint_path": str(checkpoint),
                "task_checkpoint": True,
                "task_adapted": True,
                "task_inference_ready": True,
                "offline_evaluation_eligible": True,
                "unified_evaluation_completed": True,
                "inference_cost_measured": True,
                "route_eligible": False,
            }
        ]
    ).to_csv(run_dir / "registration_record.csv", index=False)
    entries = [
        {"split": split, "relative_path": f"{split}/class/{split}-{index}.png"}
        for split in ("train", "val", "test")
        for index in range(2)
    ]
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps({"manifest_sha256": EXPECTED_MANIFEST_SHA256, "entries": entries}),
        encoding="utf-8",
    )
    pd.DataFrame([{"cost_status": "measured"}]).to_csv(
        run_dir / "forward_cost_summary.csv", index=False
    )
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    report = qualify(run_dir, reference_path)

    registration = pd.read_csv(run_dir / "registration_record.csv").iloc[0]
    assert report.is_file()
    assert registration["compatibility_status"] == "ready_for_pairing"
    assert bool(registration["route_eligible"])
