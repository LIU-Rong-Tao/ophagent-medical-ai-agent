from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.routing import run_model_hub_inference_job


class _FixedAdapter:
    def predict_proba(self, images):
        probabilities = np.array(
            [
                [0.8, 0.1, 0.05, 0.03, 0.02],
                [0.1, 0.1, 0.7, 0.05, 0.05],
            ],
            dtype=float,
        )
        return probabilities[: len(images)]


def _request(tmp_path, manifest_path, checkpoint_path):
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "job_id": "external-test",
        "task_id": "aptos_dr_5class",
        "dataset_id": "deepdrid_v1.1_regular_fundus",
        "artifact_id": "convnext_tiny",
        "loader_id": "aptos_registered_adapter_v1",
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_id": "registered",
        "label_structure": "ordinal",
        "num_classes": 5,
        "batch_size": 2,
        "device": "cpu",
        "precision": "fp32",
        "data_root": str(tmp_path / "images"),
        "input_manifest_path": str(manifest_path),
        "output_dir": str(tmp_path / "output"),
        "adapter_spec": {
            "adapter_type": "timm_classifier",
            "architecture": "unused",
            "preprocessing_id": "registered",
            "checkpoint_path": str(checkpoint_path),
        },
    }


def test_registered_adapter_exports_frozen_external_prediction_asset(
    tmp_path,
    monkeypatch,
):
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (12, 12), color=(120, 80, 40)).save(image_root / name)
    manifest = pd.DataFrame(
        {
            "case_id": ["case-a", "case-b"],
            "patient_id": ["patient-a", "patient-b"],
            "split": ["external", "external"],
            "y_true": [0, 2],
            "relative_image_path": ["a.png", "b.png"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        run_model_hub_inference_job,
        "load_registered_aptos_adapter",
        lambda spec, device: _FixedAdapter(),
    )

    output = run_model_hub_inference_job.run_request(
        _request(tmp_path, manifest_path, checkpoint_path)
    )

    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions["case_id"].tolist() == ["case-a", "case-b"]
    assert predictions["patient_id"].tolist() == ["patient-a", "patient-b"]
    assert predictions["y_pred"].tolist() == [0, 2]
    assert np.allclose(
        predictions[[f"prob_{index}" for index in range(5)]].sum(axis=1),
        1.0,
    )
    inference_manifest = json.loads(
        (output / "inference_manifest.json").read_text(encoding="utf-8")
    )
    assert inference_manifest["evaluation_design"] == "frozen_external_transfer"
    assert inference_manifest["model_selection_on_external_data"] is False
    assert inference_manifest["route_eligible"] is False


def test_frozen_manifest_rejects_duplicate_case_ids(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (12, 12)).save(image_root / "a.png")
    manifest = pd.DataFrame(
        {
            "case_id": ["same", "same"],
            "patient_id": ["a", "b"],
            "split": ["external", "external"],
            "y_true": [0, 1],
            "relative_image_path": ["a.png", "a.png"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="case_id"):
        run_model_hub_inference_job.run_request(
            _request(tmp_path, manifest_path, checkpoint_path)
        )
