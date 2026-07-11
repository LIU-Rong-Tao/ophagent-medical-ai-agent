from pathlib import Path

import numpy as np

from app.ophbench_task_adapter import (
    ARTIFACT_ID,
    OphBenchLinearProbeTaskAdapter,
    build_prediction_frame,
    registration_record,
)


class FakeEncoder:
    def preprocess(self, image):
        return image

    def encode_image(self, batch):
        return batch


class FakeHead:
    def predict_proba(self, embeddings):
        rows = len(embeddings)
        return np.tile(np.array([[0.6, 0.1, 0.1, 0.1, 0.1]]), (rows, 1))


def test_task_adapter_returns_five_class_probabilities():
    import torch

    adapter = OphBenchLinearProbeTaskAdapter(FakeEncoder(), FakeHead())
    probabilities = adapter.predict_proba([torch.ones(4), torch.zeros(4)])
    assert probabilities.shape == (2, 5)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_prediction_frame_uses_normalized_ophagent_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    probabilities = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]])
    frame = build_prediction_frame(["a.png", "b.png"], [0, 2], probabilities)
    assert list(frame["pred_label"]) == [0, 2]
    assert {f"prob_{index}" for index in range(5)}.issubset(frame.columns)
    assert {"confidence", "margin", "entropy"}.issubset(frame.columns)


def test_registration_is_a_route_eligible_local_task_artifact(tmp_path):
    record = registration_record(
        output_dir=tmp_path,
        prediction_path=tmp_path / "predictions.csv",
        head_checkpoint=tmp_path / "head.joblib",
        encoder_sha256="abc",
    )
    assert record["artifact_id"] == ARTIFACT_ID
    assert record["base_model_provider"] == "ophbench"
    assert record["task_checkpoint"] is True
    assert record["task_inference_ready"] is True
    assert record["route_eligible"] is True
    assert Path(record["checkpoint_path"]).name == "head.joblib"
