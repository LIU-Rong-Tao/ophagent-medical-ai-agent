"""OphBench image encoder plus OphAgent task-head inference contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.routing.timm_adapter_runtime import normalize_prediction_frame


ARTIFACT_ID = "aptos2019-retfound-cfp-linear-probe-v1"
STANDARD_ARTIFACT_ID = "aptos2019-retfound-cfp-linear-probe-v2"
LABELS = ("No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR")


@dataclass
class OphBenchLinearProbeTaskAdapter:
    base_adapter: object
    classifier: object
    labels: tuple[str, ...] = LABELS

    @classmethod
    def load(cls, *, encoder_checkpoint, head_checkpoint, device="cpu"):
        import joblib
        from ophbench import load_adapter

        encoder = load_adapter(
            model_id="retfound",
            checkpoint_id="retfound-cfp",
            checkpoint_path=encoder_checkpoint,
            device=device,
        ).load()
        return cls(encoder, joblib.load(head_checkpoint))

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        import torch

        tensors = [self.base_adapter.preprocess(image) for image in images]
        embeddings = self.base_adapter.encode_image(torch.stack(tensors)).detach().cpu().numpy()
        probabilities = np.asarray(self.classifier.predict_proba(embeddings), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.labels):
            raise ValueError("Task head probability columns do not match the registered label space")
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-6
        ):
            raise ValueError("Task head returned invalid class probabilities")
        return probabilities


def build_prediction_frame(image_paths, true_labels, probabilities) -> pd.DataFrame:
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "image_path": [str(path) for path in image_paths],
            "true_label": list(true_labels),
            "pred_label": predictions,
        }
    )
    for index in range(probabilities.shape[1]):
        frame[f"prob_{index}"] = probabilities[:, index]
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "predictions.csv"
        frame.to_csv(temporary, index=False)
        return normalize_prediction_frame(temporary, num_classes=probabilities.shape[1])


def registration_record(
    *,
    output_dir,
    prediction_path,
    head_checkpoint,
    encoder_sha256,
    artifact_id=ARTIFACT_ID,
    evaluation_role="integration_smoke",
    lifecycle_status="superseded",
    route_eligible=False,
    provenance=None,
    research_claim_status="not_for_scientific_comparison",
    cost_status="unmeasured",
):
    record = {
        "model_id": f"aptos_dr_5class::{artifact_id}",
        "task_id": "aptos_dr_5class",
        "dataset_id": "APTOS2019",
        "dataset_display_name": "APTOS 2019",
        "dataset_source": "public",
        "artifact_id": artifact_id,
        "model_family": "retfound",
        "architecture": "retfound-mae-vit-large-patch16-256",
        "label_space": "dr_icdr_0_4",
        "n_classes": 5,
        "prediction_source": "adapter",
        "prediction_path": str(prediction_path),
        "adapter_status": "completed",
        "compatibility_status": "ready_for_pairing",
        "role_candidates": "scout|expert",
        "pretraining_source": "ophbench::retfound::retfound-cfp",
        "checkpoint_path": str(head_checkpoint),
        "base_model_provider": "ophbench",
        "base_model_id": "retfound",
        "base_checkpoint_id": "retfound-cfp",
        "encoder_checkpoint_sha256": encoder_sha256,
        "task_checkpoint": True,
        "task_inference_ready": bool(route_eligible),
        "route_eligible": bool(route_eligible),
        "output_dir": str(output_dir),
        "evaluation_role": evaluation_role,
        "lifecycle_status": lifecycle_status,
        "research_claim_status": research_claim_status,
        "cost_status": cost_status,
    }
    record.update(dict(provenance or {}))
    return record
