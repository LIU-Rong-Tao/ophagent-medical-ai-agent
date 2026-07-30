#!/usr/bin/env python3
"""Extract FLAIR pre-head representations from the registered frozen runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.aptos_replay_adapters import (  # noqa: E402
    ReplayAdapterSpec,
    load_registered_aptos_adapter,
)
from app.flair_task_adapter import preprocess_flair_image  # noqa: E402
from scripts.training.run_aptos_frozen_encoder_probe import (  # noqa: E402
    MODEL_SPECS,
    ManifestDataset,
    _load_task_config,
    _manifests,
)


DEFAULT_OUTPUT_DIR = Path(
    "/training_data/lizekun/ophagent_assets/experiments/help_or_harm/"
    "scout_representation_v0_1"
)
SPLIT_MAP = {"validation": "val", "test": "test"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registered_manifest = MODEL_SPECS["flair"]["registered_manifest"]
    payload = json.loads(registered_manifest.read_text(encoding="utf-8"))
    adapter_spec = ReplayAdapterSpec(**payload["adapter_spec"])
    adapter = load_registered_aptos_adapter(adapter_spec, device=args.device)
    manifests, dataset_manifest_sha256 = _manifests(_load_task_config(None))
    outputs: dict[str, dict[str, object]] = {}

    for prediction_split, manifest_split in SPLIT_MAP.items():
        manifest = manifests[manifest_split].sort_values("image_key")
        dataset = ManifestDataset(manifest, preprocess_flair_image)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        feature_batches: list[np.ndarray] = []
        probability_batches: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        case_ids: list[str] = []
        elapsed_seconds = 0.0
        with torch.inference_mode():
            for images, target, keys in loader:
                images = images.to(args.device, non_blocking=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                features = adapter.encoder(images)
                probabilities = torch.softmax(
                    adapter.classifier(features).float(),
                    dim=1,
                )
                host_features = features.float().cpu().numpy()
                host_probabilities = probabilities.cpu().numpy()
                torch.cuda.synchronize()
                elapsed_seconds += time.perf_counter() - started
                feature_batches.append(host_features)
                probability_batches.append(host_probabilities)
                labels.append(target.numpy())
                case_ids.extend(str(key) for key in keys)

        feature_matrix = np.concatenate(feature_batches).astype(np.float32)
        probability_matrix = np.concatenate(probability_batches)
        label_vector = np.concatenate(labels).astype(np.int64)
        frozen_predictions = pd.read_csv(
            PROJECT_ROOT
            / "experiments/opening_risk_routing_closure/replays/flair/"
            f"full_replay_20260722/predictions/{prediction_split}_predictions.csv"
        ).set_index("case_id")
        reference = frozen_predictions.loc[case_ids]
        maximum_probability_error = float(
            np.max(
                np.abs(
                    probability_matrix
                    - reference[
                        [f"prob_{index}" for index in range(5)]
                    ].to_numpy()
                )
            )
        )
        if not np.array_equal(label_vector, reference["y_true"].to_numpy()):
            raise ValueError(f"{prediction_split}: labels do not reproduce.")
        if not np.array_equal(
            probability_matrix.argmax(axis=1),
            reference["y_pred"].to_numpy(),
        ):
            raise ValueError(f"{prediction_split}: predictions do not reproduce.")
        # cuDNN batch scheduling can cause sub-millipercent floating-point
        # differences while preserving every frozen predicted class.
        if maximum_probability_error > 5e-4:
            raise ValueError(
                f"{prediction_split}: probability error "
                f"{maximum_probability_error} exceeds tolerance."
            )
        metadata = {
            "schema_version": "ophagent.frozen_scout_representation.v0_1",
            "model_id": "flair",
            "split": prediction_split,
            "representation": "classification_head_input",
            "embedding_dim": int(feature_matrix.shape[1]),
            "case_count": int(len(case_ids)),
            "encoder_checkpoint_sha256": file_sha256(
                Path(adapter_spec.checkpoint_path)
            ),
            "task_checkpoint_sha256": file_sha256(
                Path(str(adapter_spec.task_checkpoint_path))
            ),
            "preprocessing_id": adapter_spec.preprocessing_id,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "source_prediction_sha256": file_sha256(
                PROJECT_ROOT
                / "experiments/opening_risk_routing_closure/replays/flair/"
                f"full_replay_20260722/predictions/"
                f"{prediction_split}_predictions.csv"
            ),
            "prediction_reproduction_max_abs_error": (
                maximum_probability_error
            ),
            "prediction_class_exact_match": True,
            "one_time_retrospective_forward_ms_per_image": (
                elapsed_seconds * 1000.0 / len(case_ids)
            ),
            "incremental_online_encoder_forward_ms_per_image": 0.0,
            "incremental_online_contract": (
                "retain the classification-head input from the same Scout "
                "forward; no second encoder call"
            ),
        }
        target_path = args.output_dir / (
            f"flair_{prediction_split}_representations.npz"
        )
        np.savez_compressed(
            target_path,
            case_ids=np.asarray(case_ids),
            y_true=label_vector,
            embeddings=feature_matrix,
            metadata_json=np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            ),
        )
        outputs[prediction_split] = {
            **metadata,
            "asset_path": str(target_path),
            "asset_size_bytes": target_path.stat().st_size,
            "asset_sha256": file_sha256(target_path),
        }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
