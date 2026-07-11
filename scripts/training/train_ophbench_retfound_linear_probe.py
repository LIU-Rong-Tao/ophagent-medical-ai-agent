#!/usr/bin/env python3
"""Train a frozen RETFound CFP feature linear probe without touching test labels for selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ophbench_task_adapter import (  # noqa: E402
    ARTIFACT_ID,
    LABELS,
    build_prediction_frame,
    registration_record,
)
from ophbench import __version__ as ophbench_version  # noqa: E402
from ophbench import load_adapter  # noqa: E402
from scripts.routing.timm_adapter_runtime import sha256_file  # noqa: E402


def extract_split(adapter, root: Path, *, batch_size: int, workers: int):
    dataset = ImageFolder(root, transform=adapter.preprocess)
    if tuple(dataset.classes) != (
        "anodr",
        "bmilddr",
        "cmoderatedr",
        "dseveredr",
        "eproliferativedr",
    ):
        raise ValueError(f"Unexpected APTOS class order: {dataset.classes}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    features, labels, paths = [], [], []
    offset = 0
    started = time.perf_counter()
    for images, targets in loader:
        encoded = adapter.encode_image(images).detach().cpu().numpy()
        features.append(encoded)
        labels.append(targets.numpy())
        paths.extend(path for path, _ in dataset.samples[offset : offset + len(targets)])
        offset += len(targets)
    return (
        np.concatenate(features),
        np.concatenate(labels),
        paths,
        time.perf_counter() - started,
    )


def metrics(y_true, probabilities):
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "quadratic_kappa": float(cohen_kappa_score(y_true, prediction, weights="quadratic")),
        "n": int(len(y_true)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(
        model_id="retfound",
        checkpoint_id="retfound-cfp",
        checkpoint_path=args.encoder_checkpoint,
        device=args.device,
    ).load()
    extracted = {
        split: extract_split(
            adapter, args.data_root / split, batch_size=args.batch_size, workers=args.workers
        )
        for split in ("train", "val", "test")
    }
    train_x, train_y, _, _ = extracted["train"]
    val_x, val_y, _, _ = extracted["val"]
    candidates = []
    for c_value in (0.01, 0.1, 1.0):
        candidate = LogisticRegression(C=c_value, max_iter=2000, random_state=args.seed)
        candidate.fit(train_x, train_y)
        candidates.append((metrics(val_y, candidate.predict_proba(val_x))["macro_f1"], candidate))
    _, classifier = max(candidates, key=lambda item: item[0])
    head_path = args.output_dir / "linear_probe.joblib"
    joblib.dump(classifier, head_path)

    split_metrics = {}
    prediction_path = args.output_dir / "test_predictions.csv"
    for split, (features, labels, paths, elapsed) in extracted.items():
        probabilities = classifier.predict_proba(features)
        split_metrics[split] = {**metrics(labels, probabilities), "feature_seconds": elapsed}
        if split == "test":
            build_prediction_frame(paths, labels, probabilities).to_csv(prediction_path, index=False)

    encoder_sha = sha256_file(args.encoder_checkpoint)
    record = registration_record(
        output_dir=args.output_dir,
        prediction_path=prediction_path,
        head_checkpoint=head_path,
        encoder_sha256=encoder_sha,
    )
    pd.DataFrame([record]).to_csv(args.output_dir / "registration_record.csv", index=False)
    manifest = {
        "artifact_id": ARTIFACT_ID,
        "base_model_provider": "ophbench",
        "base_model_id": "retfound",
        "base_checkpoint_id": "retfound-cfp",
        "encoder_checkpoint_sha256": encoder_sha,
        "head_checkpoint_sha256": sha256_file(head_path),
        "preprocessing": "RETFoundCFPAdapter v0.2.0 official eval transform",
        "labels": LABELS,
        "selection": "validation macro-F1 over C=[0.01,0.1,1.0]; test evaluated once after selection",
        "seed": args.seed,
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ophbench": ophbench_version,
        "metrics": split_metrics,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
