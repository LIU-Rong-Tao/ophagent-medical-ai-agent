"""Shared, task-only utilities for controlled APTOS downstream adaptations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from torchvision.datasets import ImageFolder
import yaml

from scripts.routing.timm_adapter_runtime import normalize_prediction_frame


APTOS_CLASSES = (
    "anodr",
    "bmilddr",
    "cmoderatedr",
    "dseveredr",
    "eproliferativedr",
)
APTOS_LABELS = (
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
)
APTOS_MANIFEST_SHA256 = "4d3332aab0f010ccf1fefa23af51e65fd2764558bc5a6d6c153ba13379949765"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path | str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("APTOS 训练协议必须为 mapping")
    return payload


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_manifest(data_root: Path | str) -> tuple[dict[str, Any], str]:
    root = Path(data_root)
    entries: list[dict[str, Any]] = []
    distributions: dict[str, Any] = {}
    split_keys: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        dataset = ImageFolder(root / split)
        if tuple(dataset.classes) != APTOS_CLASSES:
            raise ValueError(f"{split} 类别顺序错误：{dataset.classes}")
        counts = {str(index): 0 for index in range(len(APTOS_CLASSES))}
        keys: set[str] = set()
        for path, label in dataset.samples:
            key = Path(path).stem
            if key in keys:
                raise ValueError(f"{split} 存在重复 image_key：{key}")
            keys.add(key)
            counts[str(label)] += 1
            entries.append(
                {
                    "split": split,
                    "relative_path": Path(path).relative_to(root).as_posix(),
                    "label": int(label),
                }
            )
        split_keys[split] = keys
        distributions[split] = {
            "samples": len(dataset),
            "class_distribution": counts,
        }
    splits = tuple(split_keys)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = split_keys[left] & split_keys[right]
            if overlap:
                raise ValueError(f"{left}/{right} 存在重复图像键：{sorted(overlap)[:5]}")
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "splits": distributions,
        "entries": entries,
        "manifest_sha256": digest,
    }, digest


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=float)[labels]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "quadratic_kappa": float(cohen_kappa_score(labels, predictions, weights="quadratic")),
        "macro_auc_ovr": float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")),
        "n": int(len(labels)),
    }


def prediction_frame(dataset, probabilities: np.ndarray, data_root: Path | str) -> pd.DataFrame:
    root = Path(data_root)
    labels = np.asarray(dataset.targets, dtype=int)
    frame = pd.DataFrame(
        {
            "image_path": [Path(path).relative_to(root).as_posix() for path, _ in dataset.samples],
            "true_label": labels,
            "pred_label": probabilities.argmax(axis=1),
        }
    )
    for index in range(probabilities.shape[1]):
        frame[f"prob_{index}"] = probabilities[:, index]
    with tempfile.TemporaryDirectory(prefix="ophagent-aptos-") as directory:
        path = Path(directory) / "predictions.csv"
        frame.to_csv(path, index=False)
        return normalize_prediction_frame(path, num_classes=probabilities.shape[1])
