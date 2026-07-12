#!/usr/bin/env python3
"""标准 RETFound CFP 冻结特征线性探针 trainer adapter。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ophbench_task_adapter import (  # noqa: E402
    LABELS,
    STANDARD_ARTIFACT_ID,
    build_prediction_frame,
    registration_record,
)
from ophbench import __version__ as ophbench_version  # noqa: E402
from ophbench import load_adapter  # noqa: E402
from scripts.routing.timm_adapter_runtime import sha256_file  # noqa: E402

EXPECTED_CLASSES = (
    "anodr",
    "bmilddr",
    "cmoderatedr",
    "dseveredr",
    "eproliferativedr",
)
REQUIRED_SPLITS = ("train", "val", "test")
STANDARD_RUN_FILES = {
    "base_recipe.yaml",
    "submitted_config.yaml",
    "effective_config.yaml",
    "validation_report.json",
    "dataset_manifest.json",
    "run_manifest.json",
    "metrics.json",
    "linear_probe.joblib",
    "test_predictions.csv",
    "registration_record.csv",
}
REGISTRATION_PROVENANCE_FIELDS = {
    "evaluation_role",
    "recipe_id",
    "run_id",
    "base_recipe_sha256",
    "effective_config_sha256",
    "dataset_manifest_sha256",
    "selected_C",
    "classifier_type",
    "prediction_schema_version",
    "ophbench_version",
    "adapter_version",
    "encoder_checkpoint_sha256",
    "head_checkpoint_sha256",
}


def validate_standard_run(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    missing = sorted(name for name in STANDARD_RUN_FILES if not (output_dir / name).is_file())
    if missing:
        raise ValueError(f"标准运行目录缺少文件：{missing}")
    registration = pd.read_csv(output_dir / "registration_record.csv")
    if len(registration) != 1:
        raise ValueError("registration_record 必须且只能包含一条任务产物")
    missing_fields = sorted(REGISTRATION_PROVENANCE_FIELDS - set(registration.columns))
    if missing_fields:
        raise ValueError(f"registration provenance 缺少字段：{missing_fields}")
    row = registration.iloc[0]
    if not bool(row["task_checkpoint"]) or not bool(row["task_inference_ready"]):
        raise ValueError("标准任务产物尚未达到任务推理就绪状态")
    if not bool(row["route_eligible"]):
        raise ValueError("标准任务产物尚未获得路由资格")
    return {"ok": True, "artifact_id": str(row["artifact_id"]), "files": len(STANDARD_RUN_FILES)}


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("effective_config 必须为 mapping")
    return payload


def _dataset_manifest(data_root: Path) -> tuple[dict[str, Any], str]:
    entries = []
    distributions = {}
    split_keys: dict[str, set[str]] = {}
    for split in REQUIRED_SPLITS:
        dataset = ImageFolder(data_root / split)
        if tuple(dataset.classes) != EXPECTED_CLASSES:
            raise ValueError(f"{split} 类别顺序错误：{dataset.classes}")
        counts = {str(index): 0 for index in range(5)}
        keys = set()
        for path, label in dataset.samples:
            relative = Path(path).relative_to(data_root).as_posix()
            image_key = Path(path).stem
            if image_key in keys:
                raise ValueError(f"{split} 存在重复 image_key：{image_key}")
            keys.add(image_key)
            counts[str(label)] += 1
            entries.append({"split": split, "relative_path": relative, "label": int(label)})
        split_keys[split] = keys
        distributions[split] = {"samples": len(dataset), "class_distribution": counts}
    for left_index, left in enumerate(REQUIRED_SPLITS):
        for right in REQUIRED_SPLITS[left_index + 1 :]:
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


def strict_preflight(config: dict[str, Any]) -> dict[str, Any]:
    version = tuple(int(part) for part in ophbench_version.split(".")[:2])
    if version < (0, 2):
        raise ValueError(f"ophbench>=0.2.0 required, found {ophbench_version}")
    foundation = config["foundation"]
    if foundation.get("encoder_frozen") is not True:
        raise ValueError("RETFound 标准线性探针必须冻结 encoder")
    checkpoint = Path(foundation["encoder_checkpoint_path"])
    if not checkpoint.is_file():
        raise ValueError(f"基础 checkpoint 不存在：{checkpoint}")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != foundation["encoder_checkpoint_sha256"]:
        raise ValueError("基础 checkpoint SHA256 不匹配")
    environment = load_adapter(
        model_id="retfound",
        checkpoint_id="retfound-cfp",
        checkpoint_path=checkpoint,
    ).check_environment()
    if not environment.available:
        raise ValueError(environment.message)
    manifest, manifest_sha = _dataset_manifest(Path(config["data"]["root"]))
    if config["classifier"]["selection_metric"] != "macro_f1":
        raise ValueError("C 只能使用 val macro_f1 选择")
    return {
        "strict_preflight": True,
        "ophbench_version": ophbench_version,
        "encoder_checkpoint_sha256": actual_sha,
        "dataset_manifest_sha256": manifest_sha,
        "split_sizes": {key: value["samples"] for key, value in manifest["splits"].items()},
        "selection_split": "val",
        "test_used_for_selection": False,
        "encoder_frozen": True,
    }


def _extract_split(adapter, root: Path, *, batch_size: int, workers: int):
    dataset = ImageFolder(root, transform=adapter.preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    features, labels, paths = [], [], []
    offset = 0
    started = time.perf_counter()
    for images, targets in loader:
        features.append(adapter.encode_image(images).detach().cpu().numpy())
        labels.append(targets.numpy())
        paths.extend(path for path, _ in dataset.samples[offset : offset + len(targets)])
        offset += len(targets)
    return np.concatenate(features), np.concatenate(labels), paths, time.perf_counter() - started


def _metrics(y_true, probabilities):
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "quadratic_kappa": float(cohen_kappa_score(y_true, prediction, weights="quadratic")),
        "n": int(len(y_true)),
    }


def run_training(config_path: Path) -> Path:
    config_path = Path(config_path)
    config = _load_config(config_path)
    preflight = strict_preflight(config)
    output_dir = Path(config["output"]["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(config["data"]["root"])
    manifest, manifest_sha = _dataset_manifest(data_root)
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    foundation = config["foundation"]
    adapter = load_adapter(
        model_id="retfound",
        checkpoint_id="retfound-cfp",
        checkpoint_path=foundation["encoder_checkpoint_path"],
        device=config["runtime"]["device"],
    ).load()
    extracted = {
        split: _extract_split(
            adapter,
            data_root / split,
            batch_size=int(config["training"]["batch_size"]),
            workers=int(config["runtime"]["num_workers"]),
        )
        for split in REQUIRED_SPLITS
    }
    train_x, train_y, _, _ = extracted["train"]
    val_x, val_y, _, _ = extracted["val"]
    candidates = []
    for c_value in config["classifier"]["c_candidates"]:
        classifier = LogisticRegression(
            C=float(c_value),
            max_iter=int(config["classifier"]["max_iter"]),
            random_state=int(config["training"]["seed"]),
        )
        classifier.fit(train_x, train_y)
        score = _metrics(val_y, classifier.predict_proba(val_x))["macro_f1"]
        candidates.append((score, float(c_value), classifier))
    _, selected_c, classifier = max(candidates, key=lambda item: item[0])
    if classifier.classes_.tolist() != [0, 1, 2, 3, 4]:
        raise ValueError(f"线性头类别错误：{classifier.classes_.tolist()}")
    head_path = output_dir / "linear_probe.joblib"
    joblib.dump(classifier, head_path)
    prediction_path = output_dir / "test_predictions.csv"
    split_metrics = {}
    for split, (features, labels, paths, elapsed) in extracted.items():
        probabilities = classifier.predict_proba(features)
        split_metrics[split] = {**_metrics(labels, probabilities), "feature_seconds": elapsed}
        if split == "test":
            build_prediction_frame(paths, labels, probabilities).to_csv(prediction_path, index=False)
    metrics_payload = {
        "selection_split": "val",
        "selection_metric": "macro_f1",
        "selected_C": selected_c,
        "test_used_for_selection": False,
        "splits": split_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_id = output_dir.name
    provenance = {
        "evaluation_role": "controlled_task_adaptation",
        "recipe_id": config["recipe"]["recipe_id"],
        "run_id": run_id,
        "base_recipe_sha256": sha256_file(output_dir / "base_recipe.yaml"),
        "effective_config_sha256": sha256_file(output_dir / "effective_config.yaml"),
        "dataset_manifest_sha256": manifest_sha,
        "selected_C": selected_c,
        "classifier_type": "logistic_regression",
        "prediction_schema_version": "ophagent_v1",
        "ophbench_version": ophbench_version,
        "adapter_version": foundation["adapter_version"],
        "head_checkpoint_sha256": sha256_file(head_path),
    }
    record = registration_record(
        output_dir=output_dir,
        prediction_path=prediction_path,
        head_checkpoint=head_path,
        encoder_sha256=preflight["encoder_checkpoint_sha256"],
        artifact_id=STANDARD_ARTIFACT_ID,
        evaluation_role="integration_validation",
        lifecycle_status="active",
        route_eligible=True,
        research_claim_status="not_for_scientific_comparison",
        cost_status="unmeasured",
        provenance=provenance,
    )
    pd.DataFrame([record]).to_csv(output_dir / "registration_record.csv", index=False)
    run_manifest = {
        **provenance,
        "artifact_id": STANDARD_ARTIFACT_ID,
        "evaluation_role": "integration_validation",
        "research_claim_status": "not_for_scientific_comparison",
        "cost_status": "unmeasured",
        "base_model_provider": "ophbench",
        "base_model_id": "retfound",
        "base_checkpoint_id": "retfound-cfp",
        "encoder_checkpoint_sha256": preflight["encoder_checkpoint_sha256"],
        "head_checkpoint_sha256": provenance["head_checkpoint_sha256"],
        "labels": LABELS,
        "seed": config["training"]["seed"],
        "batch_size": config["training"]["batch_size"],
        "workers": config["runtime"]["num_workers"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "metrics": metrics_payload,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    validate_standard_run(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
