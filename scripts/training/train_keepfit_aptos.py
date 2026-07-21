#!/usr/bin/env python3
"""Fit the official KeepFIT linear probe on the controlled APTOS split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.keepfit_task_adapter import (  # noqa: E402
    KEEPFIT_CHECKPOINTS,
    KEEPFIT_SOURCE_COMMIT,
    KeepFITAptosTaskAdapter,
    load_keepfit_vision,
    preprocess_keepfit_image,
    sha256_file,
)
from scripts.training.aptos_downstream_common import (  # noqa: E402
    APTOS_LABELS,
    APTOS_MANIFEST_SHA256,
    classification_metrics,
    dataset_manifest,
    load_config,
    prediction_frame,
    utc_now,
)
from scripts.training.train_flair_aptos import task_probabilities, torch_classifier  # noqa: E402


def strict_preflight(config):
    checkpoint_id = str(config["foundation"]["base_checkpoint_id"])
    checkpoint_spec = KEEPFIT_CHECKPOINTS.get(checkpoint_id)
    if checkpoint_spec is None:
        raise ValueError(f"不支持的 KeepFIT CFP checkpoint：{checkpoint_id}")
    source = Path(config["foundation"]["source_root"])
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != KEEPFIT_SOURCE_COMMIT:
        raise ValueError(f"KeepFIT 实际源码 commit 不匹配：{commit}")
    actual_sha = sha256_file(config["foundation"]["checkpoint_path"])
    if actual_sha != checkpoint_spec["sha256"]:
        raise ValueError("KeepFIT checkpoint SHA256 不匹配")
    manifest, digest = dataset_manifest(config["data"]["root"])
    if digest != APTOS_MANIFEST_SHA256:
        raise ValueError("APTOS 数据清单与冻结划分不一致")
    classifier = config["classifier"]
    expected = {"c": 0.316, "max_iter": 1000, "class_weight": "balanced", "random_state": 0}
    if any(classifier[key] != value for key, value in expected.items()):
        raise ValueError("KeepFIT LinearProbe 必须保留官方固定参数")
    if config["foundation"]["project_features"] is not False:
        raise ValueError("KeepFIT 官方迁移入口默认使用未投影的 2048 维视觉特征")
    if config["evaluation"]["selection_split"] != "none_fixed_official_c":
        raise ValueError("固定官方 C 不允许再用 test 或 validation 搜索")
    return {
        "strict_preflight": True,
        "source_commit": commit,
        "checkpoint_sha256": actual_sha,
        "dataset_manifest_sha256": digest,
        "split_sizes": {key: value["samples"] for key, value in manifest["splits"].items()},
        "official_aptos_recipe_available": False,
        "test_used_for_selection": False,
    }


def build_datasets(config):
    root = Path(config["data"]["root"])
    return {
        split: ImageFolder(root / split, transform=preprocess_keepfit_image)
        for split in ("train", "val", "test")
    }


@torch.inference_mode()
def extract_features(encoder, dataset, config):
    loader = DataLoader(
        dataset,
        batch_size=int(config["runtime"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=True,
    )
    device = config["runtime"]["device"]
    features, labels = [], []
    for images, targets in loader:
        features.append(encoder(images.to(device, non_blocking=True)).cpu().numpy())
        labels.append(targets.numpy())
    import numpy as np

    return np.concatenate(features), np.concatenate(labels)


def run_training(config_path: Path, overrides=None):
    config = load_config(config_path)
    for dotted, value in dict(overrides or {}).items():
        if value is not None:
            section, key = dotted.split(".", 1)
            config[section][key] = value
    checkpoint_id = str(config["foundation"]["base_checkpoint_id"])
    checkpoint_spec = KEEPFIT_CHECKPOINTS[checkpoint_id]
    artifact_id = str(checkpoint_spec["artifact_id"])
    preflight = strict_preflight(config)
    output = Path(config["output"]["run_dir"])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "base_protocol.yaml")
    (output / "effective_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest, _ = dataset_manifest(config["data"]["root"])
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "validation_report.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    datasets = build_datasets(config)
    encoder = load_keepfit_vision(
        config["foundation"]["checkpoint_path"], checkpoint_id, config["runtime"]["device"]
    )
    train_x, train_y = extract_features(encoder, datasets["train"], config)
    val_x, val_y = extract_features(encoder, datasets["val"], config)
    test_x, test_y = extract_features(encoder, datasets["test"], config)
    probe = LogisticRegression(
        random_state=0,
        C=0.316,
        max_iter=1000,
        class_weight="balanced",
    ).fit(train_x, train_y)
    classifier = torch_classifier(probe).eval()
    device = config["runtime"]["device"]
    val_probabilities = task_probabilities(classifier, val_x, device)
    test_probabilities = task_probabilities(classifier, test_x, device)
    metrics = {
        "selection_split": "none_fixed_official_c",
        "test_used_for_selection": False,
        "validation": classification_metrics(val_y, val_probabilities),
        "test": classification_metrics(test_y, test_probabilities),
        "classifier_iterations": [int(value) for value in probe.n_iter_],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    prediction_path = output / "test_predictions.csv"
    prediction_frame(datasets["test"], test_probabilities, config["data"]["root"]).to_csv(
        prediction_path, index=False
    )
    checkpoint_path = output / "keepfit_aptos_task_checkpoint.pth"
    torch.save(
        {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "labels": APTOS_LABELS,
            "source_commit": KEEPFIT_SOURCE_COMMIT,
            "base_checkpoint_id": checkpoint_id,
            "encoder_checkpoint_sha256": checkpoint_spec["sha256"],
            "dataset_manifest_sha256": APTOS_MANIFEST_SHA256,
            "classifier_state_dict": classifier.cpu().state_dict(),
            "official_linear_probe_c": 0.316,
            "project_features": False,
        },
        checkpoint_path,
    )
    record = {
        "model_id": f"aptos_dr_5class::{artifact_id}",
        "task_id": "aptos_dr_5class",
        "dataset_id": "APTOS2019",
        "dataset_display_name": "APTOS 2019",
        "dataset_source": "public",
        "artifact_id": artifact_id,
        "model_family": "keepfit",
        "architecture": str(checkpoint_spec["display_name"]),
        "label_space": "dr_icdr_0_4",
        "n_classes": 5,
        "prediction_source": "adapter",
        "prediction_path": str(prediction_path),
        "adapter_status": "completed",
        "compatibility_status": "offline_evaluation_ready",
        "role_candidates": "scout|expert",
        "checkpoint_path": str(checkpoint_path),
        "base_model_provider": "ophbench",
        "base_model_id": "keepfit",
        "base_checkpoint_id": checkpoint_id,
        "encoder_checkpoint_sha256": checkpoint_spec["sha256"],
        "task_checkpoint": True,
        "task_adapted": True,
        "task_inference_ready": True,
        "offline_evaluation_eligible": True,
        "unified_evaluation_completed": True,
        "inference_cost_measured": False,
        "route_eligible": False,
        "output_dir": str(output),
        "evaluation_role": "official_framework_project_downstream_adaptation",
        "lifecycle_status": "candidate",
        "research_claim_status": "official_framework_project_downstream_adaptation",
        "cost_status": "unmeasured",
        "selection_split": "none_fixed_official_c",
        "selection_metric": "fixed_official_c",
        "test_used_for_selection": False,
        "trainer_adapter": "keepfit_aptos_official_lp_project_v1",
        "task_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    pd.DataFrame([record]).to_csv(output / "registration_record.csv", index=False)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": utc_now(),
                "artifact_id": artifact_id,
                "base_checkpoint_id": checkpoint_id,
                "claim_status": "official_framework_project_downstream_adaptation",
                "official_aptos_recipe_available": False,
                "official_components_reused": [
                    "KeepFIT ResNet-50 unprojected 2048-dimensional vision features",
                    "512 aspect-preserving canvas resize without foreground crop",
                    "LinearProbe C=0.316, class_weight=balanced, max_iter=1000",
                ],
                "project_component": "controlled APTOS train/val/test mapping",
                "ablation_checkpoint": checkpoint_id.startswith("keepfit-half-"),
                "test_used_for_selection": False,
                "route_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    adapter = KeepFITAptosTaskAdapter.load(
        encoder_checkpoint=config["foundation"]["checkpoint_path"],
        task_checkpoint=checkpoint_path,
        device=config["runtime"]["device"],
    )
    images = [datasets["test"].loader(path) for path, _ in datasets["test"].samples[:2]]
    if adapter.predict_proba(images).shape != (2, 5):
        raise ValueError("KeepFIT 任务 Adapter 最终 Smoke 失败")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = run_training(
        args.config,
        overrides={
            "data.root": str(args.data_root),
            "foundation.base_checkpoint_id": args.checkpoint_id,
            "foundation.checkpoint_path": str(args.checkpoint),
            "foundation.source_root": str(args.source_root),
            "output.run_dir": str(args.output_dir),
            "runtime.device": args.device,
        },
    )
    print(f"KeepFIT APTOS adaptation completed: {output}")


if __name__ == "__main__":
    main()
