#!/usr/bin/env python3
"""Fit the official FLAIR linear probe on the controlled APTOS split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.flair_task_adapter import (  # noqa: E402
    FLAIR_CHECKPOINT_SHA256,
    FLAIR_SOURCE_COMMIT,
    FlairAptosTaskAdapter,
    load_flair_vision,
    preprocess_flair_image,
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

ARTIFACT_ID = "aptos2019-flair-resnet50-official-lp-project-v1"


def strict_preflight(config):
    source = Path(config["foundation"]["source_root"])
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != FLAIR_SOURCE_COMMIT:
        raise ValueError(f"FLAIR 实际源码 commit 不匹配：{commit}")
    if sha256_file(config["foundation"]["checkpoint_path"]) != FLAIR_CHECKPOINT_SHA256:
        raise ValueError("FLAIR checkpoint SHA256 不匹配")
    manifest, digest = dataset_manifest(config["data"]["root"])
    if digest != APTOS_MANIFEST_SHA256:
        raise ValueError("APTOS 数据清单与冻结划分不一致")
    classifier = config["classifier"]
    expected = {"c": 0.316, "max_iter": 1000, "class_weight": "balanced", "random_state": 0}
    if any(classifier[key] != value for key, value in expected.items()):
        raise ValueError("FLAIR LinearProbe 必须保留官方固定参数")
    if config["evaluation"]["selection_split"] != "none_fixed_official_c":
        raise ValueError("固定官方 C 不允许再用 test 或 validation 搜索")
    return {
        "strict_preflight": True,
        "source_commit": commit,
        "checkpoint_sha256": FLAIR_CHECKPOINT_SHA256,
        "dataset_manifest_sha256": digest,
        "split_sizes": {key: value["samples"] for key, value in manifest["splits"].items()},
        "official_aptos_recipe_available": False,
        "test_used_for_selection": False,
    }


def build_datasets(config):
    root = Path(config["data"]["root"])
    return {
        split: ImageFolder(root / split, transform=preprocess_flair_image)
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
    return np.concatenate(features), np.concatenate(labels)


def torch_classifier(probe):
    classifier = torch.nn.Linear(probe.coef_.shape[1], probe.coef_.shape[0])
    classifier.weight.data.copy_(torch.from_numpy(probe.coef_).float())
    classifier.bias.data.copy_(torch.from_numpy(probe.intercept_).float())
    return classifier


@torch.inference_mode()
def task_probabilities(classifier, features, device="cpu"):
    classifier = classifier.to(device)
    feature_tensor = torch.from_numpy(features).float().to(device)
    return torch.softmax(classifier(feature_tensor), dim=1).cpu().numpy()


def run_training(config_path: Path, overrides=None):
    config = load_config(config_path)
    for dotted, value in dict(overrides or {}).items():
        if value is not None:
            section, key = dotted.split(".", 1)
            config[section][key] = value
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
    encoder = load_flair_vision(
        config["foundation"]["checkpoint_path"], config["runtime"]["device"]
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
    checkpoint_path = output / "flair_aptos_task_checkpoint.pth"
    torch.save(
        {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "labels": APTOS_LABELS,
            "source_commit": FLAIR_SOURCE_COMMIT,
            "encoder_checkpoint_sha256": FLAIR_CHECKPOINT_SHA256,
            "dataset_manifest_sha256": APTOS_MANIFEST_SHA256,
            "classifier_state_dict": classifier.state_dict(),
            "official_linear_probe_c": 0.316,
        },
        checkpoint_path,
    )
    record = {
        "model_id": f"aptos_dr_5class::{ARTIFACT_ID}",
        "task_id": "aptos_dr_5class",
        "dataset_id": "APTOS2019",
        "dataset_display_name": "APTOS 2019",
        "dataset_source": "public",
        "artifact_id": ARTIFACT_ID,
        "model_family": "flair",
        "architecture": "FLAIR ResNet-50",
        "label_space": "dr_icdr_0_4",
        "n_classes": 5,
        "prediction_source": "adapter",
        "prediction_path": str(prediction_path),
        "adapter_status": "completed",
        "compatibility_status": "offline_evaluation_ready",
        "role_candidates": "scout|expert",
        "checkpoint_path": str(checkpoint_path),
        "base_model_provider": "ophbench",
        "base_model_id": "flair",
        "base_checkpoint_id": "flair-default",
        "encoder_checkpoint_sha256": FLAIR_CHECKPOINT_SHA256,
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
        "trainer_adapter": "flair_aptos_official_lp_project_v1",
        "task_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    pd.DataFrame([record]).to_csv(output / "registration_record.csv", index=False)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": utc_now(),
                "artifact_id": ARTIFACT_ID,
                "claim_status": "official_framework_project_downstream_adaptation",
                "official_aptos_recipe_available": False,
                "official_components_reused": [
                    "FLAIR ResNet-50 projected normalized vision features",
                    "512 foreground crop and aspect-preserving canvas resize",
                    "LinearProbe C=0.316, class_weight=balanced, max_iter=1000",
                ],
                "project_component": "controlled APTOS train/val/test mapping",
                "test_used_for_selection": False,
                "route_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    adapter = FlairAptosTaskAdapter.load(
        encoder_checkpoint=config["foundation"]["checkpoint_path"],
        task_checkpoint=checkpoint_path,
        device=config["runtime"]["device"],
    )
    sample_images = [datasets["test"].loader(path) for path, _ in datasets["test"].samples[:2]]
    if adapter.predict_proba(sample_images).shape != (2, 5):
        raise ValueError("FLAIR 任务 Adapter 最终 Smoke 失败")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    output = run_training(
        args.config,
        overrides={
            "data.root": str(args.data_root) if args.data_root else None,
            "foundation.checkpoint_path": str(args.checkpoint) if args.checkpoint else None,
            "foundation.source_root": str(args.source_root) if args.source_root else None,
            "output.run_dir": str(args.output_dir) if args.output_dir else None,
            "runtime.device": args.device,
        },
    )
    print(f"FLAIR APTOS adaptation completed: {output}")


if __name__ == "__main__":
    main()
