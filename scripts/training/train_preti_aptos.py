#!/usr/bin/env python3
"""PRETI paper-anchored APTOS downstream adaptation for OphAgent."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.preti_task_adapter import (  # noqa: E402
    ENCODER_PREFIXES,
    LABELS,
    PRETI_SOURCE_COMMIT,
    PretiAptosTaskAdapter,
    PretiClassifier,
    encoder_state_dict,
    load_preti_foundation,
    set_encoder_trainable,
    sha256_file,
)
from scripts.routing.timm_adapter_runtime import (  # noqa: E402
    normalize_prediction_frame,
)

EXPECTED_CLASSES = (
    "anodr",
    "bmilddr",
    "cmoderatedr",
    "dseveredr",
    "eproliferativedr",
)
REQUIRED_SPLITS = ("train", "val", "test")
ARTIFACT_ID = "aptos2019-preti-vitb-paper-anchored-v1"


@dataclass
class CandidateResult:
    learning_rate: float
    best_epoch: int
    validation_metrics: dict[str, float]
    encoder_state: dict[str, object]
    classifier_state: dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PRETI 配置必须为 mapping")
    return payload


def dataset_manifest(data_root: Path) -> tuple[dict[str, Any], str]:
    entries = []
    distributions = {}
    split_keys: dict[str, set[str]] = {}
    for split in REQUIRED_SPLITS:
        dataset = ImageFolder(data_root / split)
        if tuple(dataset.classes) != EXPECTED_CLASSES:
            raise ValueError(f"{split} 类别顺序错误：{dataset.classes}")
        counts = {str(index): 0 for index in range(len(EXPECTED_CLASSES))}
        keys = set()
        for path, label in dataset.samples:
            relative = Path(path).relative_to(data_root).as_posix()
            image_key = Path(path).stem
            if image_key in keys:
                raise ValueError(f"{split} 存在重复 image_key：{image_key}")
            keys.add(image_key)
            counts[str(label)] += 1
            entries.append(
                {"split": split, "relative_path": relative, "label": int(label)}
            )
        split_keys[split] = keys
        distributions[split] = {
            "samples": len(dataset),
            "class_distribution": counts,
        }
    for index, left in enumerate(REQUIRED_SPLITS):
        for right in REQUIRED_SPLITS[index + 1 :]:
            overlap = split_keys[left] & split_keys[right]
            if overlap:
                raise ValueError(
                    f"{left}/{right} 存在重复图像键：{sorted(overlap)[:5]}"
                )
    canonical = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "splits": distributions,
        "entries": entries,
        "manifest_sha256": digest,
    }, digest


def strict_preflight(config: dict[str, Any]) -> dict[str, Any]:
    foundation = config["foundation"]
    source_root = Path(foundation["source_root"])
    checkpoint = Path(foundation["checkpoint_path"])
    data_root = Path(config["data"]["root"])
    if foundation["source_commit"] != PRETI_SOURCE_COMMIT:
        raise ValueError("PRETI 源码 commit 与冻结运行契约不一致")
    actual_source_commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_source_commit != PRETI_SOURCE_COMMIT:
        raise ValueError(
            "PRETI 实际源码 commit 与冻结运行契约不一致："
            f"{actual_source_commit}"
        )
    actual_sha = sha256_file(checkpoint)
    if actual_sha != foundation["checkpoint_sha256"]:
        raise ValueError("PRETI checkpoint SHA256 不匹配")
    manifest, manifest_sha = dataset_manifest(data_root)
    expected_manifest = str(config["data"].get("expected_manifest_sha256", ""))
    if expected_manifest and manifest_sha != expected_manifest:
        raise ValueError("APTOS 数据清单与既有受控划分不一致")
    if config["evaluation"]["selection_split"] != "val":
        raise ValueError("PRETI checkpoint 和超参数只能使用 validation 选择")
    if config["evaluation"]["save_best_by"] != "quadratic_kappa":
        raise ValueError("APTOS 当前注册主指标必须为 validation QWK")
    if config["training"]["epochs"] != 50:
        raise ValueError("论文锚定运行必须保持 50 epochs")
    if config["training"]["batch_size"] != 16:
        raise ValueError("论文锚定运行必须保持 batch size 16")
    return {
        "strict_preflight": True,
        "source_commit": actual_source_commit,
        "checkpoint_sha256": actual_sha,
        "dataset_manifest_sha256": manifest_sha,
        "split_sizes": {
            name: detail["samples"] for name, detail in manifest["splits"].items()
        },
        "selection_split": "val",
        "selection_metric": "quadratic_kappa",
        "test_used_for_selection": False,
        "official_downstream_recipe_complete": False,
    }


def build_transforms(config: dict[str, Any]):
    augmentation = config["augmentation"]
    normalization = transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (224, 224),
                scale=tuple(augmentation["crop_scale"]),
                ratio=tuple(augmentation["crop_ratio"]),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip_probability"])
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            normalization,
        ]
    )
    return train_transform, evaluation_transform


def build_loaders(config: dict[str, Any]):
    data_root = Path(config["data"]["root"])
    train_transform, evaluation_transform = build_transforms(config)
    datasets = {
        "train": ImageFolder(data_root / "train", transform=train_transform),
        "val": ImageFolder(data_root / "val", transform=evaluation_transform),
        "test": ImageFolder(data_root / "test", transform=evaluation_transform),
    }
    seed = int(config["training"]["seed"])
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["runtime"]["num_workers"]),
        "pin_memory": True,
    }
    return datasets, {
        "train": DataLoader(
            datasets["train"], shuffle=True, generator=generator, **common
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def metrics_from_probabilities(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "quadratic_kappa": float(
            cohen_kappa_score(labels, predictions, weights="quadratic")
        ),
        "n": int(len(labels)),
    }


def choose_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    if not candidates:
        raise ValueError("没有可选 PRETI 训练候选")
    return max(
        candidates,
        key=lambda item: (
            item.validation_metrics["quadratic_kappa"],
            item.validation_metrics["macro_f1"],
            -item.learning_rate,
        ),
    )


def _set_learning_rate(optimizer, base_lr: float, epoch: int, config) -> float:
    epochs = int(config["training"]["epochs"])
    warmup = int(config["scheduler"]["warmup_epochs"])
    minimum = float(config["scheduler"]["minimum_learning_rate"])
    if warmup and epoch < warmup:
        learning_rate = base_lr * float(epoch + 1) / warmup
    else:
        denominator = max(epochs - warmup - 1, 1)
        progress = float(epoch - warmup) / denominator
        learning_rate = minimum + 0.5 * (base_lr - minimum) * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _train_epoch(model, loader, optimizer, device: str, amp: bool) -> float:
    model.train()
    losses = []
    loss_function = torch.nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.startswith("cuda"),
        ):
            logits = model(images)
            loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def _evaluate(model, loader, device: str):
    model.eval()
    probabilities, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(targets.numpy())
    probability_array = np.concatenate(probabilities)
    label_array = np.concatenate(labels)
    return metrics_from_probabilities(label_array, probability_array), probability_array


def _copy_state(state: dict[str, object]) -> dict[str, object]:
    return {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in state.items()
    }


def _train_candidate(config, loaders, learning_rate: float):
    foundation = config["foundation"]
    device = str(config["runtime"]["device"])
    _set_seed(int(config["training"]["seed"]))
    encoder = load_preti_foundation(
        source_root=foundation["source_root"],
        checkpoint_path=foundation["checkpoint_path"],
        device=device,
    )
    model = PretiClassifier(encoder, num_classes=len(LABELS)).to(device)
    trainable = set_encoder_trainable(model.encoder) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        betas=tuple(float(value) for value in config["optimizer"]["betas"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    rows = []
    best = None
    for epoch in range(int(config["training"]["epochs"])):
        current_lr = _set_learning_rate(optimizer, learning_rate, epoch, config)
        started = time.perf_counter()
        train_loss = _train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            bool(config["training"]["amp"]),
        )
        validation_metrics, _ = _evaluate(model, loaders["val"], device)
        row = {
            "learning_rate_candidate": learning_rate,
            "epoch": epoch + 1,
            "effective_learning_rate": current_lr,
            "train_loss": train_loss,
            "elapsed_seconds": time.perf_counter() - started,
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        rows.append(row)
        score = (
            validation_metrics["quadratic_kappa"],
            validation_metrics["macro_f1"],
        )
        if best is None or score > best[0]:
            best = (
                score,
                epoch + 1,
                copy.deepcopy(validation_metrics),
                encoder_state_dict(model.encoder),
                _copy_state(model.head.state_dict()),
            )
        print(
            f"lr={learning_rate:.2e} epoch={epoch + 1:02d}/"
            f"{config['training']['epochs']} loss={train_loss:.4f} "
            f"val_qwk={validation_metrics['quadratic_kappa']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return CandidateResult(
        learning_rate=learning_rate,
        best_epoch=best[1],
        validation_metrics=best[2],
        encoder_state=best[3],
        classifier_state=best[4],
    ), rows


def _prediction_frame(dataset, labels, probabilities, data_root: Path):
    relative_paths = [
        Path(path).relative_to(data_root).as_posix() for path, _ in dataset.samples
    ]
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "image_path": relative_paths,
            "true_label": labels,
            "pred_label": predictions,
        }
    )
    for index in range(probabilities.shape[1]):
        frame[f"prob_{index}"] = probabilities[:, index]
    with tempfile.TemporaryDirectory(prefix="ophagent-preti-") as directory:
        temporary = Path(directory) / "predictions.csv"
        frame.to_csv(temporary, index=False)
        return normalize_prediction_frame(temporary, num_classes=len(LABELS))


def build_registration_record(
    *, output_dir: Path, prediction_path: Path, checkpoint_path: Path, config
):
    return {
        "model_id": f"aptos_dr_5class::{ARTIFACT_ID}",
        "task_id": "aptos_dr_5class",
        "dataset_id": "APTOS2019",
        "dataset_display_name": "APTOS 2019",
        "dataset_source": "public",
        "artifact_id": ARTIFACT_ID,
        "model_family": "preti",
        "architecture": "PRETI ViT-B/16",
        "label_space": "dr_icdr_0_4",
        "n_classes": 5,
        "prediction_source": "adapter",
        "prediction_path": str(prediction_path),
        "adapter_status": "completed",
        "compatibility_status": "offline_evaluation_ready",
        "role_candidates": "scout|expert",
        "pretraining_source": "ophbench::preti::preti-default",
        "checkpoint_path": str(checkpoint_path),
        "base_model_provider": "ophbench",
        "base_model_id": "preti",
        "base_checkpoint_id": "preti-default",
        "encoder_checkpoint_sha256": config["foundation"]["checkpoint_sha256"],
        "task_checkpoint": True,
        "task_adapted": True,
        "task_inference_ready": True,
        "offline_evaluation_eligible": True,
        "unified_evaluation_completed": True,
        "inference_cost_measured": False,
        "route_eligible": False,
        "output_dir": str(output_dir),
        "evaluation_role": "project_downstream_adaptation",
        "lifecycle_status": "candidate",
        "research_claim_status": "official_model_project_downstream_adaptation",
        "cost_status": "unmeasured",
        "selection_split": "val",
        "selection_metric": "quadratic_kappa",
        "test_used_for_selection": False,
        "official_downstream_recipe_complete": False,
        "trainer_adapter": "preti_aptos_paper_anchored_v1",
    }


def run_training(config_path: Path, *, overrides=None) -> Path:
    config_path = Path(config_path)
    config = _load_config(config_path)
    for dotted_key, value in dict(overrides or {}).items():
        if value is None:
            continue
        section, key = dotted_key.split(".", 1)
        config.setdefault(section, {})[key] = value
    preflight = strict_preflight(config)
    output_dir = Path(config["output"]["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "base_recipe.yaml")
    (output_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest, manifest_sha = dataset_manifest(Path(config["data"]["root"]))
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    datasets, loaders = build_loaders(config)
    candidates, history = [], []
    for learning_rate in config["optimizer"]["learning_rate_candidates"]:
        candidate, rows = _train_candidate(config, loaders, float(learning_rate))
        candidates.append(candidate)
        history.extend(rows)
    selected = choose_candidate(candidates)

    foundation = config["foundation"]
    device = str(config["runtime"]["device"])
    encoder = load_preti_foundation(
        source_root=foundation["source_root"],
        checkpoint_path=foundation["checkpoint_path"],
        device="cpu",
    )
    encoder.load_state_dict(selected.encoder_state, strict=False)
    final_model = PretiClassifier(encoder, num_classes=len(LABELS))
    final_model.head.load_state_dict(selected.classifier_state)
    final_model = final_model.to(device)
    test_metrics, test_probabilities = _evaluate(final_model, loaders["test"], device)
    test_labels = np.asarray(datasets["test"].targets, dtype=int)
    prediction_path = output_dir / "test_predictions.csv"
    _prediction_frame(
        datasets["test"],
        test_labels,
        test_probabilities,
        Path(config["data"]["root"]),
    ).to_csv(prediction_path, index=False)
    checkpoint_path = output_dir / "preti_aptos_task_checkpoint.pth"
    torch.save(
        {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "labels": LABELS,
            "source_commit": PRETI_SOURCE_COMMIT,
            "encoder_checkpoint_sha256": foundation["checkpoint_sha256"],
            "dataset_manifest_sha256": manifest_sha,
            "selected_learning_rate": selected.learning_rate,
            "selected_epoch": selected.best_epoch,
            "encoder_prefixes": ENCODER_PREFIXES,
            "encoder_state_dict": selected.encoder_state,
            "classifier_state_dict": selected.classifier_state,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    metrics = {
        "selection_split": "val",
        "selection_metric": "quadratic_kappa",
        "test_used_for_selection": False,
        "selected_learning_rate": selected.learning_rate,
        "selected_epoch": selected.best_epoch,
        "validation": selected.validation_metrics,
        "test": test_metrics,
        "candidates": [
            {
                "learning_rate": item.learning_rate,
                "best_epoch": item.best_epoch,
                "validation": item.validation_metrics,
            }
            for item in candidates
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record = build_registration_record(
        output_dir=output_dir,
        prediction_path=prediction_path,
        checkpoint_path=checkpoint_path,
        config=config,
    )
    record["task_checkpoint_sha256"] = sha256_file(checkpoint_path)
    pd.DataFrame([record]).to_csv(output_dir / "registration_record.csv", index=False)
    run_manifest = {
        "created_at_utc": _utc_now(),
        "artifact_id": ARTIFACT_ID,
        "paper_anchor": "MICCAI 2025 PRETI",
        "paper_disclosed": {
            "input_size": 224,
            "augmentation": "random crop and flip",
            "batch_size": 16,
            "epochs": 50,
            "checkpoint_selection": "best validation model",
        },
        "ophagent_declared_due_to_missing_official_downstream_recipe": {
            "optimizer": config["optimizer"],
            "scheduler": config["scheduler"],
            "selection_metric": "quadratic_kappa",
        },
        "test_used_for_selection": False,
        "route_eligible": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    adapter = PretiAptosTaskAdapter.load(
        source_root=foundation["source_root"],
        encoder_checkpoint=foundation["checkpoint_path"],
        task_checkpoint=checkpoint_path,
        device=device,
    )
    sample_images = [datasets["test"].loader(path) for path, _ in datasets["test"].samples[:2]]
    smoke_probabilities = adapter.predict_proba(sample_images)
    if smoke_probabilities.shape != (2, len(LABELS)):
        raise ValueError("PRETI 任务 Adapter 最终 Smoke 失败")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
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
            "foundation.checkpoint_path": (
                str(args.checkpoint) if args.checkpoint else None
            ),
            "foundation.source_root": (
                str(args.source_root) if args.source_root else None
            ),
            "output.run_dir": str(args.output_dir) if args.output_dir else None,
            "runtime.device": args.device,
        },
    )
    print(f"PRETI APTOS adaptation completed: {output}")


if __name__ == "__main__":
    main()
