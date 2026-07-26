#!/usr/bin/env python3
"""训练通用 timm ImageFolder 分类模型并发布标准实验产物。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from app.training_config import dump_yaml, flatten_effective_config
from models.datasets.imagefolder_classification import (
    DatasetInspection,
    build_imagefolder_loaders,
    inspect_imagefolder_dataset,
)
from scripts.routing.model_metadata import timm_pretraining_source


@dataclass(frozen=True)
class TrainingConfig:
    task_id: str
    dataset_id: str
    data_root: str
    architecture: str
    num_classes: int
    image_size: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    pretrained: bool
    seed: int
    output_dir: str
    weight_decay: float = 0.01
    label_smoothing: float = 0.0
    num_workers: int = 4
    label_structure: str = "nominal"
    device: str = "auto"
    artifact_id: str = ""
    source_artifact_id: str = ""
    model_family: str = ""
    label_space: str = ""
    initialization_source: str = "timm_pretrained"
    source_checkpoint_path: str = ""
    source_num_classes: int = 0
    trainer_adapter: str = "timm_imagefolder_v1"
    freeze_backbone: bool = False
    optimizer_name: str = "adamw"
    optimizer_momentum: float = 0.9
    scheduler_name: str = "none"
    scheduler_warmup_epochs: int = 0
    scheduler_minimum_learning_rate: float = 0.0
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    amp: bool = False
    grad_accum_steps: int = 1
    early_stopping_patience: int = 0
    class_weights: Any = None
    random_resized_crop: bool = False
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 10.0
    color_jitter: float = 0.0
    save_best_by: str = "accuracy"
    defer_test_evaluation: bool = False


REQUIRED_FIELDS = {
    "task_id",
    "dataset_id",
    "data_root",
    "architecture",
    "num_classes",
    "image_size",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "pretrained",
    "seed",
    "output_dir",
}


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("训练配置根节点必须是 mapping")
    return data


def load_training_config(path: Path | str) -> TrainingConfig:
    mapping = _read_mapping(Path(path))
    if mapping.get("schema_version") == 1 and "identity" in mapping:
        mapping = flatten_effective_config(mapping)
    missing = REQUIRED_FIELDS - set(mapping)
    if missing:
        raise ValueError(f"训练配置缺少字段：{sorted(missing)}")
    config = TrainingConfig(**{key: value for key, value in mapping.items() if key in TrainingConfig.__dataclass_fields__})
    inspection = inspect_imagefolder_dataset(config.data_root)
    if len(inspection.class_to_idx) != int(config.num_classes):
        raise ValueError(
            f"num_classes={config.num_classes} 与数据集类别数 {len(inspection.class_to_idx)} 不一致"
        )
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        allowed_existing = {
            (output_dir / "configs" / name).resolve()
            for name in (
                "base_recipe.yaml",
                "submitted_config.yaml",
                "effective_config.yaml",
                "validation_report.json",
            )
        }
        unexpected = [
            existing
            for existing in output_dir.rglob("*")
            if existing.is_file() and existing.resolve() not in allowed_existing
        ]
        if unexpected:
            raise ValueError(f"输出目录已存在既有训练产物，拒绝覆盖：{unexpected[0]}")
    if min(config.image_size, config.batch_size, config.num_epochs, config.learning_rate) <= 0:
        raise ValueError("image_size、batch_size、num_epochs 和 learning_rate 必须大于 0")
    if config.device != "auto" and config.device != "cpu" and not config.device.startswith("cuda"):
        raise ValueError("device 必须为 auto、cpu 或 cuda[:n]")
    return config


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("训练 recipe 要求 CUDA，但当前环境不可用 CUDA")
    return torch.device(requested)


def extract_checkpoint_state(raw: Any) -> dict[str, Any]:
    current = raw
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in raw and isinstance(raw[key], dict):
                current = raw[key]
                break
    if not isinstance(current, dict):
        raise ValueError("源 checkpoint 不包含可加载的 state_dict")
    state = dict(current)
    for prefix in ("module.", "model."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix) :]: value for key, value in state.items()}
    return state


def freeze_model_backbone(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not hasattr(model, "get_classifier"):
        raise RuntimeError("当前模型不支持自动识别分类头，不能执行 freeze_backbone")
    classifier = model.get_classifier()
    if classifier is None or not hasattr(classifier, "parameters"):
        raise RuntimeError("当前模型分类头不可训练，不能执行 freeze_backbone")
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def build_optimizer(model: Any, config: Any):
    from torch.optim import Adam, AdamW, SGD

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("没有可训练参数")
    common = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
    }
    name = str(config.optimizer_name).lower()
    if name == "adamw":
        return AdamW(parameters, **common)
    if name == "adam":
        return Adam(parameters, **common)
    if name == "sgd":
        return SGD(parameters, momentum=float(config.optimizer_momentum), **common)
    raise ValueError(f"不支持 optimizer={name}")


def build_scheduler(optimizer: Any, config: Any):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR

    name = str(config.scheduler_name).lower()
    if name == "none":
        return None
    warmup_epochs = int(config.scheduler_warmup_epochs)
    if name == "cosine":
        main = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config.num_epochs) - warmup_epochs),
            eta_min=float(config.scheduler_minimum_learning_rate),
        )
    elif name == "step":
        main = StepLR(
            optimizer,
            step_size=int(config.scheduler_step_size),
            gamma=float(config.scheduler_gamma),
        )
    else:
        raise ValueError(f"不支持 scheduler={name}")
    if warmup_epochs <= 0:
        return main
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])


def compute_class_weights(targets: Any, *, num_classes: int, device: Any):
    import torch

    counts = torch.bincount(torch.as_tensor(targets, dtype=torch.long), minlength=int(num_classes)).float()
    if torch.any(counts <= 0):
        raise ValueError("自动类别权重要求训练集中每个类别至少有一个样本")
    weights = counts.sum() / (float(num_classes) * counts)
    return weights.to(device)


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, criterion, device):
    import torch

    model.eval()
    losses = []
    labels = []
    probabilities = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            losses.append(float(criterion(logits, targets).item()))
            labels.extend(targets.cpu().numpy().tolist())
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return float(np.mean(losses)), np.asarray(labels, dtype=int), np.concatenate(probabilities, axis=0)


def _prediction_frame(dataset, truth: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    prediction = probabilities.argmax(axis=1)
    ordered = np.sort(probabilities, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2] if probabilities.shape[1] > 1 else ordered[:, -1]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=1)
    entropy /= math.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else 1.0
    sample_paths = [Path(path) for path, _ in dataset.samples]
    frame = pd.DataFrame(
        {
            "image_path": [str(path) for path in sample_paths],
            "image_key": [f"{path.parent.name}/{path.name}" for path in sample_paths],
            "true_label": truth,
            "pred_label": prediction,
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
        }
    )
    for index in range(probabilities.shape[1]):
        frame[f"prob_{index}"] = probabilities[:, index]
    return frame


def summarize_forward_cost(
    *,
    total_forward_ms: list[float],
    n_images: int,
    artifact_id: str,
    task_id: str,
    dataset_id: str,
    model_family: str,
    batch_size: int,
    device: str,
    checkpoint_mb: float,
    peak_allocated_memory_mb: float,
    parameter_count: int | None = None,
    trainable_parameter_count: int | None = None,
) -> dict[str, Any]:
    if n_images <= 0 or len(total_forward_ms) < 2:
        raise ValueError("forward-only 成本汇总至少需要 2 次完整数据集计时")
    values = np.asarray(total_forward_ms, dtype=float) / float(n_images)
    mean = float(values.mean())
    median = float(np.median(values))
    std = float(values.std(ddof=1))
    return {
        "artifact_id": artifact_id,
        "cost_profile_id": f"{artifact_id}_fp32_bs{int(batch_size)}",
        "task_id": task_id,
        "dataset_id": dataset_id,
        "model_family": model_family,
        "n_images": int(n_images),
        "batch_size": int(batch_size),
        "device": device,
        "precision": "fp32",
        "timed_runs": int(len(values)),
        "estimated_forward_ms_per_image": median,
        "mean_ms_per_image": mean,
        "median_ms_per_image": median,
        "std_ms_per_image": std,
        "cv_ms_per_image": std / mean if mean > 0 else np.nan,
        "images_per_second": 1000.0 / median if median > 0 else np.nan,
        "peak_allocated_memory_mb": float(peak_allocated_memory_mb),
        "checkpoint_mb": float(checkpoint_mb),
        "parameter_count": int(parameter_count) if parameter_count is not None else np.nan,
        "trainable_parameter_count": (
            int(trainable_parameter_count) if trainable_parameter_count is not None else np.nan
        ),
        "timing_scope": "forward_only",
        "cost_status": "measured",
        "timing_source": "训练完成后 5 次完整 test split forward-only 重复计时",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_training_run(
    config: TrainingConfig,
    *,
    checkpoint_path: Path,
    predictions_path: Path,
    metrics_path: Path,
    config_path: Path,
    class_mapping_path: Path,
    cost_path: Path,
    metrics: dict[str, Any],
    validation_predictions_path: Path | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(config.output_dir)
    artifact_id = config.artifact_id or config.architecture
    run_manifest_path = dump_yaml(
        output_dir / "run_manifest.yaml",
        {
            "schema_version": 1,
            "run": {
                "task_id": config.task_id,
                "dataset_id": config.dataset_id,
                "artifact_id": artifact_id,
                "source_artifact_id": config.source_artifact_id,
                "trainer_adapter": config.trainer_adapter,
                "status": "completed",
            },
            "model": {
                "family": config.model_family or config.architecture.split("_")[0],
                "architecture": config.architecture,
                "initialization_source": config.initialization_source,
            },
            "selection": {
                "save_best_by": config.save_best_by,
                "checkpoint_path": str(checkpoint_path.resolve()),
            },
            "outputs": {
                "predictions": str(predictions_path.resolve()),
                "validation_predictions": (
                    str(validation_predictions_path.resolve())
                    if validation_predictions_path is not None
                    else None
                ),
                "metrics": str(metrics_path.resolve()),
                "forward_cost": str(cost_path.resolve()),
                "training_config": str(config_path.resolve()),
            },
        },
    )
    published = [
        ("checkpoint", checkpoint_path),
        ("predictions", predictions_path),
        ("metrics", metrics_path),
        ("training_config", config_path),
        ("class_to_idx", class_mapping_path),
        ("forward_cost", cost_path),
        ("run_manifest", run_manifest_path),
    ]
    if validation_predictions_path is not None and validation_predictions_path.resolve() != predictions_path.resolve():
        published.append(("validation_predictions", validation_predictions_path))
    for name in ("base_recipe.yaml", "submitted_config.yaml", "effective_config.yaml", "validation_report.json"):
        candidate = output_dir / "configs" / name
        if candidate.is_file() and candidate.resolve() != config_path.resolve():
            published.append((name.rsplit(".", 1)[0], candidate))
    manifest = pd.DataFrame(
        [
            {
                "artifact_name": name,
                "published_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "status": "published",
            }
            for name, path in published
        ]
    )
    manifest_path = output_dir / "artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    cost = pd.read_csv(cost_path).iloc[0]
    model_family = config.model_family or config.architecture.split("_")[0]
    registration = pd.DataFrame(
        [
            {
                "model_id": f"{config.task_id}::{artifact_id}",
                "task_id": config.task_id,
                "dataset_id": config.dataset_id,
                "artifact_id": artifact_id,
                "source_artifact_id": config.source_artifact_id,
                "model_family": model_family,
                "architecture": config.architecture,
                "pretraining_source": (
                    timm_pretraining_source(config.architecture)
                    if config.initialization_source == "timm_pretrained"
                    else config.initialization_source
                ),
                "label_space": config.label_space,
                "n_classes": int(config.num_classes),
                "role_candidates": "scout|expert|adapter_scout",
                "prediction_source": "adapter",
                "prediction_path": str(predictions_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "qwk": metrics.get("qwk"),
                "forward_cost_ms_per_image": cost.get("estimated_forward_ms_per_image"),
                "checkpoint_mb": cost.get("checkpoint_mb"),
                "parameter_count": cost.get("parameter_count"),
                "trainable_parameter_count": cost.get("trainable_parameter_count"),
                "cost_scope": "forward_only",
                "cost_status": "measured",
                "adapter_status": "completed",
                "onboarding_status": "completed",
                "compatibility_status": (
                    "validation_selection_ready"
                    if config.defer_test_evaluation
                    else "ready_for_pairing"
                ),
                "validation_selection_eligible": True,
                "task_inference_ready": True,
                "route_eligible": False,
                "registration_source": str((output_dir / "registration_record.csv").resolve()),
            }
        ]
    )
    registration_path = output_dir / "registration_record.csv"
    registration.to_csv(registration_path, index=False, encoding="utf-8-sig")
    return manifest_path, registration_path


def _benchmark_forward_only(
    model: Any,
    loader: Any,
    device: Any,
    *,
    config: TrainingConfig,
    checkpoint_path: Path,
    n_images: int,
    timed_runs: int = 5,
) -> dict[str, Any]:
    import torch

    model.eval()
    first_batch = next(iter(loader), None)
    if first_batch is None:
        raise ValueError("评估 split 为空，无法执行 forward-only 成本测量")
    warmup_images = first_batch[0].to(device)
    with torch.inference_mode():
        for _ in range(3):
            model(warmup_images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    timings: list[float] = []
    for _ in range(timed_runs):
        seen = 0
        total_ms = 0.0
        for images, _targets in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_ms += (time.perf_counter() - started) * 1000.0
            seen += int(images.shape[0])
        if seen != n_images:
            raise ValueError(f"成本测量图像数异常：{seen} != {n_images}")
        timings.append(total_ms)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda"
        else 0.0
    )
    return summarize_forward_cost(
        total_forward_ms=timings,
        n_images=n_images,
        artifact_id=config.artifact_id or config.architecture,
        task_id=config.task_id,
        dataset_id=config.dataset_id,
        model_family=config.model_family or config.architecture.split("_")[0],
        batch_size=config.batch_size,
        device=str(device),
        checkpoint_mb=checkpoint_path.stat().st_size / 1024 / 1024,
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        trainable_parameter_count=int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        peak_allocated_memory_mb=peak_memory,
    )


def create_timm_model_with_pretrained_fallback(
    timm_module: Any,
    architecture: str,
    *,
    pretrained: bool,
    num_classes: int,
) -> Any:
    """Prefer timm's normal source, then retry through its official URL."""

    arguments = {"pretrained": bool(pretrained), "num_classes": int(num_classes)}
    try:
        return timm_module.create_model(architecture, **arguments)
    except Exception as primary_error:
        if not pretrained:
            raise
        try:
            pretrained_cfg = timm_module.get_pretrained_cfg(architecture)
        except Exception:
            raise primary_error
        official_url = (
            pretrained_cfg.get("url", "")
            if isinstance(pretrained_cfg, dict)
            else getattr(pretrained_cfg, "url", "")
        )
        if not str(official_url).strip():
            raise primary_error
        try:
            return timm_module.create_model(
                architecture,
                **arguments,
                pretrained_cfg_overlay={"hf_hub_id": None},
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"预训练权重加载失败：{architecture}；"
                f"Hub={type(primary_error).__name__}: {primary_error}；"
                f"官方地址={type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error


def run_training(config_path: Path | str) -> Path:
    import timm
    import torch
    from torch import nn

    config = load_training_config(config_path)
    inspection: DatasetInspection = inspect_imagefolder_dataset(config.data_root)
    output_dir = Path(config.output_dir)
    checkpoints = output_dir / "checkpoints"
    configs = output_dir / "configs"
    logs = output_dir / "logs"
    validation_evaluation = output_dir / "evaluation" / "validation"
    test_evaluation = output_dir / "evaluation" / "test"
    directories = [checkpoints, configs, logs, validation_evaluation]
    if not config.defer_test_evaluation:
        directories.append(test_evaluation)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)
    datasets_by_split, loaders = build_imagefolder_loaders(config)
    device = _resolve_device(config.device)
    if config.initialization_source == "registered_checkpoint":
        model = timm.create_model(
            config.architecture,
            pretrained=False,
            num_classes=int(config.source_num_classes),
        )
        raw_checkpoint = torch.load(config.source_checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(extract_checkpoint_state(raw_checkpoint), strict=True)
        if not hasattr(model, "reset_classifier"):
            raise RuntimeError(f"模型 {config.architecture} 不支持受控分类头替换")
        model.reset_classifier(int(config.num_classes))
    else:
        model = create_timm_model_with_pretrained_fallback(
            timm,
            config.architecture,
            pretrained=bool(config.pretrained),
            num_classes=int(config.num_classes),
        )
    if config.freeze_backbone:
        freeze_model_backbone(model)
    model = model.to(device)
    class_weights = None
    if config.class_weights == "auto":
        class_weights = compute_class_weights(
            datasets_by_split["train"].targets,
            num_classes=config.num_classes,
            device=device,
        )
    elif isinstance(config.class_weights, list):
        class_weights = torch.as_tensor(config.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config.label_smoothing),
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = checkpoints / f"{config.architecture}_best.pth"
    started = time.time()
    for epoch in range(1, int(config.num_epochs) + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        total_batches = len(loaders["train"])
        for batch_index, (images, targets) in enumerate(loaders["train"], start=1):
            images = images.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(images), targets)
                scaled_loss = loss / int(config.grad_accum_steps)
            scaler.scale(scaled_loss).backward()
            if batch_index % int(config.grad_accum_steps) == 0 or batch_index == total_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        val_loss, val_truth, val_probabilities = _evaluate(model, loaders["val"], criterion, device)
        val_prediction = val_probabilities.argmax(axis=1)
        val_accuracy = float(accuracy_score(val_truth, val_prediction))
        val_macro_f1 = float(f1_score(val_truth, val_prediction, average="macro", zero_division=0))
        val_qwk = (
            float(cohen_kappa_score(val_truth, val_prediction, weights="quadratic"))
            if config.label_structure == "ordinal"
            else None
        )
        objective_values = {
            "accuracy": val_accuracy,
            "macro_f1": val_macro_f1,
            "qwk": val_qwk,
            "val_loss": -val_loss,
        }
        current_score = objective_values.get(config.save_best_by)
        if current_score is None:
            raise ValueError(f"当前任务无法计算 save_best_by={config.save_best_by}")
        improved = float(current_score) > best_score
        if improved:
            best_score = float(current_score)
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "val_qwk": val_qwk,
                "selection_metric": config.save_best_by,
                "selection_score": float(current_score),
                "best_selection_score": best_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        pd.DataFrame(history).to_csv(logs / "train_log.csv", index=False)
        if int(config.early_stopping_patience) > 0 and stale_epochs >= int(config.early_stopping_patience):
            break

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    validation_loss, validation_truth, validation_probabilities = _evaluate(
        model,
        loaders["val"],
        criterion,
        device,
    )
    validation_predictions = _prediction_frame(
        datasets_by_split["val"],
        validation_truth,
        validation_probabilities,
    )
    validation_predictions_path = validation_evaluation / "validation_predictions.csv"
    validation_predictions.to_csv(validation_predictions_path, index=False)
    validation_pred = validation_predictions["pred_label"].to_numpy()
    validation_metrics = {
        "accuracy": float(accuracy_score(validation_truth, validation_pred)),
        "macro_f1": float(f1_score(validation_truth, validation_pred, average="macro", zero_division=0)),
        "qwk": (
            float(cohen_kappa_score(validation_truth, validation_pred, weights="quadratic"))
            if config.label_structure == "ordinal"
            else None
        ),
        "validation_loss": validation_loss,
        "n_images": len(validation_predictions),
    }
    (validation_evaluation / "metrics.json").write_text(
        json.dumps(validation_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions_path = validation_predictions_path
    metrics = validation_metrics
    metrics_path = validation_evaluation / "metrics.json"
    test_metrics = None
    if not config.defer_test_evaluation:
        test_loss, truth, probabilities = _evaluate(model, loaders["test"], criterion, device)
        predictions = _prediction_frame(datasets_by_split["test"], truth, probabilities)
        predictions_path = test_evaluation / "test_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        pred = predictions["pred_label"].to_numpy()
        test_metrics = {
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "qwk": (
                float(cohen_kappa_score(truth, pred, weights="quadratic"))
                if config.label_structure == "ordinal"
                else None
            ),
            "test_loss": test_loss,
            "n_images": len(predictions),
        }
        metrics = test_metrics
        metrics_path = test_evaluation / "metrics.json"
    submitted_config_path = Path(config_path).resolve()
    if submitted_config_path.name == "effective_config.yaml" and submitted_config_path.parent == configs.resolve():
        written_config_path = submitted_config_path
    else:
        written_config_path = configs / "config.json"
        written_config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    class_mapping_path = configs / "class_to_idx.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    class_mapping_path.write_text(json.dumps(inspection.class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")
    env_info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (configs / "env_info.json").write_text(json.dumps(env_info, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "task_id": config.task_id,
        "dataset_id": config.dataset_id,
        "architecture": config.architecture,
        "best_epoch": best_epoch,
        "save_best_by": config.save_best_by,
        "best_validation_score": best_score,
        "total_train_time_sec": time.time() - started,
        "split_sizes": inspection.split_sizes,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_evaluation_deferred": config.defer_test_evaluation,
        "checkpoint_path": str(checkpoint_path),
    }
    (logs / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    cost_summary = _benchmark_forward_only(
        model,
        loaders["val"] if config.defer_test_evaluation else loaders["test"],
        device,
        config=config,
        checkpoint_path=checkpoint_path,
        n_images=len(datasets_by_split["val"] if config.defer_test_evaluation else datasets_by_split["test"]),
    )
    cost_path = output_dir / "forward_cost_summary.csv"
    pd.DataFrame([cost_summary]).to_csv(cost_path, index=False, encoding="utf-8-sig")
    publish_training_run(
        config,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        config_path=written_config_path,
        class_mapping_path=class_mapping_path,
        cost_path=cost_path,
        metrics=metrics,
        validation_predictions_path=validation_predictions_path,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
