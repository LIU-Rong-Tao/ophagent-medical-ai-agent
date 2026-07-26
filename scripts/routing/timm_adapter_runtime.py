#!/usr/bin/env python3
"""v0.8.5c timm 分类模型真实推理与 forward-only 成本运行时。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class AdapterStageError(RuntimeError):
    """带固定阶段状态的 adapter 失败。"""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class AdapterBackendResult:
    predictions: pd.DataFrame
    cost_runs: pd.DataFrame
    device: str
    precision: str
    checkpoint_mb: float
    actual_device_name: str
    parameter_count: int | None = None
    trainable_parameter_count: int | None = None


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def resolve_path(value: Any, *, base: Path = REPO_ROOT) -> Path:
    path = Path(clean_text(value)).expanduser()
    return path if path.is_absolute() else base / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_columns(frame: pd.DataFrame) -> list[str]:
    numeric: list[tuple[int, str]] = []
    named: list[str] = []
    for column in frame.columns:
        if not column.startswith("prob_"):
            continue
        suffix = column.removeprefix("prob_")
        if suffix.isdigit():
            numeric.append((int(suffix), column))
        else:
            named.append(column)
    if numeric:
        return [column for _, column in sorted(numeric)]
    return named


def _numeric_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> pd.Series:
    for column in candidates:
        if column not in frame.columns:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().all():
            return converted.astype(int)
    raise AdapterStageError("failed_manifest", f"输入 CSV 缺少数值型 {label} 字段")


def normalize_prediction_frame(path: Path, *, num_classes: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise AdapterStageError("failed_manifest", "输入 prediction CSV 为空")
    image_column = next(
        (column for column in ("image_path", "path", "filename") if column in frame.columns),
        None,
    )
    if image_column is not None:
        image_paths = frame[image_column].astype(str)
    elif "case_id" in frame.columns:
        # Frozen prediction assets may intentionally omit the original image path.
        image_paths = frame["case_id"].astype(str)
    else:
        raise AdapterStageError(
            "failed_manifest",
            "输入 CSV 缺少 image_path/path/filename 或 case_id",
        )
    if "image_key" in frame.columns:
        image_keys = frame["image_key"].astype(str)
    elif "case_id" in frame.columns:
        image_keys = frame["case_id"].astype(str)
    else:
        image_keys = image_paths.map(lambda value: Path(value).stem)
    true_label = _numeric_column(frame, ("true_idx", "true_label", "y_true"), "true label")
    pred_label = _numeric_column(frame, ("pred_idx", "pred_label", "y_pred"), "pred label")
    prob_cols = probability_columns(frame)
    if len(prob_cols) != num_classes:
        raise AdapterStageError(
            "failed_manifest",
            f"概率列数量与 num_classes 不一致：{len(prob_cols)} != {num_classes}",
        )
    normalized = pd.DataFrame(
        {
            "image_key": image_keys,
            "image_path": image_paths,
            "true_label": true_label,
            "pred_label": pred_label,
        }
    )
    for index, column in enumerate(prob_cols):
        normalized[f"prob_{index}"] = pd.to_numeric(frame[column], errors="raise")
    probs = normalized[[f"prob_{index}" for index in range(num_classes)]].to_numpy(float)
    ordered = np.sort(probs, axis=1)
    safe = np.clip(probs, np.finfo(float).tiny, 1.0)
    normalized["confidence"] = ordered[:, -1]
    normalized["margin"] = ordered[:, -1] - ordered[:, -2]
    normalized["entropy"] = -(probs * np.log(safe)).sum(axis=1) / math.log(num_classes)
    return normalized


def _image_index(data_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not data_root.exists():
        return index
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem, path)
            try:
                index.setdefault(path.relative_to(data_root).as_posix(), path)
            except ValueError:
                pass
    return index


def build_input_manifest(job: pd.Series, output_dir: Path) -> tuple[pd.DataFrame, Path]:
    legacy_path = resolve_path(job.get("legacy_prediction_path", ""))
    if not legacy_path.exists():
        raise AdapterStageError("skipped_missing_input_csv", f"历史 prediction 不存在：{legacy_path}")
    num_classes = int(job["num_classes"])
    normalized = normalize_prediction_frame(legacy_path, num_classes=num_classes)
    data_root = resolve_path(job.get("data_root", ""))
    lookup: dict[str, Path] | None = None
    resolved_paths: list[str] = []
    missing: list[str] = []
    for _, row in normalized.iterrows():
        raw = Path(str(row["image_path"]))
        candidates = [raw]
        if not raw.is_absolute() and clean_text(job.get("data_root", "")):
            candidates.append(data_root / raw)
        found = next((candidate for candidate in candidates if candidate.exists()), None)
        if found is None and clean_text(job.get("data_root", "")):
            if lookup is None:
                lookup = _image_index(data_root)
            keys = [str(row["image_key"]), Path(str(row["image_key"])).stem, raw.stem]
            found = next((lookup[key] for key in keys if key in lookup), None)
        if found is None:
            missing.append(str(row["image_key"]))
            resolved_paths.append(str(raw))
        else:
            resolved_paths.append(str(found.resolve()))
    if missing:
        raise AdapterStageError(
            "skipped_missing_image_files",
            f"找不到 {len(missing)} 张图像，前 5 个：{missing[:5]}",
        )
    manifest = normalized[["image_key", "true_label"]].copy()
    manifest.insert(1, "image_path", resolved_paths)
    if manifest["image_key"].duplicated().any():
        duplicates = manifest.loc[manifest["image_key"].duplicated(), "image_key"].head(5).tolist()
        raise AdapterStageError("failed_manifest", f"image_key 重复：{duplicates}")
    path = output_dir / "input_manifests" / f"{job['job_id']}_input_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return manifest, path


def validate_job(job: pd.Series, output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame | None, Path | None]:
    base = {
        "job_id": job["job_id"],
        "task_id": job["task_id"],
        "artifact_id": job["artifact_id"],
    }
    checks = (
        ("checkpoint_path", "skipped_missing_checkpoint"),
        ("class_to_idx_path", "skipped_missing_class_mapping"),
    )
    for field, status in checks:
        path = resolve_path(job.get(field, ""))
        if not clean_text(job.get(field, "")) or not path.exists():
            return {**base, "status": status, "notes": f"{field} 不存在：{path}"}, None, None
    mapping_path = resolve_path(job["class_to_idx_path"])
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        values = sorted(int(value) for value in mapping.values())
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {**base, "status": "skipped_missing_class_mapping", "notes": str(exc)}, None, None
    if values != list(range(int(job["num_classes"]))):
        return {
            **base,
            "status": "skipped_missing_class_mapping",
            "notes": f"class_to_idx 必须连续覆盖 0..{int(job['num_classes']) - 1}",
        }, None, None
    try:
        manifest, manifest_path = build_input_manifest(job, output_dir)
    except AdapterStageError as exc:
        return {**base, "status": exc.status, "notes": str(exc)}, None, None
    return {
        **base,
        "status": "ready_for_adapter",
        "n_images": len(manifest),
        "input_manifest_path": str(manifest_path),
        "notes": "checkpoint、class mapping 与冻结 test manifest 已确认",
    }, manifest, manifest_path


def classification_metrics(frame: pd.DataFrame, *, label_structure: str) -> dict[str, Any]:
    prob_cols = probability_columns(frame)
    y_true = frame["true_label"].astype(int).to_numpy()
    y_pred = frame["pred_label"].astype(int).to_numpy()
    probs = frame[prob_cols].astype(float).to_numpy()
    aurocs: list[float] = []
    auprs: list[float] = []
    for index in range(probs.shape[1]):
        binary = (y_true == index).astype(int)
        if binary.min() == binary.max():
            continue
        aurocs.append(float(roc_auc_score(binary, probs[:, index])))
        auprs.append(float(average_precision_score(binary, probs[:, index])))
    ordinal = clean_text(label_structure).lower() == "ordinal"
    return {
        "n_images": len(frame),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if ordinal else np.nan,
        "qwk_status": "computed" if ordinal else "not_applicable",
        "macro_auroc_ovr": float(np.mean(aurocs)) if aurocs else np.nan,
        "macro_aupr_ovr": float(np.mean(auprs)) if auprs else np.nan,
        "n_error": int((y_true != y_pred).sum()),
    }


def summarize_cost_runs(job: pd.Series, result: AdapterBackendResult) -> pd.DataFrame:
    runs = result.cost_runs.copy()
    expected = int(job["timed_runs"])
    if len(runs) != expected:
        raise AdapterStageError(
            "failed_inference",
            f"forward cost 重复次数不完整：expected={expected}, observed={len(runs)}",
        )
    values = runs["ms_per_image"].astype(float)
    if (values <= 0).any():
        raise AdapterStageError("failed_inference", "forward cost 必须大于 0")
    mean = float(values.mean())
    median = float(values.median())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return pd.DataFrame(
        [
            {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "cost_scope": "forward_only",
                "device": result.device,
                "actual_device_name": result.actual_device_name,
                "precision": result.precision,
                "batch_size": int(job["batch_size"]),
                "warmup_runs": int(job["warmup_runs"]),
                "timed_runs": expected,
                "mean_ms_per_image": mean,
                "median_ms_per_image": median,
                "std_ms_per_image": std,
                "cv_ms_per_image": std / mean if mean else np.nan,
                "images_per_second": 1000.0 / median,
                "peak_allocated_memory_mb": float(runs["peak_allocated_memory_mb"].max()),
                "checkpoint_mb": result.checkpoint_mb,
                "parameter_count": result.parameter_count,
                "trainable_parameter_count": result.trainable_parameter_count,
                "notes": "多次 single-GPU forward-only benchmark；不含读取、预处理、传输和服务开销",
            }
        ]
    )


def validate_predictions(job: pd.Series, manifest: pd.DataFrame, predictions: pd.DataFrame) -> None:
    num_classes = int(job["num_classes"])
    required = {
        "job_id",
        "task_id",
        "artifact_id",
        "image_key",
        "image_path",
        "true_label",
        "pred_label",
        "confidence",
        "margin",
        "entropy",
        "source",
        *(f"prob_{index}" for index in range(num_classes)),
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise AdapterStageError("failed_inference", f"predictions 缺少字段：{missing}")
    if len(predictions) != len(manifest) or predictions.empty:
        raise AdapterStageError(
            "failed_inference",
            f"prediction 行数不一致：{len(predictions)} != {len(manifest)}",
        )
    if predictions["image_key"].astype(str).tolist() != manifest["image_key"].astype(str).tolist():
        raise AdapterStageError("failed_inference", "prediction image_key 顺序与 manifest 不一致")
    prob_cols = [f"prob_{index}" for index in range(num_classes)]
    probs = predictions[prob_cols].astype(float).to_numpy()
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise AdapterStageError("failed_inference", "prob_* 行和不为 1")
    if not np.array_equal(probs.argmax(axis=1), predictions["pred_label"].astype(int).to_numpy()):
        raise AdapterStageError("failed_inference", "pred_label 与 prob_* argmax 不一致")
    if set(predictions["source"].astype(str)) != {"adapter_generated"}:
        raise AdapterStageError("failed_inference", "prediction source 必须是 adapter_generated")


def write_adapter_outputs(
    job: pd.Series,
    manifest: pd.DataFrame,
    manifest_path: Path,
    result: AdapterBackendResult,
    output_dir: Path,
) -> dict[str, Any]:
    validate_predictions(job, manifest, result.predictions)
    job_dir = output_dir / "onboarded_models" / str(job["job_id"])
    job_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = job_dir / "predictions.csv"
    baseline_path = job_dir / "model_baseline.csv"
    cost_runs_path = job_dir / "forward_cost_runs.csv"
    cost_path = job_dir / "forward_cost_summary.csv"
    adapter_manifest_path = job_dir / "adapter_manifest.csv"
    result.predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics = classification_metrics(
        result.predictions,
        label_structure=clean_text(job.get("label_structure", "nominal")),
    )
    baseline = pd.DataFrame(
        [
            {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "split": "test",
                **metrics,
                "source": "adapter_generated",
                "notes": "由 v0.8.5c 真实 timm adapter 生成",
            }
        ]
    )
    baseline.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    result.cost_runs.to_csv(cost_runs_path, index=False, encoding="utf-8-sig")
    cost = summarize_cost_runs(job, result)
    cost.to_csv(cost_path, index=False, encoding="utf-8-sig")
    checkpoint_path = resolve_path(job["checkpoint_path"])
    adapter_manifest = pd.DataFrame(
        [
            {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "adapter_id": "timm_classifier_v1",
                "adapter_type": "timm_classifier",
                "checkpoint_path": str(checkpoint_path),
                "config_path": clean_text(job.get("config_path", "")),
                "data_root": clean_text(job.get("data_root", "")),
                "input_manifest_path": str(manifest_path),
                "class_to_idx_path": clean_text(job.get("class_to_idx_path", "")),
                "predictions_path": str(predictions_path),
                "model_baseline_path": str(baseline_path),
                "forward_cost_summary_path": str(cost_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "parameter_count": result.parameter_count,
                "trainable_parameter_count": result.trainable_parameter_count,
                "predictions_sha256": sha256_file(predictions_path),
                "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "notes": "严格加载 checkpoint；sanity check 不等同于 strict reproduction",
            }
        ]
    )
    adapter_manifest.to_csv(adapter_manifest_path, index=False, encoding="utf-8-sig")
    return {
        "job_id": job["job_id"],
        "task_id": job["task_id"],
        "artifact_id": job["artifact_id"],
        "adapter_id": "timm_classifier_v1",
        "status": "completed",
        "n_images": metrics["n_images"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "qwk": metrics["qwk"],
        "estimated_forward_ms_per_image": float(cost.iloc[0]["median_ms_per_image"]),
        "cost_scope": "forward_only",
        "predictions_path": str(predictions_path),
        "outputs_dir": str(job_dir),
        "notes": "真实 adapter 推理完成",
    }


def _extract_state_dict(raw: Any, checkpoint_key: str) -> dict[str, Any]:
    current = raw
    if checkpoint_key:
        for part in checkpoint_key.split("."):
            current = current[part]
    elif isinstance(raw, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in raw and isinstance(raw[key], dict):
                current = raw[key]
                break
    if not isinstance(current, dict):
        raise AdapterStageError("failed_checkpoint_load", "checkpoint 不包含 state_dict")
    state = dict(current)
    for prefix in ("module.", "model."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix) :]: value for key, value in state.items()}
    return state


def timm_model_create_kwargs(job: pd.Series) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "pretrained": False,
        "num_classes": int(job["num_classes"]),
    }
    global_pool = clean_text(job.get("global_pool", ""))
    if global_pool:
        kwargs["global_pool"] = global_pool
    return kwargs


def execute_timm_backend(job: pd.Series, manifest: pd.DataFrame) -> AdapterBackendResult:
    try:
        import torch
        import timm
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except Exception as exc:  # pragma: no cover - server runtime dependency
        raise AdapterStageError("failed_inference", f"缺少 timm/torch 运行依赖：{exc}") from exc

    class ManifestDataset(Dataset):
        def __init__(self, rows: pd.DataFrame, transform: Any):
            self.rows = rows.reset_index(drop=True)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            row = self.rows.iloc[index]
            image = Image.open(row["image_path"]).convert("RGB")
            return self.transform(image), index

    checkpoint_path = resolve_path(job["checkpoint_path"])
    try:
        model = timm.create_model(clean_text(job["arch"]), **timm_model_create_kwargs(job))
        if bool(job.get("allow_argparse_namespace", False)):
            from argparse import Namespace

            with torch.serialization.safe_globals([Namespace]):
                raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        else:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = _extract_state_dict(raw, clean_text(job.get("checkpoint_key", "")))
        model.load_state_dict(state, strict=True)
    except AdapterStageError:
        raise
    except Exception as exc:
        raise AdapterStageError("failed_checkpoint_load", str(exc)) from exc

    requested_device = clean_text(job.get("device", "cuda")) or "cuda"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise AdapterStageError("failed_inference", "配置要求 CUDA，但当前 CUDA 不可用")
    device = torch.device(requested_device)
    precision = clean_text(job.get("precision", "fp32")) or "fp32"
    if precision not in {"fp32", "amp_fp16"}:
        raise AdapterStageError("failed_inference", f"不支持 precision={precision}")
    norm = clean_text(job.get("norm", "imagenet")) or "imagenet"
    if norm != "imagenet":
        raise AdapterStageError("failed_inference", f"当前 timm adapter 不支持 norm={norm}")
    image_size = int(job["input_size"])
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    loader = DataLoader(
        ManifestDataset(manifest, transform),
        batch_size=int(job["batch_size"]),
        shuffle=False,
        num_workers=int(job.get("num_workers", 0) or 0),
    )
    model.to(device).eval()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(batch: Any) -> Any:
        if precision == "amp_fp16" and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(batch)
        return model(batch)

    output_probs: list[np.ndarray] = []
    output_indices: list[int] = []
    try:
        with torch.inference_mode():
            for tensors, indices in loader:
                tensors = tensors.to(device)
                probs = torch.softmax(forward(tensors), dim=1).detach().cpu().numpy()
                output_probs.append(probs)
                output_indices.extend(indices.tolist())
        probs = np.concatenate(output_probs, axis=0)
        if output_indices != list(range(len(manifest))):
            raise AdapterStageError("failed_inference", "DataLoader 输出顺序异常")
        ordered = np.sort(probs, axis=1)
        safe = np.clip(probs, np.finfo(float).tiny, 1.0)
        predictions = pd.DataFrame(
            {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "image_key": manifest["image_key"].astype(str),
                "image_path": manifest["image_path"].astype(str),
                "true_label": manifest["true_label"].astype(int),
                "pred_label": probs.argmax(axis=1),
                "confidence": ordered[:, -1],
                "margin": ordered[:, -1] - ordered[:, -2],
                "entropy": -(probs * np.log(safe)).sum(axis=1) / math.log(probs.shape[1]),
                "source": "adapter_generated",
            }
        )
        for index in range(probs.shape[1]):
            predictions[f"prob_{index}"] = probs[:, index]

        first_batch, _ = next(iter(loader))
        first_batch = first_batch.to(device)
        with torch.inference_mode():
            for _ in range(int(job["warmup_runs"])):
                _ = forward(first_batch)
            sync()
        cost_rows: list[dict[str, Any]] = []
        for repeat in range(1, int(job["timed_runs"]) + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            total_ms = 0.0
            with torch.inference_mode():
                for tensors, _ in loader:
                    tensors = tensors.to(device)
                    sync()
                    started = time.perf_counter()
                    _ = forward(tensors)
                    sync()
                    total_ms += (time.perf_counter() - started) * 1000.0
            peak = (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
                if device.type == "cuda"
                else 0.0
            )
            cost_rows.append(
                {
                    "repeat_index": repeat,
                    "total_forward_ms": total_ms,
                    "ms_per_image": total_ms / len(manifest),
                    "peak_allocated_memory_mb": peak,
                }
            )
    except AdapterStageError:
        raise
    except Exception as exc:
        raise AdapterStageError("failed_inference", str(exc)) from exc
    actual_device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    return AdapterBackendResult(
        predictions=predictions,
        cost_runs=pd.DataFrame(cost_rows),
        device=str(device),
        precision=precision,
        checkpoint_mb=checkpoint_path.stat().st_size / 1024 / 1024,
        actual_device_name=actual_device_name,
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        trainable_parameter_count=int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    )


Backend = Callable[[pd.Series, pd.DataFrame], AdapterBackendResult]
