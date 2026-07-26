#!/usr/bin/env python3
"""从已登记 timm checkpoint 生成模型中转台标准评测输出。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
from models.datasets.imagefolder_classification import IMAGE_SUFFIXES, inspect_imagefolder_dataset  # noqa: E402
from app.aptos_replay_adapters import (  # noqa: E402
    ReplayAdapterSpec,
    load_registered_aptos_adapter,
)
from scripts.routing.timm_adapter_runtime import (  # noqa: E402
    classification_metrics,
    execute_timm_backend,
    summarize_cost_runs,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_frozen_manifest(
    request: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    source = Path(request["input_manifest_path"])
    data_root = Path(request["data_root"])
    frame = pd.read_csv(source)
    required = {"case_id", "patient_id", "split", "y_true", "relative_image_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"冻结 manifest 缺少字段：{missing}")
    if frame.empty:
        raise ValueError("冻结 manifest 为空")
    if frame["case_id"].astype(str).duplicated().any():
        raise ValueError("冻结 manifest 的 case_id 不唯一")
    labels = pd.to_numeric(frame["y_true"], errors="coerce")
    if labels.isna().any():
        raise ValueError("冻结 manifest 存在缺失标签")
    expected = set(range(int(request["num_classes"])))
    if not set(labels.astype(int)).issubset(expected):
        raise ValueError("冻结 manifest 标签超出登记类别顺序")
    paths = [data_root / str(value) for value in frame["relative_image_path"]]
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"冻结 manifest 缺少 {len(missing_paths)} 张图像")
    manifest = pd.DataFrame(
        {
            "image_key": frame["case_id"].astype(str),
            "case_id": frame["case_id"].astype(str),
            "patient_id": frame["patient_id"].astype(str),
            "split": frame["split"].astype(str),
            "image_path": [str(path) for path in paths],
            "relative_image_path": frame["relative_image_path"].astype(str),
            "true_label": labels.astype(int),
        }
    )
    path = output_dir / "input_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return manifest, path


def _build_manifest(request: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, Path]:
    if request.get("input_manifest_path"):
        return _build_frozen_manifest(request, output_dir)
    data_root = Path(request["data_root"])
    inspection = inspect_imagefolder_dataset(data_root)
    if len(inspection.class_to_idx) != int(request["num_classes"]):
        raise ValueError("数据集类别数与任务注册信息不一致")
    rows = []
    for class_name, class_index in inspection.class_to_idx.items():
        class_dir = data_root / "test" / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append(
                    {
                        "image_key": image_path.stem,
                        "image_path": str(image_path.resolve()),
                        "true_label": int(class_index),
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("测试集没有可推理图像")
    if manifest["image_key"].duplicated().any():
        raise ValueError("测试集 image_key 不唯一")
    path = output_dir / "input_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return manifest, path


def _run_registered_aptos_adapter(
    request: dict[str, Any],
    manifest: pd.DataFrame,
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    from PIL import Image

    identity_path = Path(
        request["adapter_spec"].get("task_checkpoint_path")
        or request["adapter_spec"]["checkpoint_path"]
    )
    if _sha256_file(identity_path) != request["checkpoint_sha256"]:
        raise ValueError("冻结任务 checkpoint SHA256 与请求不一致")
    adapter = load_registered_aptos_adapter(
        ReplayAdapterSpec(**request["adapter_spec"]),
        device=str(request["device"]),
    )
    batch_size = int(request.get("batch_size", 16))
    probability_batches: list[np.ndarray] = []
    for start in range(0, len(manifest), batch_size):
        batch = manifest.iloc[start : start + batch_size]
        images = []
        try:
            for image_path in batch["image_path"]:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
            probability_batches.append(np.asarray(adapter.predict_proba(images), dtype=float))
        finally:
            for image in images:
                image.close()
    probabilities = np.concatenate(probability_batches, axis=0)
    num_classes = int(request["num_classes"])
    if probabilities.shape != (len(manifest), num_classes):
        raise ValueError(
            f"任务 Adapter 输出形状错误：{probabilities.shape} != "
            f"({len(manifest)}, {num_classes})"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("任务 Adapter 输出含 NaN/Inf")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("任务 Adapter 概率和不为 1")
    ordered = np.sort(probabilities, axis=1)
    safe = np.clip(probabilities, np.finfo(float).tiny, 1.0)
    evaluation_design = str(
        request.get("evaluation_design", "frozen_external_transfer")
    )
    predictions = pd.DataFrame(
        {
            "case_id": manifest["case_id"],
            "patient_id": manifest["patient_id"],
            "split": manifest["split"],
            "relative_image_path": manifest["relative_image_path"],
            "image_key": manifest["image_key"],
            "image_path": manifest["image_path"],
            "true_label": manifest["true_label"],
            "pred_label": probabilities.argmax(axis=1),
            "y_true": manifest["true_label"],
            "y_pred": probabilities.argmax(axis=1),
            "confidence": ordered[:, -1],
            "margin": ordered[:, -1] - ordered[:, -2],
            "entropy": -(probabilities * np.log(safe)).sum(axis=1)
            / math.log(num_classes),
            "model_id": request["artifact_id"],
            "checkpoint_sha256": request["checkpoint_sha256"],
            "preprocessing_id": request["preprocessing_id"],
            "inference_run_id": request["job_id"],
            "inference_dtype": request.get("precision", "fp32"),
            "source": evaluation_design,
        }
    )
    for index in range(num_classes):
        predictions[f"prob_{index}"] = probabilities[:, index]
    predictions_path = output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics = classification_metrics(
        predictions,
        label_structure=request["label_structure"],
    )
    pd.DataFrame(
        [
            {
                **metrics,
                "task_id": request["task_id"],
                "dataset_id": request["dataset_id"],
                "artifact_id": request["artifact_id"],
                "split": str(manifest["split"].iloc[0]),
                "evaluation_design": evaluation_design,
                "test_used_for_selection": False,
            }
        ]
    ).to_csv(output_dir / "model_baseline.csv", index=False, encoding="utf-8-sig")
    checkpoint_paths = {
        key: value
        for key, value in {
            "checkpoint": request["adapter_spec"].get("checkpoint_path"),
            "task_checkpoint": request["adapter_spec"].get("task_checkpoint_path"),
            "probe": request["adapter_spec"].get("probe_path"),
        }.items()
        if value
    }
    artifact_rows = [
        {
            "artifact_type": "prediction_asset",
            "artifact_id": request["artifact_id"],
            "path": str(predictions_path),
            "sha256": _sha256_file(predictions_path),
            "source_manifest_sha256": _sha256_file(Path(request["input_manifest_path"])),
            "route_eligible": False,
        }
    ]
    for kind, raw_path in checkpoint_paths.items():
        path = Path(raw_path)
        artifact_rows.append(
            {
                "artifact_type": kind,
                "artifact_id": request["artifact_id"],
                "path": str(path),
                "sha256": _sha256_file(path),
                "source_manifest_sha256": "",
                "route_eligible": False,
            }
        )
    pd.DataFrame(artifact_rows).to_csv(
        output_dir / "artifact_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_json(
        output_dir / "inference_manifest.json",
        {
            **request,
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": _sha256_file(Path(request["input_manifest_path"])),
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256_file(predictions_path),
            "evaluation_design": evaluation_design,
            "model_selection_on_external_data": False,
            "recalibration_on_external_data": False,
            "route_eligible": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output_dir


def run_request(request: dict[str, Any]) -> Path:
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, manifest_path = _build_manifest(request, output_dir)
    if request.get("loader_id") in {
        "aptos_registered_adapter_v1",
        "registered_task_adapter_v1",
    }:
        return _run_registered_aptos_adapter(
            request,
            manifest,
            manifest_path,
            output_dir,
        )
    job = pd.Series(
        {
            "job_id": request["job_id"],
            "task_id": request["task_id"],
            "artifact_id": request["artifact_id"],
            "checkpoint_path": request["checkpoint_path"],
            "checkpoint_key": request.get("checkpoint_key", ""),
            "arch": request["architecture"],
            "num_classes": request["num_classes"],
            "global_pool": request.get("global_pool", ""),
            "allow_argparse_namespace": request.get("allow_argparse_namespace", False),
            "device": request["device"],
            "precision": request["precision"],
            "norm": request.get("norm", "imagenet"),
            "input_size": request["input_size"],
            "batch_size": request["batch_size"],
            "num_workers": request["num_workers"],
            "warmup_runs": request["warmup_runs"],
            "timed_runs": request["timed_runs"],
            "label_structure": request["label_structure"],
        }
    )
    result = execute_timm_backend(job, manifest)
    predictions_path = output_dir / "predictions.csv"
    baseline_path = output_dir / "model_baseline.csv"
    cost_path = output_dir / "forward_cost_summary.csv"
    result.predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics = classification_metrics(result.predictions, label_structure=request["label_structure"])
    pd.DataFrame([{**metrics, "task_id": request["task_id"], "artifact_id": request["artifact_id"], "split": "test"}]).to_csv(
        baseline_path, index=False, encoding="utf-8-sig"
    )
    cost = summarize_cost_runs(job, result)
    cost.to_csv(cost_path, index=False, encoding="utf-8-sig")
    registration = pd.DataFrame(
        [
            {
                "model_id": request["model_id"],
                "task_id": request["task_id"],
                "dataset_id": request["dataset_id"],
                "artifact_id": request["artifact_id"],
                "model_family": request["model_family"],
                "architecture": request["architecture"],
                "pretraining_source": request["pretraining_source"],
                "label_space": request["label_space"],
                "n_classes": request["num_classes"],
                "role_candidates": request["role_candidates"],
                "prediction_source": "checkpoint_generated",
                "prediction_path": str(predictions_path.resolve()),
                "checkpoint_path": str(Path(request["checkpoint_path"]).resolve()),
                "checkpoint_status": "found",
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "qwk": metrics.get("qwk"),
                "forward_cost_ms_per_image": cost.iloc[0].get("median_ms_per_image"),
                "checkpoint_mb": result.checkpoint_mb,
                "parameter_count": result.parameter_count,
                "trainable_parameter_count": result.trainable_parameter_count,
                "cost_scope": "forward_only",
                "cost_status": "measured",
                "adapter_status": "completed",
                "onboarding_status": "completed",
                "compatibility_status": "ready_for_pairing",
                "registration_source": str((output_dir / "registration_record.csv").resolve()),
                "notes": "由已登记 checkpoint 后台重新生成评测输出",
            }
        ]
    )
    registration.to_csv(output_dir / "registration_record.csv", index=False, encoding="utf-8-sig")
    _write_json(
        output_dir / "inference_manifest.json",
        {
            **request,
            "input_manifest": str(manifest_path.resolve()),
            "predictions": str(predictions_path.resolve()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    try:
        _write_json(args.status, {"job_id": request["job_id"], "status": "running", "output_dir": request["output_dir"]})
        output_dir = run_request(request)
    except Exception as exc:
        _write_json(
            args.status,
            {
                "job_id": request.get("job_id"),
                "status": "failed",
                "output_dir": request.get("output_dir"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    _write_json(args.status, {"job_id": request["job_id"], "status": "succeeded", "output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
