#!/usr/bin/env python3
"""从已登记 timm checkpoint 生成模型中转台标准评测输出。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The repository-root bootstrap above must run before these imports.
from models.datasets.imagefolder_classification import IMAGE_SUFFIXES, inspect_imagefolder_dataset  # noqa: E402
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


def _build_manifest(request: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, Path]:
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


def run_request(request: dict[str, Any]) -> Path:
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, manifest_path = _build_manifest(request, output_dir)
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
