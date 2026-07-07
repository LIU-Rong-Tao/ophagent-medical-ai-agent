"""模型中转台 checkpoint 批量推理任务提交与状态读取。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

import pandas as pd

from app.training_jobs import prepare_training_subprocess_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_JOBS_ROOT = PROJECT_ROOT / "experiments/model_hub/runtime/inference_jobs"
INFERENCE_RUNS_ROOT = PROJECT_ROOT / "experiments/model_hub/runs/inference"
RUNNER = PROJECT_ROOT / "scripts/routing/run_model_hub_inference_job.py"
TIMM_CLASSIFIER_FAMILIES = {"convnext", "swin", "vit"}
KNOWN_CHECKPOINT_LOADERS = {
    "retfound_mae_cfp_official_protocol": {
        "loader_id": "retfound_mae_cfp_timm_v1",
        "architecture": "vit_large_patch16_224",
        "checkpoint_key": "model",
        "global_pool": "avg",
        "norm": "imagenet",
        "allow_argparse_namespace": True,
    }
}


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _resolve(value: object) -> Path:
    path = Path(_clean(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_loader_spec(row: pd.Series) -> dict[str, Any] | None:
    artifact_id = _clean(row.get("artifact_id"))
    if artifact_id in KNOWN_CHECKPOINT_LOADERS:
        return dict(KNOWN_CHECKPOINT_LOADERS[artifact_id])
    family = _clean(row.get("model_family")).lower()
    if family not in TIMM_CLASSIFIER_FAMILIES:
        return None
    architecture = _clean(row.get("architecture"))
    if not architecture:
        return None
    return {
        "loader_id": "timm_classifier_v1",
        "architecture": architecture,
        "checkpoint_key": _clean(row.get("checkpoint_key")),
        "global_pool": _clean(row.get("global_pool")),
        "norm": _clean(row.get("norm")) or "imagenet",
        "allow_argparse_namespace": False,
    }


def checkpoint_inference_capability(row: pd.Series) -> tuple[bool, str]:
    loader = checkpoint_loader_spec(row)
    if loader is None:
        return False, "当前模型尚未登记可执行的 checkpoint 推理 Loader"
    checkpoint = _resolve(row.get("checkpoint_path"))
    if not _clean(row.get("checkpoint_path")) or not checkpoint.is_file():
        return False, "未找到已登记 checkpoint"
    return True, "可从已登记 checkpoint 重新生成评测输出"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def submit_checkpoint_inference(row: pd.Series, task: pd.Series, *, device: str = "cuda:2") -> str:
    supported, reason = checkpoint_inference_capability(row)
    if not supported:
        raise ValueError(reason)
    loader = checkpoint_loader_spec(row)
    if loader is None:  # pragma: no cover - capability check keeps this unreachable
        raise ValueError("当前模型尚未登记可执行的 checkpoint 推理 Loader")
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    artifact_id = _clean(row.get("artifact_id"))
    task_id = _clean(task.get("task_id"))
    job_dir = INFERENCE_JOBS_ROOT / job_id
    output_dir = INFERENCE_RUNS_ROOT / task_id / artifact_id / datetime.now().strftime("%Y%m%d-%H%M%S")
    request = {
        "job_id": job_id,
        "task_id": task_id,
        "dataset_id": _clean(task.get("dataset_id")),
        "artifact_id": artifact_id,
        "model_id": f"{task_id}::{artifact_id}",
        "model_family": _clean(row.get("model_family")),
        "loader_id": loader["loader_id"],
        "architecture": loader["architecture"],
        "checkpoint_key": loader["checkpoint_key"],
        "global_pool": loader["global_pool"],
        "norm": loader["norm"],
        "allow_argparse_namespace": loader["allow_argparse_namespace"],
        "pretraining_source": _clean(row.get("pretraining_source")),
        "role_candidates": _clean(row.get("role_candidates")),
        "checkpoint_path": str(_resolve(row.get("checkpoint_path"))),
        "data_root": _clean(task.get("data_root")),
        "label_space": _clean(task.get("label_space")),
        "label_structure": _clean(task.get("label_structure")) or "nominal",
        "num_classes": int(task.get("num_classes")),
        "input_size": int(pd.to_numeric(pd.Series([row.get("input_size", row.get("img_size", 224))]), errors="coerce").fillna(224).iloc[0]),
        "batch_size": 32,
        "num_workers": 4,
        "device": device,
        "precision": "fp32",
        "warmup_runs": 3,
        "timed_runs": 3,
        "output_dir": str(output_dir),
    }
    request_path = job_dir / "request.json"
    status_path = job_dir / "status.json"
    log_path = job_dir / "job.log"
    _write_json(request_path, request)
    _write_json(status_path, {"job_id": job_id, "status": "queued", "output_dir": str(output_dir)})
    command = [sys.executable, str(RUNNER), "--request", str(request_path), "--status", str(status_path)]
    environment, warnings = prepare_training_subprocess_environment(os.environ)
    with log_path.open("ab") as log_handle:
        subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
    if warnings:
        (job_dir / "startup_warnings.json").write_text(
            json.dumps(warnings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return job_id


def latest_inference_job(task_id: str, artifact_id: str) -> dict[str, Any] | None:
    if not INFERENCE_JOBS_ROOT.is_dir():
        return None
    matches: list[dict[str, Any]] = []
    for request_path in INFERENCE_JOBS_ROOT.glob("*/request.json"):
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if request.get("task_id") != task_id or request.get("artifact_id") != artifact_id:
            continue
        status_path = request_path.parent / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {"status": "unknown"}
        matches.append({**request, **status, "job_dir": str(request_path.parent)})
    return max(matches, key=lambda item: str(item.get("job_id", ""))) if matches else None
