"""后台训练任务的预检、提交、状态读取与取消。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd

from app.training_config import compile_effective_config, dump_yaml
from models.datasets.imagefolder_classification import DatasetInspection, inspect_imagefolder_dataset
from scripts.routing.model_metadata import canonical_timm_artifact_id, resolve_timm_pretrained_architecture


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROXY_ENVIRONMENT_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True)
class TrainingRequest:
    task_id: str
    dataset_id: str
    artifact_id: str
    model_family: str
    architecture: str
    data_root: str
    num_classes: int
    output_dir: str
    recipe_id: str
    label_structure: str = "nominal"
    source_artifact_id: str = ""
    label_space: str = ""
    initialization_source: str = "timm_pretrained"
    source_checkpoint_path: str = ""
    source_num_classes: int = 0
    training_overrides: dict[str, Any] = field(default_factory=dict)
    source_task_id: str = ""
    trainer_adapter: str = "timm_imagefolder_v1"
    display_metrics: list[str] = field(default_factory=list)
    base_recipe: dict[str, Any] = field(default_factory=dict)
    submitted_config: dict[str, Any] = field(default_factory=dict)
    base_model_provider: str = ""
    base_model_id: str = ""
    base_checkpoint_id: str = ""
    encoder_checkpoint_sha256: str = ""


@dataclass(frozen=True)
class TrainingPreflight:
    request: TrainingRequest
    recipe: dict[str, Any]
    inspection: DatasetInspection
    generated_config: dict[str, Any]
    base_recipe: dict[str, Any] = field(default_factory=dict)
    submitted_config: dict[str, Any] = field(default_factory=dict)
    validation_report: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _proxy_endpoint_reachable(proxy_url: str) -> bool:
    parsed = urlparse(str(proxy_url))
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def prepare_training_subprocess_environment(
    source_environment: Mapping[str, str] | None = None,
    *,
    proxy_reachable: Callable[[str], bool] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Copy the parent environment while dropping stale loopback proxies."""

    environment = dict(os.environ if source_environment is None else source_environment)
    checker = proxy_reachable or _proxy_endpoint_reachable
    unreachable_values: set[str] = set()
    for key in PROXY_ENVIRONMENT_KEYS:
        value = str(environment.get(key, "")).strip()
        if not value:
            continue
        host = (urlparse(value).hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            continue
        if value not in unreachable_values and checker(value):
            continue
        unreachable_values.add(value)
        environment.pop(key, None)
    warnings = [
        f"已忽略不可用的本机代理 {value}；训练任务将使用服务器直连网络。"
        for value in sorted(unreachable_values)
    ]
    return environment, warnings


def read_job_status(job_dir: Path | str) -> dict[str, Any]:
    path = Path(job_dir) / "status.json"
    if not path.is_file():
        return {"status": "unknown", "error_message": "任务状态文件不存在"}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_training_progress(output_dir: Path | str) -> dict[str, Any]:
    root = Path(output_dir)
    history_path = root / "logs" / "train_log.csv"
    history = pd.DataFrame()
    if history_path.is_file():
        try:
            history = pd.read_csv(history_path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            history = pd.DataFrame()
    return {
        "history": history,
        "summary": _read_optional_json(root / "logs" / "summary.json"),
        "test_metrics": _read_optional_json(root / "evaluation" / "test" / "metrics.json"),
    }


def update_job_status(job_dir: Path | str, status: str, **fields: Any) -> dict[str, Any]:
    directory = Path(job_dir)
    current = read_job_status(directory)
    if current.get("status") == "unknown":
        current = {"created_at_utc": _utc_now()}
    current.update(fields)
    current["status"] = status
    current["updated_at_utc"] = _utc_now()
    _write_json_atomic(directory / "status.json", current)
    return current


def _load_recipes(path: Path | str) -> pd.DataFrame:
    recipes = pd.read_csv(path)
    required = {"recipe_id", "model_family", "architecture", "enabled"}
    missing = required - set(recipes.columns)
    if missing:
        raise ValueError(f"recipe registry 缺少字段：{sorted(missing)}")
    return recipes


def _enabled(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def registered_dataset_options(
    task_registry_path: Path | str,
    task_id: str,
) -> pd.DataFrame:
    registry = pd.read_csv(task_registry_path)
    required = {"task_id", "dataset_id", "data_root", "num_classes", "enabled"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"任务注册表缺少字段：{sorted(missing)}")
    task_rows = registry.loc[
        registry["task_id"].astype(str).eq(str(task_id))
        & registry["enabled"].map(_enabled)
    ].copy()
    results: list[dict[str, Any]] = []
    for _, row in task_rows.iterrows():
        record = row.to_dict()
        root = Path(str(row.get("data_root", "")))
        if not root.is_dir():
            record.update(
                {
                    "availability_status": "blocked",
                    "availability_reason": f"已登记数据目录不存在：{root}",
                    "class_to_idx": {},
                    "split_sizes": {},
                }
            )
        else:
            try:
                inspection = inspect_imagefolder_dataset(root)
            except ValueError as exc:
                record.update(
                    {
                        "availability_status": "blocked",
                        "availability_reason": str(exc),
                        "class_to_idx": {},
                        "split_sizes": {},
                    }
                )
            else:
                expected_classes = int(row["num_classes"])
                actual_classes = len(inspection.class_to_idx)
                if expected_classes != actual_classes:
                    record.update(
                        {
                            "availability_status": "blocked",
                            "availability_reason": (
                                f"任务注册表为 {expected_classes} 类，数据目录实际为 {actual_classes} 类"
                            ),
                            "class_to_idx": inspection.class_to_idx,
                            "split_sizes": inspection.split_sizes,
                        }
                    )
                else:
                    record.update(
                        {
                            "availability_status": "ready",
                            "availability_reason": "train/val/test 与类别数预检通过",
                            "class_to_idx": inspection.class_to_idx,
                            "split_sizes": inspection.split_sizes,
                        }
                    )
        results.append(record)
    return pd.DataFrame(results)


def build_adaptation_request(
    source_model: pd.Series,
    target_task: pd.Series,
    *,
    data_root: str,
    recipe_id: str,
    output_dir: str,
    initialization_source: str = "timm_pretrained",
    training_overrides: dict[str, Any] | None = None,
) -> TrainingRequest:
    source_artifact_id = str(source_model["artifact_id"])
    target_task_id = str(target_task["task_id"])
    initialization_source = str(initialization_source).strip()
    if initialization_source not in {"timm_pretrained", "registered_checkpoint"}:
        raise ValueError(f"不支持的训练初始化方式：{initialization_source}")
    source_architecture = str(source_model.get("architecture", ""))
    target_architecture = (
        resolve_timm_pretrained_architecture(source_architecture)
        if initialization_source == "timm_pretrained"
        else source_architecture
    )
    target_artifact_id = adaptation_artifact_id(
        source_artifact_id,
        target_task_id,
        initialization_source=initialization_source,
        architecture=target_architecture,
    )
    checkpoint_path = str(source_model.get("checkpoint_path", "") or "").strip()
    checkpoint_found = str(source_model.get("checkpoint_status", "")) == "found" and bool(checkpoint_path)
    reuse_checkpoint = initialization_source == "registered_checkpoint"
    if reuse_checkpoint and not checkpoint_found:
        raise ValueError("选择现有 checkpoint 初始化时，源模型必须登记可用 checkpoint")
    return TrainingRequest(
        task_id=target_task_id,
        dataset_id=str(target_task["dataset_id"]),
        artifact_id=target_artifact_id,
        source_artifact_id=source_artifact_id,
        model_family=str(source_model.get("model_family", "")),
        architecture=target_architecture,
        data_root=str(data_root),
        num_classes=int(target_task["num_classes"]),
        output_dir=str(output_dir),
        recipe_id=str(recipe_id),
        label_structure=str(target_task.get("label_structure", "nominal")),
        label_space=str(target_task.get("label_space", "")),
        initialization_source=initialization_source,
        source_checkpoint_path=checkpoint_path if reuse_checkpoint else "",
        source_num_classes=int(source_model.get("n_classes", 0) or 0) if reuse_checkpoint else 0,
        training_overrides=dict(training_overrides or {}),
        source_task_id=str(source_model.get("task_id", "")),
        display_metrics=[
            value.strip()
            for value in str(target_task.get("display_metrics", "accuracy|macro_f1")).split("|")
            if value.strip()
        ],
    )


def adaptation_artifact_id(
    source_artifact_id: str,
    target_task_id: str,
    *,
    initialization_source: str,
    architecture: str = "",
) -> str:
    source_id = str(source_artifact_id).strip()
    if not source_id:
        raise ValueError("源 artifact_id 不能为空")
    if str(initialization_source).strip() == "timm_pretrained":
        return canonical_timm_artifact_id(architecture, target_task_id)
    source_id = re.sub(r"_+", "_", source_id).strip("_")
    return f"{source_id}_to_{str(target_task_id).strip()}_adapter"


def build_training_context(
    request: TrainingRequest,
    inspection: DatasetInspection,
) -> dict[str, Any]:
    return {
        "task_id": request.task_id,
        "dataset_id": request.dataset_id,
        "artifact_id": request.artifact_id,
        "source_artifact_id": request.source_artifact_id,
        "source_task_id": request.source_task_id,
        "trainer_adapter": request.trainer_adapter,
        "model_family": request.model_family,
        "architecture": request.architecture,
        "data_root": str(Path(request.data_root)),
        "num_classes": int(request.num_classes),
        "class_to_idx": inspection.class_to_idx,
        "label_space": request.label_space,
        "label_structure": request.label_structure,
        "output_dir": request.output_dir,
        "display_metrics": list(request.display_metrics),
        "source_checkpoint_path": request.source_checkpoint_path,
        "source_num_classes": request.source_num_classes,
        "encoder_checkpoint_sha256": request.encoder_checkpoint_sha256,
    }


def build_retry_request(
    request: TrainingRequest,
    output_dir: str,
    *,
    device_override: str | None = None,
) -> TrainingRequest:
    submitted = deepcopy(request.submitted_config)
    overrides = deepcopy(request.training_overrides)
    if submitted:
        submitted.setdefault("output", {})["run_dir"] = str(output_dir)
        if device_override:
            submitted.setdefault("runtime", {})["device"] = device_override
    if device_override:
        overrides["device"] = device_override
    return replace(
        request,
        output_dir=str(output_dir),
        training_overrides=overrides,
        submitted_config=submitted,
    )


def validate_training_request(
    request: TrainingRequest,
    recipes_path: Path | str,
) -> TrainingPreflight:
    inspection = inspect_imagefolder_dataset(request.data_root)
    if len(inspection.class_to_idx) != int(request.num_classes):
        raise ValueError(
            f"num_classes={request.num_classes} 与数据集类别数 {len(inspection.class_to_idx)} 不一致"
        )
    output_dir = Path(request.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"输出目录已存在且非空，拒绝覆盖：{output_dir}")
    if request.submitted_config:
        context = build_training_context(request, inspection)
        effective, report = compile_effective_config(request.submitted_config, context)
        if request.trainer_adapter == "ophbench_retfound_linear_probe_v1":
            from scripts.training.train_ophbench_retfound_linear_probe import strict_preflight

            report.update(strict_preflight(effective))
        return TrainingPreflight(
            request=request,
            recipe=request.base_recipe.get("recipe", {}),
            inspection=inspection,
            generated_config=effective,
            base_recipe=request.base_recipe,
            submitted_config=request.submitted_config,
            validation_report=report,
        )
    recipes = _load_recipes(recipes_path)
    match = recipes.loc[recipes["recipe_id"].astype(str).eq(request.recipe_id)]
    if match.empty:
        raise ValueError(f"未找到 recipe：{request.recipe_id}")
    recipe = match.iloc[0].to_dict()
    if not _enabled(recipe.get("enabled")):
        raise ValueError(f"recipe 未启用：{request.recipe_id}")
    if str(recipe.get("model_family")) != request.model_family:
        raise ValueError("recipe 与模型家族不匹配")
    if str(recipe.get("architecture")) != request.architecture:
        raise ValueError("recipe 与模型架构不匹配")
    source_checkpoint_path = ""
    if request.initialization_source == "registered_checkpoint":
        source_checkpoint = Path(request.source_checkpoint_path)
        if not source_checkpoint.is_absolute():
            source_checkpoint = PROJECT_ROOT / source_checkpoint
        if not source_checkpoint.is_file():
            raise ValueError(f"源 checkpoint 不存在：{source_checkpoint}")
        if int(request.source_num_classes) <= 0:
            raise ValueError("从登记权重适配时必须提供源模型类别数")
        source_checkpoint_path = str(source_checkpoint.resolve())
    generated_config = {
        "task_id": request.task_id,
        "dataset_id": request.dataset_id,
        "data_root": str(Path(request.data_root).resolve()),
        "architecture": request.architecture,
        "num_classes": int(request.num_classes),
        "image_size": int(recipe["image_size"]),
        "batch_size": int(recipe["batch_size"]),
        "num_epochs": int(recipe["num_epochs"]),
        "learning_rate": float(recipe["learning_rate"]),
        "pretrained": _enabled(recipe["pretrained"]),
        "seed": int(recipe.get("seed", 42)),
        "output_dir": str(output_dir.resolve()),
        "weight_decay": float(recipe.get("weight_decay", 0.01)),
        "label_smoothing": float(recipe.get("label_smoothing", 0.0)),
        "num_workers": int(recipe.get("num_workers", 4)),
        "label_structure": request.label_structure,
        "device": str(recipe.get("device", "auto")),
        "artifact_id": request.artifact_id,
        "source_artifact_id": request.source_artifact_id,
        "model_family": request.model_family,
        "label_space": request.label_space,
        "initialization_source": request.initialization_source,
        "source_checkpoint_path": source_checkpoint_path,
        "source_num_classes": int(request.source_num_classes),
    }
    allowed_overrides = {
        "image_size",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "label_smoothing",
        "num_workers",
        "seed",
        "device",
    }
    unknown_overrides = set(request.training_overrides) - allowed_overrides
    if unknown_overrides:
        raise ValueError(f"不支持的训练参数：{sorted(unknown_overrides)}")
    generated_config.update(request.training_overrides)
    if min(
        int(generated_config["image_size"]),
        int(generated_config["batch_size"]),
        int(generated_config["num_epochs"]),
        float(generated_config["learning_rate"]),
    ) <= 0:
        raise ValueError("image_size、batch_size、num_epochs 和 learning_rate 必须大于 0")
    if float(generated_config["weight_decay"]) < 0:
        raise ValueError("weight_decay 不能小于 0")
    if not 0 <= float(generated_config["label_smoothing"]) < 1:
        raise ValueError("label_smoothing 必须在 [0, 1) 范围内")
    if int(generated_config["num_workers"]) < 0 or int(generated_config["seed"]) < 0:
        raise ValueError("num_workers 和 seed 不能小于 0")
    device = str(generated_config["device"])
    if device != "auto" and device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device 必须为 auto、cpu 或 cuda[:n]")
    return TrainingPreflight(request, recipe, inspection, generated_config)


def submit_training_job(
    request: TrainingRequest,
    jobs_root: Path | str,
    recipes_path: Path | str,
) -> str:
    preflight = validate_training_request(request, recipes_path)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    job_dir = Path(jobs_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(job_dir / "request.json", asdict(request))
    effective_config_path: Path
    if request.submitted_config:
        configs_dir = Path(request.output_dir)
        if request.trainer_adapter != "ophbench_retfound_linear_probe_v1":
            configs_dir = configs_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=False)
        dump_yaml(configs_dir / "base_recipe.yaml", preflight.base_recipe)
        dump_yaml(configs_dir / "submitted_config.yaml", preflight.submitted_config)
        effective_config_path = dump_yaml(
            configs_dir / "effective_config.yaml",
            preflight.generated_config,
        )
        _write_json_atomic(
            configs_dir / "validation_report.json",
            preflight.validation_report,
        )
    else:
        effective_config_path = job_dir / "generated_config.json"
        _write_json_atomic(effective_config_path, preflight.generated_config)
    child_environment, startup_warnings = prepare_training_subprocess_environment()
    update_job_status(
        job_dir,
        "queued",
        job_id=job_id,
        artifact_id=request.artifact_id,
        task_id=request.task_id,
        output_dir=request.output_dir,
        effective_config_path=str(effective_config_path.resolve()),
        startup_warnings=startup_warnings,
    )
    runner = Path(__file__).resolve().parents[1] / "scripts" / "training" / "run_training_job.py"
    command = [sys.executable, str(runner), "--job-dir", str(job_dir.resolve())]
    _write_json_atomic(job_dir / "command.json", {"command": command, "shell": False})
    log_handle = (job_dir / "train.log").open("a", encoding="utf-8")
    launch_options: dict[str, Any] = {
        "cwd": str(Path(__file__).resolve().parents[1]),
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "shell": False,
        "env": child_environment,
    }
    if os.name == "nt":
        launch_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        launch_options["start_new_session"] = True
    try:
        subprocess.Popen(command, **launch_options)
    finally:
        log_handle.close()
    return job_id


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        os.kill(pid, signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(pid, signal.SIGTERM)


def cancel_training_job(job_dir: Path | str) -> None:
    directory = Path(job_dir)
    status = read_job_status(directory)
    if status.get("status") in TERMINAL_STATUSES:
        raise ValueError("任务已结束，不能取消")
    pid = status.get("pid")
    if not pid:
        raise ValueError("任务还未记录可取消的进程 PID")
    _terminate_pid(int(pid))
    update_job_status(directory, "cancelled", cancelled_at_utc=_utc_now())


def archive_training_job(job_dir: Path | str, *, archived: bool) -> None:
    directory = Path(job_dir)
    status = read_job_status(directory)
    if status.get("status") not in TERMINAL_STATUSES:
        raise ValueError("运行中的任务不能归档")
    fields: dict[str, Any] = {"archived": bool(archived)}
    if archived:
        fields["archived_at_utc"] = _utc_now()
    else:
        fields["archived_at_utc"] = None
        fields["restored_at_utc"] = _utc_now()
    update_job_status(directory, str(status["status"]), **fields)


def _require_controlled_child(path: Path, root: Path, *, label: str) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"{label}不在受控训练目录内，已阻止删除：{resolved_path}")
    return resolved_path


def delete_training_job_and_outputs(
    job_dir: Path | str,
    *,
    jobs_root: Path | str,
    runs_root: Path | str,
) -> None:
    directory = _require_controlled_child(Path(job_dir), Path(jobs_root), label="任务记录")
    status = read_job_status(directory)
    if status.get("status") not in TERMINAL_STATUSES:
        raise ValueError("运行中的任务不能删除")
    output_value = str(status.get("output_dir", "")).strip()
    if output_value:
        output_dir = _require_controlled_child(Path(output_value), Path(runs_root), label="实验产物")
        if output_dir.exists():
            shutil.rmtree(output_dir)
    if directory.exists():
        shutil.rmtree(directory)


def list_training_jobs(
    jobs_root: Path | str,
    *,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    root = Path(jobs_root)
    if not root.is_dir():
        return []
    jobs = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        status = read_job_status(directory)
        status["archived"] = bool(status.get("archived", False))
        if status["archived"] and not include_archived:
            continue
        status["job_id"] = status.get("job_id", directory.name)
        status["job_dir"] = str(directory)
        output_dir = str(status.get("output_dir", "")).strip()
        status["output_exists"] = bool(output_dir) and Path(output_dir).is_dir()
        jobs.append(status)
    return jobs


def read_job_log_tail(job_dir: Path | str, max_lines: int = 80) -> str:
    path = Path(job_dir) / "train.log"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, int(max_lines)) :])
