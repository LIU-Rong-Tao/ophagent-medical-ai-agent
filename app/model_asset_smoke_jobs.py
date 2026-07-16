"""Background batch jobs for checkpoint-level model asset smoke validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
from uuid import uuid4

import pandas as pd

from app.model_asset_runtime import DEFAULT_ASSET_ROOT, build_asset_readiness
from app.training_jobs import prepare_training_subprocess_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_SMOKE_JOBS_ROOT = PROJECT_ROOT / "experiments/model_hub/runtime/asset_smoke_jobs"
WORKER_RUNNER = PROJECT_ROOT / "scripts/routing/run_model_asset_smoke.py"
PARENT_RUNNER = PROJECT_ROOT / "scripts/routing/run_model_asset_smoke_job.py"

PARENT_TERMINAL_STATUSES = {
    "succeeded",
    "completed_with_blockers",
    "failed",
    "cancelled",
}
CHILD_TERMINAL_STATUSES = {
    "runtime_smoke_passed",
    "asset_probe_passed",
    "resource_blocked",
    "failed",
    "timeout",
    "not_applicable",
    "cancelled",
}
CHILD_SUCCESS_STATUSES = {
    "runtime_smoke_passed",
    "asset_probe_passed",
    "not_applicable",
}


@dataclass(frozen=True)
class AssetSmokeRequest:
    job_id: str
    checkpoint_ids: list[str]
    selection_mode: str
    device: str
    device_selection: str
    resolved_device: str
    timeout_seconds: int
    force: bool = False
    parent_job_id: str | None = None
    source_kind: str = "background_job"
    source_reference: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        return dict(default or {})
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _validate_device(device: str) -> str:
    value = str(device).strip().lower()
    if value in {"auto", "cpu"} or value == "cuda" or value.startswith("cuda:"):
        return value
    raise ValueError("device 必须为 auto、cpu 或 cuda[:n]")


def resolve_smoke_device(device: str) -> str:
    """Resolve auto to the CUDA device with the most reported free memory."""

    requested = _validate_device(device)
    if requested != "auto":
        return "cuda:0" if requested == "cuda" else requested
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        candidates = []
        for line in completed.stdout.splitlines():
            index, free_memory = (part.strip() for part in line.split(",", 1))
            candidates.append((int(free_memory), int(index)))
        if completed.returncode == 0 and candidates:
            return f"cuda:{max(candidates)[1]}"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def read_asset_smoke_status(job_dir: Path | str) -> dict[str, Any]:
    directory = Path(job_dir)
    status = _read_json(
        directory / "status.json",
        {"status": "unknown", "error_message": "任务状态文件不存在"},
    )
    status["job_id"] = status.get("job_id", directory.name)
    return status


def update_asset_smoke_status(
    job_dir: Path | str, status: str, **fields: Any
) -> dict[str, Any]:
    directory = Path(job_dir)
    current = read_asset_smoke_status(directory)
    if current.get("status") == "unknown":
        current = {"job_id": directory.name, "created_at_utc": _utc_now()}
    current.update(fields)
    current["status"] = status
    current["updated_at_utc"] = _utc_now()
    _write_json_atomic(directory / "status.json", current)
    return current


def _flag(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _normalise_child_status(result: dict[str, Any]) -> str:
    if _flag(result.get("resource_blocked")) or result.get(
        "achieved_stage"
    ) == "resource_blocked":
        return "resource_blocked"
    raw_status = str(result.get("status", "failed"))
    if raw_status == "timeout":
        return "timeout"
    if raw_status == "failed":
        return "failed"
    if _flag(result.get("runtime_smoke_passed")):
        return "runtime_smoke_passed"
    if _flag(result.get("asset_probe_passed")):
        return "asset_probe_passed"
    if raw_status == "skipped":
        return "not_applicable"
    return "failed"


def _child_result_path(job_dir: Path, checkpoint_id: str) -> Path:
    return job_dir / "checkpoints" / f"{checkpoint_id}.json"


def _child_rows(job_dir: Path, request: AssetSmokeRequest) -> list[dict[str, Any]]:
    rows = []
    for checkpoint_id in request.checkpoint_ids:
        result = _read_json(
            _child_result_path(job_dir, checkpoint_id),
            {"checkpoint_id": checkpoint_id, "status": "pending"},
        )
        result["checkpoint_id"] = checkpoint_id
        rows.append(result)
    return rows


def _progress_payload(
    job_dir: Path, request: AssetSmokeRequest, *, active_checkpoint_id: str | None = None
) -> dict[str, Any]:
    rows = _child_rows(job_dir, request)
    counts = {
        status: sum(str(row.get("status")) == status for row in rows)
        for status in (
            "pending",
            "running",
            "runtime_smoke_passed",
            "asset_probe_passed",
            "resource_blocked",
            "failed",
            "timeout",
            "not_applicable",
            "cancelled",
        )
    }
    completed_count = sum(
        str(row.get("status")) in CHILD_TERMINAL_STATUSES for row in rows
    )
    return {
        "job_id": request.job_id,
        "total_count": len(rows),
        "completed_count": completed_count,
        "active_checkpoint_id": active_checkpoint_id,
        "counts": counts,
        "children": rows,
        "updated_at_utc": _utc_now(),
    }


def _update_progress(
    job_dir: Path, request: AssetSmokeRequest, *, active_checkpoint_id: str | None = None
) -> dict[str, Any]:
    progress = _progress_payload(
        job_dir, request, active_checkpoint_id=active_checkpoint_id
    )
    _write_json_atomic(job_dir / "progress.json", progress)
    return progress


def _summary_from_progress(
    progress: dict[str, Any], *, parent_status: str
) -> dict[str, Any]:
    counts = dict(progress.get("counts", {}))
    children = list(progress.get("children", []))
    return {
        "job_id": progress.get("job_id"),
        "status": parent_status,
        "target_count": int(progress.get("total_count", 0)),
        "completed_count": int(progress.get("completed_count", 0)),
        "runtime_smoke_passed_count": sum(
            _flag(child.get("runtime_smoke_passed")) for child in children
        ),
        "asset_probe_passed_count": sum(
            _flag(child.get("asset_probe_passed")) for child in children
        ),
        "resource_blocked_count": int(counts.get("resource_blocked", 0)),
        "failed_count": int(counts.get("failed", 0)),
        "timeout_count": int(counts.get("timeout", 0)),
        "not_applicable_count": int(counts.get("not_applicable", 0)),
        "cancelled_count": int(counts.get("cancelled", 0)),
        "task_inference_ready_count": 0,
        "route_eligible_count": 0,
        "completed_at_utc": _utc_now(),
    }


def _parent_status(progress: dict[str, Any]) -> str:
    counts = dict(progress.get("counts", {}))
    blockers = sum(
        int(counts.get(status, 0))
        for status in ("resource_blocked", "failed", "timeout", "cancelled")
    )
    return "completed_with_blockers" if blockers else "succeeded"


def _launch_parent(job_dir: Path, *, resume: bool = False) -> int:
    command = [sys.executable, str(PARENT_RUNNER), "--job-dir", str(job_dir.resolve())]
    if resume:
        command.append("--resume")
    _write_json_atomic(job_dir / "command.json", {"command": command, "shell": False})
    environment, warnings = prepare_training_subprocess_environment(os.environ)
    if warnings:
        _write_json_atomic(job_dir / "startup_warnings.json", {"warnings": warnings})
    with (job_dir / "job.log").open("a", encoding="utf-8") as log_handle:
        launch_options: dict[str, Any] = {
            "cwd": PROJECT_ROOT,
            "env": environment,
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "shell": False,
        }
        if os.name == "nt":
            launch_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            launch_options["start_new_session"] = True
        process = subprocess.Popen(command, **launch_options)
    update_asset_smoke_status(job_dir, "queued", pid=process.pid)
    return process.pid


def submit_asset_smoke_job(
    *,
    checkpoint_ids: list[str] | None = None,
    selection_mode: str = "all_local",
    device: str = "auto",
    timeout_seconds: int = 600,
    force: bool = False,
    parent_job_id: str | None = None,
    jobs_root: Path | str = ASSET_SMOKE_JOBS_ROOT,
) -> str:
    readiness = build_asset_readiness()
    local_ids = readiness.loc[
        readiness["local_asset_exists"].astype(bool), "checkpoint_id"
    ].astype(str).tolist()
    selected = local_ids if checkpoint_ids is None else list(dict.fromkeys(checkpoint_ids))
    unknown = sorted(set(selected) - set(local_ids))
    if unknown:
        raise ValueError(f"checkpoint 不属于当前本机资产：{unknown}")
    if not selected:
        raise ValueError("至少选择一个本机 checkpoint")
    timeout_value = int(timeout_seconds)
    if timeout_value <= 0:
        raise ValueError("timeout_seconds 必须大于 0")
    requested_device = _validate_device(device)
    resolved_device = resolve_smoke_device(requested_device)
    job_id = (
        "asset-smoke-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid4().hex[:8]
    )
    request = AssetSmokeRequest(
        job_id=job_id,
        checkpoint_ids=selected,
        selection_mode=str(selection_mode),
        device=requested_device,
        device_selection=("auto" if requested_device == "auto" else "user_selected"),
        resolved_device=resolved_device,
        timeout_seconds=timeout_value,
        force=bool(force),
        parent_job_id=parent_job_id,
    )
    job_dir = Path(jobs_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "checkpoints").mkdir()
    _write_json_atomic(job_dir / "request.json", asdict(request))
    for checkpoint_id in request.checkpoint_ids:
        _write_json_atomic(
            _child_result_path(job_dir, checkpoint_id),
            {"checkpoint_id": checkpoint_id, "status": "pending"},
        )
    _update_progress(job_dir, request)
    _write_json_atomic(
        job_dir / "runtime_environment.json",
        {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "requested_device": request.device,
            "device_selection": request.device_selection,
            "resolved_device": request.resolved_device,
            "created_at_utc": _utc_now(),
        },
    )
    update_asset_smoke_status(
        job_dir,
        "queued",
        job_id=job_id,
        job_type="asset_smoke",
        target_count=len(selected),
        completed_count=0,
        parent_job_id=parent_job_id,
        resolved_device=resolved_device,
        archived=False,
    )
    try:
        _launch_parent(job_dir)
    except Exception as exc:
        update_asset_smoke_status(
            job_dir,
            "failed",
            error_type=type(exc).__name__,
            error_message="后台父任务进程启动失败",
            completed_at_utc=_utc_now(),
        )
        raise
    return job_id


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        process.terminate()


def _run_child_worker(
    *,
    job_dir: Path,
    row: pd.Series,
    request: AssetSmokeRequest,
    cancel_requested: Any,
) -> dict[str, Any]:
    checkpoint_id = str(row["checkpoint_id"])
    spec = {
        "checkpoint_id": checkpoint_id,
        "model_id": str(row["model_id"]),
        "probe_profile": str(row["probe_profile"]),
        "asset_path": str(DEFAULT_ASSET_ROOT / str(row["asset_relative_path"])),
        "device": request.resolved_device,
    }
    with tempfile.TemporaryDirectory(
        prefix="ophagent-asset-smoke-", dir=job_dir
    ) as directory:
        spec_path = Path(directory) / "spec.json"
        result_path = Path(directory) / "result.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        command = [
            sys.executable,
            str(WORKER_RUNNER),
            "--worker-spec",
            str(spec_path),
            "--worker-result",
            str(result_path),
        ]
        launch_options: dict[str, Any] = {
            "cwd": PROJECT_ROOT,
            "stdout": None,
            "stderr": None,
            "shell": False,
            "env": os.environ.copy(),
        }
        if os.name == "nt":
            launch_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            launch_options["start_new_session"] = True
        started = time.monotonic()
        process = subprocess.Popen(command, **launch_options)
        while process.poll() is None:
            if cancel_requested():
                _terminate_process(process)
                process.wait(timeout=10)
                return {
                    "checkpoint_id": checkpoint_id,
                    "model_id": str(row["model_id"]),
                    "status": "cancelled",
                    "elapsed_seconds": time.monotonic() - started,
                    "task_inference_ready": False,
                    "route_eligible": False,
                }
            if time.monotonic() - started > request.timeout_seconds:
                _terminate_process(process)
                process.wait(timeout=10)
                return {
                    "checkpoint_id": checkpoint_id,
                    "model_id": str(row["model_id"]),
                    "status": "timeout",
                    "elapsed_seconds": time.monotonic() - started,
                    "task_inference_ready": False,
                    "route_eligible": False,
                }
            time.sleep(0.2)
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "status": "failed",
                "error_type": "WorkerExitedWithoutResult",
                "error_detail": "子进程未生成可解析结果",
            }
        result.update(
            {
                "checkpoint_id": checkpoint_id,
                "model_id": str(row["model_id"]),
                "probe_profile": str(row["probe_profile"]),
                "worker_returncode": process.returncode,
                "elapsed_seconds": result.get(
                    "elapsed_seconds", time.monotonic() - started
                ),
                "task_inference_ready": False,
                "route_eligible": False,
            }
        )
        result["status"] = _normalise_child_status(result)
        return result


def _write_result_tables(job_dir: Path, request: AssetSmokeRequest) -> None:
    rows = _child_rows(job_dir, request)
    columns = [
        "model_id",
        "checkpoint_id",
        "probe_profile",
        "status",
        "elapsed_seconds",
        "asset_probe_passed",
        "runtime_smoke_passed",
        "task_inference_ready",
        "route_eligible",
        "error_type",
        "error_detail",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = None
    _write_csv_atomic(job_dir / "smoke_results.csv", frame[columns])
    failures = frame.loc[frame["status"].isin(["failed", "timeout"]), columns]
    _write_csv_atomic(job_dir / "smoke_failures.csv", failures)


def run_asset_smoke_job(job_dir: Path | str, *, resume: bool = False) -> str:
    directory = Path(job_dir)
    request = AssetSmokeRequest(**_read_json(directory / "request.json"))
    cancelled = False

    def request_cancel(*_: Any) -> None:
        nonlocal cancelled
        cancelled = True

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_cancel)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, request_cancel)
    update_asset_smoke_status(
        directory,
        "running",
        pid=os.getpid(),
        started_at_utc=_utc_now(),
        resumed=bool(resume),
    )
    readiness = build_asset_readiness()
    by_checkpoint = readiness.set_index("checkpoint_id", drop=False)
    try:
        for checkpoint_id in request.checkpoint_ids:
            path = _child_result_path(directory, checkpoint_id)
            current = _read_json(path, {"status": "pending"})
            current_status = str(current.get("status", "pending"))
            if current_status in CHILD_TERMINAL_STATUSES and not (
                resume and current_status == "cancelled"
            ) and not request.force:
                continue
            if cancelled:
                break
            if checkpoint_id not in by_checkpoint.index:
                result = {
                    "checkpoint_id": checkpoint_id,
                    "status": "failed",
                    "error_type": "CheckpointNotFound",
                    "error_detail": "当前运行准备矩阵中不存在该 checkpoint",
                    "task_inference_ready": False,
                    "route_eligible": False,
                }
            else:
                row = by_checkpoint.loc[checkpoint_id]
                _write_json_atomic(
                    path,
                    {
                        "checkpoint_id": checkpoint_id,
                        "model_id": str(row["model_id"]),
                        "probe_profile": str(row["probe_profile"]),
                        "status": "running",
                        "started_at_utc": _utc_now(),
                    },
                )
                _update_progress(
                    directory, request, active_checkpoint_id=checkpoint_id
                )
                result = _run_child_worker(
                    job_dir=directory,
                    row=row,
                    request=request,
                    cancel_requested=lambda: cancelled,
                )
            result["completed_at_utc"] = _utc_now()
            _write_json_atomic(path, result)
            _update_progress(directory, request)
        if cancelled:
            for checkpoint_id in request.checkpoint_ids:
                path = _child_result_path(directory, checkpoint_id)
                current = _read_json(path, {"status": "pending"})
                if current.get("status") == "running":
                    current.update(
                        {"status": "cancelled", "completed_at_utc": _utc_now()}
                    )
                    _write_json_atomic(path, current)
            progress = _update_progress(directory, request)
            _write_result_tables(directory, request)
            summary = _summary_from_progress(progress, parent_status="cancelled")
            _write_json_atomic(directory / "summary.json", summary)
            update_asset_smoke_status(
                directory,
                "cancelled",
                completed_count=summary["completed_count"],
                cancelled_at_utc=_utc_now(),
            )
            return "cancelled"
        progress = _update_progress(directory, request)
        parent_status = _parent_status(progress)
        summary = _summary_from_progress(progress, parent_status=parent_status)
        _write_result_tables(directory, request)
        _write_json_atomic(directory / "summary.json", summary)
        update_asset_smoke_status(
            directory,
            parent_status,
            completed_count=summary["completed_count"],
            summary=summary,
            completed_at_utc=_utc_now(),
        )
        return parent_status
    except Exception as exc:
        update_asset_smoke_status(
            directory,
            "failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            completed_at_utc=_utc_now(),
        )
        raise


def list_asset_smoke_jobs(
    jobs_root: Path | str = ASSET_SMOKE_JOBS_ROOT,
    *,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    root = Path(jobs_root)
    if not root.is_dir():
        return []
    jobs = []
    for directory in (path for path in root.iterdir() if path.is_dir()):
        status = read_asset_smoke_status(directory)
        status["job_type"] = "asset_smoke"
        status["job_dir"] = str(directory)
        status["archived"] = bool(status.get("archived", False))
        if status["archived"] and not include_archived:
            continue
        status["progress"] = _read_json(directory / "progress.json")
        status["summary"] = _read_json(directory / "summary.json")
        status["source_available"] = _source_reference_available(directory)
        jobs.append(status)
    jobs.sort(
        key=lambda job: str(
            job.get("completed_at_utc")
            or job.get("updated_at_utc")
            or job.get("created_at_utc")
            or ""
        ),
        reverse=True,
    )
    return jobs


def _source_reference_available(job_dir: Path) -> bool:
    request = _read_json(job_dir / "request.json")
    reference = request.get("source_reference")
    if not reference:
        return True
    return (PROJECT_ROOT / str(reference)).exists()


def read_asset_smoke_log_tail(job_dir: Path | str, max_lines: int = 80) -> str:
    path = Path(job_dir) / "job.log"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, int(max_lines)) :])


def cancel_asset_smoke_job(job_dir: Path | str) -> None:
    directory = Path(job_dir)
    status = read_asset_smoke_status(directory)
    if status.get("status") in PARENT_TERMINAL_STATUSES:
        raise ValueError("任务已结束，不能取消")
    pid = status.get("pid")
    if not pid:
        raise ValueError("任务还未记录可取消的进程 PID")
    if os.name == "nt":
        os.kill(int(pid), signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(int(pid), signal.SIGTERM)
    update_asset_smoke_status(directory, "cancelled", cancelled_at_utc=_utc_now())


def resume_asset_smoke_job(job_dir: Path | str) -> int:
    directory = Path(job_dir)
    status = read_asset_smoke_status(directory)
    if status.get("status") != "cancelled":
        raise ValueError("只有已取消的 Smoke 批次可以恢复")
    update_asset_smoke_status(directory, "queued", resumed_at_utc=_utc_now())
    return _launch_parent(directory, resume=True)


def retry_asset_smoke_subset(
    job_dir: Path | str,
    *,
    statuses: set[str],
    jobs_root: Path | str = ASSET_SMOKE_JOBS_ROOT,
) -> str:
    directory = Path(job_dir)
    progress = _read_json(directory / "progress.json")
    checkpoint_ids = [
        str(child["checkpoint_id"])
        for child in progress.get("children", [])
        if str(child.get("status")) in statuses
    ]
    if not checkpoint_ids:
        raise ValueError("当前批次没有符合条件的子任务")
    request = _read_json(directory / "request.json")
    return submit_asset_smoke_job(
        checkpoint_ids=checkpoint_ids,
        selection_mode="retry_" + "_".join(sorted(statuses)),
        device=str(request.get("device", "auto")),
        timeout_seconds=int(request.get("timeout_seconds", 600)),
        force=True,
        parent_job_id=directory.name,
        jobs_root=jobs_root,
    )


def archive_asset_smoke_job(job_dir: Path | str, *, archived: bool) -> None:
    directory = Path(job_dir)
    status = read_asset_smoke_status(directory)
    if status.get("status") not in PARENT_TERMINAL_STATUSES:
        raise ValueError("运行中的任务不能归档")
    update_asset_smoke_status(
        directory,
        str(status["status"]),
        archived=bool(archived),
        archived_at_utc=_utc_now() if archived else None,
    )


def import_legacy_smoke_run(
    run_dir: Path | str,
    *,
    jobs_root: Path | str = ASSET_SMOKE_JOBS_ROOT,
) -> str:
    source = Path(run_dir)
    summary = _read_json(source / "summary.json")
    results_path = source / "results.csv"
    if not summary or not results_path.is_file():
        raise ValueError("历史 Smoke 目录缺少 summary.json 或 results.csv")
    results = pd.read_csv(results_path)
    run_id = str(summary.get("run_id") or source.name)
    job_id = "asset-smoke-legacy-" + run_id
    job_dir = Path(jobs_root) / job_id
    if job_dir.exists():
        return job_id
    checkpoint_ids = results["checkpoint_id"].astype(str).tolist()
    request = AssetSmokeRequest(
        job_id=job_id,
        checkpoint_ids=checkpoint_ids,
        selection_mode="legacy_import",
        device="unknown",
        device_selection="historical_record",
        resolved_device="unknown",
        timeout_seconds=0,
        source_kind="legacy_reference",
        source_reference=_safe_relative(source),
    )
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "checkpoints").mkdir()
    _write_json_atomic(job_dir / "request.json", asdict(request))
    concise_rows = []
    for record in results.to_dict(orient="records"):
        child = {
            "model_id": str(record.get("model_id", "")),
            "checkpoint_id": str(record["checkpoint_id"]),
            "probe_profile": str(record.get("probe_profile", "")),
            "status": _normalise_child_status(record),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "asset_probe_passed": _flag(record.get("asset_probe_passed")),
            "runtime_smoke_passed": _flag(record.get("runtime_smoke_passed")),
            "task_inference_ready": False,
            "route_eligible": False,
            "source_reference": _safe_relative(results_path),
            "completed_at_utc": summary.get("created_at_utc"),
        }
        _write_json_atomic(
            _child_result_path(job_dir, child["checkpoint_id"]), child
        )
        concise_rows.append(child)
    progress = _update_progress(job_dir, request)
    parent_status = _parent_status(progress)
    normalised_summary = _summary_from_progress(progress, parent_status=parent_status)
    normalised_summary["source_reference"] = _safe_relative(source)
    normalised_summary["completed_at_utc"] = (
        summary.get("created_at_utc") or normalised_summary["completed_at_utc"]
    )
    _write_json_atomic(job_dir / "summary.json", normalised_summary)
    _write_result_tables(job_dir, request)
    _write_json_atomic(
        job_dir / "runtime_environment.json",
        {
            "source_kind": "legacy_reference",
            "source_reference": _safe_relative(source),
            "imported_at_utc": _utc_now(),
        },
    )
    update_asset_smoke_status(
        job_dir,
        parent_status,
        job_id=job_id,
        job_type="asset_smoke",
        target_count=len(concise_rows),
        completed_count=len(concise_rows),
        summary=normalised_summary,
        source_kind="legacy_reference",
        archived=False,
        completed_at_utc=summary.get("created_at_utc") or _utc_now(),
    )
    return job_id


def checkpoint_smoke_evidence(
    jobs_root: Path | str = ASSET_SMOKE_JOBS_ROOT,
) -> dict[str, dict[str, dict[str, Any] | None]]:
    """Return latest result and latest successful evidence for each checkpoint."""

    evidence: dict[str, dict[str, dict[str, Any] | None]] = {}
    jobs = list_asset_smoke_jobs(jobs_root, include_archived=True)
    completed_jobs = [
        job
        for job in jobs
        if str(job.get("status")) in PARENT_TERMINAL_STATUSES
        and str(job.get("status")) != "failed"
    ]
    completed_jobs.sort(
        key=lambda job: str(
            job.get("completed_at_utc") or job.get("updated_at_utc") or ""
        )
    )
    for job in completed_jobs:
        for child in job.get("progress", {}).get("children", []):
            if str(child.get("status")) not in CHILD_TERMINAL_STATUSES:
                continue
            checkpoint_id = str(child.get("checkpoint_id", ""))
            if not checkpoint_id:
                continue
            record = {
                **child,
                "job_id": job.get("job_id"),
                "job_status": job.get("status"),
                "job_completed_at_utc": job.get("completed_at_utc"),
            }
            entry = evidence.setdefault(
                checkpoint_id, {"latest": None, "latest_success": None}
            )
            entry["latest"] = record
            if str(child.get("status")) in CHILD_SUCCESS_STATUSES:
                entry["latest_success"] = record
    return evidence
