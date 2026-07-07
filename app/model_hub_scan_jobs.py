"""模型中转台全局候选扫描后台任务。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

import pandas as pd

from app.model_hub_data import estimate_global_composition_count, scan_global_composition_candidates
from app.training_jobs import prepare_training_subprocess_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_HUB_OUTPUT_DIR = PROJECT_ROOT / "experiments/v0_8_6_interactive_model_hub/outputs"
SCAN_JOBS_ROOT = PROJECT_ROOT / "experiments/model_hub/runtime/global_scan_jobs"
SCAN_RUNS_ROOT = PROJECT_ROOT / "experiments/model_hub/runs/global_scan"
RUNNER = PROJECT_ROOT / "scripts/routing/run_model_hub_global_scan_job.py"


@dataclass(frozen=True)
class GlobalScanRequest:
    job_id: str
    task_id: str
    scout_ids: list[str]
    expert_ids: list[str]
    budgets: list[float]
    max_scouts: int
    max_experts: int
    primary_metric: str
    top_n: int
    output_dir: str
    model_hub_output_dir: str = str(MODEL_HUB_OUTPUT_DIR)
    model_hub_root: str = str(PROJECT_ROOT / "experiments/model_hub")
    display_metrics: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_global_scan_status(job_dir: Path | str) -> dict[str, Any]:
    path = Path(job_dir) / "status.json"
    if not path.is_file():
        return {"status": "unknown", "error_message": "任务状态文件不存在"}
    return json.loads(path.read_text(encoding="utf-8"))


def update_global_scan_status(job_dir: Path | str, status: str, **fields: Any) -> dict[str, Any]:
    directory = Path(job_dir)
    current = read_global_scan_status(directory)
    if current.get("status") == "unknown":
        current = {"created_at_utc": _utc_now()}
    current.update(fields)
    current["status"] = status
    current["updated_at_utc"] = _utc_now()
    _write_json(directory / "status.json", current)
    return current


def run_global_scan_request(request: GlobalScanRequest, models: pd.DataFrame) -> Path:
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimated_points = estimate_global_composition_count(
        n_scouts=len(request.scout_ids),
        n_experts=len(request.expert_ids),
        max_scouts=request.max_scouts,
        max_experts=request.max_experts,
        n_budgets=len(request.budgets),
    )
    results = scan_global_composition_candidates(
        models,
        task_id=request.task_id,
        scout_ids=request.scout_ids,
        expert_ids=request.expert_ids,
        budgets=request.budgets,
        max_scouts=request.max_scouts,
        max_experts=request.max_experts,
        primary_metric=request.primary_metric,
    )
    results_path = output_dir / "global_scan_results.csv"
    top_path = output_dir / "global_scan_top.csv"
    manifest_path = output_dir / "scan_manifest.json"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    completed = results.loc[results.get("scan_status", pd.Series(dtype=str)).astype(str).eq("completed")].copy()
    if "global_rank_primary" in completed.columns:
        completed = completed.sort_values("global_rank_primary", na_position="last")
    completed.head(max(1, int(request.top_n))).to_csv(top_path, index=False, encoding="utf-8-sig")
    _write_json(
        manifest_path,
        {
            **asdict(request),
            "created_at_utc": _utc_now(),
            "estimated_points": int(estimated_points),
            "n_rows": int(len(results)),
            "n_completed": int(len(completed)),
            "results_path": str(results_path.resolve()),
            "top_path": str(top_path.resolve()),
        },
    )
    return output_dir


def submit_global_scan_job(
    *,
    task_id: str,
    scout_ids: list[str],
    expert_ids: list[str],
    budgets: list[float],
    max_scouts: int,
    max_experts: int,
    primary_metric: str,
    top_n: int,
    display_metrics: list[str] | None = None,
    jobs_root: Path | str = SCAN_JOBS_ROOT,
    runs_root: Path | str = SCAN_RUNS_ROOT,
) -> str:
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    output_dir = Path(runs_root) / str(task_id) / job_id
    request = GlobalScanRequest(
        job_id=job_id,
        task_id=str(task_id),
        scout_ids=list(scout_ids),
        expert_ids=list(expert_ids),
        budgets=[float(value) for value in budgets],
        max_scouts=int(max_scouts),
        max_experts=int(max_experts),
        primary_metric=str(primary_metric),
        top_n=int(top_n),
        output_dir=str(output_dir),
        display_metrics=list(display_metrics or []),
    )
    job_dir = Path(jobs_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    _write_json(job_dir / "request.json", asdict(request))
    estimated_points = estimate_global_composition_count(
        n_scouts=len(request.scout_ids),
        n_experts=len(request.expert_ids),
        max_scouts=request.max_scouts,
        max_experts=request.max_experts,
        n_budgets=len(request.budgets),
    )
    update_global_scan_status(
        job_dir,
        "queued",
        job_id=job_id,
        job_type="global_scan",
        task_id=request.task_id,
        output_dir=str(output_dir),
        estimated_points=int(estimated_points),
    )
    command = [sys.executable, str(RUNNER), "--request", str((job_dir / "request.json").resolve()), "--status", str((job_dir / "status.json").resolve())]
    _write_json(job_dir / "command.json", {"command": command, "shell": False})
    environment, warnings = prepare_training_subprocess_environment(os.environ)
    if warnings:
        _write_json(job_dir / "startup_warnings.json", {"warnings": warnings})
    with (job_dir / "job.log").open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
    update_global_scan_status(job_dir, "queued", pid=process.pid)
    return job_id


def list_global_scan_jobs(
    jobs_root: Path | str = SCAN_JOBS_ROOT,
    *,
    include_archived: bool = False,
) -> list[dict[str, Any]]:
    root = Path(jobs_root)
    if not root.is_dir():
        return []
    jobs: list[dict[str, Any]] = []
    for directory in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        status = read_global_scan_status(directory)
        status["job_id"] = status.get("job_id", directory.name)
        status["job_type"] = "global_scan"
        status["job_dir"] = str(directory)
        status["archived"] = bool(status.get("archived", False))
        if status["archived"] and not include_archived:
            continue
        output_dir = str(status.get("output_dir", "")).strip()
        status["output_exists"] = bool(output_dir) and Path(output_dir).is_dir()
        jobs.append(status)
    return jobs


def read_global_scan_log_tail(job_dir: Path | str, max_lines: int = 80) -> str:
    path = Path(job_dir) / "job.log"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def load_global_scan_results(job: dict[str, Any]) -> pd.DataFrame:
    path = Path(str(job.get("output_dir", ""))) / "global_scan_results.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_completed_global_scan(
    task_id: str,
    jobs_root: Path | str = SCAN_JOBS_ROOT,
) -> dict[str, Any] | None:
    matches = [
        job
        for job in list_global_scan_jobs(jobs_root, include_archived=False)
        if str(job.get("task_id")) == str(task_id) and str(job.get("status")) == "succeeded"
    ]
    if not matches:
        return None
    return max(matches, key=lambda job: str(job.get("job_id", "")))
