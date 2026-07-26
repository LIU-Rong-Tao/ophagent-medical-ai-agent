#!/usr/bin/env python3
"""Launch one resumable parent job for independent frozen-encoder task children."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[2]
JOB_ROOT = ROOT / "experiments/model_hub/runtime/aptos_frozen_encoder_jobs"
WORKER = ROOT / "scripts/training/run_aptos_frozen_encoder_probe.py"
CHILDREN = (
    ("eyeclip_cfp", "0"),
    ("keepfit_cfp", "1"),
    ("ret_clip", "2"),
    ("retizero", "3"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    task_config = None
    if args.task_config:
        task_config = yaml.safe_load(args.task_config.read_text(encoding="utf-8"))
        models = tuple(task_config["models"])
        job_root = Path(task_config["job_root"])
        prepare = subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--task-config",
                str(args.task_config),
                "--prepare-task-only",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if prepare.returncode:
            raise RuntimeError(prepare.stderr or prepare.stdout)
        task_id = str(task_config["task_id"])
    else:
        models = tuple(model for model, _ in CHILDREN)
        job_root = JOB_ROOT
        task_id = "aptos_frozen_encoder_linear_probe"
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("至少需要一个 GPU")
    job_prefix = task_id.replace("_", "-")
    job_id = args.job_id or (
        f"{job_prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
        f"{uuid4().hex[:8]}"
    )
    job = job_root / job_id
    if job.exists() and not args.resume:
        raise FileExistsError(f"任务已存在：{job_id}")
    job.mkdir(parents=True, exist_ok=True)
    (job / "children").mkdir(exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    _write(job / "request.json", {"job_id": job_id, "task": task_id, "children": [{"model": model} for model in models], "available_gpus": list(gpus), "task_config": str(args.task_config) if args.task_config else None, "created_at_utc": _now()})
    _write(job / "status.json", {"job_id": job_id, "status": "running", "created_at_utc": _now(), "route_eligible": False})
    pending = []
    for model in models:
        child = job / "children" / f"{model}.json"
        if args.resume and child.is_file() and json.loads(child.read_text(encoding="utf-8")).get("status") == "completed":
            continue
        pending.append(model)
    active = {}
    completed = []
    while pending or active:
        free_gpus = [gpu for gpu in gpus if gpu not in active]
        while pending and free_gpus:
            model = pending.pop(0)
            gpu = free_gpus.pop(0)
            child = job / "children" / f"{model}.json"
            log = job / "logs" / f"{model}.log"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["HF_HOME"] = "/training_data/lizekun/model_cache/huggingface"
            environment["TRANSFORMERS_CACHE"] = "/training_data/lizekun/model_cache/huggingface/hub"
            command = [sys.executable, str(WORKER), "--model", model, "--device", "cuda:0"]
            if args.task_config:
                command.extend(["--task-config", str(args.task_config)])
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            _write(child, {"model_id": model, "status": "running", "physical_gpu": gpu, "pid": process.pid, "log": str(log), "started_at_utc": _now(), "route_eligible": False})
            active[gpu] = (model, process, child, log, handle)
        finished_gpus = []
        for gpu, (model, process, child, log, handle) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            text = log.read_text(encoding="utf-8", errors="replace")
            output = text.strip().splitlines()[-1] if return_code == 0 and text.strip() else ""
            row = {"model_id": model, "status": "completed" if return_code == 0 else "failed", "physical_gpu": gpu, "pid": process.pid, "return_code": return_code, "output_dir": output, "log": str(log), "finished_at_utc": _now(), "route_eligible": False}
            _write(child, row)
            completed.append(row)
            finished_gpus.append(gpu)
        for gpu in finished_gpus:
            del active[gpu]
        if pending or active:
            time.sleep(5)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((job / "children").glob("*.json"))]
    successful = all(row.get("status") == "completed" for row in rows)
    _write(job / "status.json", {"job_id": job_id, "status": "succeeded" if successful else "completed_with_blockers", "children": rows, "finished_at_utc": _now(), "route_eligible": False})
    print(job_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
