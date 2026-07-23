#!/usr/bin/env python3
"""Launch one resumable parent job for independent frozen-encoder APTOS children."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


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
    args = parser.parse_args()
    job_id = args.job_id or f"aptos-frozen-encoder-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    job = JOB_ROOT / job_id
    if job.exists() and not args.resume:
        raise FileExistsError(f"任务已存在：{job_id}")
    job.mkdir(parents=True, exist_ok=True)
    (job / "children").mkdir(exist_ok=True)
    (job / "logs").mkdir(exist_ok=True)
    _write(job / "request.json", {"job_id": job_id, "task": "aptos_frozen_encoder_linear_probe", "children": [{"model": model, "physical_gpu": gpu} for model, gpu in CHILDREN], "created_at_utc": _now()})
    _write(job / "status.json", {"job_id": job_id, "status": "running", "created_at_utc": _now(), "route_eligible": False})
    children = []
    for model, gpu in CHILDREN:
        child = job / "children" / f"{model}.json"
        log = job / "logs" / f"{model}.log"
        if args.resume and child.is_file() and json.loads(child.read_text(encoding="utf-8")).get("status") == "completed":
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["HF_HOME"] = "/training_data/lizekun/model_cache/huggingface"
        environment["TRANSFORMERS_CACHE"] = "/training_data/lizekun/model_cache/huggingface/hub"
        command = [sys.executable, str(WORKER), "--model", model, "--device", "cuda:0"]
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        _write(child, {"model_id": model, "status": "running", "physical_gpu": gpu, "pid": process.pid, "log": log.relative_to(ROOT).as_posix(), "started_at_utc": _now(), "route_eligible": False})
        children.append((model, gpu, process, child, log))
    for model, gpu, process, child, log in children:
        return_code = process.wait()
        text = log.read_text(encoding="utf-8", errors="replace")
        output = text.strip().splitlines()[-1] if return_code == 0 and text.strip() else ""
        _write(child, {"model_id": model, "status": "completed" if return_code == 0 else "failed", "physical_gpu": gpu, "pid": process.pid, "return_code": return_code, "output_dir": output, "log": log.relative_to(ROOT).as_posix(), "finished_at_utc": _now(), "route_eligible": False})
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((job / "children").glob("*.json"))]
    successful = all(row.get("status") == "completed" for row in rows)
    _write(job / "status.json", {"job_id": job_id, "status": "succeeded" if successful else "completed_with_blockers", "children": rows, "finished_at_utc": _now(), "route_eligible": False})
    print(job_id, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
