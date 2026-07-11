#!/usr/bin/env python3
"""在独立进程中执行一个已通过预检的训练任务。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Callable

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training_jobs import read_job_status, update_job_status  # noqa: E402
from scripts.training.train_timm_classifier import run_training  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_job(
    job_dir: Path | str,
    training_callable: Callable[[Path], Path] | None = None,
) -> Path:
    directory = Path(job_dir)
    queued_status = read_job_status(directory)
    configured_path = str(queued_status.get("effective_config_path", "")).strip()
    config_path = Path(configured_path) if configured_path else directory / "generated_config.json"
    if training_callable is None:
        if config_path.suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            adapter = str(payload.get("identity", {}).get("trainer_adapter", ""))
        else:
            adapter = "timm_imagefolder_v1"
        if adapter == "ophbench_retfound_linear_probe_v1":
            from scripts.training.train_ophbench_retfound_linear_probe import (
                run_training as run_retfound_linear_probe,
            )

            training_callable = run_retfound_linear_probe
        else:
            training_callable = run_training
    update_job_status(directory, "running", pid=os.getpid(), started_at_utc=_utc_now())
    try:
        output_dir = training_callable(config_path)
    except Exception as exc:
        update_job_status(
            directory,
            "failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            finished_at_utc=_utc_now(),
        )
        raise
    update_job_status(
        directory,
        "succeeded",
        output_dir=str(output_dir),
        finished_at_utc=_utc_now(),
    )
    return Path(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    execute_job(args.job_dir)


if __name__ == "__main__":
    main()
