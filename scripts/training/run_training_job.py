#!/usr/bin/env python3
"""在独立进程中执行一个已通过预检的训练任务。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training_jobs import read_job_status, update_job_status
from scripts.training.train_timm_classifier import run_training


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_job(
    job_dir: Path | str,
    training_callable: Callable[[Path], Path] = run_training,
) -> Path:
    directory = Path(job_dir)
    queued_status = read_job_status(directory)
    configured_path = str(queued_status.get("effective_config_path", "")).strip()
    config_path = Path(configured_path) if configured_path else directory / "generated_config.json"
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
