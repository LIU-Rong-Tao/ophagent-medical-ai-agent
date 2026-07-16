"""Run one parent model asset smoke job."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_asset_smoke_jobs import run_asset_smoke_job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status = run_asset_smoke_job(args.job_dir, resume=args.resume)
    return 0 if status in {"succeeded", "completed_with_blockers", "cancelled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
