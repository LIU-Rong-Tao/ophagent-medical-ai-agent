#!/usr/bin/env python3
"""运行模型中转台全局候选扫描后台任务。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_hub_data import load_model_hub_outputs
from app.model_hub_scan_jobs import GlobalScanRequest, run_global_scan_request, update_global_scan_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    request = GlobalScanRequest(**payload)
    job_dir = args.status.parent
    try:
        update_global_scan_status(job_dir, "running")
        data = load_model_hub_outputs(
            Path(request.model_hub_output_dir),
            model_hub_root=Path(request.model_hub_root),
        )
        output_dir = run_global_scan_request(request, data["models"])
    except Exception as exc:
        update_global_scan_status(
            job_dir,
            "failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1
    update_global_scan_status(job_dir, "succeeded", output_dir=str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
