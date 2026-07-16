"""Run checkpoint probes in isolated subprocesses and save non-qualifying evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_asset_runtime import (  # noqa: E402
    DEFAULT_ASSET_ROOT,
    REPRESENTATIVE_CHECKPOINTS,
    RUNTIME_ROOT,
    build_asset_readiness,
    run_probe_worker,
)


def _run_worker(row: pd.Series, *, asset_root: Path, device: str, timeout: int) -> dict:
    spec = {
        "checkpoint_id": str(row["checkpoint_id"]),
        "model_id": str(row["model_id"]),
        "probe_profile": str(row["probe_profile"]),
        "asset_path": str(asset_root / str(row["asset_relative_path"])),
        "device": device,
    }
    with tempfile.TemporaryDirectory(prefix="ophagent-asset-smoke-") as directory:
        spec_path = Path(directory) / "spec.json"
        result_path = Path(directory) / "result.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-spec",
            str(spec_path),
            "--worker-result",
            str(result_path),
        ]
        environment = os.environ.copy()
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error_type": "TimeoutExpired"}
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "status": "failed",
                "error_type": "WorkerExitedWithoutResult",
                "error_detail": completed.stderr[-2000:],
            }
        result["worker_returncode"] = completed.returncode
        return result


def _worker(spec_path: Path, result_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        result = run_probe_worker(spec)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_detail": str(exc).replace(str(Path(spec["asset_path"]).parent), "<asset_dir>"),
            "task_inference_ready": False,
            "route_eligible": False,
            "asset_probe_passed": False,
            "runtime_smoke_passed": False,
        }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("status") in {"passed", "skipped"} else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("representative", "all"), default="representative")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--worker-result", type=Path)
    args = parser.parse_args()
    if args.worker_spec:
        if not args.worker_result:
            parser.error("--worker-result is required with --worker-spec")
        return _worker(args.worker_spec, args.worker_result)

    readiness = build_asset_readiness(args.asset_root)
    selected = readiness.loc[readiness["local_asset_exists"]].copy()
    if args.scope == "representative":
        selected = selected.loc[selected["checkpoint_id"].isin(REPRESENTATIVE_CHECKPOINTS)]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{args.scope}"
    output_dir = args.runtime_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    readiness.to_csv(output_dir / "readiness.csv", index=False)
    results = []
    for _, row in selected.iterrows():
        base = {
            "model_id": row["model_id"],
            "checkpoint_id": row["checkpoint_id"],
            "probe_profile": row["probe_profile"],
            "runtime_smoke_eligible": bool(row["runtime_smoke_eligible"]),
            "task_inference_ready": False,
            "route_eligible": False,
        }
        if not bool(row["probe_eligible"]):
            result = {
                **base,
                "status": "skipped",
                "achieved_stage": "metadata_only",
                "error_detail": row["blocked_reason"],
                "asset_probe_passed": False,
                "runtime_smoke_passed": False,
            }
        else:
            result = {
                **base,
                **_run_worker(
                    row,
                    asset_root=args.asset_root,
                    device=args.device,
                    timeout=args.timeout,
                ),
            }
        results.append(result)
        print(
            f"{result['checkpoint_id']}: {result['status']} "
            f"({result.get('achieved_stage', result.get('error_type', 'unknown'))})",
            flush=True,
        )
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_dir / "results.csv", index=False)
    summary = {
        "run_id": run_id,
        "scope": args.scope,
        "asset_count": int(len(readiness)),
        "local_asset_count": int(readiness["local_asset_exists"].sum()),
        "probe_eligible_count": int(readiness["probe_eligible"].sum()),
        "runtime_smoke_eligible_count": int(readiness["runtime_smoke_eligible"].sum()),
        "selected_count": int(len(selected)),
        "passed_count": int(result_frame.get("status", pd.Series(dtype=str)).eq("passed").sum()),
        "asset_probe_passed_count": int(
            result_frame.get("asset_probe_passed", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "runtime_smoke_passed_count": int(
            result_frame.get("runtime_smoke_passed", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "failed_count": int(result_frame.get("status", pd.Series(dtype=str)).eq("failed").sum()),
        "skipped_count": int(result_frame.get("status", pd.Series(dtype=str)).eq("skipped").sum()),
        "timeout_count": int(result_frame.get("status", pd.Series(dtype=str)).eq("timeout").sum()),
        "task_inference_ready_count": 0,
        "route_eligible_count": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not summary["failed_count"] and not summary["timeout_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
