from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import pytest

import app.model_asset_smoke_jobs as jobs
from app.model_asset_smoke_jobs import AssetSmokeRequest


def _readiness(*checkpoint_ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": checkpoint_id.split("-")[0],
                "checkpoint_id": checkpoint_id,
                "local_asset_exists": True,
                "asset_relative_path": f"model/{checkpoint_id}/model.pth",
                "probe_profile": "test_runtime",
            }
            for checkpoint_id in checkpoint_ids
        ]
    )


def _request(job_id: str, checkpoint_ids: list[str]) -> AssetSmokeRequest:
    return AssetSmokeRequest(
        job_id=job_id,
        checkpoint_ids=checkpoint_ids,
        selection_mode="selected",
        device="cpu",
        device_selection="user_selected",
        resolved_device="cpu",
        timeout_seconds=10,
    )


def _prepare_job(tmp_path: Path, request: AssetSmokeRequest) -> Path:
    job_dir = tmp_path / request.job_id
    (job_dir / "checkpoints").mkdir(parents=True)
    jobs._write_json_atomic(job_dir / "request.json", asdict(request))
    for checkpoint_id in request.checkpoint_ids:
        jobs._write_json_atomic(
            jobs._child_result_path(job_dir, checkpoint_id),
            {"checkpoint_id": checkpoint_id, "status": "pending"},
        )
    jobs._update_progress(job_dir, request)
    jobs.update_asset_smoke_status(job_dir, "queued", job_id=request.job_id)
    return job_dir


def test_submit_rejects_unknown_checkpoint_before_starting_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "build_asset_readiness", lambda: _readiness("known"))
    launched = []
    monkeypatch.setattr(jobs, "_launch_parent", lambda *args, **kwargs: launched.append(args))

    with pytest.raises(ValueError, match="不属于当前本机资产"):
        jobs.submit_asset_smoke_job(
            checkpoint_ids=["unknown"], jobs_root=tmp_path
        )

    assert launched == []
    assert list(tmp_path.iterdir()) == []


def test_submit_persists_device_resolution_and_one_parent_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        jobs, "build_asset_readiness", lambda: _readiness("one", "two")
    )
    monkeypatch.setattr(jobs, "resolve_smoke_device", lambda value: "cuda:3")
    monkeypatch.setattr(jobs, "_launch_parent", lambda job_dir, resume=False: 123)

    job_id = jobs.submit_asset_smoke_job(jobs_root=tmp_path, device="auto")

    directories = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert [path.name for path in directories] == [job_id]
    request = json.loads((directories[0] / "request.json").read_text(encoding="utf-8"))
    assert request["device"] == "auto"
    assert request["device_selection"] == "auto"
    assert request["resolved_device"] == "cuda:3"
    assert request["checkpoint_ids"] == ["one", "two"]


def test_parent_process_start_failure_is_recorded_as_framework_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "build_asset_readiness", lambda: _readiness("one"))
    monkeypatch.setattr(jobs, "resolve_smoke_device", lambda value: "cpu")
    monkeypatch.setattr(
        jobs,
        "_launch_parent",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot spawn")),
    )

    with pytest.raises(OSError, match="cannot spawn"):
        jobs.submit_asset_smoke_job(jobs_root=tmp_path, device="cpu")

    job_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    status = jobs.read_asset_smoke_status(job_dir)
    assert status["status"] == "failed"
    assert status["error_type"] == "OSError"
    assert status["error_message"] == "后台父任务进程启动失败"


def test_child_failure_does_not_stop_remaining_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("asset-smoke-test", ["first", "second"])
    job_dir = _prepare_job(tmp_path, request)
    monkeypatch.setattr(
        jobs, "build_asset_readiness", lambda: _readiness("first", "second")
    )
    calls = []

    def fake_worker(*, row, **kwargs):
        checkpoint_id = str(row["checkpoint_id"])
        calls.append(checkpoint_id)
        if checkpoint_id == "first":
            return {
                "checkpoint_id": checkpoint_id,
                "status": "failed",
                "task_inference_ready": False,
                "route_eligible": False,
            }
        return {
            "checkpoint_id": checkpoint_id,
            "status": "runtime_smoke_passed",
            "runtime_smoke_passed": True,
            "asset_probe_passed": True,
            "task_inference_ready": False,
            "route_eligible": False,
        }

    monkeypatch.setattr(jobs, "_run_child_worker", fake_worker)

    status = jobs.run_asset_smoke_job(job_dir)

    assert calls == ["first", "second"]
    assert status == "completed_with_blockers"
    summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_count"] == 1
    assert summary["runtime_smoke_passed_count"] == 1
    assert summary["task_inference_ready_count"] == 0
    assert summary["route_eligible_count"] == 0


def test_child_timeout_does_not_stop_remaining_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("asset-smoke-timeout", ["slow", "next"])
    job_dir = _prepare_job(tmp_path, request)
    monkeypatch.setattr(
        jobs, "build_asset_readiness", lambda: _readiness("slow", "next")
    )
    calls = []

    def fake_worker(*, row, **kwargs):
        checkpoint_id = str(row["checkpoint_id"])
        calls.append(checkpoint_id)
        return {
            "checkpoint_id": checkpoint_id,
            "status": "timeout" if checkpoint_id == "slow" else "runtime_smoke_passed",
            "runtime_smoke_passed": checkpoint_id == "next",
            "asset_probe_passed": checkpoint_id == "next",
            "task_inference_ready": False,
            "route_eligible": False,
        }

    monkeypatch.setattr(jobs, "_run_child_worker", fake_worker)

    assert jobs.run_asset_smoke_job(job_dir) == "completed_with_blockers"
    assert calls == ["slow", "next"]
    summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["timeout_count"] == 1
    assert summary["runtime_smoke_passed_count"] == 1


def test_resource_blocked_is_not_parent_framework_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("asset-smoke-blocked", ["visionunite-default"])
    job_dir = _prepare_job(tmp_path, request)
    monkeypatch.setattr(
        jobs,
        "build_asset_readiness",
        lambda: _readiness("visionunite-default"),
    )
    monkeypatch.setattr(
        jobs,
        "_run_child_worker",
        lambda **kwargs: {
            "checkpoint_id": "visionunite-default",
            "status": "resource_blocked",
            "resource_blocked": True,
            "asset_probe_passed": True,
            "runtime_smoke_passed": False,
            "task_inference_ready": False,
            "route_eligible": False,
        },
    )

    assert jobs.run_asset_smoke_job(job_dir) == "completed_with_blockers"
    status = jobs.read_asset_smoke_status(job_dir)
    assert status["status"] == "completed_with_blockers"


def test_cancelled_job_can_resume_same_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("asset-smoke-cancelled", ["one"])
    job_dir = _prepare_job(tmp_path, request)
    jobs.update_asset_smoke_status(job_dir, "cancelled", pid=123)
    launched = []
    monkeypatch.setattr(
        jobs,
        "_launch_parent",
        lambda directory, resume=False: launched.append((Path(directory), resume)) or 456,
    )

    pid = jobs.resume_asset_smoke_job(job_dir)

    assert pid == 456
    assert launched == [(job_dir, True)]
    assert jobs.read_asset_smoke_status(job_dir)["status"] == "queued"


def test_cancel_marks_active_child_cancelled_and_keeps_pending_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("asset-smoke-cancel", ["active", "pending"])
    job_dir = _prepare_job(tmp_path, request)
    monkeypatch.setattr(
        jobs, "build_asset_readiness", lambda: _readiness("active", "pending")
    )
    handlers = {}
    monkeypatch.setattr(
        jobs.signal,
        "signal",
        lambda event, handler: handlers.__setitem__(event, handler),
    )

    def fake_worker(**kwargs):
        handlers[jobs.signal.SIGTERM]()
        return {
            "checkpoint_id": "active",
            "status": "cancelled",
            "task_inference_ready": False,
            "route_eligible": False,
        }

    monkeypatch.setattr(jobs, "_run_child_worker", fake_worker)

    assert jobs.run_asset_smoke_job(job_dir) == "cancelled"
    progress = json.loads((job_dir / "progress.json").read_text(encoding="utf-8"))
    by_id = {child["checkpoint_id"]: child for child in progress["children"]}
    assert by_id["active"]["status"] == "cancelled"
    assert by_id["pending"]["status"] == "pending"
    assert jobs.read_asset_smoke_status(job_dir)["status"] == "cancelled"


def test_retry_failed_subset_creates_new_job_with_parent_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("asset-smoke-parent", ["failed", "passed"])
    job_dir = _prepare_job(tmp_path, request)
    jobs._write_json_atomic(
        jobs._child_result_path(job_dir, "failed"),
        {"checkpoint_id": "failed", "status": "failed"},
    )
    jobs._write_json_atomic(
        jobs._child_result_path(job_dir, "passed"),
        {"checkpoint_id": "passed", "status": "runtime_smoke_passed"},
    )
    jobs._update_progress(job_dir, request)
    captured = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return "asset-smoke-child"

    monkeypatch.setattr(jobs, "submit_asset_smoke_job", fake_submit)

    new_job_id = jobs.retry_asset_smoke_subset(
        job_dir, statuses={"failed"}, jobs_root=tmp_path
    )

    assert new_job_id == "asset-smoke-child"
    assert captured["checkpoint_ids"] == ["failed"]
    assert captured["parent_job_id"] == "asset-smoke-parent"
    assert captured["force"] is True


def test_import_legacy_run_references_source_without_copying_details(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy" / "20260716T021348Z-all"
    source.mkdir(parents=True)
    (source / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "20260716T021348Z-all",
                "created_at_utc": "2026-07-16T02:21:31+00:00",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model_id": "retfound",
                "checkpoint_id": "retfound-cfp",
                "probe_profile": "retfound_runtime",
                "status": "passed",
                "asset_probe_passed": True,
                "runtime_smoke_passed": True,
                "details": "large evidence should stay in source",
            },
            {
                "model_id": "visionunite",
                "checkpoint_id": "visionunite-default",
                "probe_profile": "visionunite_resource_audit",
                "status": "skipped",
                "achieved_stage": "resource_blocked",
                "resource_blocked": True,
                "asset_probe_passed": True,
                "runtime_smoke_passed": False,
            },
        ]
    ).to_csv(source / "results.csv", index=False)
    jobs_root = tmp_path / "jobs"

    job_id = jobs.import_legacy_smoke_run(source, jobs_root=jobs_root)

    job_dir = jobs_root / job_id
    status = jobs.read_asset_smoke_status(job_dir)
    assert status["status"] == "completed_with_blockers"
    assert status["summary"]["runtime_smoke_passed_count"] == 1
    assert status["summary"]["asset_probe_passed_count"] == 2
    assert status["summary"]["resource_blocked_count"] == 1
    assert not (job_dir / "results.csv").exists()
    child_text = (job_dir / "checkpoints" / "retfound-cfp.json").read_text(
        encoding="utf-8"
    )
    assert "large evidence" not in child_text
    assert "task_inference_ready" in child_text


def test_missing_legacy_reference_is_reported_without_losing_summary(
    tmp_path: Path,
) -> None:
    request = _request("asset-smoke-legacy", ["one"])
    job_dir = _prepare_job(tmp_path, request)
    payload = asdict(request)
    payload.update(
        {"source_kind": "legacy_reference", "source_reference": "missing/run"}
    )
    jobs._write_json_atomic(job_dir / "request.json", payload)
    jobs.update_asset_smoke_status(job_dir, "succeeded")

    listed = jobs.list_asset_smoke_jobs(tmp_path)

    assert len(listed) == 1
    assert listed[0]["source_available"] is False
    assert listed[0]["status"] == "succeeded"


def test_latest_failure_and_latest_success_are_both_preserved(
    tmp_path: Path,
) -> None:
    for index, child_status in enumerate(["runtime_smoke_passed", "failed"], start=1):
        request = _request(f"asset-smoke-{index}", ["retfound-cfp"])
        job_dir = _prepare_job(tmp_path, request)
        jobs._write_json_atomic(
            jobs._child_result_path(job_dir, "retfound-cfp"),
            {"checkpoint_id": "retfound-cfp", "status": child_status},
        )
        jobs._update_progress(job_dir, request)
        jobs.update_asset_smoke_status(
            job_dir,
            "succeeded" if index == 1 else "completed_with_blockers",
            completed_at_utc=f"2026-07-16T0{index}:00:00+00:00",
        )

    evidence = jobs.checkpoint_smoke_evidence(tmp_path)["retfound-cfp"]

    assert evidence["latest"]["status"] == "failed"
    assert evidence["latest_success"]["status"] == "runtime_smoke_passed"


def test_jobs_are_sorted_by_evidence_time_not_legacy_directory_name(
    tmp_path: Path,
) -> None:
    records = [
        ("asset-smoke-legacy-old", "2026-07-16T02:00:00+00:00"),
        ("asset-smoke-20260716-new", "2026-07-16T07:00:00+00:00"),
    ]
    for job_id, completed_at in records:
        request = _request(job_id, ["one"])
        job_dir = _prepare_job(tmp_path, request)
        jobs.update_asset_smoke_status(
            job_dir, "succeeded", completed_at_utc=completed_at
        )

    listed = jobs.list_asset_smoke_jobs(tmp_path)

    assert [job["job_id"] for job in listed] == [
        "asset-smoke-20260716-new",
        "asset-smoke-legacy-old",
    ]


def test_delete_terminal_smoke_job_keeps_external_source(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    source = tmp_path / "legacy" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text("external evidence", encoding="utf-8")
    request = _request("asset-smoke-delete", ["one"])
    job_dir = _prepare_job(jobs_root, request)
    payload = asdict(request)
    payload["source_reference"] = str(source)
    jobs._write_json_atomic(job_dir / "request.json", payload)
    jobs.update_asset_smoke_status(job_dir, "completed_with_blockers")

    jobs.delete_asset_smoke_job(job_dir, jobs_root=jobs_root)

    assert not job_dir.exists()
    assert source.read_text(encoding="utf-8") == "external evidence"


def test_delete_running_smoke_job_is_rejected(tmp_path: Path) -> None:
    request = _request("asset-smoke-running", ["one"])
    job_dir = _prepare_job(tmp_path, request)
    jobs.update_asset_smoke_status(job_dir, "running")

    with pytest.raises(ValueError, match="运行中的 Smoke 批次不能删除"):
        jobs.delete_asset_smoke_job(job_dir, jobs_root=tmp_path)

    assert job_dir.exists()


def test_delete_smoke_job_outside_controlled_root_is_rejected(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside" / "asset-smoke-outside"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="不在受控 Smoke 任务目录"):
        jobs.delete_asset_smoke_job(outside, jobs_root=tmp_path / "jobs")

    assert outside.exists()


def test_smoke_job_ui_does_not_render_server_paths_or_grant_route_qualification() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "app" / "model_hub_engineering.py"
    ).read_text(encoding="utf-8")
    assert "任务推理可用" in source
    assert "可进入路由池" in source
    assert "job.get('job_dir')" not in source
    smoke_ui = source.split("def _render_asset_smoke_job_records", 1)[1].split(
        "def _render_global_scan_job_records", 1
    )[0]
    assert "/data/" not in smoke_ui
    assert "输出目录" not in smoke_ui
    assert "task_inference_ready_count" in source
    assert "route_eligible_count" in source
    assert "我确认永久删除该 Smoke 批次" in smoke_ui
    assert "模型权重不会被删除" in smoke_ui
