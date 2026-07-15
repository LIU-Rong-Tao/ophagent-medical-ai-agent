from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repository-root bootstrap above must run before these imports.
from scripts.routing.benchmark_artifact_forward_cost import (  # noqa: E402
    BenchmarkError,
    aggregate_cost_runs,
    load_benchmark_config,
    validate_runtime_device,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/routing/benchmark_artifact_forward_cost.py"


def valid_config() -> dict:
    return {
        "benchmark_id": "fixture_cost",
        "timing_scope": "forward_only",
        "device": "cuda",
        "precision": "fp32",
        "warmup_runs": 3,
        "timed_runs": 5,
        "artifacts": [
            {
                "artifact_id": "fixture_scout",
                "cost_profile_id": "rtx4090_fp32_bs32_forward_only_v1",
                "adapter": "glaucoma_convnext_tiny",
                "task_id": "glaucoma_3class",
                "dataset_id": "glaucoma_fundus",
                "model_family": "convnext",
                "batch_size": 32,
                "checkpoint_path": "/server/scout.pth",
                "data_root": "/server/data",
                "split": "test",
                "model_config": "/server/scout.yaml",
                "class_to_idx_path": "/server/class_to_idx.json",
            },
            {
                "artifact_id": "fixture_expert",
                "cost_profile_id": "rtx4090_fp32_bs24_forward_only_v1",
                "adapter": "glaucoma_retfound_dinov2",
                "task_id": "glaucoma_3class",
                "dataset_id": "glaucoma_fundus",
                "model_family": "retfound_dinov2",
                "batch_size": 24,
                "checkpoint_path": "/server/expert.pth",
                "data_root": "/server/data",
                "split": "test",
                "retfound_root": "/server/RETFound",
            },
        ],
    }


def test_aggregate_cost_runs_reports_robust_multi_run_statistics():
    runs = pd.DataFrame(
        [
            {
                "artifact_id": "fixture_scout",
                "cost_profile_id": "profile",
                "task_id": "task",
                "dataset_id": "dataset",
                "model_family": "convnext",
                "repeat_index": 1,
                "n_images": 100,
                "batch_size": 32,
                "device": "cuda:0",
                "precision": "fp32",
                "warmup_runs": 3,
                "timed_runs": 3,
                "total_forward_ms": 90.0,
                "ms_per_image": 0.9,
                "peak_allocated_memory_mb": 500.0,
                "checkpoint_mb": 100.0,
                "timing_scope": "forward_only",
                "timing_source": "fixture",
            },
            {
                "artifact_id": "fixture_scout",
                "cost_profile_id": "profile",
                "task_id": "task",
                "dataset_id": "dataset",
                "model_family": "convnext",
                "repeat_index": 2,
                "n_images": 100,
                "batch_size": 32,
                "device": "cuda:0",
                "precision": "fp32",
                "warmup_runs": 3,
                "timed_runs": 3,
                "total_forward_ms": 100.0,
                "ms_per_image": 1.0,
                "peak_allocated_memory_mb": 510.0,
                "checkpoint_mb": 100.0,
                "timing_scope": "forward_only",
                "timing_source": "fixture",
            },
            {
                "artifact_id": "fixture_scout",
                "cost_profile_id": "profile",
                "task_id": "task",
                "dataset_id": "dataset",
                "model_family": "convnext",
                "repeat_index": 3,
                "n_images": 100,
                "batch_size": 32,
                "device": "cuda:0",
                "precision": "fp32",
                "warmup_runs": 3,
                "timed_runs": 3,
                "total_forward_ms": 110.0,
                "ms_per_image": 1.1,
                "peak_allocated_memory_mb": 505.0,
                "checkpoint_mb": 100.0,
                "timing_scope": "forward_only",
                "timing_source": "fixture",
            },
        ]
    )

    summary = aggregate_cost_runs(runs)
    row = summary.iloc[0]

    assert len(summary) == 1
    assert row["estimated_forward_ms_per_image"] == pytest.approx(1.0)
    assert row["mean_ms_per_image"] == pytest.approx(1.0)
    assert row["median_ms_per_image"] == pytest.approx(1.0)
    assert row["std_ms_per_image"] == pytest.approx(0.1)
    assert row["cv_ms_per_image"] == pytest.approx(0.1)
    assert row["images_per_second"] == pytest.approx(1000.0)
    assert row["peak_allocated_memory_mb"] == pytest.approx(510.0)
    assert row["n_repeats"] == 3
    assert row["cost_status"] == "measured"


def test_load_benchmark_config_rejects_duplicate_artifact_profile(tmp_path: Path):
    payload = valid_config()
    payload["artifacts"].append(dict(payload["artifacts"][0]))
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="artifact_id.*cost_profile_id"):
        load_benchmark_config(path)


def test_cost_profile_id_cannot_mix_batch_sizes(tmp_path: Path):
    payload = valid_config()
    payload["artifacts"][1]["cost_profile_id"] = payload["artifacts"][0][
        "cost_profile_id"
    ]
    payload["artifacts"][1]["batch_size"] = 16
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="cost_profile_id.*batch_size"):
        load_benchmark_config(path)


@pytest.mark.parametrize(
    ("artifact_index", "field"),
    [(0, "model_config"), (0, "class_to_idx_path"), (1, "retfound_root")],
)
def test_load_benchmark_config_checks_adapter_specific_fields(
    tmp_path: Path, artifact_index: int, field: str
):
    payload = valid_config()
    del payload["artifacts"][artifact_index][field]
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkError, match=field):
        load_benchmark_config(path)


def test_dry_run_does_not_import_torch_or_require_server_paths(tmp_path: Path):
    config_path = tmp_path / "benchmark.json"
    runs_path = tmp_path / "work" / "runs.csv"
    summary_path = tmp_path / "outputs" / "summary.csv"
    config_path.write_text(json.dumps(valid_config()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config_path),
            "--runs-output",
            str(runs_path),
            "--summary-output",
            str(summary_path),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0, result.stderr
    assert "fixture_scout" in result.stdout
    assert "fixture_expert" in result.stdout
    assert not runs_path.exists()
    assert not summary_path.exists()


def test_aggregate_cost_runs_rejects_incomplete_repeats():
    rows = []
    for repeat_index, latency in [(1, 1.0), (2, 1.1)]:
        rows.append(
            {
                "artifact_id": "fixture",
                "cost_profile_id": "profile",
                "task_id": "task",
                "dataset_id": "dataset",
                "model_family": "convnext",
                "repeat_index": repeat_index,
                "n_images": 100,
                "batch_size": 32,
                "device": "cuda:0",
                "precision": "fp32",
                "warmup_runs": 3,
                "timed_runs": 3,
                "total_forward_ms": latency * 100,
                "ms_per_image": latency,
                "peak_allocated_memory_mb": 500.0,
                "checkpoint_mb": 100.0,
                "timing_scope": "forward_only",
                "timing_source": "fixture",
            }
        )

    with pytest.raises(BenchmarkError, match="重复测量数量"):
        aggregate_cost_runs(pd.DataFrame(rows))


def test_runtime_device_must_match_declared_cost_profile():
    config = valid_config()
    config["expected_device_name_contains"] = "4090"

    assert validate_runtime_device(config, "NVIDIA GeForce RTX 4090") == (
        "NVIDIA GeForce RTX 4090"
    )
    with pytest.raises(BenchmarkError, match="GPU.*4090"):
        validate_runtime_device(config, "NVIDIA H100 80GB HBM3")
