from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from scripts.routing.run_timm_adapter_activation import run_protocol
from scripts.routing.timm_adapter_runtime import AdapterBackendResult, AdapterStageError


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_RUNNER = ROOT / "scripts/routing/run_controlled_protocol.py"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fake_backend(job: pd.Series, manifest: pd.DataFrame) -> AdapterBackendResult:
    n_classes = int(job["num_classes"])
    rows: list[dict[str, object]] = []
    for index, item in manifest.reset_index(drop=True).iterrows():
        true_label = int(item["true_label"])
        pred_label = true_label if index != 1 else (true_label + 1) % n_classes
        probs = np.full(n_classes, 0.1 / max(1, n_classes - 1), dtype=float)
        probs[pred_label] = 0.9
        ordered = np.sort(probs)
        row: dict[str, object] = {
            "job_id": job["job_id"],
            "task_id": job["task_id"],
            "artifact_id": job["artifact_id"],
            "image_key": item["image_key"],
            "image_path": item["image_path"],
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": float(ordered[-1]),
            "margin": float(ordered[-1] - ordered[-2]),
            "entropy": 0.2,
            "source": "adapter_generated",
        }
        for class_index, value in enumerate(probs):
            row[f"prob_{class_index}"] = float(value)
        rows.append(row)
    predictions = pd.DataFrame(rows)
    cost_runs = pd.DataFrame(
        [
            {
                "repeat_index": index,
                "total_forward_ms": 4.0 + index * 0.1,
                "ms_per_image": 1.0 + index * 0.025,
                "peak_allocated_memory_mb": 12.0 + index,
            }
            for index in range(1, 6)
        ]
    )
    return AdapterBackendResult(
        predictions=predictions,
        cost_runs=cost_runs,
        device="cpu",
        precision="fp32",
        checkpoint_mb=1.0,
        actual_device_name="test-cpu",
    )


def failing_checkpoint_backend(job: pd.Series, manifest: pd.DataFrame) -> AdapterBackendResult:
    raise AdapterStageError("failed_checkpoint_load", "strict load mismatch")


def create_fixture(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    legacy_rows: list[dict[str, object]] = []
    expert_rows: list[dict[str, object]] = []
    for index in range(4):
        image_path = images / f"case_{index}.png"
        image_path.write_bytes(b"test-image")
        true_label = index % 3
        scout_pred = true_label if index != 1 else (true_label + 1) % 3
        expert_pred = true_label
        legacy_rows.append(
            {
                "image_path": str(image_path),
                "image_key": image_path.stem,
                "true_label": true_label,
                "pred_label": scout_pred,
                "prob_0": 0.8 if scout_pred == 0 else 0.1,
                "prob_1": 0.8 if scout_pred == 1 else 0.1,
                "prob_2": 0.8 if scout_pred == 2 else 0.1,
            }
        )
        expert_rows.append(
            {
                "image_path": str(image_path),
                "image_key": image_path.stem,
                "true_label": true_label,
                "pred_label": expert_pred,
                "prob_0": 0.8 if expert_pred == 0 else 0.1,
                "prob_1": 0.8 if expert_pred == 1 else 0.1,
                "prob_2": 0.8 if expert_pred == 2 else 0.1,
            }
        )

    configs = tmp_path / "configs"
    configs.mkdir()
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    class_mapping = tmp_path / "class_to_idx.json"
    class_mapping.write_text(json.dumps({"a": 0, "b": 1, "c": 2}), encoding="utf-8")
    config_json = tmp_path / "config.json"
    config_json.write_text(json.dumps({"image_size": 224}), encoding="utf-8")
    legacy_path = tmp_path / "legacy_predictions.csv"
    expert_path = tmp_path / "legacy_expert_predictions.csv"
    write_csv(legacy_path, legacy_rows)
    write_csv(expert_path, expert_rows)

    jobs = [
        {
            "job_id": "ready_nominal_job",
            "task_id": "mock_3class",
            "dataset_id": "mock_data",
            "disease_family": "mock",
            "artifact_id": "mock_scout",
            "role_candidates": "scout",
            "arch": "mock_arch",
            "checkpoint_path": str(checkpoint),
            "config_path": str(config_json),
            "legacy_prediction_path": str(legacy_path),
            "data_root": str(images),
            "class_to_idx_path": str(class_mapping),
            "num_classes": 3,
            "input_size": 224,
            "norm": "imagenet",
            "batch_size": 2,
            "device": "cpu",
            "precision": "fp32",
            "warmup_runs": 3,
            "timed_runs": 5,
            "label_structure": "nominal",
            "enabled": "true",
        },
        {
            "job_id": "missing_checkpoint_job",
            "task_id": "mock_3class",
            "dataset_id": "mock_data",
            "disease_family": "mock",
            "artifact_id": "missing_checkpoint",
            "role_candidates": "scout",
            "arch": "mock_arch",
            "checkpoint_path": str(tmp_path / "missing.pth"),
            "config_path": str(config_json),
            "legacy_prediction_path": str(legacy_path),
            "data_root": str(images),
            "class_to_idx_path": str(class_mapping),
            "num_classes": 3,
            "input_size": 224,
            "norm": "imagenet",
            "batch_size": 2,
            "device": "cpu",
            "precision": "fp32",
            "warmup_runs": 3,
            "timed_runs": 5,
            "label_structure": "nominal",
            "enabled": "true",
        },
        {
            "job_id": "missing_manifest_job",
            "task_id": "mock_3class",
            "dataset_id": "mock_data",
            "disease_family": "mock",
            "artifact_id": "missing_manifest",
            "role_candidates": "scout",
            "arch": "mock_arch",
            "checkpoint_path": str(checkpoint),
            "config_path": str(config_json),
            "legacy_prediction_path": str(tmp_path / "missing_predictions.csv"),
            "data_root": str(images),
            "class_to_idx_path": str(class_mapping),
            "num_classes": 3,
            "input_size": 224,
            "norm": "imagenet",
            "batch_size": 2,
            "device": "cpu",
            "precision": "fp32",
            "warmup_runs": 3,
            "timed_runs": 5,
            "label_structure": "nominal",
            "enabled": "true",
        },
    ]
    jobs_path = configs / "timm_adapter_jobs.csv"
    write_csv(jobs_path, jobs)

    replays = [
        {
            "replay_id": "mixed_replay",
            "task_id": "mock_3class",
            "scout_job_id": "ready_nominal_job",
            "expert_artifact_id": "mock_expert",
            "expert_legacy_prediction_path": str(expert_path),
            "policies": "low_confidence|low_margin|high_entropy",
            "budgets": "0.5",
            "prediction_source_mode": "mixed_adapter_legacy",
            "enabled": "true",
        },
        {
            "replay_id": "missing_expert_replay",
            "task_id": "mock_3class",
            "scout_job_id": "ready_nominal_job",
            "expert_artifact_id": "missing_expert",
            "expert_legacy_prediction_path": str(tmp_path / "missing_expert.csv"),
            "policies": "low_confidence",
            "budgets": "0.5",
            "prediction_source_mode": "mixed_adapter_legacy",
            "enabled": "true",
        },
    ]
    replays_path = configs / "routing_replay_protocols.csv"
    write_csv(replays_path, replays)

    output_dir = tmp_path / "outputs"
    protocol = configs / "protocol.yaml"
    protocol.write_text(
        "\n".join(
            [
                "protocol_id: fixture_v085c",
                f"timm_adapter_jobs: {jobs_path}",
                f"routing_replay_protocols: {replays_path}",
                f"output_dir: {output_dir}",
                "report: summary.html",
            ]
        ),
        encoding="utf-8",
    )
    return protocol, output_dir


def test_audit_builds_exact_manifest_and_explicit_skip_statuses(tmp_path: Path):
    protocol, output_dir = create_fixture(tmp_path)

    run_protocol(protocol, output_dir=output_dir, stage="audit")

    inventory = read_csv(output_dir / "model_inventory.csv")
    statuses = dict(zip(inventory["job_id"], inventory["status"]))
    assert statuses["ready_nominal_job"] == "ready_for_adapter"
    assert statuses["missing_checkpoint_job"] == "skipped_missing_checkpoint"
    assert statuses["missing_manifest_job"] == "skipped_missing_input_csv"
    manifest = read_csv(output_dir / "input_manifests/ready_nominal_job_input_manifest.csv")
    assert list(manifest.columns) == ["image_key", "image_path", "true_label"]
    assert len(manifest) == 4
    assert manifest["image_path"].map(lambda value: Path(value).exists()).all()


def test_onboarding_writes_real_schema_metrics_cost_and_manifest(tmp_path: Path):
    protocol, output_dir = create_fixture(tmp_path)

    run_protocol(protocol, output_dir=output_dir, stage="all", backend=fake_backend)

    jobs = read_csv(output_dir / "adapter_job_summary.csv")
    completed = jobs.loc[jobs["job_id"] == "ready_nominal_job"].iloc[0]
    assert completed["status"] == "completed"
    assert completed["n_images"] == 4
    assert completed["cost_scope"] == "forward_only"
    predictions = read_csv(output_dir / "onboarded_models/ready_nominal_job/predictions.csv")
    assert {"image_path", "prob_0", "prob_1", "prob_2", "source"} <= set(predictions)
    assert np.allclose(predictions[["prob_0", "prob_1", "prob_2"]].sum(axis=1), 1.0)
    baseline = read_csv(output_dir / "onboarded_models/ready_nominal_job/model_baseline.csv")
    assert baseline.loc[0, "source"] == "adapter_generated"
    assert baseline.loc[0, "qwk_status"] == "not_applicable"
    assert pd.isna(baseline.loc[0, "qwk"])
    cost = read_csv(output_dir / "onboarded_models/ready_nominal_job/forward_cost_summary.csv")
    assert cost.loc[0, "timed_runs"] == 5
    assert cost.loc[0, "median_ms_per_image"] > 0
    assert cost.loc[0, "cv_ms_per_image"] > 0
    manifest = read_csv(output_dir / "onboarded_models/ready_nominal_job/adapter_manifest.csv")
    assert manifest.loc[0, "checkpoint_sha256"]
    assert manifest.loc[0, "predictions_sha256"]
    assert manifest.loc[0, "input_manifest_path"].endswith("input_manifest.csv")


def test_onboarding_reuses_audit_outputs_without_changing_their_signatures(tmp_path: Path):
    protocol, output_dir = create_fixture(tmp_path)
    run_protocol(protocol, output_dir=output_dir, stage="audit")
    audit_paths = [
        output_dir / "model_inventory.csv",
        output_dir / "input_manifests/ready_nominal_job_input_manifest.csv",
    ]
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in audit_paths
    }
    time.sleep(0.01)

    run_protocol(protocol, output_dir=output_dir, stage="onboarding", backend=fake_backend)

    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in audit_paths
    }
    assert after == before


def test_checkpoint_failure_is_structured_and_does_not_emit_predictions(tmp_path: Path):
    protocol, output_dir = create_fixture(tmp_path)

    run_protocol(protocol, output_dir=output_dir, stage="onboarding", backend=failing_checkpoint_backend)

    jobs = read_csv(output_dir / "adapter_job_summary.csv")
    failed = jobs.loc[jobs["job_id"] == "ready_nominal_job"].iloc[0]
    assert failed["status"] == "failed_checkpoint_load"
    assert "strict load mismatch" in failed["notes"]
    assert not (output_dir / "onboarded_models/ready_nominal_job/predictions.csv").exists()


def test_sanity_checks_and_mixed_replay_are_published(tmp_path: Path):
    protocol, output_dir = create_fixture(tmp_path)

    run_protocol(protocol, output_dir=output_dir, stage="all", backend=fake_backend)

    prediction_check = read_csv(output_dir / "adapter_vs_legacy_prediction_check.csv")
    assert prediction_check.loc[0, "n_overlap"] == 4
    assert prediction_check.loc[0, "status"] in {
        "matched",
        "close_but_not_identical",
        "different_but_explained",
    }
    baseline_check = read_csv(output_dir / "adapter_vs_legacy_baseline_check.csv")
    assert {"accuracy_diff", "macro_f1_diff", "status"} <= set(baseline_check)
    replay = read_csv(output_dir / "routing_replay_summary.csv")
    completed = replay.loc[replay["replay_id"] == "mixed_replay"].iloc[0]
    assert completed["status"] == "completed"
    assert completed["prediction_source_mode"] == "mixed_adapter_legacy"
    missing = replay.loc[replay["replay_id"] == "missing_expert_replay"].iloc[0]
    assert missing["status"] == "skipped_missing_expert_predictions"


def test_dry_run_does_not_write_outputs(tmp_path: Path):
    protocol, output_dir = create_fixture(tmp_path)

    run_protocol(protocol, output_dir=output_dir, stage="all", dry_run=True)

    assert not output_dir.exists()


def test_controlled_runner_second_resume_skips_all_v085c_stages(tmp_path: Path):
    protocol, _ = create_fixture(tmp_path)
    work = tmp_path / "work" / "activation"
    published = tmp_path / "published"
    wrapper = tmp_path / "fixture_activation_runner.py"
    wrapper.write_text(
        """
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scripts.routing.run_timm_adapter_activation import run_protocol
from scripts.routing.timm_adapter_runtime import AdapterBackendResult

def backend(job, manifest):
    rows = []
    n_classes = int(job['num_classes'])
    for _, item in manifest.reset_index(drop=True).iterrows():
        pred = int(item['true_label'])
        probs = np.full(n_classes, 0.1 / (n_classes - 1), dtype=float)
        probs[pred] = 0.9
        row = {
            'job_id': job['job_id'], 'task_id': job['task_id'],
            'artifact_id': job['artifact_id'], 'image_key': item['image_key'],
            'image_path': item['image_path'], 'true_label': int(item['true_label']),
            'pred_label': pred, 'confidence': 0.9, 'margin': 0.85,
            'entropy': 0.2, 'source': 'adapter_generated',
        }
        for index, value in enumerate(probs): row[f'prob_{index}'] = float(value)
        rows.append(row)
    runs = pd.DataFrame([
        {'repeat_index': i, 'total_forward_ms': 4.0, 'ms_per_image': 1.0,
         'peak_allocated_memory_mb': 10.0} for i in range(1, 6)
    ])
    return AdapterBackendResult(pd.DataFrame(rows), runs, 'cpu', 'fp32', 1.0, 'test-cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--stage', type=str, required=True)
args = parser.parse_args()
run_protocol(Path(args.config), output_dir=Path(args.output_dir), stage=args.stage, backend=backend)
""".strip(),
        encoding="utf-8",
    )
    controlled = tmp_path / "controlled.json"
    controlled.write_text(
        json.dumps(
            {
                "protocol_id": "fixture_v085c_resume",
                "mode": "exploratory",
                "selection_split": "test",
                "evaluation_split": "test",
                "output_dir": str(published),
                "stages": [
                    {
                        "id": "audit",
                        "kind": "routing",
                        "env": {"PYTHONPATH": str(ROOT)},
                        "command": ["{python}", str(wrapper), "--config", str(protocol), "--output-dir", str(work), "--stage", "audit"],
                        "inputs": [str(wrapper), str(protocol)],
                        "outputs": [str(work / "model_inventory.csv"), str(work / "input_manifests/ready_nominal_job_input_manifest.csv")],
                    },
                    {
                        "id": "onboarding",
                        "kind": "routing",
                        "env": {"PYTHONPATH": str(ROOT)},
                        "depends_on": ["audit"],
                        "command": ["{python}", str(wrapper), "--config", str(protocol), "--output-dir", str(work), "--stage", "onboarding"],
                        "inputs": [str(wrapper), str(protocol), str(work / "model_inventory.csv")],
                        "outputs": [str(work / "adapter_job_summary.csv"), str(work / "model_baselines_from_adapters.csv")],
                    },
                    {
                        "id": "replay",
                        "kind": "routing",
                        "env": {"PYTHONPATH": str(ROOT)},
                        "depends_on": ["onboarding"],
                        "command": ["{python}", str(wrapper), "--config", str(protocol), "--output-dir", str(work), "--stage", "replay"],
                        "inputs": [str(wrapper), str(protocol), str(work / "adapter_job_summary.csv")],
                        "outputs": [str(work / "routing_replay_summary.csv"), str(work / "summary.html")],
                    },
                ],
                "publish": {"artifacts": []},
            }
        ),
        encoding="utf-8",
    )

    command = [sys.executable, str(CONTROLLED_RUNNER), "--config", str(controlled), "--resume"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")

    assert first.returncode == 0, first.stderr
    assert first.stdout.count("[RUNNING]") == 3
    assert second.returncode == 0, second.stderr
    assert second.stdout.count("[SKIPPED]") == 3
    assert second.stdout.count("fingerprint unchanged") == 3


def test_readme_documents_activation_boundaries():
    readme = (
        Path(__file__).resolve().parents[1]
        / "experiments/v0_8_5c_timm_adapter_activation/README.md"
    )
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "v0.8.5b" in text and "v0.8.5c" in text
    assert "RETFound" in text and "本版本不启用" in text
    assert "forward-only cost" in text
    assert "不是真实部署端到端延迟" in text
    assert "mixed_adapter_legacy" in text
    assert "sanity check" in text
    assert "strict reproduction" in text
