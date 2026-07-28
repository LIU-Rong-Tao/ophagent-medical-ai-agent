from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.model_hub_index import (
    DATASET_LABELS,
    EXPERIMENT_LABELS,
    build_dataset_index,
    build_job_index,
    build_model_capability_index,
    build_route_run_index,
    build_task_asset_index,
)


def _write_prediction_registry(root: Path) -> Path:
    validation = root / "assets/validation.csv"
    test = root / "assets/test.csv"
    validation.parent.mkdir(parents=True)
    validation.write_text("case_id,y_true,y_pred\nv1,0,0\n", encoding="utf-8")
    test.write_text("case_id,y_true,y_pred\nt1,0,0\n", encoding="utf-8")
    registry = root / "experiments/example/configs/example_prediction_assets.csv"
    registry.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "task_id": "aptos_dr_5class",
                "artifact_id": "model_a",
                "model_family": "Example",
                "validation_prediction_path": "assets/validation.csv",
                "test_prediction_path": "assets/test.csv",
                "checkpoint_path": "",
                "current_run_reproducible": True,
                "route_eligible": False,
            }
        ]
    ).to_csv(registry, index=False)
    return registry


def test_task_asset_index_uses_registry_and_existing_assets(tmp_path: Path) -> None:
    registry = _write_prediction_registry(tmp_path)

    assets = build_task_asset_index(tmp_path)

    assert len(assets) == 1
    assert assets.iloc[0]["registry_path"] == registry.relative_to(tmp_path).as_posix()
    assert bool(assets.iloc[0]["validation_asset_exists"])
    assert bool(assets.iloc[0]["test_asset_exists"])
    assert not bool(assets.iloc[0]["route_eligible"])


def test_deepdrid_dataset_and_experiment_semantics_are_separate() -> None:
    assert DATASET_LABELS["deepdrid_dr_5class_external"] == "DeepDRiD"
    assert DATASET_LABELS["deepdrid_dr_5class_native"] == "DeepDRiD"
    assert (
        EXPERIMENT_LABELS["deepdrid_dr_5class_external"]
        == "APTOS 模型冻结迁移"
    )
    assert (
        EXPERIMENT_LABELS["deepdrid_dr_5class_native"]
        == "DeepDRiD 原生任务适配"
    )


def test_route_index_reads_existing_result_package(tmp_path: Path) -> None:
    output = tmp_path / "experiments/example/model_hub_validation"
    output.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "task_id": "aptos_dr_5class",
                "pairing_id": "a_to_b",
                "status": "completed",
            },
            {
                "task_id": "aptos_dr_5class",
                "pairing_id": "a_to_b",
                "status": "completed",
            },
        ]
    ).to_csv(output / "pairing_results.csv", index=False)
    pd.DataFrame([{"artifact_id": "a"}, {"artifact_id": "b"}]).to_csv(
        output / "model_hub_snapshot.csv",
        index=False,
    )
    (output / "artifact_manifest.csv").write_text("path,sha256\n", encoding="utf-8")
    (output / "run_config.yaml").write_text(
        "protocol_version: example_v1\nroute_eligible: false\n",
        encoding="utf-8",
    )

    runs = build_route_run_index(tmp_path)

    assert len(runs) == 1
    assert runs.iloc[0]["pairing_count"] == 1
    assert runs.iloc[0]["result_rows"] == 2
    assert runs.iloc[0]["model_count"] == 2
    assert runs.iloc[0]["stage"] == "Validation 候选扫描"


def test_dataset_index_keeps_admission_separate_from_task_assets(
    tmp_path: Path,
) -> None:
    _write_prediction_registry(tmp_path)
    coverage = (
        tmp_path
        / "experiments/opening_risk_routing_closure/model_hub_coverage_matrix.csv"
    )
    coverage.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset_id": "REFUGE",
                "current_qualification": (
                    "admitted_for_labeled_train_validation_only"
                ),
            }
        ]
    ).to_csv(coverage, index=False)
    assets = build_task_asset_index(tmp_path)
    routes = build_route_run_index(tmp_path)

    datasets = build_dataset_index(tmp_path, assets, routes)

    assert set(datasets["dataset_label"]) == {"APTOS2019", "REFUGE"}
    refuge = datasets.loc[datasets["dataset_label"].eq("REFUGE")].iloc[0]
    assert refuge["prediction_assets"] == 0
    assert not bool(refuge["route_eligible"])


def test_job_index_includes_each_runtime_namespace(tmp_path: Path) -> None:
    smoke = tmp_path / "experiments/model_hub/runtime/asset_smoke_jobs/smoke-1"
    inference = tmp_path / "experiments/model_hub/runtime/inference_jobs/infer-1"
    smoke.mkdir(parents=True)
    inference.mkdir(parents=True)
    (smoke / "status.json").write_text(
        json.dumps({"status": "succeeded"}),
        encoding="utf-8",
    )
    (inference / "status.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )

    jobs = build_job_index(tmp_path)

    assert set(jobs["job_id"]) == {"smoke-1", "infer-1"}
    assert set(jobs["status"]) == {"succeeded", "running"}


def test_model_capability_keeps_cost_protocols_distinct() -> None:
    assets = pd.DataFrame(
        [
            {
                "task_id": "task_a",
                "artifact_id": "batch16",
                "cost_scope": "H100 GPU forward-only batch16",
                "cost_status": "measured",
                "forward_cost_ms_per_image": 1.0,
            },
            {
                "task_id": "task_a",
                "artifact_id": "batch32",
                "cost_scope": "H100 forward-only validation batch32",
                "cost_status": "measured",
                "forward_cost_ms_per_image": 1.0,
            },
            {
                "task_id": "task_a",
                "artifact_id": "ambiguous",
                "cost_scope": "H100 forward-only",
                "cost_status": "measured",
                "forward_cost_ms_per_image": 1.0,
            },
        ]
    )

    capabilities = build_model_capability_index(
        assets,
        pd.DataFrame(),
    ).set_index("artifact_id")

    assert capabilities.loc["batch16", "cost_protocol_id"] == (
        "h100_fp32_forward_only_batch1_batch16_w10_r30_v1"
    )
    assert capabilities.loc["batch32", "cost_protocol_id"] == (
        "h100_fp32_forward_only_batch32_split5_v1"
    )
    assert capabilities.loc["ambiguous", "cost_protocol_id"] == (
        "cost_protocol_unavailable"
    )
