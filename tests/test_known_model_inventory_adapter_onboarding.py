from __future__ import annotations

import csv
from pathlib import Path

from scripts.routing.run_known_model_inventory_adapter_onboarding import run_protocol


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def create_fixture(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    outputs = tmp_path / "outputs"
    legacy = tmp_path / "legacy_outputs"
    v085 = tmp_path / "v085_outputs"
    experiments = tmp_path / "experiments"
    checkpoints = tmp_path / "checkpoints"
    existing_checkpoint = checkpoints / "mock_scout.pth"
    existing_checkpoint.parent.mkdir(parents=True)
    existing_checkpoint.write_text("not a real checkpoint", encoding="utf-8")
    class_mapping = checkpoints / "class_to_idx.json"
    class_mapping.write_text('{"normal": 0, "disease": 1}', encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()

    write_csv(
        legacy / "model_baselines.csv",
        ["artifact_id", "task_id", "accuracy", "macro_f1"],
        [
            {
                "artifact_id": "legacy_retinal_model",
                "task_id": "mock_task",
                "accuracy": 0.75,
                "macro_f1": 0.7,
            }
        ],
    )
    write_csv(
        legacy / "routing_results.csv",
        ["protocol_id", "task_id", "scout_artifact", "expert_artifact", "budget", "policy", "accuracy", "macro_f1"],
        [
            {
                "protocol_id": "legacy_protocol",
                "task_id": "mock_task",
                "scout_artifact": "legacy_retinal_model",
                "expert_artifact": "legacy_expert",
                "budget": 0.5,
                "policy": "low_confidence",
                "accuracy": 0.8,
                "macro_f1": 0.76,
            }
        ],
    )
    write_csv(
        legacy / "case_audit.csv",
        ["image_key", "true_label"],
        [{"image_key": "case_1", "true_label": 0}],
    )
    write_csv(
        v085 / "registered_tasks.csv",
        ["task_id", "dataset_id"],
        [{"task_id": "mock_task", "dataset_id": "mock_dataset"}],
    )
    write_csv(
        v085 / "registered_models.csv",
        ["artifact_id", "task_id"],
        [{"artifact_id": "pending_timm_model", "task_id": "mock_task"}],
    )
    write_csv(
        v085 / "route_protocol_summary.csv",
        ["protocol_id", "task_id"],
        [{"protocol_id": "mock_route", "task_id": "mock_task"}],
    )
    write_csv(
        v085 / "model_baselines_all.csv",
        ["artifact_id", "task_id", "accuracy", "macro_f1"],
        [
            {
                "artifact_id": "pending_timm_model",
                "task_id": "mock_task",
                "accuracy": 0.72,
                "macro_f1": 0.68,
            },
            {
                "artifact_id": "mixed_only_model",
                "task_id": "",
                "accuracy": 0.61,
                "macro_f1": 0.57,
            },
        ],
    )
    write_csv(
        v085 / "routing_results_all.csv",
        ["protocol_id", "task_id", "scout_artifact", "expert_artifact", "budget", "policy", "accuracy", "macro_f1"],
        [
            {
                "protocol_id": "mock_route",
                "task_id": "mock_task",
                "scout_artifact": "pending_timm_model",
                "expert_artifact": "mock_expert",
                "budget": 0.5,
                "policy": "low_confidence",
                "accuracy": 0.78,
                "macro_f1": 0.74,
            }
        ],
    )
    write_csv(
        v085 / "risk_results_all.csv",
        ["protocol_id", "risk_event"],
        [{"protocol_id": "mock_route", "risk_event": "generic_miss"}],
    )
    write_csv(
        v085 / "case_audit_all.csv",
        ["image_key", "true_label"],
        [{"image_key": "case_2", "true_label": 1}],
    )
    write_csv(
        v085 / "artifact_manifest.csv",
        ["artifact_name", "source_path"],
        [{"artifact_name": "model_baselines_all", "source_path": str(v085 / "model_baselines_all.csv")}],
    )
    (v085 / "summary.html").write_text("<html>\n<body>mock, summary, with commas</body>\n</html>", encoding="utf-8")

    write_csv(
        configs / "inventory_sources.csv",
        [
            "source_id",
            "source_type",
            "source_path",
            "task_id",
            "dataset_id",
            "disease_family",
            "source_schema",
            "enabled",
            "notes",
        ],
        [
            {
                "source_id": "legacy_mock",
                "source_type": "legacy_outputs",
                "source_path": str(legacy),
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "source_schema": "fixture",
                "enabled": "true",
                "notes": "旧产物来源",
            },
            {
                "source_id": "v085_registry_outputs",
                "source_type": "registry_outputs",
                "source_path": str(v085),
                "task_id": "mixed",
                "dataset_id": "mixed",
                "disease_family": "mixed",
                "source_schema": "v085",
                "enabled": "true",
                "notes": "v0.8.5 registry 输出",
            },
            {
                "source_id": "missing_source",
                "source_type": "legacy_outputs",
                "source_path": str(tmp_path / "missing"),
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "source_schema": "fixture",
                "enabled": "true",
                "notes": "缺失来源",
            },
            {
                "source_id": "scan_experiments",
                "source_type": "scan_root",
                "source_path": str(experiments),
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "source_schema": "scan",
                "enabled": "true",
                "notes": "空扫描目录",
            },
        ],
    )
    write_csv(
        configs / "adapter_registry.csv",
        [
            "adapter_id",
            "adapter_type",
            "model_family",
            "supported_backbones",
            "status",
            "enabled",
            "notes",
        ],
        [
            {
                "adapter_id": "synthetic_mock_v1",
                "adapter_type": "synthetic_mock",
                "model_family": "mock",
                "supported_backbones": "mock_scout|mock_expert|mock_scout_b",
                "status": "available",
                "enabled": "true",
                "notes": "测试用合成适配器",
            },
            {
                "adapter_id": "retfound_adapter_v1",
                "adapter_type": "retfound",
                "model_family": "retfound",
                "supported_backbones": "retfound_dinov2",
                "status": "needs_loader_audit",
                "enabled": "true",
                "notes": "需要 loader 审计",
            },
        ],
    )
    write_csv(
        configs / "onboarding_jobs.csv",
        [
            "job_id",
            "task_id",
            "dataset_id",
            "disease_family",
            "artifact_id",
            "role_candidates",
            "adapter_id",
            "model_family",
            "backbone",
            "checkpoint_path",
            "config_path",
            "data_root",
            "class_to_idx_path",
            "num_classes",
            "input_size",
            "batch_size",
            "device",
            "precision",
            "enabled",
            "run_adapter",
            "notes",
        ],
        [
            {
                "job_id": "mock_scout_job",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "artifact_id": "mock_scout",
                "role_candidates": "scout",
                "adapter_id": "synthetic_mock_v1",
                "model_family": "mock",
                "backbone": "mock_scout",
                "checkpoint_path": str(existing_checkpoint),
                "config_path": "",
                "data_root": str(data_root),
                "class_to_idx_path": str(class_mapping),
                "num_classes": 2,
                "input_size": 224,
                "batch_size": 4,
                "device": "cpu",
                "precision": "fp32",
                "enabled": "true",
                "run_adapter": "true",
                "notes": "可运行 mock scout",
            },
            {
                "job_id": "mock_scout_b_job",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "artifact_id": "mock_scout_b",
                "role_candidates": "scout",
                "adapter_id": "synthetic_mock_v1",
                "model_family": "mock",
                "backbone": "mock_scout_b",
                "checkpoint_path": str(existing_checkpoint),
                "config_path": "",
                "data_root": str(data_root),
                "class_to_idx_path": str(class_mapping),
                "num_classes": 2,
                "input_size": 224,
                "batch_size": 4,
                "device": "cpu",
                "precision": "fp32",
                "enabled": "true",
                "run_adapter": "true",
                "notes": "可运行第二 scout",
            },
            {
                "job_id": "mock_expert_job",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "artifact_id": "mock_expert",
                "role_candidates": "expert",
                "adapter_id": "synthetic_mock_v1",
                "model_family": "mock",
                "backbone": "mock_expert",
                "checkpoint_path": str(existing_checkpoint),
                "config_path": "",
                "data_root": str(data_root),
                "class_to_idx_path": str(class_mapping),
                "num_classes": 2,
                "input_size": 224,
                "batch_size": 4,
                "device": "cpu",
                "precision": "fp32",
                "enabled": "true",
                "run_adapter": "true",
                "notes": "可运行 mock expert",
            },
            {
                "job_id": "missing_checkpoint_job",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "artifact_id": "missing_model",
                "role_candidates": "scout",
                "adapter_id": "synthetic_mock_v1",
                "model_family": "mock",
                "backbone": "mock_missing",
                "checkpoint_path": str(tmp_path / "missing.pth"),
                "config_path": "",
                "data_root": str(data_root),
                "class_to_idx_path": str(class_mapping),
                "num_classes": 2,
                "input_size": 224,
                "batch_size": 4,
                "device": "cpu",
                "precision": "fp32",
                "enabled": "true",
                "run_adapter": "true",
                "notes": "缺 checkpoint",
            },
            {
                "job_id": "pending_timm_job",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "artifact_id": "pending_timm_model",
                "role_candidates": "scout",
                "adapter_id": "synthetic_mock_v1",
                "model_family": "mock",
                "backbone": "mock_scout",
                "checkpoint_path": str(existing_checkpoint),
                "config_path": "",
                "data_root": "",
                "class_to_idx_path": str(class_mapping),
                "num_classes": 2,
                "input_size": 224,
                "batch_size": 4,
                "device": "cpu",
                "precision": "fp32",
                "enabled": "true",
                "run_adapter": "false",
                "notes": "checkpoint 与 adapter 已确认，但 data_root 待确认",
            },
            {
                "job_id": "needs_loader_job",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "artifact_id": "retfound_like",
                "role_candidates": "expert",
                "adapter_id": "retfound_adapter_v1",
                "model_family": "retfound",
                "backbone": "retfound_dinov2",
                "checkpoint_path": str(existing_checkpoint),
                "config_path": "",
                "data_root": str(data_root),
                "class_to_idx_path": str(class_mapping),
                "num_classes": 2,
                "input_size": 224,
                "batch_size": 4,
                "device": "cpu",
                "precision": "fp32",
                "enabled": "true",
                "run_adapter": "true",
                "notes": "loader 待审计",
            },
        ],
    )
    write_csv(
        configs / "routing_replay_protocols.csv",
        [
            "replay_id",
            "task_id",
            "dataset_id",
            "disease_family",
            "routing_type",
            "scout_artifact_ids",
            "expert_artifact_id",
            "policies",
            "budgets",
            "prediction_source_mode",
            "enabled",
            "notes",
        ],
        [
            {
                "replay_id": "single_mock",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "routing_type": "single_scout",
                "scout_artifact_ids": "mock_scout",
                "expert_artifact_id": "mock_expert",
                "policies": "low_confidence|low_margin|high_entropy",
                "budgets": "0.5",
                "prediction_source_mode": "adapter_generated",
                "enabled": "true",
                "notes": "单 scout replay",
            },
            {
                "replay_id": "multi_mock",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "routing_type": "multi_scout",
                "scout_artifact_ids": "mock_scout|mock_scout_b",
                "expert_artifact_id": "mock_expert",
                "policies": "mean_uncertainty|max_uncertainty|disagreement_then_uncertainty",
                "budgets": "0.5",
                "prediction_source_mode": "adapter_generated",
                "enabled": "true",
                "notes": "多 scout replay",
            },
            {
                "replay_id": "missing_predictions",
                "task_id": "mock_task",
                "dataset_id": "mock_dataset",
                "disease_family": "mock_disease",
                "routing_type": "single_scout",
                "scout_artifact_ids": "missing_model",
                "expert_artifact_id": "mock_expert",
                "policies": "low_confidence",
                "budgets": "0.5",
                "prediction_source_mode": "adapter_generated",
                "enabled": "true",
                "notes": "缺 predictions",
            },
        ],
    )
    protocol = configs / "protocol.yaml"
    protocol.write_text(
        "\n".join(
            [
                "protocol_id: fixture_v085b",
                f"inventory_sources: {configs / 'inventory_sources.csv'}",
                f"adapter_registry: {configs / 'adapter_registry.csv'}",
                f"onboarding_jobs: {configs / 'onboarding_jobs.csv'}",
                f"routing_replay_protocols: {configs / 'routing_replay_protocols.csv'}",
                f"output_dir: {outputs}",
                "report: summary.html",
            ]
        ),
        encoding="utf-8",
    )
    return protocol


def test_inventory_stage_records_missing_and_legacy_status(tmp_path: Path):
    protocol = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"

    run_protocol(protocol, output_dir=output_dir, stage="inventory")

    inventory = read_csv(output_dir / "model_inventory.csv")
    assert {row["artifact_id"] for row in inventory} >= {
        "legacy_retinal_model",
        "mock_scout",
        "missing_model",
        "retfound_like",
    }
    missing = next(row for row in inventory if row["artifact_id"] == "missing_model")
    assert missing["checkpoint_status"] == "missing"
    assert missing["can_onboard"] == "false"
    assert missing["onboarding_status"] == "missing_checkpoint"
    legacy = next(row for row in inventory if row["artifact_id"] == "legacy_retinal_model")
    assert legacy["legacy_artifact_available"] == "true"
    assert legacy["can_onboard"] == "false"
    assert legacy["onboarding_status"] == "legacy_replay_only"
    loader = next(row for row in inventory if row["artifact_id"] == "retfound_like")
    assert loader["adapter_status"] == "needs_loader_audit"
    assert loader["can_onboard"] == "false"
    pending = next(
        row
        for row in inventory
        if row["artifact_id"] == "pending_timm_model" and row["task_id"] == "mock_task"
    )
    assert pending["checkpoint_status"] == "found"
    assert pending["adapter_status"] == "available"
    assert pending["legacy_artifact_available"] == "true"
    assert pending["can_onboard"] == "false"
    assert pending["onboarding_status"] == "incomplete_metadata"
    assert pending["missing_reason"] == "data_root_or_input_csv_required"
    assert not any(
        row["artifact_id"] == "pending_timm_model" and row["task_id"] == "mixed"
        for row in inventory
    )
    assert any(
        row["artifact_id"] == "mixed_only_model" and row["task_id"] == "mixed"
        for row in inventory
    )

    index_rows = read_csv(output_dir / "artifact_source_index.csv")
    assert any(row["file_type"] == "model_baselines" and row["exists"] == "true" for row in index_rows)
    assert any(
        row["source_id"] == "v085_registry_outputs"
        and row["file_type"] == "model_baselines"
        and row["file_path"].endswith("model_baselines_all.csv")
        and row["exists"] == "true"
        for row in index_rows
    )
    assert any(
        row["source_id"] == "v085_registry_outputs"
        and row["file_type"] == "routing_results"
        and row["file_path"].endswith("routing_results_all.csv")
        and row["exists"] == "true"
        for row in index_rows
    )
    assert any(
        row["source_id"] == "v085_registry_outputs"
        and row["file_type"] == "registered_models"
        and row["exists"] == "true"
        for row in index_rows
    )
    assert any(
        row["source_id"] == "v085_registry_outputs"
        and row["file_type"] == "summary_html"
        and row["exists"] == "true"
        and row["n_rows"] == ""
        for row in index_rows
    )
    assert any(row["source_id"] == "missing_source" and row["exists"] == "false" for row in index_rows)
    summary = read_csv(output_dir / "inventory_summary.csv")
    keys = {row["metric"]: row["value"] for row in summary}
    assert int(keys["total_candidates"]) >= 5
    assert int(keys["missing_checkpoint_n"]) >= 1


def test_all_stage_generates_adapter_outputs_replay_and_sanity(tmp_path: Path):
    protocol = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"

    run_protocol(protocol, output_dir=output_dir, stage="all")

    for name in [
        "adapter_job_summary.csv",
        "onboarded_models.csv",
        "model_baselines_from_adapters.csv",
        "forward_cost_summary_from_adapters.csv",
        "adapter_manifest.csv",
        "single_scout_routing_results_from_adapters.csv",
        "multi_scout_routing_results_from_adapters.csv",
        "routing_replay_summary.csv",
        "adapter_vs_legacy_baseline_check.csv",
        "adapter_vs_legacy_routing_check.csv",
        "summary.html",
    ]:
        assert (output_dir / name).exists(), name

    job_summary = read_csv(output_dir / "adapter_job_summary.csv")
    statuses = {row["job_id"]: row["status"] for row in job_summary}
    assert statuses["mock_scout_job"] == "completed"
    assert statuses["missing_checkpoint_job"] == "skipped_missing_checkpoint"
    assert statuses["needs_loader_job"] == "skipped_needs_loader_audit"

    predictions = read_csv(output_dir / "onboarded_models" / "mock_scout_job" / "predictions.csv")
    assert {"prob_0", "prob_1", "confidence", "margin", "entropy"} <= set(predictions[0])
    assert predictions[0]["source"] == "adapter_generated"

    replay_summary = read_csv(output_dir / "routing_replay_summary.csv")
    assert any(row["replay_id"] == "single_mock" and row["status"] == "completed" for row in replay_summary)
    assert any(row["replay_id"] == "multi_mock" and row["status"] == "completed" for row in replay_summary)
    assert any(
        row["replay_id"] == "missing_predictions"
        and row["status"] == "skipped_missing_predictions"
        for row in replay_summary
    )

    html = (output_dir / "summary.html").read_text(encoding="utf-8")
    assert "v0.8.5b 已知模型清单与适配器接入摘要" in html
    assert "不训练" in html
    assert "不伪造 checkpoint" in html
    assert "forward-only cost" in html
    assert "真实部署端到端延迟" in html
    assert "strict reproduction" not in html


def test_dry_run_writes_no_outputs(tmp_path: Path):
    protocol = create_fixture(tmp_path)
    output_dir = tmp_path / "dry_outputs"

    run_protocol(protocol, output_dir=output_dir, stage="all", dry_run=True)

    assert not output_dir.exists()


def test_readme_documents_boundaries():
    readme = (
        Path(__file__).resolve().parents[1]
        / "experiments/v0_8_5b_known_model_inventory_adapter_onboarding/README.md"
    )
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "v0.8.5：已有产物注册" in text
    assert "v0.8.5b：已知模型 inventory" in text
    assert "adapter 负责 checkpoint" in text
    assert "routing replay 负责 predictions" in text
    assert "legacy_replay_only" in text
    assert "RETFound-DINOv2" in text
    assert "sanity comparison" in text
    assert "strict reproduction" in text
    assert "不训练" in text
    assert "不微调" in text
    assert "不做 UI" in text
    assert "不做 Agent" in text
