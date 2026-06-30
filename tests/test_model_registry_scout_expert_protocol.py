from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.routing.run_model_registry_scout_expert_protocol import (
    RegistryProtocolError,
    run_protocol,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def create_source_output(
    root: Path,
    *,
    protocol_id: str,
    task_id: str,
    scout: str,
    expert: str,
    risk_columns: bool,
    empty_risk_results: bool = False,
    write_risk_results: bool = True,
) -> Path:
    source = root / task_id
    write_csv(
        source / "model_baselines.csv",
        [
            "protocol_id",
            "task_id",
            "artifact_id",
            "role",
            "accuracy",
            "macro_f1",
            "cost_status",
            "estimated_forward_ms_per_image",
        ],
        [
            {
                "protocol_id": protocol_id,
                "task_id": task_id,
                "artifact_id": scout,
                "role": "scout",
                "accuracy": 0.8,
                "macro_f1": 0.7,
                "cost_status": "measured",
                "estimated_forward_ms_per_image": 0.5,
            },
            {
                "protocol_id": protocol_id,
                "task_id": task_id,
                "artifact_id": expert,
                "role": "expert",
                "accuracy": 0.9,
                "macro_f1": 0.8,
                "cost_status": "measured",
                "estimated_forward_ms_per_image": 4.0,
            },
        ],
    )
    routing_fields = [
        "protocol_id",
        "task_id",
        "method_kind",
        "protocol_name",
        "scout_artifact",
        "expert_artifact",
        "budget",
        "policy",
        "non_deployable",
        "accuracy",
        "macro_f1",
        "cost_status",
        "estimated_forward_ms_per_image",
    ]
    routing_row = {
        "protocol_id": protocol_id,
        "task_id": task_id,
        "method_kind": "uncertainty",
        "protocol_name": f"{scout}_to_{expert}",
        "scout_artifact": scout,
        "expert_artifact": expert,
        "budget": 0.3,
        "policy": "low_confidence",
        "non_deployable": "False",
        "accuracy": 0.85,
        "macro_f1": 0.82,
        "cost_status": "estimated_from_measured_models",
        "estimated_forward_ms_per_image": 1.7,
    }
    if risk_columns:
        routing_fields.extend(
            [
                "large_undergrading_event_recall_fixed_pool",
                "severe_pdr_miss_event_recall_fixed_pool",
            ]
        )
        routing_row["large_undergrading_event_recall_fixed_pool"] = 0.75
        routing_row["severe_pdr_miss_event_recall_fixed_pool"] = 0.6
    write_csv(source / "routing_results.csv", routing_fields, [routing_row])
    if not write_risk_results:
        pass
    elif empty_risk_results:
        write_csv(source / "risk_results.csv", ["protocol_id", "task_id"], [])
    elif risk_columns:
        write_csv(
            source / "risk_results.csv",
            ["protocol_id", "task_id", "risk_event", "event_recall"],
            [
                {
                    "protocol_id": protocol_id,
                    "task_id": task_id,
                    "risk_event": "large_undergrading",
                    "event_recall": 0.75,
                }
            ],
        )
    write_csv(
        source / "case_audit.csv",
        [
            "protocol_id",
            "task_id",
            "image_key",
            "true_label",
            "selected_for_expert",
            "cost_status",
        ],
        [
            {
                "protocol_id": protocol_id,
                "task_id": task_id,
                "image_key": "case_001",
                "true_label": 1,
                "selected_for_expert": "True",
                "cost_status": "missing",
            }
        ],
    )
    return source


def create_protocol_fixture(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    outputs = tmp_path / "outputs"
    sources = tmp_path / "sources"
    glaucoma_source = create_source_output(
        sources,
        protocol_id="glaucoma_source_protocol",
        task_id="glaucoma_3class",
        scout="convnext_tiny_glaucoma_scout",
        expert="retfound_dinov2_glaucoma_expert",
        risk_columns=False,
        empty_risk_results=True,
    )
    dr_source = create_source_output(
        sources,
        protocol_id="dr_source_protocol",
        task_id="aptos_dr_5class",
        scout="convnext_tiny",
        expert="retfound_mae_cfp_official_protocol",
        risk_columns=True,
        write_risk_results=False,
    )

    write_csv(
        configs / "task_registry.csv",
        [
            "task_id",
            "disease_family",
            "dataset_id",
            "label_space",
            "num_classes",
            "risk_mode",
            "source_schema",
            "risk_source_mode",
            "source_output_dir",
            "enabled",
            "notes",
        ],
        [
            {
                "task_id": "glaucoma_3class",
                "disease_family": "glaucoma",
                "dataset_id": "Glaucoma_fundus",
                "label_space": "glaucoma_normal_early_advanced",
                "num_classes": 3,
                "risk_mode": "none",
                "source_schema": "v084b_task_agnostic",
                "risk_source_mode": "empty_schema",
                "source_output_dir": str(glaucoma_source),
                "enabled": "true",
                "notes": "青光眼 generic multiclass 示例",
            },
            {
                "task_id": "aptos_dr_5class",
                "disease_family": "diabetic_retinopathy",
                "dataset_id": "APTOS2019",
                "label_space": "dr_icdr_0_4",
                "num_classes": 5,
                "risk_mode": "dr_risk_events",
                "source_schema": "legacy_v083_controlled_replay",
                "risk_source_mode": "embedded_in_routing",
                "source_output_dir": str(dr_source),
                "enabled": "true",
                "notes": "DR 风险事件示例",
            },
        ],
    )
    write_csv(
        configs / "model_registry.csv",
        [
            "artifact_id",
            "task_id",
            "model_family",
            "role_candidates",
            "prediction_source",
            "baseline_source",
            "cost_profile_id",
            "cost_source",
            "source_version",
            "enabled",
            "notes",
        ],
        [
            {
                "artifact_id": "convnext_tiny_glaucoma_scout",
                "task_id": "glaucoma_3class",
                "model_family": "convnext",
                "role_candidates": "scout",
                "prediction_source": "",
                "baseline_source": str(glaucoma_source / "model_baselines.csv"),
                "cost_profile_id": "profile",
                "cost_source": str(glaucoma_source / "model_baselines.csv"),
                "source_version": "fixture",
                "enabled": "true",
                "notes": "scout only",
            },
            {
                "artifact_id": "retfound_dinov2_glaucoma_expert",
                "task_id": "glaucoma_3class",
                "model_family": "retfound_dinov2",
                "role_candidates": "expert",
                "prediction_source": "",
                "baseline_source": str(glaucoma_source / "model_baselines.csv"),
                "cost_profile_id": "profile",
                "cost_source": str(glaucoma_source / "model_baselines.csv"),
                "source_version": "fixture",
                "enabled": "true",
                "notes": "expert only",
            },
            {
                "artifact_id": "convnext_tiny",
                "task_id": "aptos_dr_5class",
                "model_family": "convnext",
                "role_candidates": "scout|expert",
                "prediction_source": "",
                "baseline_source": str(dr_source / "model_baselines.csv"),
                "cost_profile_id": "profile",
                "cost_source": str(dr_source / "model_baselines.csv"),
                "source_version": "fixture",
                "enabled": "true",
                "notes": "role can be both",
            },
            {
                "artifact_id": "retfound_mae_cfp_official_protocol",
                "task_id": "aptos_dr_5class",
                "model_family": "retfound",
                "role_candidates": "expert",
                "prediction_source": "",
                "baseline_source": str(dr_source / "model_baselines.csv"),
                "cost_profile_id": "profile",
                "cost_source": str(dr_source / "model_baselines.csv"),
                "source_version": "fixture",
                "enabled": "true",
                "notes": "DR expert",
            },
        ],
    )
    write_csv(
        configs / "route_protocols.csv",
        [
            "protocol_id",
            "task_id",
            "scout_artifact_id",
            "expert_artifact_id",
            "routing_policies",
            "budgets",
            "enabled",
            "notes",
        ],
        [
            {
                "protocol_id": "glaucoma_convnext_to_retfound",
                "task_id": "glaucoma_3class",
                "scout_artifact_id": "convnext_tiny_glaucoma_scout",
                "expert_artifact_id": "retfound_dinov2_glaucoma_expert",
                "routing_policies": "low_confidence",
                "budgets": "0.3",
                "enabled": "true",
                "notes": "青光眼示例",
            },
            {
                "protocol_id": "dr_convnext_to_retfound",
                "task_id": "aptos_dr_5class",
                "scout_artifact_id": "convnext_tiny",
                "expert_artifact_id": "retfound_mae_cfp_official_protocol",
                "routing_policies": "low_confidence",
                "budgets": "0.3",
                "enabled": "true",
                "notes": "DR 示例",
            },
        ],
    )
    write_csv(
        configs / "cost_registry.csv",
        [
            "artifact_id",
            "cost_profile_id",
            "cost_scope",
            "cost_source",
            "device",
            "precision",
            "batch_size",
            "enabled",
            "notes",
        ],
        [
            {
                "artifact_id": "convnext_tiny_glaucoma_scout",
                "cost_profile_id": "profile",
                "cost_scope": "forward_only",
                "cost_source": str(glaucoma_source / "model_baselines.csv"),
                "device": "fixture",
                "precision": "fp32",
                "batch_size": 8,
                "enabled": "true",
                "notes": "仅前向传播计算成本",
            }
        ],
    )
    protocol = configs / "protocol.yaml"
    protocol.write_text(
        "\n".join(
            [
                "protocol_id: fixture_v085",
                f"task_registry: {configs / 'task_registry.csv'}",
                f"model_registry: {configs / 'model_registry.csv'}",
                f"route_protocols: {configs / 'route_protocols.csv'}",
                f"cost_registry: {configs / 'cost_registry.csv'}",
                f"output_dir: {outputs}",
                "report: summary.html",
            ]
        ),
        encoding="utf-8",
    )
    return protocol


def test_registry_protocol_aggregates_outputs_and_preserves_boundaries(tmp_path: Path):
    protocol = create_protocol_fixture(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_protocol(protocol, output_dir=output_dir, dry_run=False)

    assert result.output_dir == output_dir
    assert (output_dir / "registered_tasks.csv").exists()
    assert (output_dir / "registered_models.csv").exists()
    assert (output_dir / "route_protocol_summary.csv").exists()
    assert (output_dir / "model_baselines_all.csv").exists()
    assert (output_dir / "routing_results_all.csv").exists()
    assert (output_dir / "risk_results_all.csv").exists()
    assert (output_dir / "case_audit_all.csv").exists()
    assert (output_dir / "artifact_manifest.csv").exists()
    assert (output_dir / "summary.html").exists()

    tasks = read_csv(output_dir / "registered_tasks.csv")
    assert {row["task_id"] for row in tasks} == {"glaucoma_3class", "aptos_dr_5class"}

    routing_rows = read_csv(output_dir / "routing_results_all.csv")
    assert {row["task_id"] for row in routing_rows} == {
        "glaucoma_3class",
        "aptos_dr_5class",
    }
    glaucoma_rows = [row for row in routing_rows if row["task_id"] == "glaucoma_3class"]
    assert all(row["cost_status"] == "estimated_from_measured_models" for row in glaucoma_rows)
    assert all(
        row.get("large_undergrading_event_recall_fixed_pool", "") == ""
        for row in glaucoma_rows
    )

    summary = read_csv(output_dir / "route_protocol_summary.csv")
    glaucoma_summary = next(
        row for row in summary if row["protocol_id"] == "glaucoma_convnext_to_retfound"
    )
    assert glaucoma_summary["best_non_oracle_policy_by_macro_f1"] == "low_confidence"
    assert glaucoma_summary["best_non_oracle_cost_status"] == "estimated_from_measured_models"
    assert glaucoma_summary["has_risk_events"] == "false"
    assert glaucoma_summary["risk_source_mode"] == "empty_schema"
    dr_summary = next(row for row in summary if row["protocol_id"] == "dr_convnext_to_retfound")
    assert dr_summary["has_risk_events"] == "true"
    assert dr_summary["risk_source_mode"] == "embedded_in_routing"

    risk_rows = read_csv(output_dir / "risk_results_all.csv")
    dr_risk_rows = [row for row in risk_rows if row["task_id"] == "aptos_dr_5class"]
    assert dr_risk_rows
    assert all(row["risk_source_mode"] == "embedded_in_routing" for row in dr_risk_rows)
    assert all(row["source_schema"] == "legacy_v083_controlled_replay" for row in dr_risk_rows)
    assert "large_undergrading_event_recall_fixed_pool" in dr_risk_rows[0]

    html = (output_dir / "summary.html").read_text(encoding="utf-8")
    assert "注册表级接入" in html
    assert "forward-only cost" in html
    assert "真实部署端到端延迟" in html


def test_dry_run_validates_without_writing_outputs(tmp_path: Path):
    protocol = create_protocol_fixture(tmp_path)
    output_dir = tmp_path / "dry_outputs"

    run_protocol(protocol, output_dir=output_dir, dry_run=True)

    assert not output_dir.exists()


def test_route_protocol_rejects_unknown_model(tmp_path: Path):
    protocol = create_protocol_fixture(tmp_path)
    route_path = tmp_path / "configs" / "route_protocols.csv"
    rows = read_csv(route_path)
    rows[0]["expert_artifact_id"] = "missing_expert"
    write_csv(route_path, list(rows[0].keys()), rows)

    with pytest.raises(RegistryProtocolError, match="missing_expert"):
        run_protocol(protocol, output_dir=tmp_path / "outputs", dry_run=True)


def test_route_protocol_rejects_unsupported_role(tmp_path: Path):
    protocol = create_protocol_fixture(tmp_path)
    route_path = tmp_path / "configs" / "route_protocols.csv"
    rows = read_csv(route_path)
    rows[0]["scout_artifact_id"] = "retfound_dinov2_glaucoma_expert"
    rows[0]["expert_artifact_id"] = "convnext_tiny_glaucoma_scout"
    write_csv(route_path, list(rows[0].keys()), rows)

    with pytest.raises(RegistryProtocolError, match="不支持.*scout"):
        run_protocol(protocol, output_dir=tmp_path / "outputs", dry_run=True)


def test_readme_documents_plugin_level_boundaries():
    readme = (
        Path(__file__).resolve().parents[1]
        / "experiments/v0_8_5_model_registry_scout_expert_protocol/README.md"
    )
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "registry-level" in text
    assert "adapter-level" in text
    assert "training-level" in text
    assert "forward-only cost" in text
    assert "真实部署端到端延迟" in text
