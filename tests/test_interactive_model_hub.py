from __future__ import annotations

# ruff: noqa: E402 - project imports follow the explicit repository path bootstrap below.

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.routing.run_interactive_model_hub import (
    HUB_COLUMNS,
    PairingContext,
    PairingSkip,
    cost_summary,
    normalize_and_validate_prediction,
    risk_proxy_summary,
    run_protocol,
)
from scripts.routing.run_controlled_protocol import load_config
from scripts.routing.model_metadata import normalized_model_metadata
from app.model_hub_data import (
    available_routing_policies,
    build_global_model_catalog,
    enrich_cost_curve,
    derive_budget_preview,
    dr_risk_summary,
    evaluate_exploratory_composition,
    estimate_global_composition_count,
    load_model_hub_outputs,
    load_registered_training_models,
    _load_model_prediction,
    scan_global_composition_candidates,
    select_operating_points,
    split_task_models,
    task_metric_profile,
    task_evaluation_summary,
)
import app.model_hub_data as model_hub_data_module
from app.model_hub_scan_jobs import GlobalScanRequest, run_global_scan_request
import app.model_hub_engineering as engineering_module
from app.model_hub_engineering import (
    current_task_candidate_catalog,
    filter_global_model_catalog,
    _job_record_title,
    official_profile_for_model,
    training_capability,
)
from app.model_hub_research import (
    _comparison_plot_frame,
    build_comparison_table,
    build_proxy_event_table_html,
)
from app.model_hub_ui import human_model, source_status


def test_official_profile_selection_is_compact_and_architecture_specific() -> None:
    convnext = official_profile_for_model("convnext", "convnext_tiny")
    swin = official_profile_for_model("swin", "swin_tiny_patch4_window7_224")
    unknown = official_profile_for_model("vit", "vit_huge_patch14_224")

    assert convnext is not None
    assert convnext["profile"]["profile_id"] == "convnext_tiny_official_anchor"
    assert convnext["profile"]["display_name"] == "ConvNeXt 官方仓库迁移锚点（Tiny 适配）"
    assert convnext["search_plan"]["display_name"] == "固定预算 6 次 LR×WD 验证集搜索"
    assert len(convnext["trials"]) == 6
    assert swin is not None
    assert swin["profile"]["profile_id"] == "swin_tiny_official_anchor"
    assert unknown is None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(
    artifact_id: str,
    *,
    predictions: list[int],
    true_labels: list[int] | None = None,
) -> list[dict[str, object]]:
    true_labels = true_labels or [0, 1, 2, 0]
    rows: list[dict[str, object]] = []
    for index, (truth, prediction) in enumerate(zip(true_labels, predictions)):
        probabilities = np.full(3, 0.1, dtype=float)
        probabilities[prediction] = 0.8
        ordered = np.sort(probabilities)
        rows.append(
            {
                "task_id": "task_a",
                "artifact_id": artifact_id,
                "image_key": f"case_{index}",
                "image_path": f"/fixture/case_{index}.png",
                "true_label": truth,
                "pred_label": prediction,
                "confidence": ordered[-1],
                "margin": ordered[-1] - ordered[-2],
                "entropy": 0.4 + index * 0.01,
                "source": "fixture",
                "prob_0": probabilities[0],
                "prob_1": probabilities[1],
                "prob_2": probabilities[2],
            }
        )
    return rows


def create_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    adapter_dir = source / "v085c"
    onboarded = adapter_dir / "onboarded_models"

    scout_a_path = onboarded / "scout_a_job" / "predictions.csv"
    scout_b_path = onboarded / "scout_b_job" / "predictions.csv"
    expert_a_path = source / "expert_a.csv"
    expert_b_path = source / "expert_b.csv"
    write_csv(scout_a_path, prediction_rows("scout_a", predictions=[0, 2, 2, 0]))
    write_csv(scout_b_path, prediction_rows("scout_b", predictions=[0, 1, 1, 0]))
    write_csv(expert_a_path, prediction_rows("expert_a", predictions=[0, 1, 2, 0]))
    expert_b_rows = prediction_rows("expert_b", predictions=[0, 1, 2, 0])
    for row in expert_b_rows:
        row["task_id"] = "task_b"
    write_csv(expert_b_path, expert_b_rows)

    write_csv(
        adapter_dir / "adapter_job_summary.csv",
        [
            {
                "job_id": "scout_a_job",
                "task_id": "task_a",
                "artifact_id": "scout_a",
                "adapter_id": "timm_classifier_v1",
                "status": "completed",
                "n_images": 4,
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "qwk": "",
                "estimated_forward_ms_per_image": 1.0,
                "cost_scope": "forward_only",
                "predictions_path": str(scout_a_path),
                "outputs_dir": str(scout_a_path.parent),
                "notes": "fixture",
            },
            {
                "job_id": "scout_b_job",
                "task_id": "task_a",
                "artifact_id": "scout_b",
                "adapter_id": "timm_classifier_v1",
                "status": "completed",
                "n_images": 4,
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "qwk": "",
                "estimated_forward_ms_per_image": 2.0,
                "cost_scope": "forward_only",
                "predictions_path": str(scout_b_path),
                "outputs_dir": str(scout_b_path.parent),
                "notes": "fixture",
            },
        ],
    )
    write_csv(
        adapter_dir / "onboarded_models.csv",
        [
            {
                "job_id": "scout_a_job",
                "task_id": "task_a",
                "artifact_id": "scout_a",
                "predictions_path": str(scout_a_path),
                "baseline_path": "",
                "cost_path": "",
                "manifest_path": "",
            },
            {
                "job_id": "scout_b_job",
                "task_id": "task_a",
                "artifact_id": "scout_b",
                "predictions_path": str(scout_b_path),
                "baseline_path": "",
                "cost_path": "",
                "manifest_path": "",
            },
        ],
    )
    write_csv(
        adapter_dir / "model_baselines_from_adapters.csv",
        [
            {"job_id": "scout_a_job", "task_id": "task_a", "artifact_id": "scout_a", "split": "test", "n_images": 4, "accuracy": 0.75, "macro_f1": 0.7, "qwk": "", "qwk_status": "not_applicable"},
            {"job_id": "scout_b_job", "task_id": "task_a", "artifact_id": "scout_b", "split": "test", "n_images": 4, "accuracy": 0.75, "macro_f1": 0.7, "qwk": "", "qwk_status": "not_applicable"},
        ],
    )
    write_csv(
        adapter_dir / "forward_cost_summary_from_adapters.csv",
        [
            {"job_id": "scout_a_job", "task_id": "task_a", "artifact_id": "scout_a", "cost_scope": "forward_only", "median_ms_per_image": 1.0, "mean_ms_per_image": 1.0},
            {"job_id": "scout_b_job", "task_id": "task_a", "artifact_id": "scout_b", "cost_scope": "forward_only", "median_ms_per_image": 2.0, "mean_ms_per_image": 2.0},
        ],
    )

    replay_protocols = source / "v085c_replays.csv"
    write_csv(
        replay_protocols,
        [
            {"replay_id": "a", "task_id": "task_a", "scout_job_id": "scout_a_job", "expert_artifact_id": "expert_a", "expert_legacy_prediction_path": str(expert_a_path), "policies": "low_confidence", "budgets": "0.5", "prediction_source_mode": "mixed_adapter_legacy", "enabled": "true"},
            {"replay_id": "b", "task_id": "task_b", "scout_job_id": "", "expert_artifact_id": "expert_b", "expert_legacy_prediction_path": str(expert_b_path), "policies": "low_confidence", "budgets": "0.5", "prediction_source_mode": "legacy", "enabled": "true"},
        ],
    )

    tasks = source / "task_registry.csv"
    write_csv(
        tasks,
        [
            {
                "task_id": "task_a",
                "disease_family": "fixture_a",
                "dataset_id": "dataset_a",
                "dataset_display_name": "Fixture Dataset A",
                "dataset_source": "Fixture Archive",
                "dataset_url": "https://example.test/dataset-a",
                "provenance_status": "verified_public",
                "label_space": "classes_0_2",
                "num_classes": 3,
                "enabled": "true",
            },
            {
                "task_id": "task_b",
                "disease_family": "fixture_b",
                "dataset_id": "dataset_b",
                "dataset_display_name": "Fixture Dataset B",
                "dataset_source": "Fixture Archive",
                "dataset_url": "https://example.test/dataset-b",
                "provenance_status": "verified_public",
                "label_space": "classes_0_2_b",
                "num_classes": 3,
                "enabled": "true",
            },
        ],
    )
    registered = source / "registered_models.csv"
    write_csv(
        registered,
        [
            {"artifact_id": "scout_a", "task_id": "task_a", "model_family": "mock", "role_candidates": "scout", "prediction_source": "", "baseline_source": "", "cost_profile_id": "", "cost_source": "", "source_version": "v0.8.5c", "pretraining_source": "custom_registered_pretraining", "enabled": "true", "notes": ""},
            {"artifact_id": "scout_b", "task_id": "task_a", "model_family": "mock", "role_candidates": "scout", "prediction_source": "", "baseline_source": "", "cost_profile_id": "", "cost_source": "", "source_version": "v0.8.5c", "enabled": "true", "notes": ""},
            {"artifact_id": "expert_a", "task_id": "task_a", "model_family": "mock_expert", "role_candidates": "expert", "prediction_source": "", "baseline_source": "", "cost_profile_id": "", "cost_source": "", "source_version": "legacy", "enabled": "true", "notes": ""},
            {"artifact_id": "expert_b", "task_id": "task_b", "model_family": "mock_expert", "role_candidates": "expert", "prediction_source": "", "baseline_source": "", "cost_profile_id": "", "cost_source": "", "source_version": "legacy", "enabled": "true", "notes": ""},
        ],
    )
    legacy_baselines = source / "legacy_baselines.csv"
    write_csv(
        legacy_baselines,
        [
            {"task_id": "task_a", "artifact_id": "expert_a", "name": "expert_a", "accuracy": 1.0, "macro_f1": 1.0, "qwk": "", "estimated_forward_ms_per_image": 5.0, "timing_scope": "forward_only"},
            {"task_id": "task_b", "artifact_id": "expert_b", "name": "expert_b", "accuracy": 1.0, "macro_f1": 1.0, "qwk": "", "estimated_forward_ms_per_image": 6.0, "timing_scope": "forward_only"},
        ],
    )

    pairings = source / "pairings.csv"
    write_csv(
        pairings,
        [
            {"pairing_id": "single_valid", "task_id": "task_a", "scout_artifact_ids": "scout_a", "primary_scout_artifact_id": "scout_a", "expert_artifact_id": "expert_a", "enabled": "true", "prediction_source_mode": "mixed_adapter_legacy", "routing_policies": "low_confidence|low_margin|high_entropy", "budget_grid": "0|0.2|0.3|0.5|1", "result_semantics": "interactive_replay", "notes": ""},
            {"pairing_id": "multi_valid", "task_id": "task_a", "scout_artifact_ids": "scout_a|scout_b", "primary_scout_artifact_id": "scout_a", "expert_artifact_id": "expert_a", "enabled": "true", "prediction_source_mode": "mixed_adapter_legacy", "routing_policies": "disagreement_then_uncertainty|mean_uncertainty", "budget_grid": "0|0.2|0.3|0.5|1", "result_semantics": "interactive_replay", "notes": ""},
            {"pairing_id": "cross_task_invalid", "task_id": "task_a", "scout_artifact_ids": "scout_a", "primary_scout_artifact_id": "scout_a", "expert_artifact_id": "expert_b", "enabled": "true", "prediction_source_mode": "mixed_adapter_legacy", "routing_policies": "low_confidence", "budget_grid": "0|1", "result_semantics": "interactive_replay", "notes": ""},
        ],
    )

    config = {
        "protocol_id": "fixture_v086",
        "result_semantics": "interactive_replay",
        "task_registry": str(tasks),
        "registered_models": str(registered),
        "legacy_model_baselines": str(legacy_baselines),
        "v085c_output_dir": str(adapter_dir),
        "v085c_replay_protocols": str(replay_protocols),
        "pairing_protocols": str(pairings),
        "case_trace_budgets": [0.2, 0.3, 0.5],
        "qwk_enabled_tasks": [],
        "source_versions": ["v0.8.5", "v0.8.5c"],
    }
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def test_interactive_model_hub_generates_controlled_outputs(tmp_path: Path) -> None:
    config_path = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_protocol(config_path, output_dir=output_dir, stage="all")

    assert result.output_dir == output_dir
    expected = {
        "model_hub_snapshot.csv",
        "pairing_results.csv",
        "case_routing_trace.csv",
        "run_config.yaml",
        "artifact_manifest.csv",
        "candidate_ranking.csv",
        "candidate_selection.csv",
        "report.html",
    }
    assert expected == {path.name for path in result.files}

    hub = pd.read_csv(output_dir / "model_hub_snapshot.csv")
    assert set(hub["artifact_id"]) == {"scout_a", "scout_b", "expert_a", "expert_b"}
    assert {"model_family", "architecture", "pretraining_source"}.issubset(hub.columns)
    assert hub.loc[hub["artifact_id"] == "scout_a", "prediction_source"].item() == "adapter"
    assert hub.loc[hub["artifact_id"] == "scout_a", "pretraining_source"].item() == "custom_registered_pretraining"
    expert = hub.loc[hub["artifact_id"] == "expert_a"].iloc[0]
    assert expert["prediction_source"] == "legacy"
    assert expert["adapter_status"] != "completed"
    assert expert["dataset_display_name"] == "Fixture Dataset A"
    assert expert["dataset_source"] == "Fixture Archive"
    assert expert["provenance_status"] == "verified_public"

    pairings = pd.read_csv(output_dir / "pairing_results.csv")
    completed = pairings.loc[pairings["status"] == "completed"]
    assert set(completed["requested_budget"]) == {0.0, 0.2, 0.3, 0.5, 1.0}
    assert {
        "requested_budget",
        "selected_n",
        "realized_budget",
        "estimated_total_compute_ms_per_image",
        "estimated_parallel_latency_ms_per_image",
    }.issubset(pairings.columns)
    assert {"routed", "random", "oracle", "scout_only", "full_expert"}.issubset(
        set(pairings["evaluation_kind"])
    )
    assert not pd.read_csv(output_dir / "candidate_ranking.csv").empty
    assert not pd.read_csv(output_dir / "candidate_selection.csv").empty

    single = completed.loc[completed["pairing_id"] == "single_valid"]
    assert single.loc[single["requested_budget"] == 0, "accuracy"].iloc[0] == 0.75
    assert single.loc[single["requested_budget"] == 1, "accuracy"].iloc[0] == 1.0

    multi = completed.loc[completed["pairing_id"] == "multi_valid"]
    assert (multi["primary_scout_artifact_id"] == "scout_a").all()
    assert (multi["scout_cost_sum_ms_per_image"] == 3.0).all()
    assert (multi["scout_parallel_scenario_ms_per_image"] == 2.0).all()
    assert (multi["parallel_cost_status"] == "scenario_estimate_not_measured").all()

    skipped = pairings.loc[pairings["pairing_id"] == "cross_task_invalid"]
    assert len(skipped) == 1
    assert skipped.iloc[0]["status"] == "skipped_incompatible_task"

    trace = pd.read_csv(output_dir / "case_routing_trace.csv")
    assert not trace.empty
    assert {0.2, 0.3, 0.5}.issubset(set(trace["requested_budget"]))
    assert set(trace["final_source"]) <= {"scout", "expert", "no_prediction"}

    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "交互式工程探索" in report
    assert "不作为正式科研结论" in report
    for path in result.files:
        assert "work/" not in path.read_text(encoding="utf-8-sig", errors="ignore")

    resumed = run_protocol(config_path, output_dir=output_dir, stage="all", resume=True)
    assert {path.name for path in resumed.files} == expected


def test_controlled_publish_uses_only_stable_manifest_paths() -> None:
    config = load_config(
        ROOT
        / "experiments"
        / "v0_8_6_interactive_model_hub"
        / "configs"
        / "controlled_runner.yaml"
    )

    assert config["publish"]["stable_manifest_paths"] is True


def test_dry_run_validates_without_writing(tmp_path: Path) -> None:
    config_path = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"

    result = run_protocol(config_path, output_dir=output_dir, stage="all", dry_run=True)

    assert result.files == []
    assert not output_dir.exists()


def test_prediction_case_exclusions_are_explicit_and_complete(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.csv"
    write_csv(
        prediction_path,
        prediction_rows("scout_a", predictions=[0, 1, 2, 0]),
    )

    filtered = normalize_and_validate_prediction(
        prediction_path,
        n_classes=3,
        excluded_image_keys={"case_1"},
    )

    assert filtered["image_key"].tolist() == ["case_0", "case_2", "case_3"]
    try:
        normalize_and_validate_prediction(
            prediction_path,
            n_classes=3,
            excluded_image_keys={"missing_case"},
        )
    except PairingSkip as exc:
        assert exc.status == "skipped_incompatible_image_keys"
    else:
        raise AssertionError("missing exclusion key must stop evaluation")


def test_label_proxy_threshold_supports_glaucoma_advanced_undergrading() -> None:
    summary = risk_proxy_summary(
        np.array([2, 2, 1, 0]),
        np.array([1, 2, 1, 0]),
        np.array([2, 1, 1, 0]),
        np.array([True, True, False, False]),
        undergrading_threshold=2,
    )

    assert summary["dangerous_total"] == 1
    assert summary["dangerous_corrected"] == 1
    assert summary["dangerous_introduced"] == 1
    assert summary["net_dangerous_reduction"] == 0


def test_partial_cpu_probe_cost_is_not_presented_as_complete_total() -> None:
    pairing = pd.Series({"task_id": "task_a"})
    context = PairingContext(
        pairing=pairing,
        scout_ids=["scout"],
        primary_scout_id="scout",
        expert_id="green_probe",
        scouts={},
        expert=pd.DataFrame(),
        n_overlap=4,
        overlap_rate=1.0,
    )
    hub = pd.DataFrame(
        [
            {"task_id": "task_a", "artifact_id": "scout", "forward_cost_ms_per_image": 1.0, "cpu_postprocess_status": "not_applicable"},
            {"task_id": "task_a", "artifact_id": "green_probe", "forward_cost_ms_per_image": 2.0, "cpu_postprocess_status": "unmeasured"},
        ]
    )

    summary = cost_summary(context, hub, call_rate=0.2)

    assert summary["cost_mode"] == "partial_component_cost_cpu_probe_unmeasured"
    assert pd.isna(summary["estimated_total_compute_ms_per_image"])
    assert summary["parallel_cost_status"] == "cpu_probe_unmeasured"


def test_ui_data_supports_five_percent_preview_without_publishing(tmp_path: Path) -> None:
    config_path = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"
    run_protocol(config_path, output_dir=output_dir, stage="all")
    data = load_model_hub_outputs(output_dir)

    metrics, cases = derive_budget_preview(
        data["traces"],
        data["pairings"],
        pairing_id="single_valid",
        policy="low_confidence",
        requested_budget=0.05,
    )

    assert metrics["preview_semantics"] == "exploratory_preview_not_published"
    assert metrics["selected_n"] == 0
    assert len(cases) == 4


def test_task_model_split_keeps_ready_models_visible_and_pending_models_separate() -> None:
    models = pd.DataFrame(
        [
            {"task_id": "task_a", "artifact_id": "ready", "prediction_source": "adapter", "compatibility_status": "ready_for_pairing"},
            {"task_id": "task_a", "artifact_id": "legacy", "prediction_source": "legacy", "compatibility_status": "ready_for_pairing"},
            {"task_id": "task_a", "artifact_id": "pending", "prediction_source": "missing", "compatibility_status": "incomplete"},
            {"task_id": "task_b", "artifact_id": "other", "prediction_source": "adapter", "compatibility_status": "ready_for_pairing"},
        ]
    )

    available, pending = split_task_models(models, "task_a")

    assert set(available["artifact_id"]) == {"ready", "legacy"}
    assert set(pending["artifact_id"]) == {"pending"}


def test_registered_training_models_are_loaded_from_unified_model_hub_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "training" / "glaucoma_3class" / "vit_adapter" / "20260702-120000"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_id": "glaucoma_3class::vit_adapter",
                "task_id": "glaucoma_3class",
                "dataset_id": "Glaucoma_fundus",
                "artifact_id": "vit_adapter",
                "model_family": "vit",
                "architecture": "vit_base_patch16_224",
                "label_space": "glaucoma_normal_early_advanced",
                "n_classes": 3,
                "prediction_source": "adapter",
                "prediction_path": str(run_dir / "predictions.csv"),
                "adapter_status": "completed",
                "compatibility_status": "ready_for_pairing",
            }
        ]
    ).to_csv(run_dir / "registration_record.csv", index=False)

    registered = load_registered_training_models(tmp_path)

    assert registered["artifact_id"].tolist() == ["vit_adapter"]
    assert registered.loc[0, "registration_file"].endswith("registration_record.csv")


def test_registered_timm_run_uses_canonical_single_underscore_artifact_id(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "runs"
        / "training"
        / "glaucoma_3class"
        / "swin_tiny__aptos_dr_5class__adapter__glaucoma_3class__adapter"
        / "20260706-164246"
    )
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_id": "glaucoma_3class::swin_tiny__aptos_dr_5class__adapter__glaucoma_3class__adapter",
                "task_id": "glaucoma_3class",
                "dataset_id": "Glaucoma_fundus",
                "artifact_id": "swin_tiny__aptos_dr_5class__adapter__glaucoma_3class__adapter",
                "model_family": "swin",
                "architecture": "swin_tiny_patch4_window7_224",
                "pretraining_source": "timm_pretrained",
                "label_space": "glaucoma_normal_early_advanced",
                "n_classes": 3,
                "prediction_source": "adapter",
                "prediction_path": str(run_dir / "predictions.csv"),
                "adapter_status": "completed",
                "compatibility_status": "ready_for_pairing",
            }
        ]
    ).to_csv(run_dir / "registration_record.csv", index=False)

    registered = load_registered_training_models(tmp_path)

    assert registered.loc[0, "artifact_id"] == "swin_tiny_imagenet_glaucoma_3class_adapter"
    assert registered.loc[0, "model_id"] == "glaucoma_3class::swin_tiny_imagenet_glaucoma_3class_adapter"
    assert registered.loc[0, "pretraining_source"] == "imagenet1k"


def test_canonical_adapted_model_names_have_readable_labels() -> None:
    assert human_model("swin_tiny_imagenet_glaucoma_3class_adapter") == "Swin-Tiny（ImageNet）· 青光眼三分类"
    assert human_model("vit_b_imagenet_aptos_dr_5class_adapter") == "ViT-B/16（ImageNet）· DR 五级分级"
    assert human_model("convnext_tiny_imagenet_glaucoma_3class_adapter") == "ConvNeXt-Tiny（ImageNet）· 青光眼三分类"


def test_checkpoint_generated_prediction_is_marked_as_verified_online_chain() -> None:
    row = pd.Series(
        {
            "prediction_source": "checkpoint_generated",
            "adapter_status": "completed",
            "task_inference_ready": True,
        }
    )

    assert source_status(row) == ("在线推理链已验证", "badge-live")

    row["task_inference_ready"] = False
    assert source_status(row) == ("暂无可用预测结果", "badge-wait")


def test_model_hub_snapshot_publishes_checkpoint_loading_metadata() -> None:
    assert "checkpoint_path" in HUB_COLUMNS
    assert "checkpoint_status" in HUB_COLUMNS


def test_dr_risk_summary_reports_capture_resolution_and_residual_events() -> None:
    detail = pd.DataFrame(
        {
            "true_label": [4, 3, 2, 0],
            "primary_scout_artifact_id": ["scout_a"] * 4,
            "primary_scout_pred_label": [1, 2, 1, 0],
            "final_pred_label": [4, 2, 2, 0],
            "is_reviewed_by_expert": [True, False, True, False],
        }
    )

    metrics, enriched = dr_risk_summary(detail, task_id="aptos_dr_5class")

    assert metrics["dr_large_undergrading_event_total"] == 1
    assert metrics["dr_large_undergrading_selected_n"] == 1
    assert metrics["dr_large_undergrading_residual_n"] == 0
    assert metrics["dr_referable_miss_event_total"] == 2
    assert metrics["dr_referable_miss_selected_n"] == 2
    assert metrics["dr_referable_miss_residual_n"] == 0
    assert metrics["dr_severe_pdr_miss_event_total"] == 2
    assert metrics["dr_severe_pdr_miss_residual_n"] == 1
    assert enriched.loc[0, "dr_large_undergrading_scout_event"]
    assert not enriched.loc[0, "dr_large_undergrading_final_residual"]
    assert metrics["risk_semantics"] == "label_based_safety_proxy_not_clinical_gold_standard"


def test_dr_risk_summary_is_not_applicable_without_a_scout() -> None:
    detail = pd.DataFrame(
        {
            "true_label": [4, 3],
            "primary_scout_artifact_id": ["", ""],
            "primary_scout_pred_label": [2, 2],
            "final_pred_label": [4, 3],
            "is_reviewed_by_expert": [True, True],
        }
    )

    metrics, enriched = dr_risk_summary(detail, task_id="aptos_dr_5class")

    assert metrics == {"risk_semantics": "not_applicable"}
    assert not any(column.startswith("dr_") for column in enriched.columns)


def test_exploratory_composition_supports_scout_only_expert_only_and_expert_pool(tmp_path: Path) -> None:
    config_path = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"
    run_protocol(config_path, output_dir=output_dir, stage="all")
    models = pd.read_csv(output_dir / "model_hub_snapshot.csv")

    scout_metrics, scout_cases = evaluate_exploratory_composition(
        models,
        task_id="task_a",
        scout_ids=["scout_a", "scout_b"],
        primary_scout_id="scout_a",
        expert_ids=[],
        policy="disagreement_then_uncertainty",
        requested_budget=0.5,
    )
    assert scout_metrics["composition_mode"] == "scout_only"
    assert scout_metrics["selected_n"] == 0
    assert set(scout_cases["final_source"]) == {"scout"}

    expert_metrics, expert_cases = evaluate_exploratory_composition(
        models,
        task_id="task_a",
        scout_ids=[],
        primary_scout_id=None,
        expert_ids=["expert_a"],
        policy="dense_expert",
        requested_budget=0.0,
    )
    assert expert_metrics["composition_mode"] == "expert_only"
    assert expert_metrics["selected_n"] == 4
    assert expert_metrics["risk_semantics"] == "not_applicable"
    assert set(expert_cases["final_source"]) == {"expert"}

    pooled_metrics, pooled_cases = evaluate_exploratory_composition(
        models,
        task_id="task_a",
        scout_ids=["scout_a"],
        primary_scout_id="scout_a",
        expert_ids=["expert_a", "scout_b"],
        policy="low_confidence",
        requested_budget=0.5,
    )
    assert pooled_metrics["composition_mode"] == "scout_to_expert_pool"
    assert pooled_metrics["n_expert"] == 2
    assert pooled_metrics["expert_aggregation"] == "mean_probability"
    assert pooled_cases["expert_pred_label"].notna().all()

    fixed_metrics, fixed_cases = evaluate_exploratory_composition(
        models,
        task_id="task_a",
        scout_ids=["scout_a"],
        primary_scout_id="scout_a",
        expert_ids=["expert_a", "scout_b"],
        expert_handoff_mode="fixed_expert",
        fixed_expert_id="expert_a",
        policy="low_confidence",
        requested_budget=0.5,
    )
    assert fixed_metrics["expert_handoff_mode"] == "fixed_expert"
    assert fixed_metrics["configured_expert_ids"] == "expert_a|scout_b"
    assert fixed_metrics["active_expert_ids"] == "expert_a"
    assert fixed_metrics["n_expert"] == 1
    assert fixed_metrics["estimated_total_compute_ms_per_image"] < pooled_metrics["estimated_total_compute_ms_per_image"]
    assert fixed_cases["expert_artifact_ids"].eq("expert_a").all()


def test_prediction_loader_repairs_duplicate_stem_keys_from_imagefolder_paths(tmp_path: Path) -> None:
    path = tmp_path / "glaucoma_predictions.csv"
    rows = prediction_rows("glaucoma_model", predictions=[0, 1], true_labels=[0, 1])
    rows[0]["image_key"] = "103"
    rows[0]["image_path"] = "/data/glaucoma/test/anormal_control/103.png"
    rows[1]["image_key"] = "103"
    rows[1]["image_path"] = "/data/glaucoma/test/early_glaucoma/103.png"
    write_csv(path, rows)

    frame = _load_model_prediction(
        pd.Series(
            {
                "artifact_id": "glaucoma_model",
                "prediction_path": str(path),
                "n_classes": 3,
            }
        )
    )

    assert frame["image_key"].tolist() == [
        "anormal_control/103.png",
        "early_glaucoma/103.png",
    ]


def test_prediction_loader_relocates_legacy_project_absolute_path(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "new_server" / "ophagent-medical-ai-agent"
    relative_path = Path("experiments/demo/outputs/predictions.csv")
    prediction_path = project_root / relative_path
    write_csv(
        prediction_path,
        prediction_rows("relocated_model", predictions=[0, 1, 2, 0]),
    )
    monkeypatch.setattr(model_hub_data_module, "PROJECT_ROOT", project_root)

    frame = _load_model_prediction(
        pd.Series(
            {
                "artifact_id": "relocated_model",
                "prediction_path": (
                    "/data/LRT/ophagent-medical-ai-agent/"
                    + relative_path.as_posix()
                ),
                "n_classes": 3,
            }
        )
    )

    assert len(frame) == 4
    assert frame["image_key"].tolist() == [
        "case_0",
        "case_1",
        "case_2",
        "case_3",
    ]


def test_global_composition_scan_covers_single_and_multi_route_candidates(tmp_path: Path) -> None:
    config_path = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"
    run_protocol(config_path, output_dir=output_dir, stage="all")
    models = load_model_hub_outputs(output_dir)["models"]

    scan = scan_global_composition_candidates(
        models,
        task_id="task_a",
        scout_ids=["scout_a", "scout_b"],
        expert_ids=["expert_a"],
        budgets=[0.0, 0.5],
        max_scouts=2,
        max_experts=1,
        primary_metric="accuracy",
    )

    assert not scan.empty
    assert {1, 2}.issubset(set(scan["n_scout"]))
    assert {0.0, 0.5}.issubset(set(scan["realized_budget"]))
    assert scan["is_pareto"].any()
    assert scan["global_rank_primary"].min() == 1
    assert scan["global_scan_semantics"].eq("exploratory_current_model_pool").all()


def test_global_scan_estimate_prevents_interactive_combination_explosion() -> None:
    assert estimate_global_composition_count(
        n_scouts=7,
        n_experts=6,
        max_scouts=3,
        max_experts=3,
        n_budgets=6,
    ) == 32718
    assert estimate_global_composition_count(
        n_scouts=7,
        n_experts=6,
        max_scouts=1,
        max_experts=1,
        n_budgets=6,
    ) == 756


def test_global_scan_background_request_writes_results_and_manifest(tmp_path: Path) -> None:
    config_path = create_fixture(tmp_path)
    output_dir = tmp_path / "outputs"
    run_protocol(config_path, output_dir=output_dir, stage="all")
    models = load_model_hub_outputs(output_dir)["models"]
    request = GlobalScanRequest(
        job_id="scan_fixture",
        task_id="task_a",
        scout_ids=["scout_a", "scout_b"],
        expert_ids=["expert_a"],
        budgets=[0.0, 0.5],
        max_scouts=2,
        max_experts=1,
        primary_metric="accuracy",
        top_n=5,
        output_dir=str(tmp_path / "scan_run"),
    )

    run_dir = run_global_scan_request(request, models)
    results = pd.read_csv(run_dir / "global_scan_results.csv")
    top = pd.read_csv(run_dir / "global_scan_top.csv")
    manifest = json.loads((run_dir / "scan_manifest.json").read_text(encoding="utf-8"))

    assert not results.empty
    assert not top.empty
    assert manifest["job_id"] == "scan_fixture"
    assert manifest["n_completed"] == int(results["scan_status"].eq("completed").sum())
    assert manifest["estimated_points"] == estimate_global_composition_count(
        n_scouts=2,
        n_experts=1,
        max_scouts=2,
        max_experts=1,
        n_budgets=2,
    )


def test_glaucoma_registry_records_verified_public_provenance() -> None:
    registry = pd.read_csv(
        ROOT
        / "experiments"
        / "v0_8_5_model_registry_scout_expert_protocol"
        / "configs"
        / "task_registry.csv"
    )
    row = registry.loc[registry["task_id"] == "glaucoma_3class"].iloc[0]

    assert row["dataset_display_name"] == "Glaucoma Fundus"
    assert row["dataset_source"] == "Harvard Dataverse"
    assert row["dataset_url"] == "https://doi.org/10.7910/DVN/1YRRAC"
    assert row["provenance_status"] == "verified_public"


def test_cost_curve_marks_pareto_and_selects_named_operating_points() -> None:
    curve = pd.DataFrame(
        {
            "realized_budget": [0.0, 0.2, 0.3, 0.5, 1.0],
            "accuracy": [0.81, 0.84, 0.855, 0.85, 0.848],
            "estimated_total_compute_ms_per_image": [0.5, 1.0, 1.5, 2.2, 4.0],
        }
    )

    enriched = enrich_cost_curve(curve)
    points = select_operating_points(enriched)

    assert enriched["relative_cost"].tolist() == [1.0, 2.0, 3.0, 4.4, 8.0]
    assert enriched["is_pareto"].tolist() == [True, True, True, False, False]
    assert points["efficient"]["realized_budget"] == 0.0
    assert points["balanced"]["realized_budget"] == 0.3
    assert points["performance"]["realized_budget"] == 0.3


def test_unmeasured_cost_is_excluded_from_pareto_and_recommendations() -> None:
    curve = pd.DataFrame(
        [
            {
                "name": "measured",
                "accuracy": 0.80,
                "estimated_total_compute_ms_per_image": 2.0,
                "cost_status": "measured",
            },
            {
                "name": "integration-only",
                "accuracy": 0.99,
                "estimated_total_compute_ms_per_image": 1.0,
                "cost_status": "unmeasured",
            },
        ]
    )
    enriched = enrich_cost_curve(curve)
    unmeasured = enriched.loc[enriched["name"].eq("integration-only")].iloc[0]
    assert pd.isna(unmeasured["relative_cost"])
    assert unmeasured["is_pareto"] == False  # noqa: E712
    points = select_operating_points(enriched)
    assert all(point["name"] == "measured" for point in points.values())


def test_task_registry_drives_primary_and_display_metrics() -> None:
    registry = pd.read_csv(
        ROOT / "experiments/v0_8_5_model_registry_scout_expert_protocol/configs/task_registry.csv"
    )

    dr = task_metric_profile(registry, "aptos_dr_5class")
    glaucoma = task_metric_profile(registry, "glaucoma_3class")

    assert dr == {"primary_metric": "qwk", "display_metrics": ["accuracy", "macro_f1", "qwk"]}
    assert glaucoma == {"primary_metric": "macro_f1", "display_metrics": ["accuracy", "macro_f1"]}


def test_operating_points_can_use_task_primary_metric() -> None:
    curve = enrich_cost_curve(
        pd.DataFrame(
            {
                "realized_budget": [0.0, 0.5, 1.0],
                "accuracy": [0.90, 0.91, 0.92],
                "macro_f1": [0.60, 0.80, 0.70],
                "estimated_total_compute_ms_per_image": [1.0, 2.0, 3.0],
            }
        ),
        metric_column="macro_f1",
    )

    points = select_operating_points(curve, metric_column="macro_f1")

    assert float(points["performance"]["macro_f1"]) == 0.80


def test_comparison_table_uses_registered_metrics_and_chinese_engineering_columns() -> None:
    runs = pd.DataFrame(
        [
            {
                "组合": "组合 A",
                "scout_ids": "convnext_tiny|swin_tiny",
                "primary_scout_id": "convnext_tiny",
                "active_expert_ids": "retfound_mae_cfp_official_protocol",
                "expert_handoff_mode": "fixed_expert",
                "routing_policy": "mean_uncertainty",
                "accuracy": 0.85,
                "macro_f1": 0.72,
                "qwk": 0.90,
                "realized_budget": 0.3,
                "selected_n": 330,
                "estimated_total_compute_ms_per_image": 2.1,
            }
        ]
    )

    table = build_comparison_table(runs, ["accuracy", "macro_f1", "qwk"])

    assert table.columns.tolist() == [
        "组合",
        "路由模型",
        "默认输出模型",
        "专家模型",
        "专家接管方式",
        "路由机制",
        "Accuracy",
        "Macro-F1",
        "QWK",
        "专家调用比例",
        "入选病例数",
        "估算前向成本（ms/图）",
    ]
    assert table.loc[0, "专家接管方式"] == "固定专家接管"


def test_comparison_plot_frame_drops_internal_key_and_coerces_text_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "key": (["convnext_tiny"], ["retfound_mae_cfp_official_protocol"], "fixed_expert", np.nan),
                "组合": "ConvNeXt-Tiny → RETFound-MAE（官方协议）",
                "expert_handoff_mode": "fixed_expert",
                "realized_budget": 0.3,
                "accuracy": 0.8427,
                "estimated_total_compute_ms_per_image": 1.529,
            },
            {
                "key": np.nan,
                "组合": "ViT-B/16（ImageNet）· DR 五级分级 → RETFound-MAE（官方协议）",
                "expert_handoff_mode": "fixed_expert",
                "realized_budget": 0.3,
                "accuracy": 0.8491,
                "estimated_total_compute_ms_per_image": 1.540,
            },
        ]
    )

    plot = _comparison_plot_frame(frame)

    assert "key" not in plot.columns
    assert plot["组合"].map(type).eq(str).all()
    assert plot["专家接管方式"].map(type).eq(str).all()
    assert pd.api.types.is_numeric_dtype(plot["realized_budget"])
    assert pd.api.types.is_numeric_dtype(plot["accuracy"])
    assert pd.api.types.is_numeric_dtype(plot["estimated_total_compute_ms_per_image"])


def test_job_record_title_uses_canonical_training_artifact_name() -> None:
    job = {
        "job_id": "20260706T084306-93cc727b",
        "artifact_id": "swin_tiny__aptos_dr_5class__adapter__glaucoma_3class__adapter",
        "task_id": "glaucoma_3class",
        "status": "succeeded",
        "output_dir": "/tmp/unused",
        "request": {
            "architecture": "swin_tiny_patch4_window7_224",
            "initialization_source": "timm_pretrained",
            "target_task_id": "glaucoma_3class",
        },
    }

    title = _job_record_title(job, "已完成")

    assert "Swin-Tiny（ImageNet）· 青光眼三分类" in title
    assert "aptos_dr_5class" not in title
    assert "__" not in title


def test_task_evaluation_summary_uses_protocol_or_generic_fallback() -> None:
    dr_metrics = {
        "risk_semantics": "label_based_safety_proxy_not_clinical_gold_standard",
        "dr_large_undergrading_event_total": 10,
        "dr_large_undergrading_selected_n": 8,
        "dr_large_undergrading_resolved_n": 4,
        "dr_large_undergrading_residual_n": 6,
    }
    dr = task_evaluation_summary("aptos_dr_5class", dr_metrics, pd.DataFrame())
    assert dr["profile"] == "disease_proxy"
    assert dr["rows"].loc[0, "残余"] == 6

    detail = pd.DataFrame({"true_label": [0, 0, 1, 1, 2], "final_pred_label": [0, 1, 1, 1, 0]})
    glaucoma = task_evaluation_summary("glaucoma_3class", {}, detail)
    assert glaucoma["profile"] == "generic_multiclass"
    assert glaucoma["rows"].set_index("类别").loc[1, "召回率"] == 1.0

    unlabeled = task_evaluation_summary("new_task", {}, pd.DataFrame({"final_pred_label": [0]}))
    assert unlabeled["profile"] == "unavailable"


def test_dr_proxy_help_exposes_the_three_frozen_event_formulas() -> None:
    research = (ROOT / "app" / "model_hub_research.py").read_text(encoding="utf-8")
    rows = pd.DataFrame(
        [
            {"事件": "大跨度低估", "总事件": 41, "送专家": 31, "纠正": 16, "残余": 25},
            {"事件": "可转诊漏检", "总事件": 77, "送专家": 57, "纠正": 50, "残余": 27},
            {"事件": "重症漏检", "总事件": 65, "送专家": 47, "纠正": 21, "残余": 44},
        ]
    )
    rendered = build_proxy_event_table_html(rows)

    assert 'st.popover("大跨度低估 ？")' not in research
    assert "proxy-help-icon" in rendered
    assert "proxy-event-cards" in rendered
    assert "送专家" in rendered
    assert "残余" in rendered
    assert "参考标签 &gt;= 4 且默认输出 &lt;= 2" in rendered
    assert "参考标签 &gt;= 2 且默认输出 &lt;= 1" in rendered
    assert "参考标签 &gt;= 3 且默认输出 &lt;= 2" in rendered


def test_default_output_model_help_is_precise_and_policy_aware() -> None:
    research = (ROOT / "app" / "model_hub_research.py").read_text(encoding="utf-8")

    assert '"默认输出模型"' in research
    assert "未调用专家的病例由它给出最终输出" in research
    assert "其他路由模型仅参与分歧或平均不确定性计算" in research
    assert '"基础输出模型"' not in research


def test_research_workspace_exposes_global_candidate_scan() -> None:
    research = (ROOT / "app" / "model_hub_research.py").read_text(encoding="utf-8")

    assert "scan_global_composition_candidates" in research
    assert "estimate_global_composition_count" in research
    assert "def _render_global_scan" in research
    assert "model_hub_global_scan" in research
    assert "MAX_INTERACTIVE_GLOBAL_SCAN_EVALUATIONS" in research
    assert "submit_global_scan_job" in research
    assert "latest_completed_global_scan" in research
    assert "仅展示 Pareto 前沿和 Top-N 候选" in research
    assert "plot_view" in research


def test_global_scan_result_cards_explain_primary_metric_and_pareto_frontier() -> None:
    research = (ROOT / "app" / "model_hub_research.py").read_text(encoding="utf-8")

    assert "当前主指标" in research
    assert "来自任务注册协议" in research
    assert "Pareto 前沿" in research
    assert "不存在另一个组合同时" in research
    assert "查看组合细节" in research


def test_tradeoff_chart_uses_effect_wording_not_hardware_efficiency_wording() -> None:
    research = (ROOT / "app" / "model_hub_research.py").read_text(encoding="utf-8")

    assert "成本-效果折中图" in research
    assert "性能－成本操作点" not in research


def test_task_record_page_exposes_global_scan_jobs() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")

    assert "全局候选扫描任务" in engineering
    assert "list_global_scan_jobs" in engineering
    assert "global_scan_results.csv" in engineering


def test_global_scan_job_records_use_pending_navigation_and_chinese_preview() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")

    assert 'st.session_state["engineering_layer"] = "研究评测"' not in engineering
    assert 'st.session_state["pending_engineering_layer"] = "研究评测"' in engineering
    assert "全局排名" in engineering
    assert "路由模型" in engineering
    assert "专家模型" in engineering
    assert "路由机制" in engineering
    assert "专家调用比例" in engineering
    assert "估算前向成本（ms/图）" in engineering


def test_training_job_logs_summarize_known_scheduler_warning() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")

    assert "split_training_log_messages" in engineering
    assert "PyTorch 学习率调度器弃用提示" in engineering
    assert "已收起" in engineering



def test_training_initialization_is_explicit_in_the_ui() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")

    assert '"训练初始化"' in engineering
    assert "原始预训练权重（推荐）" in engineering
    assert "继续训练现有 checkpoint" in engineering
    assert "跨疾病 checkpoint 迁移（研究实验）" in engineering


def test_pretraining_source_prefers_explicit_metadata_then_training_config() -> None:
    explicit = normalized_model_metadata(
        "convnext_tiny",
        "convnext_tiny",
        pretraining_source="custom_retinal_pretraining",
        training_config={"pretrained": True},
    )
    inferred = normalized_model_metadata(
        "convnext_tiny",
        "convnext_tiny",
        pretraining_source="unspecified",
        training_config={"pretrained": True},
    )

    assert explicit["pretraining_source"] == "custom_retinal_pretraining"
    assert inferred["pretraining_source"] == "timm_pretrained"


def test_training_recipe_selector_explains_existing_templates_and_extension_rules() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")
    expected_recipes = {
        "timm_full_train.yaml": ("通用全量微调", "统一工程初筛"),
        "timm_quick_smoke.yaml": ("快速链路验证", "只验证数据、模型和训练链路"),
        "timm_head_only.yaml": ("冻结骨干只训分类头", "冻结骨干的低成本迁移基线"),
        "timm_low_lr_finetune.yaml": ("低学习率保守微调", "对预训练权重更保守的低学习率控制组"),
    }
    recipe_root = ROOT / "experiments" / "model_hub" / "registry" / "training_recipes"

    for filename, expected_values in expected_recipes.items():
        recipe_text = (recipe_root / filename).read_text(encoding="utf-8")
        assert all(expected in recipe_text for expected in expected_values)
    assert "工程模板说明" in engineering
    assert "科研候选实验" in engineering
    assert "六次 trial 属于同一个模型×数据集实验系列" in engineering
    assert "当前模板：" in engineering
    assert "display_name" in engineering
    assert "recipe_id" in engineering


def test_ui_uses_routing_rank_wording_and_safety_proxy_disclaimer() -> None:
    demo = (ROOT / "app" / "model_hub_demo.py").read_text(encoding="utf-8")
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")
    research = (ROOT / "app" / "model_hub_research.py").read_text(encoding="utf-8")
    clinical = (ROOT / "app" / "model_hub_clinical.py").read_text(encoding="utf-8")

    assert "模型工程" in demo
    assert "病例回放与路由解释" in demo
    assert "临床演示" not in demo
    assert "v0.8.6 产物不完整" not in demo
    assert "受控路由基线产物不完整" in demo
    assert "模型接入" in engineering
    assert "研究评测" in engineering
    assert "路由候选" in engineering
    assert "专家候选" in engineering
    assert "组合对比" in research
    assert "查看病例路由解释" in clinical
    assert "Scout 候选" not in engineering
    assert "Expert 候选" not in engineering
    assert "组合对比篮" not in research
    assert "不是临床金标准" in research


def test_training_capability_requires_enabled_matching_recipe() -> None:
    recipes = pd.DataFrame(
        [
            {
                "recipe_id": "convnext_tiny_screening",
                "model_family": "convnext",
                "architecture": "convnext_tiny",
                "enabled": 1,
            }
        ]
    )
    supported = pd.Series({"model_family": "convnext", "architecture": "convnext_tiny"})
    unsupported = pd.Series({"model_family": "retfound", "architecture": "retfound_mae"})

    assert training_capability(supported, recipes) == (True, "可提交受控微调任务")
    allowed, reason = training_capability(unsupported, recipes)
    assert not allowed
    assert "recipe" in reason


def test_global_model_catalog_keeps_cross_task_models_visible_and_marks_compatibility() -> None:
    models = pd.DataFrame(
        [
            {
                "model_id": "dr::vit",
                "task_id": "dr",
                "artifact_id": "vit_dr",
                "model_family": "vit",
                "architecture": "vit_base_patch16_224",
                "label_space": "dr_5class",
                "n_classes": 5,
                "prediction_source": "adapter",
                "adapter_status": "completed",
                "compatibility_status": "ready_for_pairing",
            },
            {
                "model_id": "glaucoma::convnext",
                "task_id": "glaucoma",
                "artifact_id": "convnext_glaucoma",
                "model_family": "convnext",
                "architecture": "convnext_tiny",
                "label_space": "glaucoma_3class",
                "n_classes": 3,
                "prediction_source": "adapter",
                "adapter_status": "completed",
                "compatibility_status": "ready_for_pairing",
            },
            {
                "model_id": "glaucoma::legacy",
                "task_id": "glaucoma",
                "artifact_id": "legacy_glaucoma",
                "model_family": "retfound",
                "architecture": "retfound_dinov2",
                "label_space": "glaucoma_3class",
                "n_classes": 3,
                "prediction_source": "legacy",
                "adapter_status": "needs_loader_audit",
                "compatibility_status": "ready_for_pairing",
            },
        ]
    )
    recipes = pd.DataFrame(
        [
            {
                "recipe_id": "vit",
                "model_family": "vit",
                "architecture": "vit_base_patch16_224",
                "enabled": 1,
            }
        ]
    )

    catalog = build_global_model_catalog(models, target_task_id="glaucoma", recipes=recipes)
    states = catalog.set_index("model_id")["target_task_status"].to_dict()

    assert len(catalog) == 3
    assert states["glaucoma::convnext"] == "offline_replay"
    assert states["glaucoma::legacy"] == "offline_replay"
    assert states["dr::vit"] == "adaptable"
    assert "5" in catalog.set_index("model_id").loc["dr::vit", "target_task_reason"]
    assert "3" in catalog.set_index("model_id").loc["dr::vit", "target_task_reason"]


def test_split_task_models_still_returns_ready_and_incomplete_groups() -> None:
    models = pd.DataFrame(
        [
            {"task_id": "dr", "artifact_id": "ready", "prediction_source": "adapter", "compatibility_status": "ready_for_pairing"},
            {"task_id": "dr", "artifact_id": "missing", "prediction_source": "missing", "compatibility_status": "incomplete"},
        ]
    )

    ready, incomplete = split_task_models(models, "dr")

    assert ready["artifact_id"].tolist() == ["ready"]
    assert incomplete["artifact_id"].tolist() == ["missing"]


def test_global_model_catalog_blocks_cross_task_model_without_training_adapter() -> None:
    models = pd.DataFrame(
        [
            {
                "model_id": "dr::retfound",
                "task_id": "dr",
                "artifact_id": "retfound_dr",
                "model_family": "retfound",
                "architecture": "retfound_mae",
                "label_space": "dr_5class",
                "n_classes": 5,
                "prediction_source": "legacy",
                "adapter_status": "needs_loader_audit",
                "compatibility_status": "ready_for_pairing",
            },
            {
                "model_id": "glaucoma::reference",
                "task_id": "glaucoma",
                "artifact_id": "reference",
                "model_family": "convnext",
                "architecture": "convnext_tiny",
                "label_space": "glaucoma_3class",
                "n_classes": 3,
                "prediction_source": "adapter",
                "adapter_status": "completed",
                "compatibility_status": "ready_for_pairing",
            },
        ]
    )

    catalog = build_global_model_catalog(models, target_task_id="glaucoma", recipes=pd.DataFrame())
    row = catalog.loc[catalog["model_id"] == "dr::retfound"].iloc[0]

    assert row["target_task_status"] == "blocked"
    assert "Adapter" in row["target_task_reason"]


def test_filter_global_model_catalog_filters_status_and_family_without_hiding_all_models() -> None:
    catalog = pd.DataFrame(
        [
            {"model_id": "dr::vit", "model_family": "vit", "target_task_status": "adaptable"},
            {"model_id": "glaucoma::convnext", "model_family": "convnext", "target_task_status": "direct_inference"},
            {"model_id": "glaucoma::retfound", "model_family": "retfound", "target_task_status": "offline_replay"},
        ]
    )

    assert len(filter_global_model_catalog(catalog)) == 3
    filtered = filter_global_model_catalog(catalog, status="adaptable", family="vit")

    assert filtered["model_id"].tolist() == ["dr::vit"]


def test_current_cfp_candidate_view_hides_legacy_non_cfp_and_generative_assets() -> None:
    catalog = pd.DataFrame(
        [
            {
                "model_id": "local::task",
                "provider_id": "local_artifact",
                "download_status": "",
                "artifact_type": "task_checkpoint",
                "modalities": "CFP",
            },
            {
                "model_id": "ophbench::retfound::retfound-cfp",
                "provider_id": "ophbench",
                "download_status": "downloaded",
                "artifact_type": "foundation_encoder",
                "modalities": "CFP",
            },
            {
                "model_id": "ophbench::retfound::retfound-oct",
                "provider_id": "ophbench",
                "download_status": "downloaded",
                "artifact_type": "foundation_encoder",
                "modalities": "OCT",
            },
            {
                "model_id": "ophbench::visionfm::visionfm-fundus",
                "provider_id": "ophbench",
                "download_status": "excluded_by_project_scope",
                "artifact_type": "foundation_encoder",
                "modalities": "CFP",
            },
            {
                "model_id": "ophbench::deretfound::sd-retina",
                "provider_id": "ophbench",
                "download_status": "downloaded",
                "artifact_type": "generative_model",
                "modalities": "CFP",
            },
        ]
    )

    visible, modality = current_task_candidate_catalog(
        catalog,
        dataset_id="APTOS2019",
    )

    assert modality == "CFP"
    assert visible["model_id"].tolist() == [
        "local::task",
        "ophbench::retfound::retfound-cfp",
    ]


def test_available_routing_policies_match_route_model_count() -> None:
    assert available_routing_policies(1, 1) == ["low_confidence", "low_margin", "high_entropy"]
    assert available_routing_policies(2, 1) == ["disagreement_then_uncertainty", "mean_uncertainty"]
    assert available_routing_policies(0, 2) == []
    assert available_routing_policies(1, 0) == []


def test_engineering_ui_uses_confirmed_background_job_workflow() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")

    assert "任务与数据集" in engineering
    assert "模型与训练配置" in engineering
    assert "完整训练配置（YAML）" in engineering
    assert "配置预检" in engineering
    assert "人工确认" in engineering
    assert "submit_training_job" in engineering
    assert "后台任务" in engineering
    assert "run_training(" not in engineering
    assert "from scripts.training.train_timm_classifier" not in engineering


def test_training_submission_defers_navigation_until_before_widget_creation() -> None:
    state = {
        "engineering_layer": "模型接入",
        "pending_engineering_layer": "任务运行记录",
    }

    engineering_module.apply_pending_engineering_navigation(state)

    assert state["engineering_layer"] == "任务运行记录"
    assert "pending_engineering_layer" not in state


def test_job_records_render_local_training_curves_without_external_tracker() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")

    assert "load_training_progress" in engineering
    assert "训练与验证损失" in engineering
    assert "验证集性能" in engineering
    assert "原始训练日志" in engineering
    assert "wandb" not in engineering.lower()


def test_engineering_ui_uses_registered_dataset_selector_instead_of_manual_root() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(encoding="utf-8")
    registry = pd.read_csv(
        ROOT / "experiments/v0_8_5_model_registry_scout_expert_protocol/configs/task_registry.csv"
    )

    assert "registered_dataset_options" in engineering
    assert 'st.text_input(\n            "ImageFolder 数据根目录"' not in engineering
    assert registry["data_root"].fillna("").str.strip().ne("").all()
    assert set(registry["label_structure"]) == {"nominal", "ordinal"}


def test_model_access_ui_separates_asset_readiness_and_routing_eligibility() -> None:
    engineering = (ROOT / "app" / "model_hub_engineering.py").read_text(
        encoding="utf-8"
    )

    assert "A. 模型资产目录" in engineering
    assert "B. 接入准备度" in engineering
    assert "C. 中转台资格" in engineering
    assert "OphBench 建库时已下载" in engineering
    assert "OphAgent 本地权重" in engineering
    assert "基础加载已通过" in engineering
    assert "任务适配待完成" in engineering
    assert "任务推理暂不可用" in engineering
    assert "暂不可路由" in engineering
    assert "可交接 Checkpoint" in engineering
    assert "VisionFM legacy 资产" in engineering
    assert "current_task_candidate_catalog" in engineering
    assert "详细错误：`{detail}`" in engineering
    assert "seed_unverified" not in engineering
