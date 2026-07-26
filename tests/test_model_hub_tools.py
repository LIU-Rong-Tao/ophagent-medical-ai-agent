from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from app.model_hub_tools import (
    TOOL_NAMES,
    ToolContext,
    ToolRequest,
    ToolRuntime,
    capability_matrix,
)


def _context(tmp_path: Path) -> ToolContext:
    prediction_path = tmp_path / "validation_predictions.csv"
    pd.DataFrame(
        {
            "case_id": ["source-1"],
            "y_true": [2],
            "y_pred": [2],
            "prob_0": [0.05],
            "prob_1": [0.10],
            "prob_2": [0.70],
            "prob_3": [0.10],
            "prob_4": [0.05],
        }
    ).to_csv(prediction_path, index=False)
    registry = pd.DataFrame(
        [
            {
                "task_id": "aptos_dr_5class",
                "artifact_id": "offline_model",
                "display_name": "Offline Model",
                "architecture": "ViT",
                "qualification": "analytical_asset_only",
                "qualification_reason": "prediction only",
                "validation_prediction_path": str(prediction_path),
                "checkpoint_sha256": "a" * 64,
                "preprocessing_id": "test_preprocess",
                "forward_cost_ms_per_image": 1.2,
                "cost_scope": "test",
                "route_eligible": False,
            }
        ]
    )
    return ToolContext(
        project_root=tmp_path,
        asset_registry=registry,
        online_artifacts={},
    )


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "fundus.png"
    Image.new("RGB", (64, 64), color=(32, 24, 20)).save(path)
    return path


def test_six_tools_have_one_contract_and_capability_status(tmp_path: Path) -> None:
    matrix = capability_matrix(_context(tmp_path))

    assert tuple(matrix["tool_name"]) == TOOL_NAMES
    assert matrix["status"].str.startswith("implemented").all()


def test_case_validation_accepts_local_image_and_blocks_locked_test(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    image = _image(tmp_path)
    runtime = ToolRuntime(context)
    accepted = runtime.run(
        ToolRequest(
            "case_input.validate",
            task_id="aptos_dr_5class",
            case_id="CASE-0001",
            payload={"image_paths": [str(image)], "split": "validation"},
        )
    )

    assert accepted.ok
    assert accepted.data["image_count"] == 1
    assert accepted.data["input_complete"] is True

    blocked_runtime = ToolRuntime(context)
    blocked = blocked_runtime.run(
        ToolRequest(
            "case_input.validate",
            task_id="aptos_dr_5class",
            case_id="CASE-0001",
            payload={"image_paths": [str(image)], "split": "test"},
        )
    )
    assert not blocked.ok
    assert blocked.code == "TEST_LOCKED"


def test_prediction_asset_is_read_only_and_uses_source_case_key(
    tmp_path: Path,
) -> None:
    runtime = ToolRuntime(_context(tmp_path))
    result = runtime.run(
        ToolRequest(
            "prediction_asset.validate",
            task_id="aptos_dr_5class",
            case_id="DISPLAY-0001",
            payload={
                "artifact_id": "offline_model",
                "split": "validation",
                "source_case_key": "source-1",
            },
        )
    )

    assert result.ok
    assert result.qualification == "analytical_asset_only"
    assert result.data["read_only"] is True
    assert result.data["pred_label"] == 2
    assert "y_true" not in result.data
    assert "source-1" not in str(runtime.to_dict())


def test_offline_asset_cannot_run_original_image_and_halts_downstream(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    image = _image(tmp_path)
    runtime = ToolRuntime(context)
    blocked = runtime.run(
        ToolRequest(
            "model_inference.run",
            task_id="aptos_dr_5class",
            case_id="CASE-0001",
            payload={
                "artifact_id": "offline_model",
                "image_paths": [str(image)],
            },
        )
    )
    downstream = runtime.run(
        ToolRequest(
            "result_risk_audit.run",
            task_id="aptos_dr_5class",
            case_id="CASE-0001",
            payload={"predictions": []},
        )
    )

    assert blocked.code == "QUALIFICATION_BLOCKED"
    assert blocked.qualification == "analytical_asset_only"
    assert downstream.code == "UPSTREAM_FAILED"
    assert runtime.events[-1].status == "skipped"


def test_case_risk_audit_reuses_probability_risk_metrics(tmp_path: Path) -> None:
    runtime = ToolRuntime(_context(tmp_path))
    result = runtime.run(
        ToolRequest(
            "result_risk_audit.run",
            task_id="aptos_dr_5class",
            case_id="CASE-0001",
            payload={
                "predictions": [
                    {
                        "artifact_id": "a",
                        "probabilities": [0.05, 0.10, 0.70, 0.10, 0.05],
                    },
                    {
                        "artifact_id": "b",
                        "probabilities": [0.05, 0.10, 0.20, 0.55, 0.10],
                    },
                ]
            },
        )
    )

    assert result.ok
    assert result.data["model_disagreement"] is True
    assert result.data["semantics"] == "model_output_error_risk_not_clinical_consequence"
    assert result.data["task_proxy"]["name"] == "DR 重症等级概率质量代理"


def test_cached_route_trace_is_research_only_and_never_reads_test(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    trace_path = (
        tmp_path
        / "experiments/opening_risk_routing_closure/outputs/"
        "model_hub_validation_expanded_pool/case_routing_trace.csv"
    )
    trace_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pairing_id": "aptos_dr_5class__flair__ret_clip__to__retfound_cfp",
                "task_id": "aptos_dr_5class",
                "image_key": "source-1",
                "primary_scout_artifact_id": "flair",
                "expert_artifact_id": "retfound_cfp",
                "routing_policy": "disagreement_then_uncertainty",
                "requested_budget": 0.2,
                "realized_budget": 0.2,
                "scout_pred_labels": "{}",
                "scout_confidences": "{}",
                "scout_entropies": "{}",
                "scout_margins": "{}",
                "scout_disagreement": True,
                "routing_score": 0.8,
                "is_reviewed_by_expert": True,
                "expert_pred_label": 3,
                "final_pred_label": 3,
                "final_source": "expert",
            }
        ]
    ).to_csv(trace_path, index=False)
    protocol_path = (
        tmp_path
        / "experiments/opening_risk_routing_closure/configs/protocols/"
        "aptos_h100_ten_model_frozen_test_protocol.json"
    )
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text('{"route_eligible": false}', encoding="utf-8")
    runtime = ToolRuntime(context)
    result = runtime.run(
        ToolRequest(
            "routing_protocol.evaluate",
            task_id="aptos_dr_5class",
            case_id="CASE-0001",
            payload={"split": "validation", "source_case_key": "source-1"},
        )
    )

    assert result.ok
    assert result.data["evaluation_design"] == "research_simulation"
    assert result.data["route_eligible"] is False
    assert result.data["test_content_used"] is False
    assert "true_label" not in result.data
