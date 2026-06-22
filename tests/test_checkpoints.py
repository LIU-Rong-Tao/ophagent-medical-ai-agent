import json
import builtins
from pathlib import Path

import pandas as pd
import pytest

from app.checkpoints import (
    compute_file_sha256,
    discover_model_artifacts,
    resolve_capabilities,
    summarize_frozen_model_finding,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(
    root: Path,
    *,
    experiment_root: str = "aptos_convnext_tiny",
    run_name: str = "run_a",
    checkpoint_count: int = 1,
    with_predictions: bool = True,
) -> Path:
    run = root / "experiments" / experiment_root / run_name
    _write_json(
        run / "configs" / "config.json",
        {
            "backbone": "convnext_tiny",
            "num_classes": 5,
            "image_size": 224,
        },
    )
    _write_json(
        run / "configs" / "class_to_idx.json",
        {
            "anodr": 0,
            "bmilddr": 1,
            "cmoderatedr": 2,
            "dseveredr": 3,
            "eproliferativedr": 4,
        },
    )
    _write_json(run / "configs" / "env_info.json", {"python": "3.10"})
    _write_json(run / "evaluation" / "test" / "metrics.json", {"accuracy": 0.8})
    _write_json(run / "logs" / "summary.json", {"best_epoch": 3})
    for index in range(checkpoint_count):
        checkpoint = run / "checkpoints" / f"model_{index}_best.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"weight-{index}".encode())
    if with_predictions:
        prediction = run / "evaluation" / "test" / "test_predictions.csv"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text(
            "image_path,true_idx,pred_idx,prob_0,prob_1,prob_2,prob_3,prob_4\n"
            "/data/a.png,0,0,0.8,0.1,0.05,0.03,0.02\n",
            encoding="utf-8",
        )
    return run


def test_discovery_only_scans_known_experiment_roots_and_does_not_load_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _make_run(tmp_path)
    ignored = tmp_path / "experiments" / "unrelated_model" / "run" / "checkpoints"
    ignored.mkdir(parents=True)
    (ignored / "unrelated_best.pth").write_bytes(b"ignored")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("静态发现阶段禁止导入或加载 torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    artifacts = discover_model_artifacts(tmp_path)

    assert len(artifacts) == 1
    assert artifacts[0].model_key == "convnext_tiny"
    assert "unrelated_model" not in str(artifacts[0].experiment_dir)


def test_discovery_distinguishes_complete_missing_and_ambiguous_artifacts(
    tmp_path: Path,
):
    _make_run(tmp_path, run_name="complete")
    _make_run(tmp_path, run_name="ambiguous", checkpoint_count=2)
    _make_run(tmp_path, run_name="offline", checkpoint_count=0)

    by_run = {
        artifact.experiment_dir.name: artifact
        for artifact in discover_model_artifacts(tmp_path)
    }

    assert by_run["complete"].artifact_status == "static_complete"
    assert by_run["complete"].can_attempt_load
    assert by_run["ambiguous"].artifact_status == "checkpoint_ambiguous"
    assert not by_run["ambiguous"].can_attempt_load
    assert by_run["offline"].artifact_status == "offline_only"
    assert not by_run["offline"].can_attempt_load


def test_checkpoint_hash_cache_invalidates_when_size_or_mtime_changes(tmp_path: Path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"one")
    first = compute_file_sha256(checkpoint)
    assert first == compute_file_sha256(checkpoint)

    checkpoint.write_bytes(b"two-two")
    second = compute_file_sha256(checkpoint)

    assert first != second


def test_capability_resolver_requires_explicit_dr_protocol():
    columns = {
        "pred_idx",
        "true_idx",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "prob_4",
    }
    dr = resolve_capabilities("dr_icdr_5class_proxy_v1", columns)
    generic = resolve_capabilities("generic_multiclass_v1", columns)

    assert dr["supports_probability_audit"]
    assert dr["supports_ordinal_dr_audit"]
    assert dr["supports_vtdr_miss_proxy"]
    assert generic["supports_probability_audit"]
    assert not generic["supports_ordinal_dr_audit"]
    assert not generic["supports_vtdr_miss_proxy"]
    assert not generic["supports_large_undergrading"]


def test_finding_summary_reports_rank_tie_and_gap_from_frozen_v067c(tmp_path: Path):
    path = tmp_path / "tradeoff.csv"
    pd.DataFrame(
        [
            {
                "backbone": "swin_tiny",
                "ranking_method": "gated_severe_prob_mass_only",
                "clinical_event": "vision_threatening_dr_miss",
                "review_budget": budget,
                "dangerous_error_recall_at_k": value,
            }
            for budget, value in [(0.1, 0.4), (0.2, 0.7), (0.3, 0.8)]
        ]
        + [
            {
                "backbone": "swin_tiny",
                "ranking_method": "confidence_only",
                "clinical_event": "vision_threatening_dr_miss",
                "review_budget": budget,
                "dangerous_error_recall_at_k": value,
            }
            for budget, value in [(0.1, 0.4), (0.2, 0.6), (0.3, 0.85)]
        ]
    ).to_csv(path, index=False)

    result = summarize_frozen_model_finding(
        path,
        backbone="swin_tiny",
        event="vision_threatening_dr_miss",
        method="gated_severe_prob_mass_only",
    )

    assert result["rank_at_10"] == 1
    assert result["status_at_10"] == "并列第一"
    assert result["rank_at_20"] == 1
    assert result["status_at_20"] == "第一"
    assert result["top20_recall"] == pytest.approx(0.7)
    assert result["delta_to_best_comparator_at_20"] == pytest.approx(0.1)
    assert result["rank_at_30"] == 2
    assert result["status_at_30"] == "非第一"


def test_finding_summary_returns_not_evaluated_when_target_row_missing(tmp_path: Path):
    path = tmp_path / "tradeoff.csv"
    pd.DataFrame(
        [
            {
                "backbone": "convnext_tiny",
                "ranking_method": "confidence_only",
                "clinical_event": "general_error",
                "review_budget": 0.2,
                "dangerous_error_recall_at_k": 0.5,
            }
        ]
    ).to_csv(path, index=False)

    result = summarize_frozen_model_finding(
        path,
        backbone="swin_tiny",
        event="vision_threatening_dr_miss",
        method="gated_severe_prob_mass_only",
    )

    assert result["status_at_20"] == "未评估"
