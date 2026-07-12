import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from app.benchmark_artifact_importer import load_benchmark_pilot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pilot(root: Path) -> Path:
    root.mkdir()
    run_manifest = {
        "protocol_version": "0.1",
        "evaluation_role": "pilot_protocol_validation",
        "research_claim_status": "not_for_scientific_comparison",
        "task_id": "aptos_dr_5class",
        "dataset_id": "aptos2019",
        "label_space": "dr_icdr_0_4",
        "model_id": "retfound",
        "checkpoint_id": "retfound-cfp",
        "checkpoint_sha256": "a" * 64,
        "split_manifest_sha256": "b" * 64,
        "test_used_for_selection": False,
    }
    (root / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"primary_metric": "quadratic_kappa"}))
    pd.DataFrame(
        [
            {
                "image_key": "case-1",
                "true_label": 0,
                "pred_label": 0,
                "prob_0": 0.6,
                "prob_1": 0.1,
                "prob_2": 0.1,
                "prob_3": 0.1,
                "prob_4": 0.1,
            }
        ]
    ).to_csv(root / "test_predictions.csv", index=False)
    artifacts = [
        {"name": name, "sha256": _sha256(root / name)}
        for name in ("run_manifest.json", "metrics.json", "test_predictions.csv")
    ]
    (root / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    return root


def test_benchmark_pilot_is_validated_but_never_promoted_to_routing(tmp_path):
    artifact = load_benchmark_pilot(
        _pilot(tmp_path / "pilot"),
        expected_task_id="aptos_dr_5class",
        expected_label_space="dr_icdr_0_4",
    )
    record = artifact.as_non_routable_record()
    assert record["evaluation_role"] == "pilot_protocol_validation"
    assert record["research_claim_status"] == "not_for_scientific_comparison"
    assert record["task_checkpoint"] is False
    assert record["task_inference_ready"] is False
    assert record["route_eligible"] is False


def test_benchmark_pilot_rejects_tampered_predictions(tmp_path):
    root = _pilot(tmp_path / "pilot")
    with (root / "test_predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_benchmark_pilot(root)
