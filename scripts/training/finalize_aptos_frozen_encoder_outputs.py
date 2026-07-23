#!/usr/bin/env python3
"""Index completed shared frozen-encoder APTOS runs without re-running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiments/opening_risk_routing_closure"
MODEL_INFO = {
    "eyeclip_cfp": ("EyeCLIP", "ViT-B/32", "eyeclip_frozen_encoder_linear_probe", "official EyeCLIP"),
    "keepfit_cfp": ("KeepFIT", "ResNet-50", "keepfit_frozen_encoder_linear_probe", "official KeepFIT"),
    "ret_clip": ("RET-CLIP", "ViT-B/16", "ret_clip_frozen_encoder_linear_probe", "official RET-CLIP"),
    "retizero": ("RetiZero", "RETFound LoRA", "retizero_frozen_encoder_linear_probe", "official RetiZero"),
}
QUALIFICATION_STATUS = {
    "convnext_tiny": "routing_shortlist",
    "eyeclip_cfp": "task_ready_reference",
    "flair": "routing_shortlist",
    "keepfit_cfp": "task_ready_reference",
    "preti": "routing_shortlist",
    "ret_clip": "routing_shortlist",
    "retfound_cfp": "qualification_limited",
    "retfound_green": "cost_incomplete",
    "retizero": "task_ready_reference",
    "swin_tiny": "routing_shortlist",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest_run(model: str) -> Path:
    candidates = sorted((EXPERIMENT_ROOT / "replays" / model).glob("*/run_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"completed run missing for {model}")
    return candidates[-1].parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=sorted(MODEL_INFO))
    args = parser.parse_args()
    registry_path = EXPERIMENT_ROOT / "configs/protocols/aptos_h100_prediction_assets.csv"
    registry = pd.read_csv(registry_path)
    rows = []
    expanded_models = []
    for model in args.models:
        run = _latest_run(model)
        config = json.loads((run / "config_resolved.json").read_text(encoding="utf-8"))
        validation = run / "predictions/validation_predictions.csv"
        test = run / "predictions/test_predictions.csv"
        costs = json.loads((run / "costs/forward_cost.json").read_text(encoding="utf-8"))
        test_frame = pd.read_csv(test)
        report = pd.DataFrame(classification_report(test_frame.y_true, test_frame.y_pred, output_dict=True)).transpose()
        report.to_csv(run / "metrics/class_metrics.csv")
        pd.DataFrame(confusion_matrix(test_frame.y_true, test_frame.y_pred)).to_csv(run / "metrics/confusion_matrix.csv", index=False)
        manifest = {
            "model_id": model,
            "validation_prediction_sha256": _sha256(validation),
            "test_prediction_sha256": _sha256(test),
            "checkpoint_sha256": config["checkpoint_sha256"],
            "preprocessing_id": config["preprocessing_id"],
            "task_inference_ready": False,
            "validation_selection_eligible": True,
            "offline_evaluation_eligible": True,
            "route_eligible": False,
        }
        (run / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        family, architecture, adapter, source = MODEL_INFO[model]
        rows.append({
            "task_id": "aptos_dr_5class", "artifact_id": model, "model_family": family,
            "architecture": architecture, "adapter_type": adapter, "pretraining_source": source,
            "validation_prediction_path": validation.relative_to(ROOT).as_posix(), "test_prediction_path": test.relative_to(ROOT).as_posix(),
            "checkpoint_path": str(config["checkpoint"]), "checkpoint_sha256": config["checkpoint_sha256"],
            "preprocessing_id": config["preprocessing_id"],
            "forward_cost_ms_per_image": costs["results"]["batch_16"]["median_ms_per_image"],
            "cost_scope": "H100 GPU forward-only batch16", "cost_status": "measured",
            "cpu_postprocess_ms_per_image": "", "cpu_postprocess_status": "not_applicable",
            "current_run_reproducible": True, "validation_selection_eligible": True,
            "qualification_status": "current_run_reproducible", "route_eligible": False,
            "role_candidates": "scout|expert", "source_version": "h100_frozen_encoder_probe",
            "notes": "validation-only linear-probe parameter selection; test not used for selection",
        })
        expanded_models.append({
            "model_id": model, "adapter": adapter, "checkpoint_path": str(config["checkpoint"]),
            "preprocessing_id": config["preprocessing_id"], "existing_replay": run.relative_to(EXPERIMENT_ROOT).as_posix(),
            "validation_predictions": "predictions/validation_predictions.csv", "test_predictions": "predictions/test_predictions.csv",
            "required_artifacts": ["run_summary.json", "artifact_manifest.json", "costs/forward_cost.json"],
        })
    registry = pd.concat([registry.loc[~registry.artifact_id.isin(args.models)], pd.DataFrame(rows)], ignore_index=True)
    registry.loc[
        registry["artifact_id"].isin(QUALIFICATION_STATUS), "qualification_status"
    ] = registry.loc[registry["artifact_id"].isin(QUALIFICATION_STATUS), "artifact_id"].map(
        QUALIFICATION_STATUS
    )
    registry.to_csv(registry_path, index=False)
    source_config = EXPERIMENT_ROOT / "configs/protocols/aptos_h100_six_model_pool.yaml"
    expanded = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    expanded["protocol_id"] = "aptos_h100_ten_model_expansion_validation_v1"
    expanded["output_dir"] = "experiments/opening_risk_routing_closure/selection/h100_ten_model_pool_runner"
    expanded["model_hub_output_dir"] = "experiments/opening_risk_routing_closure/outputs/model_hub_validation_expanded_pool"
    expanded["models"].extend(expanded_models)
    target = EXPERIMENT_ROOT / "configs/protocols/aptos_h100_ten_model_pool.yaml"
    target.write_text(yaml.safe_dump(expanded, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
