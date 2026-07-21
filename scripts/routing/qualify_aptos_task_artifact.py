#!/usr/bin/env python3
"""Grant APTOS routing eligibility only after artifact evidence passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_MANIFEST_SHA256 = "4d3332aab0f010ccf1fefa23af51e65fd2764558bc5a6d6c153ba13379949765"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _prediction_evidence(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(path)
    required = {"image_key", "true_label", "pred_label", *(f"prob_{i}" for i in range(5))}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"标准概率包缺少字段：{missing}")
    if len(frame) != 1100 or frame["image_key"].duplicated().any():
        raise ValueError("APTOS test 概率包必须包含 1100 个唯一病例")
    probability_columns = [f"prob_{index}" for index in range(5)]
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("标准概率包存在非有限概率")
    max_sum_error = float(np.abs(probabilities.sum(axis=1) - 1).max())
    if max_sum_error > 1e-6:
        raise ValueError(f"标准概率包行概率和不为 1：max_error={max_sum_error}")
    if not np.array_equal(frame["pred_label"].to_numpy(), probabilities.argmax(axis=1)):
        raise ValueError("pred_label 与概率 argmax 不一致")
    return frame.sort_values("image_key").reset_index(drop=True), {
        "rows": len(frame),
        "unique_cases": int(frame["image_key"].nunique()),
        "max_probability_sum_error": max_sum_error,
        "prediction_sha256": sha256_file(path),
    }


def qualify(run_dir: Path, reference_predictions: Path) -> Path:
    run_dir = run_dir.resolve()
    registration_path = run_dir / "registration_record.csv"
    registration = pd.read_csv(registration_path)
    if len(registration) != 1:
        raise ValueError("registration_record 必须且只能有一行")
    row = registration.iloc[0]
    required_true = (
        "task_checkpoint",
        "task_adapted",
        "task_inference_ready",
        "offline_evaluation_eligible",
        "unified_evaluation_completed",
        "inference_cost_measured",
    )
    failed = [field for field in required_true if not _as_bool(row.get(field, False))]
    if failed:
        raise ValueError(f"路由资格前置证据未通过：{failed}")
    if str(row.get("task_id")) != "aptos_dr_5class" or int(row.get("n_classes", 0)) != 5:
        raise ValueError("当前资格门仅适用于 APTOS DR 五分类任务")

    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("数据清单不是冻结的 APTOS 划分")
    entries = manifest.get("entries", [])
    split_keys: dict[str, set[str]] = {split: set() for split in ("train", "val", "test")}
    for entry in entries:
        split_keys[str(entry["split"])].add(Path(entry["relative_path"]).stem)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_keys[left] & split_keys[right]:
            raise ValueError(f"{left}/{right} 存在病例交叉")

    prediction_path = Path(str(row["prediction_path"]))
    candidate, prediction_evidence = _prediction_evidence(prediction_path)
    reference, _ = _prediction_evidence(reference_predictions)
    if not candidate["image_key"].equals(reference["image_key"]):
        raise ValueError("候选模型与参考模型的病例集合不一致")
    if not candidate["true_label"].equals(reference["true_label"]):
        raise ValueError("候选模型与参考模型的真实标签不一致")

    checkpoint_path = Path(str(row["checkpoint_path"]))
    if not checkpoint_path.is_file():
        raise ValueError("任务 checkpoint 不存在")
    checkpoint_sha = sha256_file(checkpoint_path)
    declared_sha = str(row.get("task_checkpoint_sha256", ""))
    if declared_sha and checkpoint_sha != declared_sha:
        raise ValueError("任务 checkpoint SHA256 与登记值不一致")
    cost_path = run_dir / "forward_cost_summary.csv"
    cost = pd.read_csv(cost_path)
    if len(cost) != 1 or str(cost.iloc[0].get("cost_status")) != "measured":
        raise ValueError("缺少有效的前向成本测量")

    report = {
        "schema_version": 1,
        "qualified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "offline_aptos_routing_pool",
        "artifact_id": str(row["artifact_id"]),
        "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "case_set_matches_reference": True,
        "true_labels_match_reference": True,
        "split_overlap_detected": False,
        "task_checkpoint_sha256": checkpoint_sha,
        "cost_status": "measured",
        "performance_threshold_applied": False,
        "qualification_does_not_imply_recommendation": True,
        **prediction_evidence,
    }
    report_path = run_dir / "route_qualification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    registration["compatibility_status"] = "ready_for_pairing"
    registration["route_eligible"] = True
    registration["lifecycle_status"] = "active"
    registration["route_qualification_path"] = str(report_path)
    registration["route_qualification_status"] = "passed"
    registration.to_csv(registration_path, index=False)
    manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest.update(
        {
            "compatibility_status": "ready_for_pairing",
            "route_eligible": True,
            "route_qualification_status": "passed",
            "route_qualification_path": str(report_path),
        }
    )
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    args = parser.parse_args()
    print(qualify(args.run_dir, args.reference_predictions))


if __name__ == "__main__":
    main()
