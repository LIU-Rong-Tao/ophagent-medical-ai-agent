"""Read-only consumer contract for OphBench research pilot artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_FILES = {
    "run_manifest.json",
    "test_predictions.csv",
    "metrics.json",
    "artifact_manifest.json",
}


@dataclass(frozen=True)
class BenchmarkPilotArtifact:
    root: Path
    run_manifest: dict[str, Any]
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    artifact_manifest: dict[str, Any]

    def as_non_routable_record(self) -> dict[str, Any]:
        """Expose the pilot for inspection without promoting it to a task checkpoint."""

        return {
            "provider_id": "ophbench_research_artifact",
            "artifact_id": (
                f"{self.run_manifest['model_id']}::{self.run_manifest['checkpoint_id']}::"
                f"{self.run_manifest['dataset_id']}::pilot"
            ),
            "task_id": self.run_manifest["task_id"],
            "dataset_id": self.run_manifest["dataset_id"],
            "label_space": self.run_manifest["label_space"],
            "protocol_version": self.run_manifest["protocol_version"],
            "evaluation_role": self.run_manifest["evaluation_role"],
            "research_claim_status": self.run_manifest["research_claim_status"],
            "task_checkpoint": False,
            "task_inference_ready": False,
            "route_eligible": False,
            "cost_status": "not_applicable",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_pilot(
    root: Path | str,
    *,
    expected_task_id: str | None = None,
    expected_label_space: str | None = None,
) -> BenchmarkPilotArtifact:
    directory = Path(root)
    missing = sorted(name for name in REQUIRED_FILES if not (directory / name).is_file())
    if missing:
        raise ValueError(f"OphBench pilot 缺少标准文件：{missing}")
    run_manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    artifact_manifest = json.loads(
        (directory / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    required_manifest = {
        "protocol_version",
        "evaluation_role",
        "research_claim_status",
        "task_id",
        "dataset_id",
        "label_space",
        "model_id",
        "checkpoint_id",
        "checkpoint_sha256",
        "split_manifest_sha256",
        "test_used_for_selection",
    }
    absent = sorted(required_manifest - set(run_manifest))
    if absent:
        raise ValueError(f"run_manifest 缺少字段：{absent}")
    if str(run_manifest["protocol_version"]) != "0.1":
        raise ValueError("当前仅支持 Frozen Feature Transfer protocol_version=0.1")
    if run_manifest["evaluation_role"] != "pilot_protocol_validation":
        raise ValueError("该产物不是协议 pilot")
    if run_manifest["research_claim_status"] != "not_for_scientific_comparison":
        raise ValueError("pilot 必须明确禁止科研比较声明")
    if run_manifest["test_used_for_selection"] is not False:
        raise ValueError("检测到 test 参与模型选择")
    if expected_task_id and run_manifest["task_id"] != expected_task_id:
        raise ValueError("task_id 与消费任务不一致")
    if expected_label_space and run_manifest["label_space"] != expected_label_space:
        raise ValueError("label_space 与消费任务不一致")
    declared_artifacts = {
        str(item["name"]): str(item["sha256"])
        for item in artifact_manifest.get("artifacts", [])
        if isinstance(item, dict) and "name" in item and "sha256" in item
    }
    for name in ("run_manifest.json", "test_predictions.csv", "metrics.json"):
        if declared_artifacts.get(name) != _sha256(directory / name):
            raise ValueError(f"artifact SHA256 不匹配：{name}")
    predictions = pd.read_csv(directory / "test_predictions.csv")
    required_columns = {"image_key", "true_label", "pred_label"}
    if not required_columns.issubset(predictions.columns):
        raise ValueError("test_predictions 缺少 sample key 或标签字段")
    if predictions.empty or predictions["image_key"].astype(str).duplicated().any():
        raise ValueError("image_key 为空或重复")
    probability_columns = sorted(
        (column for column in predictions if column.startswith("prob_")),
        key=lambda value: int(value.removeprefix("prob_")),
    )
    if probability_columns != [f"prob_{index}" for index in range(5)]:
        raise ValueError("概率列必须连续覆盖 prob_0..prob_4")
    probabilities = predictions[probability_columns].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("预测概率包含非法值或行和不为 1")
    return BenchmarkPilotArtifact(
        root=directory,
        run_manifest=run_manifest,
        metrics=metrics,
        predictions=predictions,
        artifact_manifest=artifact_manifest,
    )
