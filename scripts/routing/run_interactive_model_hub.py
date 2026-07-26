#!/usr/bin/env python3
"""生成 v0.8.6 Model Hub、受控路由 replay 与病例级轨迹。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The repository-root bootstrap above must run before these imports.
from scripts.routing.timm_adapter_runtime import normalize_prediction_frame  # noqa: E402
from scripts.routing.model_metadata import normalized_model_metadata  # noqa: E402


VALID_STAGES = {"model_hub", "pairing", "report", "all"}
OUTPUT_NAMES = (
    "model_hub_snapshot.csv",
    "pairing_results.csv",
    "case_routing_trace.csv",
    "run_config.yaml",
    "artifact_manifest.csv",
    "candidate_ranking.csv",
    "candidate_selection.csv",
    "report.html",
)


class ModelHubError(RuntimeError):
    """v0.8.6 配置、输入或输出错误。"""


class PairingSkip(ModelHubError):
    """可解释的 pairing 跳过状态。"""

    def __init__(self, status: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.status = status
        self.details = details


@dataclass(frozen=True)
class ProtocolResult:
    output_dir: Path
    stage: str
    files: list[Path]


@dataclass
class PairingContext:
    pairing: pd.Series
    scout_ids: list[str]
    primary_scout_id: str
    expert_id: str
    scouts: dict[str, pd.DataFrame]
    expert: pd.DataFrame
    n_overlap: int
    overlap_rate: float


HUB_COLUMNS = [
    "model_id",
    "task_id",
    "dataset_id",
    "dataset_display_name",
    "dataset_source",
    "dataset_url",
    "provenance_status",
    "disease_family",
    "label_space",
    "n_classes",
    "split",
    "artifact_id",
    "model_family",
    "architecture",
    "pretraining_source",
    "checkpoint_path",
    "checkpoint_status",
    "checkpoint_mb",
    "parameter_count",
    "trainable_parameter_count",
    "role_candidates",
    "source_version",
    "prediction_source",
    "prediction_path",
    "baseline_source",
    "accuracy",
    "macro_f1",
    "qwk",
    "qwk_status",
    "forward_cost_ms_per_image",
    "cost_scope",
    "cost_status",
    "adapter_status",
    "onboarding_status",
    "compatibility_status",
    "adapter_type",
    "validation_prediction_path",
    "test_prediction_path",
    "checkpoint_sha256",
    "current_run_reproducible",
    "validation_selection_eligible",
    "qualification_status",
    "route_eligible",
    "task_inference_ready",
    "offline_evaluation_eligible",
    "cpu_postprocess_ms_per_image",
    "cpu_postprocess_status",
    "excluded_case_count",
    "excluded_image_keys",
    "notes",
]


PAIRING_COLUMNS = [
    "evaluation_kind",
    "pairing_id",
    "task_id",
    "scout_artifact_ids",
    "primary_scout_artifact_id",
    "expert_artifact_id",
    "routing_policy",
    "prediction_source_mode",
    "result_semantics",
    "requested_budget",
    "selected_n",
    "realized_budget",
    "n_scout",
    "n_expert",
    "n_overlap",
    "overlap_rate",
    "n_reviewed",
    "n_auto",
    "expert_call_rate",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "qwk",
    "qwk_status",
    "scout_only_accuracy",
    "expert_only_accuracy",
    "scout_only_macro_f1",
    "expert_only_macro_f1",
    "expert_only_qwk",
    "dangerous_total",
    "dangerous_selected",
    "dangerous_corrected",
    "dangerous_introduced",
    "net_dangerous_reduction",
    "residual_dangerous_count",
    "dangerous_recall_at_k",
    "dangerous_precision_at_k",
    "vtdr_miss_count",
    "severe_grade_error_count",
    "replay_mode",
    "cost_mode",
    "scout_cost_sum_ms_per_image",
    "scout_parallel_scenario_ms_per_image",
    "expert_cost_ms_per_image",
    "estimated_total_compute_ms_per_image",
    "estimated_online_sequential_latency_ms_per_image",
    "estimated_parallel_latency_ms_per_image",
    "parallel_cost_status",
    "cost_reduction_vs_expert_only",
    "status",
    "notes",
]


TRACE_COLUMNS = [
    "pairing_id",
    "task_id",
    "image_key",
    "image_path",
    "true_label",
    "scout_artifact_ids",
    "primary_scout_artifact_id",
    "expert_artifact_id",
    "routing_policy",
    "requested_budget",
    "realized_budget",
    "scout_pred_labels",
    "scout_confidences",
    "scout_entropies",
    "scout_margins",
    "scout_disagreement",
    "routing_score",
    "review_rank",
    "is_reviewed_by_expert",
    "expert_pred_label",
    "final_pred_label",
    "final_source",
    "was_scout_correct",
    "was_expert_correct",
    "was_final_correct",
    "notes",
]


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - 由部署环境决定
            raise ModelHubError("读取 YAML 配置需要 PyYAML") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ModelHubError("protocol 配置根节点必须是 mapping")
    return data


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def truthy(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_list(value: Any) -> list[str]:
    return [item.strip() for item in clean_text(value).split("|") if item.strip()]


def parse_budgets(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = parse_list(value)
    budgets = [float(item) for item in raw]
    if not budgets or any(item < 0 or item > 1 for item in budgets):
        raise ModelHubError(f"budget 必须位于 [0, 1]：{value}")
    return sorted(dict.fromkeys(budgets))


def resolve_path(value: Any, *, config_dir: Path | None = None) -> Path:
    text = clean_text(value)
    if not text:
        return Path()
    marker = "ophagent-medical-ai-agent/"
    normalized = text.replace("\\", "/")
    if marker in normalized:
        candidate = REPO_ROOT / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    path = Path(text)
    if path.is_absolute():
        return path
    root_candidate = REPO_ROOT / path
    if root_candidate.exists() or config_dir is None:
        return root_candidate
    return config_dir / path


def read_csv(path_value: Any, *, config_dir: Path | None = None, required: bool = True) -> pd.DataFrame:
    if not clean_text(path_value):
        if required:
            raise ModelHubError("CSV 路径不能为空")
        return pd.DataFrame()
    path = resolve_path(path_value, config_dir=config_dir)
    if not path.exists():
        if required:
            raise ModelHubError(f"找不到 CSV：{path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = np.nan
    output[columns].to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_key(task_id: str, artifact_id: str) -> tuple[str, str]:
    return clean_text(task_id), clean_text(artifact_id)


def inventory_model_metadata(
    inventory_row: pd.Series | None,
    *,
    artifact_id: str,
    architecture: str,
    config_dir: Path,
    registry_row: pd.Series | None = None,
) -> dict[str, str]:
    explicit_source = clean_text(
        metric_value(
            registry_row,
            "pretraining_source",
            metric_value(inventory_row, "pretraining_source"),
        )
    )
    training_config: dict[str, Any] = {}
    config_value = clean_text(
        metric_value(registry_row, "config_path", metric_value(inventory_row, "config_path"))
    )
    if config_value:
        config_path = resolve_path(config_value, config_dir=config_dir)
        if config_path.is_file():
            try:
                training_config = load_yaml_or_json(config_path)
            except (OSError, ModelHubError):
                training_config = {}
    return normalized_model_metadata(
        artifact_id,
        architecture,
        pretraining_source=explicit_source,
        training_config=training_config,
    )


def find_row(frame: pd.DataFrame, *, task_id: str, artifact_id: str) -> pd.Series | None:
    if frame.empty:
        return None
    matches = frame.copy()
    if "task_id" in matches.columns:
        matches = matches.loc[matches["task_id"].astype(str) == task_id]
    id_columns = [column for column in ("artifact_id", "name") if column in matches.columns]
    if not id_columns:
        return None
    mask = pd.Series(False, index=matches.index)
    for column in id_columns:
        mask |= matches[column].astype(str) == artifact_id
    matches = matches.loc[mask]
    return None if matches.empty else matches.iloc[0]


def task_metadata(tasks: pd.DataFrame, task_id: str) -> dict[str, Any]:
    matches = tasks.loc[tasks["task_id"].astype(str) == task_id]
    if matches.empty:
        raise ModelHubError(f"task registry 缺少任务：{task_id}")
    row = matches.iloc[0]
    return {
        "dataset_id": clean_text(row.get("dataset_id")),
        "dataset_display_name": clean_text(row.get("dataset_display_name", row.get("dataset_id"))),
        "dataset_source": clean_text(row.get("dataset_source")),
        "dataset_url": clean_text(row.get("dataset_url")),
        "provenance_status": clean_text(row.get("provenance_status", "unverified")),
        "disease_family": clean_text(row.get("disease_family")),
        "label_space": clean_text(row.get("label_space")),
        "label_structure": clean_text(row.get("label_structure", "nominal")),
        "n_classes": int(row.get("num_classes")),
    }


def prediction_metrics(
    path: Path,
    *,
    n_classes: int,
    qwk_enabled: bool,
    excluded_image_keys: set[str] | None = None,
) -> dict[str, Any]:
    frame = normalize_and_validate_prediction(
        path,
        n_classes=n_classes,
        excluded_image_keys=excluded_image_keys,
    )
    truth = frame["true_label"].astype(int)
    prediction = frame["pred_label"].astype(int)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "qwk": (
            float(cohen_kappa_score(truth, prediction, weights="quadratic"))
            if qwk_enabled
            else np.nan
        ),
        "qwk_status": "computed" if qwk_enabled else "not_enabled_for_task",
    }


def metric_value(row: pd.Series | None, name: str, fallback: Any = np.nan) -> Any:
    if row is None or name not in row or clean_text(row.get(name)) == "":
        return fallback
    return row.get(name)


def merge_roles(*values: Any) -> str:
    roles: list[str] = []
    for value in values:
        for role in parse_list(value):
            if role not in roles:
                roles.append(role)
    return "|".join(roles)


def load_case_exclusions(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> dict[tuple[str, str], set[str]]:
    registry_value = clean_text(config.get("case_exclusion_registry"))
    if not registry_value:
        return {}
    exclusions = read_csv(registry_value, config_dir=config_path.parent)
    required = {"task_id", "split", "image_key"}
    missing = required - set(exclusions.columns)
    if missing:
        raise ModelHubError(
            "case exclusion registry missing columns: " + ", ".join(sorted(missing))
        )
    grouped: dict[tuple[str, str], set[str]] = {}
    for _, row in exclusions.iterrows():
        if "enabled" in exclusions.columns and not truthy(row.get("enabled")):
            continue
        task_id = clean_text(row["task_id"])
        split = clean_text(row["split"])
        image_key = clean_text(row["image_key"])
        if not task_id or not split or not image_key:
            raise ModelHubError("case exclusion registry contains an empty key")
        grouped.setdefault((task_id, split), set()).add(image_key)
    return grouped


def build_prediction_asset_hub(config: dict[str, Any], *, config_path: Path) -> pd.DataFrame:
    """Register frozen probability packages as first-class Model Hub assets."""
    tasks = read_csv(config["task_registry"], config_dir=config_path.parent)
    assets = read_csv(config["prediction_asset_registry"], config_dir=config_path.parent)
    required = {
        "task_id", "artifact_id", "model_family", "architecture", "adapter_type",
        "validation_prediction_path", "test_prediction_path", "checkpoint_sha256",
        "preprocessing_id", "forward_cost_ms_per_image", "cost_scope",
        "current_run_reproducible", "validation_selection_eligible", "route_eligible",
    }
    missing = required - set(assets.columns)
    if missing:
        raise ModelHubError("prediction asset registry missing columns: " + ", ".join(sorted(missing)))
    selection_split = clean_text(config.get("selection_split", "val")) or "val"
    qwk_tasks = set(config.get("qwk_enabled_tasks", []))
    exclusions = load_case_exclusions(config, config_path=config_path)
    rows: list[dict[str, Any]] = []
    for _, asset in assets.iterrows():
        task_id = clean_text(asset["task_id"])
        artifact_id = clean_text(asset["artifact_id"])
        meta = task_metadata(tasks, task_id)
        validation_path = resolve_path(asset["validation_prediction_path"], config_dir=config_path.parent)
        test_path = resolve_path(asset["test_prediction_path"], config_dir=config_path.parent)
        prediction_path = validation_path if selection_split == "val" else test_path
        excluded_image_keys = exclusions.get((task_id, selection_split), set())
        if not prediction_path.exists():
            raise ModelHubError(f"prediction asset missing: {prediction_path}")
        metrics = prediction_metrics(
            prediction_path,
            n_classes=meta["n_classes"],
            qwk_enabled=task_id in qwk_tasks,
            excluded_image_keys=excluded_image_keys,
        )
        route_eligible = truthy(asset["route_eligible"])
        validation_eligible = truthy(asset["validation_selection_eligible"])
        rows.append(
            {
                "model_id": f"{task_id}::{artifact_id}",
                "task_id": task_id,
                **meta,
                "split": selection_split,
                "artifact_id": artifact_id,
                "model_family": clean_text(asset["model_family"]),
                "architecture": clean_text(asset["architecture"]),
                "pretraining_source": clean_text(asset.get("pretraining_source", "frozen replay asset")),
                "checkpoint_path": clean_text(asset.get("checkpoint_path", "")),
                "checkpoint_status": "verified_historical_replay",
                "checkpoint_mb": np.nan,
                "parameter_count": np.nan,
                "trainable_parameter_count": np.nan,
                "role_candidates": clean_text(asset.get("role_candidates", "scout|expert")),
                "source_version": clean_text(asset.get("source_version", "h100_replay")),
                "prediction_source": "prediction_asset",
                "prediction_path": str(prediction_path),
                "baseline_source": str(prediction_path),
                **metrics,
                "forward_cost_ms_per_image": metric_value(asset, "forward_cost_ms_per_image"),
                "cost_scope": clean_text(asset["cost_scope"]),
                "cost_status": clean_text(asset.get("cost_status", "measured")),
                "adapter_status": "prediction_asset",
                "onboarding_status": "registered_frozen_probability_asset",
                "compatibility_status": "ready_for_pairing" if validation_eligible else "not_selection_eligible",
                "adapter_type": clean_text(asset["adapter_type"]),
                "validation_prediction_path": str(validation_path),
                "test_prediction_path": str(test_path),
                "checkpoint_sha256": clean_text(asset["checkpoint_sha256"]),
                "current_run_reproducible": truthy(asset["current_run_reproducible"]),
                "validation_selection_eligible": validation_eligible,
                "qualification_status": clean_text(asset.get("qualification_status", "replay_candidate")),
                "route_eligible": route_eligible,
                "task_inference_ready": False,
                "offline_evaluation_eligible": validation_eligible,
                "cpu_postprocess_ms_per_image": metric_value(asset, "cpu_postprocess_ms_per_image"),
                "cpu_postprocess_status": clean_text(asset.get("cpu_postprocess_status", "not_applicable")),
                "excluded_case_count": len(excluded_image_keys),
                "excluded_image_keys": "|".join(sorted(excluded_image_keys)),
                "notes": clean_text(asset.get("notes", "")),
            }
        )
    return pd.DataFrame(rows).sort_values(["task_id", "artifact_id"]).reset_index(drop=True)


def build_model_hub(config: dict[str, Any], *, config_path: Path) -> pd.DataFrame:
    if clean_text(config.get("prediction_asset_registry")):
        return build_prediction_asset_hub(config, config_path=config_path)
    config_dir = config_path.parent
    tasks = read_csv(config["task_registry"], config_dir=config_dir)
    registered = read_csv(config.get("registered_models", ""), config_dir=config_dir, required=False)
    inventory = read_csv(config.get("model_inventory", ""), config_dir=config_dir, required=False)
    legacy_baselines = read_csv(
        config.get("legacy_model_baselines", ""), config_dir=config_dir, required=False
    )
    adapter_dir = resolve_path(config["v085c_output_dir"], config_dir=config_dir)
    jobs = read_csv(adapter_dir / "adapter_job_summary.csv")
    onboarded = read_csv(adapter_dir / "onboarded_models.csv")
    baselines = read_csv(adapter_dir / "model_baselines_from_adapters.csv")
    costs = read_csv(adapter_dir / "forward_cost_summary_from_adapters.csv")
    replays = read_csv(config["v085c_replay_protocols"], config_dir=config_dir)
    qwk_tasks = set(config.get("qwk_enabled_tasks", []))
    excluded_artifact_ids = {clean_text(value) for value in config.get("excluded_artifact_ids", [])}
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    for _, item in onboarded.iterrows():
        task_id = clean_text(item["task_id"])
        artifact_id = clean_text(item["artifact_id"])
        if artifact_id in excluded_artifact_ids:
            continue
        meta = task_metadata(tasks, task_id)
        job = find_row(jobs, task_id=task_id, artifact_id=artifact_id)
        baseline = find_row(baselines, task_id=task_id, artifact_id=artifact_id)
        cost = find_row(costs, task_id=task_id, artifact_id=artifact_id)
        registry = find_row(registered, task_id=task_id, artifact_id=artifact_id)
        inv = find_row(inventory, task_id=task_id, artifact_id=artifact_id)
        architecture = clean_text(
            metric_value(inv, "architecture", metric_value(inv, "backbone", artifact_id))
        )
        model_meta = inventory_model_metadata(
            inv,
            artifact_id=artifact_id,
            architecture=architecture,
            config_dir=config_dir,
            registry_row=registry,
        )
        prediction_path = resolve_path(item["predictions_path"], config_dir=config_dir)
        role = merge_roles(metric_value(registry, "role_candidates", "scout"), "adapter_scout")
        qwk_status = clean_text(metric_value(baseline, "qwk_status")) or (
            "computed" if task_id in qwk_tasks else "not_enabled_for_task"
        )
        cost_value = metric_value(cost, "median_ms_per_image", metric_value(cost, "mean_ms_per_image"))
        rows[model_key(task_id, artifact_id)] = {
            "model_id": f"{task_id}::{artifact_id}",
            "task_id": task_id,
            **meta,
            "split": clean_text(metric_value(baseline, "split", "test")) or "test",
            "artifact_id": artifact_id,
            "model_family": model_meta["model_family"],
            "architecture": model_meta["architecture"],
            "pretraining_source": model_meta["pretraining_source"],
            "checkpoint_path": clean_text(metric_value(inv, "checkpoint_path")),
            "checkpoint_status": clean_text(metric_value(inv, "checkpoint_status", "unknown")),
            "checkpoint_mb": metric_value(cost, "checkpoint_mb"),
            "parameter_count": metric_value(cost, "parameter_count"),
            "trainable_parameter_count": metric_value(cost, "trainable_parameter_count"),
            "role_candidates": role,
            "source_version": "v0.8.5c",
            "prediction_source": "adapter",
            "prediction_path": str(prediction_path),
            "baseline_source": str(adapter_dir / "model_baselines_from_adapters.csv"),
            "accuracy": metric_value(baseline, "accuracy", metric_value(job, "accuracy")),
            "macro_f1": metric_value(baseline, "macro_f1", metric_value(job, "macro_f1")),
            "qwk": metric_value(baseline, "qwk", metric_value(job, "qwk")),
            "qwk_status": qwk_status,
            "forward_cost_ms_per_image": cost_value,
            "cost_scope": clean_text(metric_value(cost, "cost_scope", "forward_only")),
            "cost_status": "measured" if clean_text(cost_value) else "missing_forward_cost",
            "adapter_status": clean_text(metric_value(job, "status", "completed")),
            "onboarding_status": "completed",
            "compatibility_status": "ready_for_pairing" if prediction_path.exists() else "missing_predictions",
            "notes": clean_text(metric_value(job, "notes")),
        }

    for _, replay in replays.iterrows():
        if "enabled" in replay and not truthy(replay["enabled"]):
            continue
        task_id = clean_text(replay["task_id"])
        artifact_id = clean_text(replay["expert_artifact_id"])
        if artifact_id in excluded_artifact_ids:
            continue
        key = model_key(task_id, artifact_id)
        if key in rows:
            continue
        meta = task_metadata(tasks, task_id)
        registry = find_row(registered, task_id=task_id, artifact_id=artifact_id)
        inv = find_row(inventory, task_id=task_id, artifact_id=artifact_id)
        architecture = clean_text(
            metric_value(inv, "architecture", metric_value(inv, "backbone", artifact_id))
        )
        model_meta = inventory_model_metadata(
            inv,
            artifact_id=artifact_id,
            architecture=architecture,
            config_dir=config_dir,
            registry_row=registry,
        )
        baseline = find_row(legacy_baselines, task_id=task_id, artifact_id=artifact_id)
        prediction_path = resolve_path(replay["expert_legacy_prediction_path"], config_dir=config_dir)
        computed = (
            prediction_metrics(
                prediction_path,
                n_classes=meta["n_classes"],
                qwk_enabled=task_id in qwk_tasks,
            )
            if prediction_path.exists()
            else {}
        )
        cost_value = metric_value(baseline, "estimated_forward_ms_per_image")
        rows[key] = {
            "model_id": f"{task_id}::{artifact_id}",
            "task_id": task_id,
            **meta,
            "split": "test",
            "artifact_id": artifact_id,
            "model_family": model_meta["model_family"],
            "architecture": model_meta["architecture"],
            "pretraining_source": model_meta["pretraining_source"],
            "checkpoint_path": clean_text(metric_value(inv, "checkpoint_path")),
            "checkpoint_status": clean_text(metric_value(inv, "checkpoint_status", "unknown")),
            "checkpoint_mb": metric_value(baseline, "checkpoint_mb", metric_value(inv, "checkpoint_mb")),
            "parameter_count": metric_value(baseline, "parameter_count", metric_value(inv, "parameter_count")),
            "trainable_parameter_count": metric_value(
                baseline,
                "trainable_parameter_count",
                metric_value(inv, "trainable_parameter_count"),
            ),
            "role_candidates": merge_roles(metric_value(registry, "role_candidates", "expert"), "legacy_expert"),
            "source_version": clean_text(metric_value(registry, "source_version", "legacy")),
            "prediction_source": "legacy",
            "prediction_path": str(prediction_path),
            "baseline_source": str(resolve_path(config.get("legacy_model_baselines", ""))),
            "accuracy": metric_value(baseline, "accuracy", computed.get("accuracy")),
            "macro_f1": metric_value(baseline, "macro_f1", computed.get("macro_f1")),
            "qwk": metric_value(baseline, "qwk", computed.get("qwk")),
            "qwk_status": computed.get("qwk_status", "not_reported"),
            "forward_cost_ms_per_image": cost_value,
            "cost_scope": clean_text(metric_value(baseline, "timing_scope", "forward_only")),
            "cost_status": "measured" if clean_text(cost_value) else "missing_forward_cost",
            "adapter_status": clean_text(metric_value(inv, "adapter_status", "needs_loader_audit")),
            "onboarding_status": clean_text(metric_value(inv, "onboarding_status", "legacy_replay_only")),
            "compatibility_status": "ready_for_pairing" if prediction_path.exists() else "missing_predictions",
            "notes": "冻结 legacy prediction；未伪装为 adapter completed",
        }

    for source in (registered, inventory):
        if source.empty:
            continue
        for _, item in source.iterrows():
            if "enabled" in item and clean_text(item.get("enabled")) and not truthy(item.get("enabled")):
                continue
            task_id = clean_text(item.get("task_id"))
            artifact_id = clean_text(item.get("artifact_id"))
            if (
                not task_id
                or not artifact_id
                or artifact_id in excluded_artifact_ids
                or model_key(task_id, artifact_id) in rows
            ):
                continue
            meta = task_metadata(tasks, task_id)
            architecture = clean_text(item.get("architecture", item.get("backbone", artifact_id)))
            model_meta = inventory_model_metadata(
                item,
                artifact_id=artifact_id,
                architecture=architecture,
                config_dir=config_dir,
            )
            prediction_value = item.get("legacy_prediction_path", item.get("prediction_source", ""))
            prediction_path = resolve_path(prediction_value, config_dir=config_dir)
            has_prediction = bool(clean_text(prediction_value)) and prediction_path.exists()
            rows[model_key(task_id, artifact_id)] = {
                "model_id": f"{task_id}::{artifact_id}",
                "task_id": task_id,
                **meta,
                "split": "test",
                "artifact_id": artifact_id,
                "model_family": model_meta["model_family"],
                "architecture": model_meta["architecture"],
                "pretraining_source": model_meta["pretraining_source"],
                "checkpoint_path": clean_text(item.get("checkpoint_path")),
                "checkpoint_status": clean_text(item.get("checkpoint_status", "unknown")),
                "checkpoint_mb": metric_value(item, "checkpoint_mb"),
                "parameter_count": metric_value(item, "parameter_count"),
                "trainable_parameter_count": metric_value(item, "trainable_parameter_count"),
                "role_candidates": clean_text(item.get("role_candidates")),
                "source_version": clean_text(item.get("source_version", item.get("legacy_source", "legacy"))),
                "prediction_source": "legacy" if has_prediction else "missing",
                "prediction_path": str(prediction_path) if has_prediction else "",
                "baseline_source": clean_text(item.get("legacy_baseline_path", item.get("baseline_source", ""))),
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "qwk": np.nan,
                "qwk_status": "not_reported",
                "forward_cost_ms_per_image": np.nan,
                "cost_scope": "",
                "cost_status": "missing_forward_cost",
                "adapter_status": clean_text(item.get("adapter_status", "unknown")),
                "onboarding_status": clean_text(item.get("onboarding_status", "incomplete")),
                "compatibility_status": "ready_for_pairing" if has_prediction else "incomplete",
                "notes": clean_text(item.get("notes")),
            }

    hub = pd.DataFrame(rows.values())
    if not hub.empty:
        metadata = hub.apply(
            lambda row: normalized_model_metadata(
                clean_text(row.get("artifact_id")),
                clean_text(row.get("architecture"))
                or clean_text(row.get("model_family"))
                or clean_text(row.get("artifact_id")),
                pretraining_source=clean_text(row.get("pretraining_source")),
            ),
            axis=1,
        )
        hub["model_family"] = metadata.map(lambda item: item["model_family"])
        hub["architecture"] = metadata.map(lambda item: item["architecture"])
        hub["pretraining_source"] = metadata.map(lambda item: item["pretraining_source"])
    return hub.sort_values(["task_id", "artifact_id"]).reset_index(drop=True)


def normalize_and_validate_prediction(
    path: Path,
    *,
    n_classes: int,
    excluded_image_keys: set[str] | None = None,
) -> pd.DataFrame:
    try:
        frame = normalize_prediction_frame(path, num_classes=n_classes)
    except Exception as exc:
        raw = pd.read_csv(path)
        aliases = {"case_id": "image_key", "y_true": "true_label", "y_pred": "pred_label"}
        frame = raw.rename(columns={source: target for source, target in aliases.items() if source in raw})
        required = {"image_key", "true_label", "pred_label", *[f"prob_{index}" for index in range(n_classes)]}
        if not required.issubset(frame.columns):
            raise PairingSkip("skipped_invalid_prediction_schema", f"invalid prediction package {path}: {exc}") from exc
        frame = frame.copy()
        frame["image_key"] = frame["image_key"].astype(str)
        frame["image_path"] = frame.get("image_path", "")
        probabilities = frame[[f"prob_{index}" for index in range(n_classes)]].astype(float).to_numpy()
        ordered = np.sort(probabilities, axis=1)
        frame["confidence"] = probabilities.max(axis=1)
        frame["margin"] = ordered[:, -1] - ordered[:, -2]
        frame["entropy"] = -(
            probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))
        ).sum(axis=1) / np.log(n_classes)
    if frame["image_key"].duplicated().any():
        raise PairingSkip("skipped_incompatible_image_keys", f"prediction image_key 重复：{path}")
    excluded_image_keys = excluded_image_keys or set()
    if excluded_image_keys:
        observed = set(frame["image_key"].astype(str))
        missing_exclusions = excluded_image_keys - observed
        if missing_exclusions:
            raise PairingSkip(
                "skipped_incompatible_image_keys",
                f"prediction 缺少待排除病例：{sorted(missing_exclusions)[:5]}",
            )
        frame = frame.loc[
            ~frame["image_key"].astype(str).isin(excluded_image_keys)
        ].copy()
        if frame.empty:
            raise PairingSkip("skipped_incompatible_image_keys", "排除后 prediction 为空")
    prob_cols = [f"prob_{index}" for index in range(n_classes)]
    probabilities = frame[prob_cols].astype(float).to_numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise PairingSkip("skipped_invalid_prediction_schema", f"prediction 概率无效：{path}")
    if not np.array_equal(probabilities.argmax(axis=1), frame["pred_label"].astype(int).to_numpy()):
        raise PairingSkip("skipped_invalid_prediction_schema", f"pred_label 与概率 argmax 不一致：{path}")
    return frame.sort_values("image_key").reset_index(drop=True)


def get_hub_row(hub: pd.DataFrame, task_id: str, artifact_id: str) -> pd.Series:
    rows = hub.loc[
        (hub["task_id"].astype(str) == task_id)
        & (hub["artifact_id"].astype(str) == artifact_id)
    ]
    if rows.empty:
        cross_task = hub.loc[hub["artifact_id"].astype(str) == artifact_id]
        if not cross_task.empty:
            raise PairingSkip(
                "skipped_incompatible_task",
                f"{artifact_id} 不属于任务 {task_id}",
            )
        raise PairingSkip("skipped_missing_model", f"Model Hub 缺少模型：{artifact_id}")
    return rows.iloc[0]


def prepare_pairing(pairing: pd.Series, hub: pd.DataFrame) -> PairingContext:
    task_id = clean_text(pairing["task_id"])
    scout_ids = parse_list(pairing["scout_artifact_ids"])
    primary = clean_text(pairing["primary_scout_artifact_id"])
    expert_id = clean_text(pairing["expert_artifact_id"])
    if not scout_ids or primary not in scout_ids:
        raise PairingSkip(
            "skipped_invalid_protocol",
            "primary_scout_artifact_id 必须显式包含在 scout_artifact_ids 中",
        )
    model_rows = [get_hub_row(hub, task_id, artifact) for artifact in [*scout_ids, expert_id]]
    label_spaces = {clean_text(row["label_space"]) for row in model_rows}
    n_classes_set = {int(row["n_classes"]) for row in model_rows}
    datasets = {clean_text(row["dataset_id"]) for row in model_rows}
    splits = {clean_text(row["split"]) for row in model_rows}
    exclusion_sets = {
        tuple(sorted(parse_list(row.get("excluded_image_keys", ""))))
        for row in model_rows
    }
    if len(label_spaces) != 1 or len(n_classes_set) != 1:
        raise PairingSkip("skipped_incompatible_label_space", "模型标签空间或类别数不一致")
    if len(datasets) != 1 or len(splits) != 1:
        raise PairingSkip("skipped_incompatible_dataset_split", "模型数据集或 split 不一致")
    if len(exclusion_sets) != 1:
        raise PairingSkip("skipped_incompatible_image_keys", "模型病例排除规则不一致")
    missing = [row["artifact_id"] for row in model_rows if not clean_text(row["prediction_path"])]
    if missing:
        raise PairingSkip(
            "skipped_missing_predictions",
            "缺少 prediction：" + "|".join(map(str, missing)),
        )
    n_classes = next(iter(n_classes_set))
    excluded_image_keys = set(next(iter(exclusion_sets)))
    frames = {
        clean_text(row["artifact_id"]): normalize_and_validate_prediction(
            resolve_path(row["prediction_path"]),
            n_classes=n_classes,
            excluded_image_keys=excluded_image_keys,
        )
        for row in model_rows
    }
    key_sets = {artifact: set(frame["image_key"].astype(str)) for artifact, frame in frames.items()}
    overlap = set.intersection(*key_sets.values())
    maximum = max(len(keys) for keys in key_sets.values())
    overlap_rate = len(overlap) / maximum if maximum else 0.0
    if any(keys != overlap for keys in key_sets.values()):
        raise PairingSkip(
            "skipped_incompatible_image_keys",
            "image_key 集合不完全一致，禁止静默丢弃病例",
            n_overlap=len(overlap),
            overlap_rate=overlap_rate,
        )
    ordered_keys = sorted(overlap)
    aligned = {
        artifact: frame.set_index("image_key").loc[ordered_keys].reset_index()
        for artifact, frame in frames.items()
    }
    reference = aligned[primary]["true_label"].astype(int).to_numpy()
    if any(
        not np.array_equal(reference, frame["true_label"].astype(int).to_numpy())
        for frame in aligned.values()
    ):
        raise PairingSkip("skipped_incompatible_true_labels", "相同 image_key 的真实标签不一致")
    return PairingContext(
        pairing=pairing,
        scout_ids=scout_ids,
        primary_scout_id=primary,
        expert_id=expert_id,
        scouts={artifact: aligned[artifact] for artifact in scout_ids},
        expert=aligned[expert_id],
        n_overlap=len(overlap),
        overlap_rate=overlap_rate,
    )


def routing_ranking(context: PairingContext, policy: str) -> pd.DataFrame:
    primary = context.scouts[context.primary_scout_id]
    score_frame = pd.DataFrame({"image_key": primary["image_key"].astype(str)})
    predictions: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    for artifact in context.scout_ids:
        frame = context.scouts[artifact]
        predictions.append(frame["pred_label"].astype(int).to_numpy())
        uncertainties.append(1.0 - frame["confidence"].astype(float).to_numpy())
    disagreement = np.ptp(np.column_stack(predictions), axis=1) > 0
    if len(context.scout_ids) == 1:
        if policy == "low_confidence":
            score = 1.0 - primary["confidence"].astype(float).to_numpy()
        elif policy == "low_margin":
            score = 1.0 - primary["margin"].astype(float).to_numpy()
        elif policy == "high_entropy":
            score = primary["entropy"].astype(float).to_numpy()
        elif policy == "severe_probability_mass":
            probability_columns = [
                column for column in primary.columns if column.startswith("prob_")
            ]
            probability_columns.sort(key=lambda column: int(column.removeprefix("prob_")))
            if len(probability_columns) < 4:
                raise PairingSkip("skipped_unsupported_policy", "severe probability mass requires ordinal classes")
            score = primary[probability_columns[3:]].astype(float).sum(axis=1).to_numpy()
        else:
            raise PairingSkip("skipped_unsupported_policy", f"单 Scout 不支持策略：{policy}")
    else:
        mean_uncertainty = np.column_stack(uncertainties).mean(axis=1)
        if policy == "mean_uncertainty":
            score = mean_uncertainty
        elif policy == "disagreement_then_uncertainty":
            score = disagreement.astype(float) * 2.0 + mean_uncertainty
        else:
            raise PairingSkip("skipped_unsupported_policy", f"多 Scout 不支持策略：{policy}")
    score_frame["routing_score"] = score
    score_frame["scout_disagreement"] = disagreement
    ranked = score_frame.sort_values(
        ["routing_score", "image_key"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["review_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def selected_n_for_budget(n_cases: int, budget: float) -> int:
    return min(n_cases, max(0, int(round(n_cases * float(budget)))))


def risk_proxy_summary(
    truth: np.ndarray,
    scout: np.ndarray,
    final: np.ndarray,
    selected: np.ndarray,
    *,
    undergrading_threshold: int = 3,
    enabled: bool = True,
) -> dict[str, int | float | str]:
    """Label-defined ordered-class proxy; never a clinical-outcome inference."""
    if not enabled:
        return {
            "label_proxy_status": "not_applicable_nominal_task",
            "dangerous_total": np.nan,
            "dangerous_selected": np.nan,
            "dangerous_corrected": np.nan,
            "dangerous_introduced": np.nan,
            "net_dangerous_reduction": np.nan,
            "residual_dangerous_count": np.nan,
            "dangerous_recall_at_k": np.nan,
            "dangerous_precision_at_k": np.nan,
            "vtdr_miss_count": np.nan,
            "severe_grade_error_count": np.nan,
        }
    dangerous_scout = (truth >= undergrading_threshold) & (
        scout < undergrading_threshold
    )
    dangerous_final = (truth >= undergrading_threshold) & (
        final < undergrading_threshold
    )
    corrected = dangerous_scout & ~dangerous_final
    introduced = ~dangerous_scout & dangerous_final
    selected_dangerous = dangerous_scout & selected
    total = int(dangerous_scout.sum())
    selected_n = int(selected.sum())
    return {
        "label_proxy_status": "computed_ordinal_label_proxy",
        "dangerous_total": total,
        "dangerous_selected": int(selected_dangerous.sum()),
        "dangerous_corrected": int(corrected.sum()),
        "dangerous_introduced": int(introduced.sum()),
        "net_dangerous_reduction": int(corrected.sum() - introduced.sum()),
        "residual_dangerous_count": int(dangerous_final.sum()),
        "dangerous_recall_at_k": float(selected_dangerous.sum() / total) if total else np.nan,
        "dangerous_precision_at_k": float(selected_dangerous.sum() / selected_n) if selected_n else np.nan,
        "vtdr_miss_count": int(((truth >= 2) & (final < 2)).sum()),
        "severe_grade_error_count": int((np.abs(truth - final) >= 2).sum()),
    }


def classification_summary(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    qwk_enabled: bool,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "qwk": (
            float(cohen_kappa_score(truth, prediction, weights="quadratic"))
            if qwk_enabled
            else np.nan
        ),
        "qwk_status": "computed" if qwk_enabled else "not_enabled_for_task",
    }


def numeric_cost(hub: pd.DataFrame, task_id: str, artifact_id: str) -> float:
    row = get_hub_row(hub, task_id, artifact_id)
    try:
        value = float(row["forward_cost_ms_per_image"])
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) and value > 0 else float("nan")


def has_unmeasured_cpu_postprocess(
    hub: pd.DataFrame, task_id: str, artifact_ids: list[str]
) -> bool:
    """Avoid presenting a GPU-only component as the complete deployed cost."""
    for artifact_id in artifact_ids:
        row = get_hub_row(hub, task_id, artifact_id)
        if clean_text(row.get("cpu_postprocess_status")).lower() == "unmeasured":
            return True
    return False


def cost_summary(context: PairingContext, hub: pd.DataFrame, call_rate: float) -> dict[str, Any]:
    task_id = clean_text(context.pairing["task_id"])
    scout_costs = [numeric_cost(hub, task_id, artifact) for artifact in context.scout_ids]
    expert_cost = numeric_cost(hub, task_id, context.expert_id)
    if any(math.isnan(value) for value in scout_costs) or math.isnan(expert_cost):
        return {
            "replay_mode": "cached_prediction_replay",
            "cost_mode": "missing_forward_cost",
            "scout_cost_sum_ms_per_image": np.nan,
            "scout_parallel_scenario_ms_per_image": np.nan,
            "expert_cost_ms_per_image": expert_cost,
            "estimated_total_compute_ms_per_image": np.nan,
            "estimated_online_sequential_latency_ms_per_image": np.nan,
            "estimated_parallel_latency_ms_per_image": np.nan,
            "parallel_cost_status": "missing_forward_cost",
            "cost_reduction_vs_expert_only": np.nan,
        }
    scout_sum = float(sum(scout_costs))
    scout_parallel = float(max(scout_costs))
    if has_unmeasured_cpu_postprocess(
        hub, task_id, [*context.scout_ids, context.expert_id]
    ):
        return {
            "replay_mode": "cached_prediction_replay",
            "cost_mode": "partial_component_cost_cpu_probe_unmeasured",
            "scout_cost_sum_ms_per_image": scout_sum,
            "scout_parallel_scenario_ms_per_image": scout_parallel,
            "expert_cost_ms_per_image": expert_cost,
            "estimated_total_compute_ms_per_image": np.nan,
            "estimated_online_sequential_latency_ms_per_image": np.nan,
            "estimated_parallel_latency_ms_per_image": np.nan,
            "parallel_cost_status": "cpu_probe_unmeasured",
            "cost_reduction_vs_expert_only": np.nan,
        }
    total = scout_sum + call_rate * expert_cost
    parallel = scout_parallel + call_rate * expert_cost
    return {
        "replay_mode": "cached_prediction_replay",
        "cost_mode": "estimated_from_measured_forward_only",
        "scout_cost_sum_ms_per_image": scout_sum,
        "scout_parallel_scenario_ms_per_image": scout_parallel,
        "expert_cost_ms_per_image": expert_cost,
        "estimated_total_compute_ms_per_image": total,
        "estimated_online_sequential_latency_ms_per_image": total,
        "estimated_parallel_latency_ms_per_image": parallel,
        "parallel_cost_status": "scenario_estimate_not_measured",
        "cost_reduction_vs_expert_only": 1.0 - total / expert_cost,
    }


def evaluate_pairing(
    context: PairingContext,
    hub: pd.DataFrame,
    *,
    qwk_enabled: bool,
    trace_budgets: set[float],
    proxy_undergrading_threshold: int = 3,
    proxy_risk_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairing = context.pairing
    primary = context.scouts[context.primary_scout_id]
    truth = primary["true_label"].astype(int).to_numpy()
    primary_prediction = primary["pred_label"].astype(int).to_numpy()
    expert_prediction = context.expert["pred_label"].astype(int).to_numpy()
    scout_baseline = classification_summary(truth, primary_prediction, qwk_enabled=qwk_enabled)
    expert_baseline = classification_summary(truth, expert_prediction, qwk_enabled=qwk_enabled)
    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    policies = parse_list(pairing["routing_policies"])
    budgets = parse_budgets(pairing["budget_grid"])

    def append_control(
        *, kind: str, policy: str, budget: float, selected_mask: np.ndarray
    ) -> None:
        selected_n = int(selected_mask.sum())
        final_prediction = np.where(selected_mask, expert_prediction, primary_prediction)
        final_metrics = classification_summary(truth, final_prediction, qwk_enabled=qwk_enabled)
        risk_metrics = risk_proxy_summary(
            truth,
            primary_prediction,
            final_prediction,
            selected_mask,
            undergrading_threshold=proxy_undergrading_threshold,
            enabled=proxy_risk_enabled,
        )
        call_rate = selected_n / len(primary) if len(primary) else 0.0
        result_rows.append(
            {
                "evaluation_kind": kind,
                "pairing_id": pairing["pairing_id"],
                "task_id": pairing["task_id"],
                "scout_artifact_ids": pairing["scout_artifact_ids"],
                "primary_scout_artifact_id": context.primary_scout_id,
                "expert_artifact_id": context.expert_id,
                "routing_policy": policy,
                "prediction_source_mode": pairing["prediction_source_mode"],
                "result_semantics": pairing["result_semantics"],
                "requested_budget": budget,
                "selected_n": selected_n,
                "realized_budget": call_rate,
                "n_scout": len(primary),
                "n_expert": len(context.expert),
                "n_overlap": context.n_overlap,
                "overlap_rate": context.overlap_rate,
                "n_reviewed": selected_n,
                "n_auto": len(primary) - selected_n,
                "expert_call_rate": call_rate,
                **final_metrics,
                **risk_metrics,
                "scout_only_accuracy": scout_baseline["accuracy"],
                "expert_only_accuracy": expert_baseline["accuracy"],
                "scout_only_macro_f1": scout_baseline["macro_f1"],
                "expert_only_macro_f1": expert_baseline["macro_f1"],
                "expert_only_qwk": expert_baseline["qwk"],
                **cost_summary(context, hub, call_rate),
                "status": "completed",
                "notes": "validation-only label-defined proxy; Oracle is not deployable" if kind == "oracle" else "validation cached probability replay",
            }
        )

    append_control(kind="scout_only", policy="none", budget=0.0, selected_mask=np.zeros(len(primary), dtype=bool))
    append_control(kind="full_expert", policy="none", budget=1.0, selected_mask=np.ones(len(primary), dtype=bool))

    for policy in policies:
        ranking = routing_ranking(context, policy)
        rank_by_key = dict(zip(ranking["image_key"], ranking["review_rank"]))
        score_by_key = dict(zip(ranking["image_key"], ranking["routing_score"]))
        disagree_by_key = dict(zip(ranking["image_key"], ranking["scout_disagreement"]))
        for budget in budgets:
            selected_n = selected_n_for_budget(len(primary), budget)
            selected = set(ranking.head(selected_n)["image_key"].astype(str))
            selected_mask = primary["image_key"].astype(str).isin(selected).to_numpy()
            final_prediction = np.where(selected_mask, expert_prediction, primary_prediction)
            final_metrics = classification_summary(truth, final_prediction, qwk_enabled=qwk_enabled)
            risk_metrics = risk_proxy_summary(
                truth,
                primary_prediction,
                final_prediction,
                selected_mask,
                undergrading_threshold=proxy_undergrading_threshold,
                enabled=proxy_risk_enabled,
            )
            call_rate = selected_n / len(primary) if len(primary) else 0.0
            result_rows.append(
                {
                    "evaluation_kind": "routed",
                    "pairing_id": pairing["pairing_id"],
                    "task_id": pairing["task_id"],
                    "scout_artifact_ids": pairing["scout_artifact_ids"],
                    "primary_scout_artifact_id": context.primary_scout_id,
                    "expert_artifact_id": context.expert_id,
                    "routing_policy": policy,
                    "prediction_source_mode": pairing["prediction_source_mode"],
                    "result_semantics": pairing["result_semantics"],
                    "requested_budget": budget,
                    "selected_n": selected_n,
                    "realized_budget": call_rate,
                    "n_scout": len(primary),
                    "n_expert": len(context.expert),
                    "n_overlap": context.n_overlap,
                    "overlap_rate": context.overlap_rate,
                    "n_reviewed": selected_n,
                    "n_auto": len(primary) - selected_n,
                    "expert_call_rate": call_rate,
                    **final_metrics,
                    **risk_metrics,
                    "scout_only_accuracy": scout_baseline["accuracy"],
                    "expert_only_accuracy": expert_baseline["accuracy"],
                    "scout_only_macro_f1": scout_baseline["macro_f1"],
                    "expert_only_macro_f1": expert_baseline["macro_f1"],
                    "expert_only_qwk": expert_baseline["qwk"],
                    **cost_summary(context, hub, call_rate),
                    "status": "completed",
                    "notes": (
                        "interactive replay; not formal model selection; "
                        "100% 仅表示最终预测全部由 Expert 替换，系统成本仍包含 Scout"
                    ),
                }
            )
            if policy == policies[0]:
                seed = int(hashlib.sha256(f"{pairing['pairing_id']}:{budget}".encode()).hexdigest()[:8], 16)
                random_indices = np.random.default_rng(seed).choice(len(primary), size=selected_n, replace=False)
                random_mask = np.zeros(len(primary), dtype=bool)
                random_mask[random_indices] = True
                append_control(kind="random", policy="random_same_budget", budget=budget, selected_mask=random_mask)
                beneficial = (primary_prediction != truth) & (expert_prediction == truth)
                oracle_order = np.lexsort((primary["image_key"].astype(str).to_numpy(), ~beneficial))
                oracle_mask = np.zeros(len(primary), dtype=bool)
                oracle_mask[oracle_order[:selected_n]] = True
                append_control(kind="oracle", policy="oracle_exact_k", budget=budget, selected_mask=oracle_mask)
            if not any(abs(budget - item) < 1e-9 for item in trace_budgets):
                continue
            for index, record in primary.iterrows():
                image_key = str(record["image_key"])
                is_reviewed = image_key in selected
                scout_predictions = {
                    artifact: int(context.scouts[artifact].iloc[index]["pred_label"])
                    for artifact in context.scout_ids
                }
                scout_confidences = {
                    artifact: float(context.scouts[artifact].iloc[index]["confidence"])
                    for artifact in context.scout_ids
                }
                scout_entropies = {
                    artifact: float(context.scouts[artifact].iloc[index]["entropy"])
                    for artifact in context.scout_ids
                }
                scout_margins = {
                    artifact: float(context.scouts[artifact].iloc[index]["margin"])
                    for artifact in context.scout_ids
                }
                final_pred = int(expert_prediction[index]) if is_reviewed else int(primary_prediction[index])
                trace_rows.append(
                    {
                        "pairing_id": pairing["pairing_id"],
                        "task_id": pairing["task_id"],
                        "image_key": image_key,
                        "image_path": clean_text(record.get("image_path")),
                        "true_label": int(truth[index]),
                        "scout_artifact_ids": pairing["scout_artifact_ids"],
                        "primary_scout_artifact_id": context.primary_scout_id,
                        "expert_artifact_id": context.expert_id,
                        "routing_policy": policy,
                        "requested_budget": budget,
                        "realized_budget": call_rate,
                        "scout_pred_labels": json.dumps(scout_predictions, ensure_ascii=False),
                        "scout_confidences": json.dumps(scout_confidences, ensure_ascii=False),
                        "scout_entropies": json.dumps(scout_entropies, ensure_ascii=False),
                        "scout_margins": json.dumps(scout_margins, ensure_ascii=False),
                        "scout_disagreement": bool(disagree_by_key[image_key]),
                        "routing_score": float(score_by_key[image_key]),
                        "review_rank": int(rank_by_key[image_key]),
                        "is_reviewed_by_expert": bool(is_reviewed),
                        "expert_pred_label": int(expert_prediction[index]),
                        "final_pred_label": final_pred,
                        "final_source": "expert" if is_reviewed else "scout",
                        "was_scout_correct": bool(primary_prediction[index] == truth[index]),
                        "was_expert_correct": bool(expert_prediction[index] == truth[index]),
                        "was_final_correct": bool(final_pred == truth[index]),
                        "notes": "病例级工程 replay；真实标签仅用于研究评估界面",
                    }
                )
    return pd.DataFrame(result_rows), pd.DataFrame(trace_rows)


def skipped_pairing_row(pairing: pd.Series, error: PairingSkip) -> dict[str, Any]:
    return {
        "pairing_id": pairing.get("pairing_id", ""),
        "task_id": pairing.get("task_id", ""),
        "scout_artifact_ids": pairing.get("scout_artifact_ids", ""),
        "primary_scout_artifact_id": pairing.get("primary_scout_artifact_id", ""),
        "expert_artifact_id": pairing.get("expert_artifact_id", ""),
        "routing_policy": "",
        "prediction_source_mode": pairing.get("prediction_source_mode", ""),
        "result_semantics": pairing.get("result_semantics", "interactive_replay"),
        "requested_budget": np.nan,
        "selected_n": 0,
        "realized_budget": np.nan,
        "n_scout": error.details.get("n_scout", 0),
        "n_expert": error.details.get("n_expert", 0),
        "n_overlap": error.details.get("n_overlap", 0),
        "overlap_rate": error.details.get("overlap_rate", 0.0),
        "status": error.status,
        "notes": str(error),
    }


def expanded_pool_pairings(config: dict[str, Any], hub: pd.DataFrame) -> pd.DataFrame:
    expansion = config.get("pairing_expansion", {})
    if not expansion or not truthy(expansion.get("enabled", False)):
        return pd.DataFrame()
    task_id = clean_text(expansion["task_id"])
    available = hub.loc[
        (hub["task_id"].astype(str) == task_id)
        & hub["validation_selection_eligible"].fillna(False).astype(bool)
        & hub["compatibility_status"].astype(str).eq("ready_for_pairing"),
        "artifact_id",
    ].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    budgets = "|".join(str(value) for value in expansion["budgets"])
    single_policies = "|".join(expansion["single_scout_policies"])
    multi_policies = "|".join(expansion["multi_scout_policies"])
    result_semantics = clean_text(
        config.get("result_semantics", "validation_cached_probability_replay")
    ) or "validation_cached_probability_replay"
    for scout in available:
        for expert in available:
            if scout == expert:
                continue
            rows.append({
                "pairing_id": f"{task_id}__{scout}__to__{expert}",
                "task_id": task_id,
                "scout_artifact_ids": scout,
                "primary_scout_artifact_id": scout,
                "expert_artifact_id": expert,
                "enabled": True,
                "prediction_source_mode": "prediction_asset",
                "routing_policies": single_policies,
                "budget_grid": budgets,
                "result_semantics": result_semantics,
                "notes": "expanded from registered probability assets",
            })
    for left_index, left in enumerate(available):
        for right in available[left_index + 1 :]:
            for expert in available:
                if expert in {left, right}:
                    continue
                rows.append({
                    "pairing_id": f"{task_id}__{left}__{right}__to__{expert}",
                    "task_id": task_id,
                    "scout_artifact_ids": f"{left}|{right}",
                    "primary_scout_artifact_id": left,
                    "expert_artifact_id": expert,
                    "enabled": True,
                    "prediction_source_mode": "prediction_asset",
                    "routing_policies": multi_policies,
                    "budget_grid": budgets,
                    "result_semantics": result_semantics,
                    "notes": "expanded two-scout probability disagreement replay",
                })
    return pd.DataFrame(rows)


def evaluate_pairings(
    config: dict[str, Any],
    *,
    config_path: Path,
    hub: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairings = read_csv(config.get("pairing_protocols", ""), config_dir=config_path.parent, required=False)
    generated = expanded_pool_pairings(config, hub)
    if not generated.empty:
        pairings = pd.concat([pairings, generated], ignore_index=True)
    required = {
        "pairing_id",
        "task_id",
        "scout_artifact_ids",
        "primary_scout_artifact_id",
        "expert_artifact_id",
        "enabled",
        "prediction_source_mode",
        "routing_policies",
        "budget_grid",
        "result_semantics",
    }
    missing = required - set(pairings.columns)
    if missing:
        raise ModelHubError("pairing_protocols.csv 缺少字段：" + ", ".join(sorted(missing)))
    result_frames: list[pd.DataFrame] = []
    trace_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, Any]] = []
    qwk_tasks = set(config.get("qwk_enabled_tasks", []))
    proxy_undergrading_threshold = int(
        config.get("label_proxy_undergrading_threshold", 3)
    )
    trace_budgets = set(float(value) for value in config.get("case_trace_budgets", [0.2, 0.3, 0.5]))
    for _, pairing in pairings.iterrows():
        if not truthy(pairing["enabled"]):
            continue
        try:
            context = prepare_pairing(pairing, hub)
            results, traces = evaluate_pairing(
                context,
                hub,
                qwk_enabled=clean_text(pairing["task_id"]) in qwk_tasks,
                trace_budgets=trace_budgets,
                proxy_undergrading_threshold=proxy_undergrading_threshold,
                proxy_risk_enabled=(
                    clean_text(
                        get_hub_row(
                            hub,
                            clean_text(pairing["task_id"]),
                            context.primary_scout_id,
                        ).get("label_structure")
                    )
                    == "ordinal"
                ),
            )
            result_frames.append(results)
            trace_frames.append(traces)
        except PairingSkip as exc:
            skipped_rows.append(skipped_pairing_row(pairing, exc))
    result_parts = [*result_frames]
    if skipped_rows:
        result_parts.append(pd.DataFrame(skipped_rows))
    results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    traces = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    return results, traces


def write_run_config(
    path: Path, config: dict[str, Any], pairings: pd.DataFrame, config_path: Path
) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ModelHubError("写入 run_config.yaml 需要 PyYAML") from exc
    payload = {
        "protocol_version": config["protocol_id"],
        "protocol_sha256": sha256_file(config_path),
        "source_versions": config.get("source_versions", []),
        "enabled_pairings": pairings.loc[pairings["status"] == "completed", "pairing_id"].drop_duplicates().tolist(),
        "budget_grid": sorted(pairings.loc[pairings["status"] == "completed", "requested_budget"].dropna().unique().tolist()),
        "allowed_policies": sorted(pairings.loc[pairings["status"] == "completed", "routing_policy"].dropna().unique().tolist()),
        "cost_modes": sorted(pairings.loc[pairings["status"] == "completed", "cost_mode"].dropna().unique().tolist()),
        "result_semantics": config.get("result_semantics", "interactive_replay"),
        "selection_split": config.get("selection_split", "val"),
        "evaluation_design": config.get("evaluation_design", ""),
        "confirmatory_test": config.get("confirmatory_test", None),
        "test_result_based_selection": config.get("test_result_based_selection", ""),
        "route_eligible": config.get("route_eligible", None),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "100% 为 Expert prediction replacement；在线成本仍包含 Scout。并行情景未实测。",
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def rank_validation_candidates(
    pairings: pd.DataFrame,
    *,
    selection_status: str = "validation_candidate_not_test_evaluated",
) -> pd.DataFrame:
    candidates = pairings.loc[
        (pairings["status"] == "completed")
        & (pairings["evaluation_kind"] == "routed")
    ].copy()
    if candidates.empty:
        return candidates
    candidates["scout_count"] = candidates["scout_artifact_ids"].astype(str).str.count(r"\|") + 1
    candidates = candidates.sort_values(
        [
            "dangerous_introduced",
            "net_dangerous_reduction",
            "macro_f1",
            "qwk",
            "requested_budget",
            "estimated_total_compute_ms_per_image",
        ],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates.insert(0, "validation_rank", np.arange(1, len(candidates) + 1))
    candidates["selection_status"] = selection_status
    return candidates


def select_validation_candidates(
    pairings: pd.DataFrame,
    hub: pd.DataFrame,
    *,
    selection_status: str = "validation_candidate_not_test_evaluated",
) -> pd.DataFrame:
    """Publish a compact shortlist without using Oracle as evidence."""
    ranked = rank_validation_candidates(pairings, selection_status=selection_status)
    selections: list[pd.DataFrame] = []
    complete_cost = ranked.loc[
        ranked["cost_mode"].eq("estimated_from_measured_forward_only")
    ].copy()
    selection_pool = complete_cost if not complete_cost.empty else ranked

    def pick(label: str, frame: pd.DataFrame, *, low_budget: bool = False) -> None:
        if frame.empty:
            return
        frame = frame.loc[frame["dangerous_introduced"] == 0].copy()
        if frame.empty:
            return
        if low_budget:
            positive = frame.loc[frame["net_dangerous_reduction"] > 0]
            frame = positive if not positive.empty else frame
            frame = frame.loc[frame["requested_budget"] == frame["requested_budget"].min()]
        selected = frame.sort_values(
            ["macro_f1", "accuracy", "net_dangerous_reduction", "qwk", "requested_budget"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).head(1).copy()
        selected.insert(0, "selection_role", label)
        selections.append(selected)

    pick("primary_single_scout", selection_pool.loc[selection_pool["scout_count"] == 1])
    pick(
        "single_scout_ablation",
        selection_pool.loc[selection_pool["scout_count"] == 1],
        low_budget=True,
    )
    pick("primary_multi_scout", selection_pool.loc[selection_pool["scout_count"] >= 2])

    baselines = hub.copy()
    if not baselines.empty:
        baselines = baselines.sort_values(
            ["macro_f1", "qwk", "accuracy"], ascending=False, kind="mergesort"
        ).head(1)
        baselines.insert(0, "selection_role", "strong_single_model_baseline")
        baselines["selection_status"] = selection_status
        selections.append(baselines)
    if not selections:
        return pd.DataFrame()
    return pd.concat(selections, ignore_index=True, sort=False)


def render_report(path: Path, hub: pd.DataFrame, pairings: pd.DataFrame) -> None:
    completed = pairings.loc[pairings["status"] == "completed"] if not pairings.empty else pairings
    skipped = pairings.loc[pairings["status"] != "completed"] if not pairings.empty else pairings
    task_count = int(hub["task_id"].nunique()) if not hub.empty else 0
    adapter_count = int((hub["adapter_status"] == "completed").sum()) if not hub.empty else 0
    pairing_count = int(completed["pairing_id"].nunique()) if not completed.empty else 0
    skipped_count = int(skipped["pairing_id"].nunique()) if not skipped.empty else 0
    preview = completed[
        [
            "pairing_id",
            "routing_policy",
            "requested_budget",
            "accuracy",
            "macro_f1",
            "estimated_total_compute_ms_per_image",
        ]
    ].head(30) if not completed.empty else pd.DataFrame()
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(clean_text(value))}</td>" for value in row) + "</tr>"
        for row in preview.itertuples(index=False, name=None)
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OphAgent v0.8.6 交互式模型中转台</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Microsoft YaHei",sans-serif;background:#f4f7fa;color:#172033;margin:0}}
main{{max-width:1180px;margin:auto;padding:32px 24px 64px}} h1,h2{{color:#17324d}}
.notice{{background:#fff8e8;border-left:4px solid #b7791f;padding:14px 16px;line-height:1.7}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:22px 0}}
.metric{{background:white;border:1px solid #d8e0ea;border-top:3px solid #0f766e;padding:14px;border-radius:6px}}
.metric b{{display:block;font-size:26px;color:#17324d;margin-top:6px}}
table{{border-collapse:collapse;width:100%;background:white}} th,td{{border:1px solid #d8e0ea;padding:8px 10px;text-align:left;font-size:13px}}
th{{background:#eef3f7}} .table{{overflow:auto}}
@media(max-width:720px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<h1>OphAgent v0.8.6：Interactive Model Hub</h1>
<p>受控 Scout–Expert 组合 replay 与病例级路由轨迹。</p>
<div class="notice"><strong>边界：</strong>v0.8.6 是交互式工程探索 replay，不作为正式科研结论。验证集冻结、Random/Oracle/Learned Gate、统计检验和外部验证留到 v0.8.7。</div>
<div class="grid"><div class="metric">任务数<b>{task_count}</b></div><div class="metric">真实 Adapter 模型<b>{adapter_count}</b></div><div class="metric">完成 Pairing<b>{pairing_count}</b></div><div class="metric">兼容性跳过<b>{skipped_count}</b></div></div>
<h2>预算曲线预览</h2><div class="table"><table><thead><tr><th>Pairing</th><th>策略</th><th>预算</th><th>Accuracy</th><th>Macro-F1</th><th>估算总前向成本</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>成本口径</h2><p>总计算与顺序延迟由已测模型 forward-only 成本估算；并行延迟仅为多执行器情景估算，不是当前单卡并发实测。图像解码、预处理、I/O、排队、模型加载和服务开销均未计入。</p>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def write_manifest(output_dir: Path, config: dict[str, Any], config_path: Path) -> None:
    created = datetime.now(timezone.utc).isoformat()
    rows = []
    for name in OUTPUT_NAMES:
        if name == "artifact_manifest.csv":
            continue
        path = output_dir / name
        if not path.exists():
            continue
        rows.append(
            {
                "protocol_id": config["protocol_id"],
                "protocol_sha256": sha256_file(config_path),
                "artifact_name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "created_at_utc": created,
                "source_versions": "|".join(config.get("source_versions", [])),
                "notes": "published output; no work path",
            }
        )
    write_csv(
        output_dir / "artifact_manifest.csv",
        pd.DataFrame(rows),
        ["protocol_id", "protocol_sha256", "artifact_name", "path", "size_bytes", "sha256", "created_at_utc", "source_versions", "notes"],
    )


def verify_resume_artifacts(output_dir: Path, config_path: Path) -> None:
    manifest_path = output_dir / "artifact_manifest.csv"
    if not manifest_path.exists():
        raise ModelHubError("--resume 需要已有 artifact_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    required = {"model_hub_snapshot.csv", "pairing_results.csv", "case_routing_trace.csv"}
    present = set(manifest.get("artifact_name", pd.Series(dtype=str)).astype(str))
    missing = required - present
    if missing:
        raise ModelHubError("--resume 缺少既有结果：" + ", ".join(sorted(missing)))
    if "protocol_sha256" not in manifest.columns:
        raise ModelHubError("--resume 需要带 protocol_sha256 的新 manifest")
    if set(manifest["protocol_sha256"].dropna().astype(str)) != {sha256_file(config_path)}:
        raise ModelHubError("--resume 配置指纹不一致；请显式重新运行验证集协议")
    for row in manifest.itertuples(index=False):
        artifact = output_dir / str(getattr(row, "artifact_name"))
        if not artifact.exists() or sha256_file(artifact) != str(getattr(row, "sha256")):
            raise ModelHubError(f"--resume 产物完整性校验失败：{artifact.name}")


def validate_config(config: dict[str, Any], config_path: Path) -> None:
    if clean_text(config.get("prediction_asset_registry")):
        required = {"protocol_id", "task_registry", "prediction_asset_registry"}
        missing = required - set(config)
        if missing:
            raise ModelHubError("prediction asset protocol missing fields: " + ", ".join(sorted(missing)))
        for field in required - {"protocol_id"}:
            path = resolve_path(config[field], config_dir=config_path.parent)
            if not path.exists():
                raise ModelHubError(f"missing config input {field}: {path}")
        if not config.get("pairing_expansion") and not clean_text(config.get("pairing_protocols")):
            raise ModelHubError("prediction asset protocol requires pairing_expansion or pairing_protocols")
        return
    required = {
        "protocol_id",
        "task_registry",
        "registered_models",
        "v085c_output_dir",
        "v085c_replay_protocols",
        "pairing_protocols",
    }
    missing = required - set(config)
    if missing:
        raise ModelHubError("protocol.yaml 缺少字段：" + ", ".join(sorted(missing)))
    for field in required - {"protocol_id"}:
        path = resolve_path(config[field], config_dir=config_path.parent)
        if field == "v085c_output_dir":
            expected = path / "adapter_job_summary.csv"
            if not expected.exists():
                raise ModelHubError(f"v0.8.5c outputs 不完整：{expected}")
        elif not path.exists():
            raise ModelHubError(f"找不到配置输入 {field}：{path}")
    pairings = read_csv(config["pairing_protocols"], config_dir=config_path.parent)
    if pairings.empty:
        raise ModelHubError("pairing_protocols.csv 不能为空")
    for value in pairings.loc[pairings["enabled"].map(truthy), "budget_grid"]:
        parse_budgets(value)


def run_protocol(
    config_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    stage: str = "all",
    dry_run: bool = False,
    resume: bool = False,
) -> ProtocolResult:
    config_path = Path(config_path).resolve()
    if stage not in VALID_STAGES:
        raise ModelHubError(f"不支持 stage：{stage}")
    config = load_yaml_or_json(config_path)
    validate_config(config, config_path)
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else resolve_path(config.get("model_hub_output_dir", config.get("output_dir", "outputs")))
    )
    if dry_run:
        return ProtocolResult(output_dir=output_path, stage=stage, files=[])
    if resume:
        verify_resume_artifacts(output_path, config_path)
        return ProtocolResult(
            output_dir=output_path,
            stage=stage,
            files=[output_path / name for name in OUTPUT_NAMES if (output_path / name).exists()],
        )
    output_path.mkdir(parents=True, exist_ok=True)

    hub_path = output_path / "model_hub_snapshot.csv"
    pairing_path = output_path / "pairing_results.csv"
    trace_path = output_path / "case_routing_trace.csv"
    ranking_path = output_path / "candidate_ranking.csv"
    selection_status = (
        "exploratory_test_result_not_for_selection"
        if clean_text(config.get("test_result_based_selection")) == "exploratory_only"
        else "validation_candidate_not_test_evaluated"
    )
    if stage in {"model_hub", "all"}:
        write_csv(hub_path, build_model_hub(config, config_path=config_path), HUB_COLUMNS)
    if stage in {"pairing", "all"}:
        if not hub_path.exists():
            write_csv(hub_path, build_model_hub(config, config_path=config_path), HUB_COLUMNS)
        hub = pd.read_csv(hub_path)
        pairings, traces = evaluate_pairings(config, config_path=config_path, hub=hub)
        write_csv(pairing_path, pairings, PAIRING_COLUMNS)
        write_csv(trace_path, traces, TRACE_COLUMNS)
        rank_validation_candidates(pairings, selection_status=selection_status).to_csv(ranking_path, index=False)
        select_validation_candidates(pairings, hub, selection_status=selection_status).to_csv(
            output_path / "candidate_selection.csv", index=False
        )
    if stage in {"report", "all"}:
        if not hub_path.exists() or not pairing_path.exists() or not trace_path.exists():
            hub = build_model_hub(config, config_path=config_path)
            write_csv(hub_path, hub, HUB_COLUMNS)
            pairings, traces = evaluate_pairings(config, config_path=config_path, hub=hub)
            write_csv(pairing_path, pairings, PAIRING_COLUMNS)
            write_csv(trace_path, traces, TRACE_COLUMNS)
            rank_validation_candidates(pairings, selection_status=selection_status).to_csv(
                ranking_path, index=False
            )
            select_validation_candidates(pairings, hub, selection_status=selection_status).to_csv(
                output_path / "candidate_selection.csv", index=False
            )
        hub = pd.read_csv(hub_path)
        pairings = pd.read_csv(pairing_path)
        write_run_config(output_path / "run_config.yaml", config, pairings, config_path)
        render_report(output_path / "report.html", hub, pairings)
        write_manifest(output_path, config, config_path)

    files = [output_path / name for name in OUTPUT_NAMES if (output_path / name).exists()]
    return ProtocolResult(output_dir=output_path, stage=stage, files=files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=sorted(VALID_STAGES), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_protocol(
            args.config,
            output_dir=args.output_dir,
            stage=args.stage,
            dry_run=args.dry_run,
            resume=args.resume,
        )
    except (ModelHubError, PairingSkip) as exc:
        print(f"[ERROR] {exc}")
        return 2
    if args.dry_run:
        print("[DRY-RUN] v0.8.6 protocol valid; no files written")
    else:
        print(f"[DONE] v0.8.6 stage={args.stage} outputs={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
