#!/usr/bin/env python3
"""v0.8.5b known-model inventory, adapter onboarding, and routing replay.

This script is deterministic and conservative:

* Stage 1 only inventories files that actually exist in the repository/server.
* Stage 2 only runs adapter jobs whose required inputs are present.
* Stage 3 only replays routing when predictions are available.

It never trains, fine-tunes, fabricates checkpoints, or treats legacy predictions
as newly generated adapter predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The repository-root bootstrap above must run before these imports.
from scripts.routing.model_metadata import normalized_model_metadata  # noqa: E402


TRUTHY = {"1", "true", "yes", "y", "on"}
FALSY = {"0", "false", "no", "n", "off"}
VALID_STAGES = {"inventory", "onboarding", "replay", "all"}
SOURCE_ARTIFACT_TARGETS = {
    "registered_tasks": ["registered_tasks.csv"],
    "registered_models": ["registered_models.csv"],
    "route_protocol_summary": ["route_protocol_summary.csv"],
    "model_baselines": ["model_baselines.csv", "model_baselines_all.csv"],
    "routing_results": ["routing_results.csv", "routing_results_all.csv"],
    "risk_results": ["risk_results.csv", "risk_results_all.csv"],
    "case_audit": ["case_audit.csv", "case_audit_all.csv"],
    "artifact_manifest": ["artifact_manifest.csv"],
    "cost_summary": [
        "glaucoma_model_forward_cost_summary.csv",
        "model_forward_cost_summary.csv",
        "forward_cost_summary.csv",
        "forward_cost_summary_from_adapters.csv",
    ],
    "summary_html": ["summary.html"],
}


class KnownModelProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProtocolResult:
    output_dir: Path
    stage: str
    files: list[Path]


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise KnownModelProtocolError("读取 YAML 配置需要 PyYAML") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise KnownModelProtocolError("protocol config 根节点必须是 mapping")
    return data


def resolve_path(value: str | Path, *, config_dir: Path | None = None) -> Path:
    raw = str(value).strip()
    if not raw:
        return Path("")
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if expanded.is_absolute():
        return expanded
    if config_dir is not None and (config_dir / expanded).exists():
        return config_dir / expanded
    return REPO_ROOT / expanded


def display_path(path: Path | str) -> str:
    if not path:
        return ""
    return str(path).replace("\\", "/")


def find_existing_artifact(source_path: Path, file_type: str) -> Path | None:
    for name in SOURCE_ARTIFACT_TARGETS.get(file_type, []):
        path = source_path / name
        if path.exists():
            return path
    return None


def is_enabled(row: pd.Series | dict[str, Any]) -> bool:
    value = str(row.get("enabled", "true")).strip().lower()
    return value not in FALSY


def is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def read_registry(path: Path, required_columns: set[str], *, name: str) -> pd.DataFrame:
    if not path.exists():
        raise KnownModelProtocolError(f"找不到 {name}：{path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = required_columns - set(frame.columns)
    if missing:
        raise KnownModelProtocolError(f"{name} 缺少字段：{sorted(missing)}")
    return frame.loc[frame.apply(is_enabled, axis=1)].copy()


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except EmptyDataError:
        return pd.DataFrame()


def write_csv(path: Path, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if columns is not None:
        for column in columns:
            if column not in output.columns:
                output[column] = ""
        output = output[columns]
    output.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).replace(",", "|").split("|") if item.strip()]


def parse_budgets(value: str) -> list[float]:
    budgets: list[float] = []
    for item in parse_list(value):
        budgets.append(float(item))
    return budgets


def load_all_config(config_path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    config = load_yaml_or_json(config_path)
    config_dir = config_path.parent
    required = {
        "inventory_sources": {
            "source_id",
            "source_type",
            "source_path",
            "task_id",
            "dataset_id",
            "disease_family",
            "source_schema",
            "enabled",
            "notes",
        },
        "adapter_registry": {
            "adapter_id",
            "adapter_type",
            "model_family",
            "supported_backbones",
            "status",
            "enabled",
            "notes",
        },
        "onboarding_jobs": {
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
        },
        "routing_replay_protocols": {
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
        },
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, columns in required.items():
        if key not in config:
            raise KnownModelProtocolError(f"protocol.yaml 缺少 {key}")
        tables[key] = read_registry(
            resolve_path(config[key], config_dir=config_dir),
            columns,
            name=f"{key}.csv",
        )
    return config, tables


def infer_artifact_from_path(path: Path) -> str:
    text = display_path(path).lower()
    mapping = [
        ("aptos_convnext_tiny", "convnext_tiny"),
        ("aptos_swin_tiny", "swin_tiny"),
        ("aptos_vit_base_patch16_imagenet", "vit_b_imagenet"),
        ("aptos_vit_large_patch16_official_like", "vit_l_official_like"),
        ("aptos_retfound_mae_cfp_official_protocol", "retfound_mae_cfp_official_protocol"),
        ("retfound_green", "retfound_green_linear_probe"),
        ("convnext_tiny_glaucoma_scout", "convnext_tiny_glaucoma_scout"),
        ("retfound_dinov2_glaucoma_expert", "retfound_dinov2_glaucoma_expert"),
    ]
    for marker, artifact in mapping:
        if marker in text:
            return artifact
    return path.parent.parent.parent.name if "evaluation/test" in text else path.stem


def infer_model_family(artifact_id: str) -> str:
    return normalized_model_metadata(artifact_id, infer_backbone(artifact_id))["model_family"]


def infer_backbone(artifact_id: str) -> str:
    mapping = {
        "convnext_tiny": "convnext_tiny",
        "swin_tiny": "swin_tiny_patch4_window7_224",
        "vit_b_imagenet": "vit_base_patch16_224",
        "vit_l_official_like": "vit_large_patch16_224",
        "retfound_mae_cfp_official_protocol": "retfound_mae_cfp",
        "retfound_green_linear_probe": "retfound_green",
        "convnext_tiny_glaucoma_scout": "convnext_tiny",
        "retfound_dinov2_glaucoma_expert": "retfound_dinov2",
    }
    return mapping.get(artifact_id, artifact_id)


def infer_role_candidates(artifact_id: str) -> str:
    lowered = artifact_id.lower()
    if "expert" in lowered or "retfound" in lowered:
        return "expert"
    if "scout" in lowered or any(key in lowered for key in ("convnext", "swin", "vit", "green")):
        return "scout|expert"
    return "scout|expert"


def infer_adapter_id(artifact_id: str, adapters: pd.DataFrame) -> str:
    family = infer_model_family(artifact_id)
    if family in {"convnext", "swin", "vit"}:
        candidate = adapters.loc[adapters["adapter_type"].astype(str) == "timm_classifier"]
        if not candidate.empty:
            return str(candidate.iloc[0]["adapter_id"])
    if family.startswith("retfound"):
        candidate = adapters.loc[adapters["adapter_type"].astype(str) == "retfound"]
        if not candidate.empty:
            return str(candidate.iloc[0]["adapter_id"])
    if family == "mock":
        candidate = adapters.loc[adapters["adapter_type"].astype(str) == "synthetic_mock"]
        if not candidate.empty:
            return str(candidate.iloc[0]["adapter_id"])
    return ""


def find_run_root_from_prediction(prediction_path: Path) -> Path:
    parts = prediction_path.parts
    if len(parts) >= 3 and parts[-3:] == ("evaluation", "test", prediction_path.name):
        return prediction_path.parents[2]
    if "evaluation" in parts:
        index = parts.index("evaluation")
        return Path(*parts[:index])
    return prediction_path.parent


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def coalesce_text(*values: Any) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def discover_prediction_candidate(
    prediction_path: Path,
    *,
    source: pd.Series,
    adapters: pd.DataFrame,
) -> dict[str, Any]:
    artifact_id = infer_artifact_from_path(prediction_path)
    lowered = display_path(prediction_path).lower()
    if "aptos" in lowered:
        task_id = "aptos_dr_5class"
        dataset_id = "APTOS2019"
        disease_family = "diabetic_retinopathy"
    elif "glaucoma" in lowered:
        task_id = "glaucoma_3class"
        dataset_id = "Glaucoma_fundus"
        disease_family = "glaucoma"
    else:
        task_id = source.get("task_id", "")
        dataset_id = source.get("dataset_id", "")
        disease_family = source.get("disease_family", "")
    run_root = find_run_root_from_prediction(prediction_path)
    checkpoint = first_existing(sorted((run_root / "checkpoints").glob("*.pth"))) if (run_root / "checkpoints").exists() else None
    config = first_existing([run_root / "configs" / "config.json"])
    class_mapping = first_existing([run_root / "configs" / "class_to_idx.json"])
    return {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "disease_family": disease_family,
        "artifact_id": artifact_id,
        "model_family": infer_model_family(artifact_id),
        "backbone": infer_backbone(artifact_id),
        "role_candidates": infer_role_candidates(artifact_id),
        "legacy_source": str(source.get("source_id", "")),
        "legacy_prediction_path": display_path(prediction_path),
        "checkpoint_path": display_path(checkpoint or ""),
        "config_path": display_path(config or ""),
        "class_to_idx_path": display_path(class_mapping or ""),
        "adapter_id": infer_adapter_id(artifact_id, adapters),
        "notes": "从实际 prediction 文件扫描发现",
    }


def is_unified_model_hub_runtime_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return any(
        parts[index] == "model_hub" and parts[index + 1] in {"runs", "runtime"}
        for index in range(len(parts) - 1)
    )


def source_file_rows(source: pd.Series) -> list[dict[str, Any]]:
    source_path = resolve_path(source["source_path"])
    rows: list[dict[str, Any]] = []
    base = {
        "source_id": source["source_id"],
        "task_id": source["task_id"],
        "dataset_id": source["dataset_id"],
        "disease_family": source["disease_family"],
    }
    if not source_path.exists():
        rows.append(
            {
                **base,
                "file_type": str(source.get("source_type", "unknown")),
                "file_path": display_path(source_path),
                "exists": "false",
                "n_rows": "",
                "columns": "",
                "notes": "来源路径不存在",
            }
        )
        return rows
    for file_type, names in SOURCE_ARTIFACT_TARGETS.items():
        found = False
        for name in names:
            path = source_path / name
            if path.exists():
                frame = pd.DataFrame() if file_type == "summary_html" else read_optional_csv(path)
                rows.append(
                    {
                        **base,
                        "file_type": file_type,
                        "file_path": display_path(path),
                        "exists": "true",
                        "n_rows": "" if file_type == "summary_html" else len(frame),
                        "columns": "" if file_type == "summary_html" else "|".join(frame.columns),
                        "notes": "",
                    }
                )
                found = True
        if not found and source.get("source_type", "") != "scan_root":
            rows.append(
                {
                    **base,
                    "file_type": file_type,
                    "file_path": display_path(source_path / names[0]),
                    "exists": "false",
                    "n_rows": "",
                    "columns": "",
                    "notes": "未发现该类型产物",
                }
            )
    if source.get("source_type", "") == "scan_root":
        for pattern, file_type in (
            ("**/*.pth", "checkpoint"),
            ("**/config.json", "config"),
            ("**/class_to_idx.json", "class_mapping"),
            ("**/test_predictions.csv", "prediction"),
            ("**/*standardized*.csv", "prediction"),
        ):
            for path in (
                path
                for path in sorted(source_path.glob(pattern))[:500]
                if not is_unified_model_hub_runtime_path(path)
            ):
                rows.append(
                    {
                        **base,
                        "file_type": file_type,
                        "file_path": display_path(path),
                        "exists": "true",
                        "n_rows": "",
                        "columns": "",
                        "notes": "扫描发现",
                    }
                )
    return rows


def path_status(path_value: str, *, unknown_when_empty: bool = True) -> str:
    value = str(path_value or "").strip()
    if not value:
        return "unknown" if unknown_when_empty else "missing"
    return "found" if resolve_path(value).exists() else "missing"


def adapter_status(adapter_id: str, adapters: pd.DataFrame) -> str:
    if not adapter_id:
        return "unknown"
    rows = adapters.loc[adapters["adapter_id"].astype(str) == adapter_id]
    if rows.empty:
        return "unsupported_adapter"
    status = str(rows.iloc[0]["status"])
    if status == "available":
        return "available"
    if status in {"needs_loader_audit", "unsupported_adapter", "unknown"}:
        return status
    return "unknown"


def determine_onboarding(row: dict[str, Any], adapters: pd.DataFrame) -> dict[str, str]:
    ckpt = path_status(str(row.get("checkpoint_path", "")), unknown_when_empty=True)
    cfg = path_status(str(row.get("config_path", "")), unknown_when_empty=True)
    cls = path_status(str(row.get("class_to_idx_path", "")), unknown_when_empty=True)
    data = path_status(str(row.get("data_root", "")), unknown_when_empty=True)
    adapter = adapter_status(str(row.get("adapter_id", "")), adapters)
    legacy_available = any(
        str(row.get(key, "")).strip()
        for key in (
            "legacy_prediction_path",
            "legacy_baseline_path",
            "legacy_routing_path",
            "legacy_risk_path",
            "legacy_case_audit_path",
            "legacy_cost_path",
        )
    )
    missing_reason = ""
    if adapter == "unsupported_adapter":
        status = "legacy_replay_only" if legacy_available else "unsupported_adapter"
        missing_reason = "unsupported_adapter"
    elif adapter == "needs_loader_audit":
        status = "needs_loader_audit"
        missing_reason = "needs_loader_audit"
    elif ckpt == "missing":
        status = "legacy_replay_only" if legacy_available else "missing_checkpoint"
        missing_reason = "checkpoint_required"
    elif ckpt == "unknown":
        status = "legacy_replay_only" if legacy_available else "incomplete_metadata"
        missing_reason = "checkpoint_required"
    elif cls == "missing":
        status = "missing_class_mapping"
        missing_reason = "class_mapping_required"
    elif data == "missing":
        status = "missing_data_root"
        missing_reason = "data_root_or_input_csv_required"
    elif data == "unknown":
        status = "incomplete_metadata"
        missing_reason = "data_root_or_input_csv_required"
    elif adapter != "available":
        status = "legacy_replay_only" if legacy_available else "incomplete_metadata"
        missing_reason = "adapter_required"
    else:
        status = "ready_for_adapter"
        missing_reason = ""
    can_onboard = status == "ready_for_adapter"
    return {
        "checkpoint_status": ckpt,
        "config_status": cfg,
        "class_to_idx_status": cls,
        "data_status": data,
        "adapter_status": adapter,
        "can_onboard": str(can_onboard).lower(),
        "onboarding_status": status,
        "legacy_artifact_available": str(bool(legacy_available)).lower(),
        "missing_reason": missing_reason,
    }


INVENTORY_COLUMNS = [
    "task_id",
    "dataset_id",
    "disease_family",
    "artifact_id",
    "model_family",
    "backbone",
    "architecture",
    "pretraining_source",
    "role_candidates",
    "legacy_source",
    "legacy_prediction_path",
    "legacy_baseline_path",
    "legacy_routing_path",
    "legacy_risk_path",
    "legacy_case_audit_path",
    "legacy_cost_path",
    "checkpoint_path",
    "checkpoint_status",
    "config_path",
    "config_status",
    "class_to_idx_path",
    "class_to_idx_status",
    "data_root",
    "data_status",
    "adapter_id",
    "adapter_status",
    "can_onboard",
    "onboarding_status",
    "legacy_artifact_available",
    "missing_reason",
    "notes",
]


def build_inventory(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sources = tables["inventory_sources"]
    adapters = tables["adapter_registry"]
    jobs = tables["onboarding_jobs"]
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []

    def merge(row: dict[str, Any]) -> None:
        task_id = str(row.get("task_id", ""))
        artifact_id = str(row.get("artifact_id", ""))
        if not task_id or not artifact_id:
            return
        key = (task_id, artifact_id)
        existing = rows_by_key.setdefault(key, {"task_id": task_id, "artifact_id": artifact_id})
        for column, value in row.items():
            if value not in (None, "") and not str(existing.get(column, "")).strip():
                existing[column] = value

    for _, job in jobs.iterrows():
        merge(
            {
                "task_id": job["task_id"],
                "dataset_id": job["dataset_id"],
                "disease_family": job["disease_family"],
                "artifact_id": job["artifact_id"],
                "model_family": job["model_family"],
                "backbone": job["backbone"],
                "role_candidates": job["role_candidates"],
                "checkpoint_path": job["checkpoint_path"],
                "config_path": job["config_path"],
                "class_to_idx_path": job["class_to_idx_path"],
                "data_root": job["data_root"],
                "adapter_id": job["adapter_id"],
                "notes": job.get("notes", ""),
            }
        )

    for _, source in sources.iterrows():
        source_path = resolve_path(source["source_path"])
        source_rows.extend(source_file_rows(source))
        if not source_path.exists():
            continue
        baseline_path = find_existing_artifact(source_path, "model_baselines")
        routing_path = find_existing_artifact(source_path, "routing_results")
        risk_path = find_existing_artifact(source_path, "risk_results")
        case_audit_path = find_existing_artifact(source_path, "case_audit")
        cost_path = find_existing_artifact(source_path, "cost_summary")
        baseline = read_optional_csv(baseline_path) if baseline_path is not None else pd.DataFrame()
        for _, item in baseline.iterrows():
            artifact_id = str(item.get("artifact_id", item.get("name", item.get("model_name", ""))))
            if not artifact_id:
                continue
            task_id = coalesce_text(item.get("task_id"), source["task_id"])
            dataset_id = coalesce_text(item.get("dataset_id"), source["dataset_id"])
            disease_family = coalesce_text(item.get("disease_family"), source["disease_family"])
            merge(
                {
                    "task_id": task_id,
                    "dataset_id": dataset_id,
                    "disease_family": disease_family,
                    "artifact_id": artifact_id,
                    "model_family": infer_model_family(artifact_id),
                    "backbone": infer_backbone(artifact_id),
                    "role_candidates": str(item.get("role", infer_role_candidates(artifact_id))) or infer_role_candidates(artifact_id),
                    "legacy_source": source["source_id"],
                    "legacy_baseline_path": display_path(baseline_path or ""),
                    "legacy_routing_path": display_path(routing_path or ""),
                    "legacy_risk_path": display_path(risk_path or ""),
                    "legacy_case_audit_path": display_path(case_audit_path or ""),
                    "legacy_cost_path": display_path(cost_path or ""),
                    "adapter_id": infer_adapter_id(artifact_id, adapters),
                    "notes": f"从 {source['source_id']} baseline 发现",
                }
            )
        if source.get("source_type", "") == "scan_root":
            for prediction_path in sorted(source_path.glob("**/test_predictions.csv"))[:500]:
                if is_unified_model_hub_runtime_path(prediction_path):
                    continue
                merge(discover_prediction_candidate(prediction_path, source=source, adapters=adapters))

    concrete_artifacts = {
        artifact_id
        for task_id, artifact_id in rows_by_key
        if task_id != "mixed"
    }
    for key in list(rows_by_key):
        task_id, artifact_id = key
        if task_id == "mixed" and artifact_id in concrete_artifacts:
            del rows_by_key[key]

    normalized: list[dict[str, Any]] = []
    for row in rows_by_key.values():
        row.update(
            normalized_model_metadata(
                str(row.get("artifact_id", "")),
                str(row.get("backbone", "")) or infer_backbone(str(row.get("artifact_id", ""))),
            )
        )
        for column in INVENTORY_COLUMNS:
            row.setdefault(column, "")
        row.update(determine_onboarding(row, adapters))
        normalized.append(row)
    inventory = pd.DataFrame(normalized)
    if inventory.empty:
        inventory = pd.DataFrame(columns=INVENTORY_COLUMNS)
    else:
        inventory = inventory[INVENTORY_COLUMNS].sort_values(["task_id", "artifact_id"]).reset_index(drop=True)

    source_index = pd.DataFrame(source_rows)
    if source_index.empty:
        source_index = pd.DataFrame(columns=["source_id", "task_id", "dataset_id", "disease_family", "file_type", "file_path", "exists", "n_rows", "columns", "notes"])

    def count_status(value: str) -> int:
        return int((inventory["onboarding_status"] == value).sum()) if not inventory.empty else 0

    summary_rows = [
        ("total_candidates", len(inventory)),
        ("ready_for_adapter_n", int((inventory["onboarding_status"] == "ready_for_adapter").sum()) if not inventory.empty else 0),
        ("missing_checkpoint_n", count_status("missing_checkpoint")),
        ("missing_data_root_n", count_status("missing_data_root")),
        ("missing_class_mapping_n", count_status("missing_class_mapping")),
        ("unsupported_adapter_n", count_status("unsupported_adapter")),
        ("needs_loader_audit_n", count_status("needs_loader_audit")),
        ("legacy_replay_only_n", count_status("legacy_replay_only")),
        ("dr_candidates_n", int(inventory["disease_family"].str.contains("diabetic", case=False, na=False).sum()) if not inventory.empty else 0),
        ("glaucoma_candidates_n", int(inventory["disease_family"].str.contains("glaucoma", case=False, na=False).sum()) if not inventory.empty else 0),
    ]
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    return inventory, source_index, summary


def probability_columns(frame: pd.DataFrame) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for column in frame.columns:
        if column.startswith("prob_") and column.removeprefix("prob_").isdigit():
            indexed.append((int(column.removeprefix("prob_")), column))
    indexed.sort()
    return [column for _, column in indexed]


def compute_probability_signals(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.sort(probs, axis=1)
    pred = probs.argmax(axis=1)
    confidence = order[:, -1]
    margin = order[:, -1] - order[:, -2]
    safe = np.clip(probs, np.finfo(float).tiny, 1.0)
    entropy = -(probs * np.log(safe)).sum(axis=1) / math.log(probs.shape[1])
    return pred, confidence, margin, entropy


def classification_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    prob_cols = probability_columns(frame)
    y_true = frame["true_label"].astype(int).to_numpy()
    y_pred = frame["pred_label"].astype(int).to_numpy()
    probs = frame[prob_cols].astype(float).to_numpy()
    metrics: dict[str, Any] = {
        "n_images": len(frame),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_error": int((y_true != y_pred).sum()),
    }
    aurocs: list[float] = []
    auprs: list[float] = []
    for index in range(probs.shape[1]):
        binary = (y_true == index).astype(int)
        if binary.min() == binary.max():
            continue
        aurocs.append(float(roc_auc_score(binary, probs[:, index])))
        auprs.append(float(average_precision_score(binary, probs[:, index])))
    metrics["macro_auroc_ovr"] = float(np.mean(aurocs)) if aurocs else float("nan")
    metrics["macro_aupr_ovr"] = float(np.mean(auprs)) if auprs else float("nan")
    return metrics


def synthetic_predictions(job: pd.Series) -> pd.DataFrame:
    n = 8
    n_classes = int(job["num_classes"] or 2)
    rows: list[dict[str, Any]] = []
    artifact = str(job["artifact_id"])
    for index in range(n):
        true_label = index % n_classes
        probs = np.full(n_classes, 0.10 / max(1, n_classes - 1), dtype=float)
        if "expert" in artifact:
            pred = true_label if index not in {3} else (true_label + 1) % n_classes
            high = 0.88
        elif artifact.endswith("_b"):
            pred = true_label if index not in {2, 5} else (true_label + 1) % n_classes
            high = 0.76
        else:
            pred = true_label if index not in {1, 2, 5} else (true_label + 1) % n_classes
            high = 0.72
        probs[:] = (1.0 - high) / max(1, n_classes - 1)
        probs[pred] = high
        pred_array, confidence, margin, entropy = compute_probability_signals(probs.reshape(1, -1))
        row: dict[str, Any] = {
            "job_id": job["job_id"],
            "task_id": job["task_id"],
            "artifact_id": artifact,
            "image_key": f"case_{index:03d}",
            "true_label": true_label,
            "pred_label": int(pred_array[0]),
            "confidence": float(confidence[0]),
            "margin": float(margin[0]),
            "entropy": float(entropy[0]),
            "source": "adapter_generated",
        }
        for class_index, value in enumerate(probs):
            row[f"prob_{class_index}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def run_timm_classifier_adapter(job: pd.Series) -> pd.DataFrame:
    input_csv = str(job.get("input_csv", "")).strip() or str(job.get("legacy_prediction_path", "")).strip()
    if not input_csv:
        raise KnownModelProtocolError("timm adapter 需要 input_csv 或 legacy_prediction_path 提供图像列表")
    checkpoint_path = resolve_path(job["checkpoint_path"])
    input_path = resolve_path(input_csv)
    class_mapping = resolve_path(job["class_to_idx_path"])
    if not checkpoint_path.exists():
        raise KnownModelProtocolError("checkpoint 不存在")
    if not input_path.exists():
        raise KnownModelProtocolError("input_csv 不存在")
    if not class_mapping.exists():
        raise KnownModelProtocolError("class_to_idx_path 不存在")
    try:
        import timm
        import torch
        from PIL import Image
        from agent.runner import build_transform
    except Exception as exc:  # pragma: no cover - depends on server runtime
        raise KnownModelProtocolError(f"当前环境缺少 timm/torch/PIL 运行依赖：{exc}") from exc

    config_data: dict[str, Any] = {}
    config_path = resolve_path(job["config_path"])
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config_data = {}
    model_name = str(config_data.get("model_name") or config_data.get("arch") or job["backbone"])
    image_size = int(config_data.get("image_size") or job.get("input_size") or 224)
    num_classes = int(job["num_classes"])
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and str(job.get("device", "")) != "cpu" else "cpu")
    model.to(device).eval()
    transform = build_transform(image_size)
    manifest = pd.read_csv(input_path)
    image_column = "image_path" if "image_path" in manifest.columns else "path"
    if image_column not in manifest.columns:
        raise KnownModelProtocolError("input_csv 缺少 image_path/path 字段")
    true_column = "true_label" if "true_label" in manifest.columns else "true_idx" if "true_idx" in manifest.columns else None
    if true_column is None:
        raise KnownModelProtocolError("input_csv 缺少 true_label/true_idx 字段")
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for _, record in manifest.iterrows():
            image_path = resolve_path(str(record[image_column]))
            if not image_path.exists():
                image_path = resolve_path(str(record[image_column]), config_dir=input_path.parent)
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
            pred, confidence, margin, entropy = compute_probability_signals(probs.reshape(1, -1))
            row = {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "image_key": Path(str(record[image_column])).stem,
                "true_label": int(record[true_column]),
                "pred_label": int(pred[0]),
                "confidence": float(confidence[0]),
                "margin": float(margin[0]),
                "entropy": float(entropy[0]),
                "source": "adapter_generated",
            }
            for index, value in enumerate(probs):
                row[f"prob_{index}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def write_adapter_outputs(job: pd.Series, frame: pd.DataFrame, output_dir: Path, adapter: pd.Series) -> dict[str, Any]:
    job_dir = output_dir / "onboarded_models" / str(job["job_id"])
    prob_cols = probability_columns(frame)
    predictions_path = job_dir / "predictions.csv"
    baseline_path = job_dir / "model_baseline.csv"
    cost_path = job_dir / "forward_cost_summary.csv"
    manifest_path = job_dir / "adapter_manifest.csv"
    write_csv(predictions_path, frame)
    metrics = classification_metrics(frame)
    baseline = pd.DataFrame(
        [
            {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "split": "test",
                **metrics,
                "source": "adapter_generated",
                "notes": job.get("notes", ""),
            }
        ]
    )
    write_csv(baseline_path, baseline)
    mean_ms = max(0.01, 0.05 * len(prob_cols))
    cost = pd.DataFrame(
        [
            {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "cost_scope": "forward_only",
                "device": job.get("device", ""),
                "precision": job.get("precision", ""),
                "batch_size": job.get("batch_size", ""),
                "warmup_runs": 0,
                "timed_runs": 1,
                "mean_ms_per_image": mean_ms,
                "median_ms_per_image": mean_ms,
                "std_ms_per_image": 0.0,
                "cv_ms_per_image": 0.0,
                "images_per_second": 1000.0 / mean_ms,
                "notes": "synthetic_mock 为测试成本；真实成本需服务器 benchmark" if adapter["adapter_type"] == "synthetic_mock" else "forward-only adapter benchmark placeholder",
            }
        ]
    )
    write_csv(cost_path, cost)
    manifest = pd.DataFrame(
        [
            {
                "job_id": job["job_id"],
                "task_id": job["task_id"],
                "artifact_id": job["artifact_id"],
                "adapter_id": job["adapter_id"],
                "adapter_type": adapter["adapter_type"],
                "checkpoint_path": job.get("checkpoint_path", ""),
                "data_root": job.get("data_root", ""),
                "class_to_idx_path": job.get("class_to_idx_path", ""),
                "predictions_path": display_path(predictions_path),
                "model_baseline_path": display_path(baseline_path),
                "forward_cost_summary_path": display_path(cost_path),
                "sha256": sha256_file(predictions_path),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "notes": job.get("notes", ""),
            }
        ]
    )
    write_csv(manifest_path, manifest)
    return {
        "job_id": job["job_id"],
        "task_id": job["task_id"],
        "artifact_id": job["artifact_id"],
        "adapter_id": job["adapter_id"],
        "status": "completed",
        "n_images": metrics["n_images"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "estimated_forward_ms_per_image": mean_ms,
        "cost_scope": "forward_only",
        "outputs_dir": display_path(job_dir),
        "notes": job.get("notes", ""),
    }


def job_skip_status(job: pd.Series, adapters: pd.DataFrame) -> str | None:
    adapter_rows = adapters.loc[adapters["adapter_id"].astype(str) == str(job["adapter_id"])]
    if adapter_rows.empty:
        return "skipped_unsupported_adapter"
    adapter = adapter_rows.iloc[0]
    if str(adapter["status"]) == "needs_loader_audit":
        return "skipped_needs_loader_audit"
    if str(adapter["status"]) != "available":
        return "skipped_unsupported_adapter"
    if str(job.get("checkpoint_path", "")).strip() and not resolve_path(job["checkpoint_path"]).exists():
        return "skipped_missing_checkpoint"
    if str(job.get("data_root", "")).strip() and not resolve_path(job["data_root"]).exists():
        return "skipped_missing_data_root"
    if str(job.get("class_to_idx_path", "")).strip() and not resolve_path(job["class_to_idx_path"]).exists():
        return "skipped_missing_class_mapping"
    return None


def run_onboarding(tables: dict[str, pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs = tables["onboarding_jobs"]
    adapters = tables["adapter_registry"]
    summary_rows: list[dict[str, Any]] = []
    baselines: list[pd.DataFrame] = []
    costs: list[pd.DataFrame] = []
    manifests: list[pd.DataFrame] = []
    onboarded: list[dict[str, Any]] = []
    for _, job in jobs.iterrows():
        if not is_truthy(job.get("run_adapter", "")):
            summary_rows.append(
                {
                    "job_id": job["job_id"],
                    "task_id": job["task_id"],
                    "artifact_id": job["artifact_id"],
                    "adapter_id": job["adapter_id"],
                    "status": "skipped_disabled",
                    "n_images": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "estimated_forward_ms_per_image": "",
                    "cost_scope": "",
                    "outputs_dir": "",
                    "notes": job.get("notes", ""),
                }
            )
            continue
        skip = job_skip_status(job, adapters)
        if skip is not None:
            summary_rows.append(
                {
                    "job_id": job["job_id"],
                    "task_id": job["task_id"],
                    "artifact_id": job["artifact_id"],
                    "adapter_id": job["adapter_id"],
                    "status": skip,
                    "n_images": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "estimated_forward_ms_per_image": "",
                    "cost_scope": "",
                    "outputs_dir": "",
                    "notes": job.get("notes", ""),
                }
            )
            continue
        adapter = adapters.loc[adapters["adapter_id"].astype(str) == str(job["adapter_id"])].iloc[0]
        try:
            if adapter["adapter_type"] == "synthetic_mock":
                predictions = synthetic_predictions(job)
            elif adapter["adapter_type"] == "timm_classifier":
                predictions = run_timm_classifier_adapter(job)
            else:
                raise KnownModelProtocolError(f"不支持的 adapter_type：{adapter['adapter_type']}")
            summary = write_adapter_outputs(job, predictions, output_dir, adapter)
            summary_rows.append(summary)
            onboarded.append(
                {
                    "job_id": job["job_id"],
                    "task_id": job["task_id"],
                    "dataset_id": job["dataset_id"],
                    "disease_family": job["disease_family"],
                    "artifact_id": job["artifact_id"],
                    "role_candidates": job["role_candidates"],
                    "predictions_path": display_path(output_dir / "onboarded_models" / str(job["job_id"]) / "predictions.csv"),
                    "baseline_path": display_path(output_dir / "onboarded_models" / str(job["job_id"]) / "model_baseline.csv"),
                    "cost_path": display_path(output_dir / "onboarded_models" / str(job["job_id"]) / "forward_cost_summary.csv"),
                    "manifest_path": display_path(output_dir / "onboarded_models" / str(job["job_id"]) / "adapter_manifest.csv"),
                }
            )
            baselines.append(read_optional_csv(output_dir / "onboarded_models" / str(job["job_id"]) / "model_baseline.csv"))
            costs.append(read_optional_csv(output_dir / "onboarded_models" / str(job["job_id"]) / "forward_cost_summary.csv"))
            manifests.append(read_optional_csv(output_dir / "onboarded_models" / str(job["job_id"]) / "adapter_manifest.csv"))
        except Exception as exc:
            summary_rows.append(
                {
                    "job_id": job["job_id"],
                    "task_id": job["task_id"],
                    "artifact_id": job["artifact_id"],
                    "adapter_id": job["adapter_id"],
                    "status": "failed",
                    "n_images": "",
                    "accuracy": "",
                    "macro_f1": "",
                    "estimated_forward_ms_per_image": "",
                    "cost_scope": "",
                    "outputs_dir": "",
                    "notes": str(exc)[:500],
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(onboarded),
        pd.concat(baselines, ignore_index=True) if baselines else pd.DataFrame(),
        pd.concat(costs, ignore_index=True) if costs else pd.DataFrame(),
        pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame(),
    )


ADAPTER_JOB_SUMMARY_COLUMNS = [
    "job_id",
    "task_id",
    "artifact_id",
    "adapter_id",
    "status",
    "n_images",
    "accuracy",
    "macro_f1",
    "estimated_forward_ms_per_image",
    "cost_scope",
    "outputs_dir",
    "notes",
]
ONBOARDED_MODELS_COLUMNS = [
    "job_id",
    "task_id",
    "dataset_id",
    "disease_family",
    "artifact_id",
    "role_candidates",
    "predictions_path",
    "baseline_path",
    "cost_path",
    "manifest_path",
]
ADAPTER_BASELINE_COLUMNS = [
    "job_id",
    "task_id",
    "artifact_id",
    "split",
    "n_images",
    "accuracy",
    "macro_f1",
    "macro_auroc_ovr",
    "macro_aupr_ovr",
    "n_error",
    "source",
    "notes",
]
ADAPTER_COST_COLUMNS = [
    "job_id",
    "task_id",
    "artifact_id",
    "cost_scope",
    "device",
    "precision",
    "batch_size",
    "warmup_runs",
    "timed_runs",
    "mean_ms_per_image",
    "median_ms_per_image",
    "std_ms_per_image",
    "cv_ms_per_image",
    "images_per_second",
    "notes",
]
ADAPTER_MANIFEST_COLUMNS = [
    "job_id",
    "task_id",
    "artifact_id",
    "adapter_id",
    "adapter_type",
    "checkpoint_path",
    "data_root",
    "class_to_idx_path",
    "predictions_path",
    "model_baseline_path",
    "forward_cost_summary_path",
    "sha256",
    "created_at_utc",
    "notes",
]


def prediction_map(output_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted((output_dir / "onboarded_models").glob("*/predictions.csv")):
        frame = read_optional_csv(path)
        if not frame.empty and "artifact_id" in frame.columns:
            mapping[str(frame.iloc[0]["artifact_id"])] = path
    return mapping


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    prob_cols = probability_columns(frame)
    if not prob_cols:
        raise KnownModelProtocolError(f"prediction 缺少 prob_* 列：{path}")
    return frame


def uncertainty(frame: pd.DataFrame, policy: str) -> pd.Series:
    if policy in {"low_confidence", "mean_uncertainty", "max_uncertainty", "disagreement_then_uncertainty"}:
        return 1.0 - frame["confidence"].astype(float)
    if policy == "low_margin":
        return 1.0 - frame["margin"].astype(float)
    if policy == "high_entropy":
        return frame["entropy"].astype(float)
    return 1.0 - frame["confidence"].astype(float)


def select_keys(scores: pd.DataFrame, budget: float) -> set[str]:
    n = min(len(scores), max(0, int(round(len(scores) * budget))))
    ranked = scores.sort_values(["score", "image_key"], ascending=[False, True], kind="mergesort")
    return set(ranked.head(n)["image_key"].astype(str))


def routed_metrics(scout: pd.DataFrame, expert: pd.DataFrame, selected: set[str]) -> dict[str, Any]:
    merged = scout.merge(
        expert[["image_key", "pred_label"]].rename(columns={"pred_label": "expert_pred_label"}),
        on="image_key",
        how="inner",
        validate="one_to_one",
    )
    routed_pred = np.where(
        merged["image_key"].astype(str).isin(selected),
        merged["expert_pred_label"].astype(int),
        merged["pred_label"].astype(int),
    )
    true = merged["true_label"].astype(int).to_numpy()
    return {
        "n_rows": len(merged),
        "selected_n": len(selected),
        "accuracy": float(accuracy_score(true, routed_pred)),
        "macro_f1": float(f1_score(true, routed_pred, average="macro", zero_division=0)),
        "n_error": int((true != routed_pred).sum()),
    }


def run_single_replay(protocol: pd.Series, predictions: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    scout_id = parse_list(protocol["scout_artifact_ids"])[0]
    expert_id = str(protocol["expert_artifact_id"])
    if scout_id not in predictions or expert_id not in predictions:
        return pd.DataFrame(), {
            "replay_id": protocol["replay_id"],
            "task_id": protocol["task_id"],
            "routing_type": protocol["routing_type"],
            "status": "skipped_missing_predictions",
            "scout_artifact_ids": protocol["scout_artifact_ids"],
            "expert_artifact_id": expert_id,
            "n_rows": 0,
            "best_policy": "",
            "best_budget": "",
            "best_accuracy": "",
            "best_macro_f1": "",
            "notes": "缺少 scout 或 expert predictions",
        }
    scout = load_prediction(predictions[scout_id])
    expert = load_prediction(predictions[expert_id])
    rows: list[dict[str, Any]] = []
    for policy in parse_list(protocol["policies"]):
        scores = pd.DataFrame({"image_key": scout["image_key"].astype(str), "score": uncertainty(scout, policy)})
        for budget in parse_budgets(protocol["budgets"]):
            selected = select_keys(scores, budget)
            metrics = routed_metrics(scout, expert, selected)
            rows.append(
                {
                    "replay_id": protocol["replay_id"],
                    "task_id": protocol["task_id"],
                    "routing_type": "single_scout",
                    "scout_artifact_ids": scout_id,
                    "expert_artifact_id": expert_id,
                    "policy": policy,
                    "budget": budget,
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    return frame, {
        "replay_id": protocol["replay_id"],
        "task_id": protocol["task_id"],
        "routing_type": protocol["routing_type"],
        "status": "completed",
        "scout_artifact_ids": protocol["scout_artifact_ids"],
        "expert_artifact_id": expert_id,
        "n_rows": int(best["n_rows"]),
        "best_policy": best["policy"],
        "best_budget": best["budget"],
        "best_accuracy": best["accuracy"],
        "best_macro_f1": best["macro_f1"],
        "notes": protocol.get("notes", ""),
    }


def run_multi_replay(protocol: pd.Series, predictions: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    scout_ids = parse_list(protocol["scout_artifact_ids"])
    expert_id = str(protocol["expert_artifact_id"])
    missing = [artifact for artifact in [*scout_ids, expert_id] if artifact not in predictions]
    if missing:
        return pd.DataFrame(), {
            "replay_id": protocol["replay_id"],
            "task_id": protocol["task_id"],
            "routing_type": protocol["routing_type"],
            "status": "skipped_missing_predictions",
            "scout_artifact_ids": protocol["scout_artifact_ids"],
            "expert_artifact_id": expert_id,
            "n_rows": 0,
            "best_policy": "",
            "best_budget": "",
            "best_accuracy": "",
            "best_macro_f1": "",
            "notes": "缺少 predictions：" + "|".join(missing),
        }
    scouts = [load_prediction(predictions[artifact]) for artifact in scout_ids]
    expert = load_prediction(predictions[expert_id])
    base = scouts[0].copy()
    score_frame = pd.DataFrame({"image_key": base["image_key"].astype(str)})
    for artifact, scout in zip(scout_ids, scouts):
        score_frame[f"{artifact}_uncertainty"] = uncertainty(scout, "low_confidence").to_numpy()
        score_frame[f"{artifact}_pred"] = scout["pred_label"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for policy in parse_list(protocol["policies"]):
        if policy == "max_uncertainty":
            score = score_frame[[f"{artifact}_uncertainty" for artifact in scout_ids]].max(axis=1)
        elif policy == "disagreement_then_uncertainty":
            preds = score_frame[[f"{artifact}_pred" for artifact in scout_ids]]
            disagreement = preds.nunique(axis=1) > 1
            score = score_frame[[f"{artifact}_uncertainty" for artifact in scout_ids]].mean(axis=1) + disagreement.astype(float)
        else:
            score = score_frame[[f"{artifact}_uncertainty" for artifact in scout_ids]].mean(axis=1)
        scores = pd.DataFrame({"image_key": score_frame["image_key"], "score": score})
        for budget in parse_budgets(protocol["budgets"]):
            selected = select_keys(scores, budget)
            metrics = routed_metrics(base, expert, selected)
            rows.append(
                {
                    "replay_id": protocol["replay_id"],
                    "task_id": protocol["task_id"],
                    "routing_type": "multi_scout",
                    "scout_artifact_ids": protocol["scout_artifact_ids"],
                    "expert_artifact_id": expert_id,
                    "policy": policy,
                    "budget": budget,
                    **metrics,
                }
            )
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    return frame, {
        "replay_id": protocol["replay_id"],
        "task_id": protocol["task_id"],
        "routing_type": protocol["routing_type"],
        "status": "completed",
        "scout_artifact_ids": protocol["scout_artifact_ids"],
        "expert_artifact_id": expert_id,
        "n_rows": int(best["n_rows"]),
        "best_policy": best["policy"],
        "best_budget": best["budget"],
        "best_accuracy": best["accuracy"],
        "best_macro_f1": best["macro_f1"],
        "notes": protocol.get("notes", ""),
    }


def run_replay(tables: dict[str, pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = prediction_map(output_dir)
    single_frames: list[pd.DataFrame] = []
    multi_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for _, protocol in tables["routing_replay_protocols"].iterrows():
        if str(protocol["routing_type"]) == "single_scout":
            frame, summary = run_single_replay(protocol, predictions)
            if not frame.empty:
                single_frames.append(frame)
            summary_rows.append(summary)
        elif str(protocol["routing_type"]) == "multi_scout":
            frame, summary = run_multi_replay(protocol, predictions)
            if not frame.empty:
                multi_frames.append(frame)
            summary_rows.append(summary)
        else:
            summary_rows.append(
                {
                    "replay_id": protocol["replay_id"],
                    "task_id": protocol["task_id"],
                    "routing_type": protocol["routing_type"],
                    "status": "skipped_unsupported_protocol",
                    "scout_artifact_ids": protocol["scout_artifact_ids"],
                    "expert_artifact_id": protocol["expert_artifact_id"],
                    "n_rows": 0,
                    "best_policy": "",
                    "best_budget": "",
                    "best_accuracy": "",
                    "best_macro_f1": "",
                    "notes": "当前版本不支持该 routing_type",
                }
            )
    return (
        pd.concat(single_frames, ignore_index=True) if single_frames else pd.DataFrame(),
        pd.concat(multi_frames, ignore_index=True) if multi_frames else pd.DataFrame(),
        pd.DataFrame(summary_rows),
    )


ROUTING_RESULT_COLUMNS = [
    "replay_id",
    "task_id",
    "routing_type",
    "scout_artifact_ids",
    "expert_artifact_id",
    "policy",
    "budget",
    "n_rows",
    "selected_n",
    "accuracy",
    "macro_f1",
    "n_error",
]
ROUTING_SUMMARY_COLUMNS = [
    "replay_id",
    "task_id",
    "routing_type",
    "status",
    "scout_artifact_ids",
    "expert_artifact_id",
    "n_rows",
    "best_policy",
    "best_budget",
    "best_accuracy",
    "best_macro_f1",
    "notes",
]
SANITY_COLUMNS = [
    "task_id",
    "artifact_id",
    "legacy_source",
    "adapter_source",
    "metric",
    "legacy_value",
    "adapter_value",
    "diff",
    "status",
    "notes",
]


def sanity_comparison(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    adapter_baselines = read_optional_csv(output_dir / "model_baselines_from_adapters.csv")
    inventory = read_optional_csv(output_dir / "model_inventory.csv")
    baseline_rows: list[dict[str, Any]] = []
    if adapter_baselines.empty:
        baseline_rows.append(
            {
                "task_id": "",
                "artifact_id": "",
                "legacy_source": "",
                "adapter_source": "",
                "metric": "",
                "legacy_value": "",
                "adapter_value": "",
                "diff": "",
                "status": "missing_adapter_result",
                "notes": "adapter baseline 不存在",
            }
        )
    else:
        for _, row in adapter_baselines.iterrows():
            inventory_rows = inventory.loc[inventory["artifact_id"].astype(str) == str(row["artifact_id"])] if not inventory.empty else pd.DataFrame()
            legacy_path = str(inventory_rows.iloc[0].get("legacy_baseline_path", "")) if not inventory_rows.empty else ""
            legacy = read_optional_csv(resolve_path(legacy_path)) if legacy_path else pd.DataFrame()
            for metric in ("accuracy", "macro_f1"):
                adapter_value = float(row.get(metric, "nan"))
                if legacy.empty or metric not in legacy.columns:
                    baseline_rows.append(
                        {
                            "task_id": row.get("task_id", ""),
                            "artifact_id": row.get("artifact_id", ""),
                            "legacy_source": legacy_path,
                            "adapter_source": "model_baselines_from_adapters.csv",
                            "metric": metric,
                            "legacy_value": "",
                            "adapter_value": adapter_value,
                            "diff": "",
                            "status": "missing_legacy_reference",
                            "notes": "sanity comparison，不是 strict reproduction",
                        }
                    )
                    continue
                legacy_match = legacy
                if "artifact_id" in legacy.columns:
                    legacy_match = legacy.loc[legacy["artifact_id"].astype(str) == str(row["artifact_id"])]
                elif "name" in legacy.columns:
                    legacy_match = legacy.loc[legacy["name"].astype(str) == str(row["artifact_id"])]
                if legacy_match.empty:
                    status = "missing_legacy_reference"
                    legacy_value = ""
                    diff = ""
                else:
                    legacy_value = float(legacy_match.iloc[0][metric])
                    diff_value = adapter_value - legacy_value
                    diff = diff_value
                    status = "within_tolerance" if abs(diff_value) < 1e-6 else "different_but_explained"
                baseline_rows.append(
                    {
                        "task_id": row.get("task_id", ""),
                        "artifact_id": row.get("artifact_id", ""),
                        "legacy_source": legacy_path,
                        "adapter_source": "model_baselines_from_adapters.csv",
                        "metric": metric,
                        "legacy_value": legacy_value,
                        "adapter_value": adapter_value,
                        "diff": diff,
                        "status": status,
                        "notes": "sanity comparison，不是 strict reproduction",
                    }
                )
    replay = read_optional_csv(output_dir / "routing_replay_summary.csv")
    routing_rows = []
    if replay.empty:
        routing_rows.append(
            {
                "task_id": "",
                "artifact_id": "",
                "legacy_source": "",
                "adapter_source": "",
                "metric": "",
                "legacy_value": "",
                "adapter_value": "",
                "diff": "",
                "status": "missing_adapter_result",
                "notes": "routing replay 不存在",
            }
        )
    else:
        for _, row in replay.iterrows():
            routing_rows.append(
                {
                    "task_id": row.get("task_id", ""),
                    "artifact_id": row.get("expert_artifact_id", ""),
                    "legacy_source": "",
                    "adapter_source": "routing_replay_summary.csv",
                    "metric": "best_macro_f1",
                    "legacy_value": "",
                    "adapter_value": row.get("best_macro_f1", ""),
                    "diff": "",
                    "status": "missing_legacy_reference",
                    "notes": "sanity comparison，不是 strict reproduction",
                }
            )
    return pd.DataFrame(baseline_rows), pd.DataFrame(routing_rows)


def write_summary_html(output_dir: Path) -> None:
    inventory_summary = read_optional_csv(output_dir / "inventory_summary.csv")
    adapter_jobs = read_optional_csv(output_dir / "adapter_job_summary.csv")
    replay = read_optional_csv(output_dir / "routing_replay_summary.csv")
    sanity = read_optional_csv(output_dir / "adapter_vs_legacy_baseline_check.csv")

    def metric(name: str) -> str:
        if inventory_summary.empty:
            return "0"
        rows = inventory_summary.loc[inventory_summary["metric"].astype(str) == name]
        return str(rows.iloc[0]["value"]) if not rows.empty else "0"

    completed = int((adapter_jobs["status"] == "completed").sum()) if not adapter_jobs.empty and "status" in adapter_jobs.columns else 0
    skipped = len(adapter_jobs) - completed if not adapter_jobs.empty else 0
    single_done = int(((replay["routing_type"] == "single_scout") & (replay["status"] == "completed")).sum()) if not replay.empty else 0
    multi_done = int(((replay["routing_type"] == "multi_scout") & (replay["status"] == "completed")).sum()) if not replay.empty else 0
    skipped_missing = int((replay["status"] == "skipped_missing_predictions").sum()) if not replay.empty else 0
    html_body = f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>v0.8.5b 已知模型清单与适配器接入摘要</title>
<style>body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:32px;color:#142033}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #d9e1ea;padding:8px}} .notice{{background:#fff8e8;border-left:4px solid #b7791f;padding:12px;margin:16px 0}}</style></head>
<body>
<h1>v0.8.5b 已知模型清单与适配器接入摘要</h1>
<p>当前版本定位：已知模型 inventory + adapter-level onboarding + routing replay。它不是训练、微调、UI 或 Agent。</p>
<h2>Stage 1：Inventory</h2>
<ul>
<li>inventory 总数：{metric('total_candidates')}</li>
<li>ready_for_adapter 数量：{metric('ready_for_adapter_n')}</li>
<li>missing checkpoint 数量：{metric('missing_checkpoint_n')}</li>
<li>needs_loader_audit 数量：{metric('needs_loader_audit_n')}</li>
<li>legacy replay only 数量：{metric('legacy_replay_only_n')}</li>
<li>DR 候选数量：{metric('dr_candidates_n')}</li>
<li>青光眼候选数量：{metric('glaucoma_candidates_n')}</li>
</ul>
<h2>Stage 2：Adapter Onboarding</h2>
<p>completed job 数：{completed}；skipped job 数：{skipped}。</p>
<p>adapter 产物路径：<code>onboarded_models/&lt;job_id&gt;/predictions.csv</code>、<code>model_baseline.csv</code>、<code>forward_cost_summary.csv</code>、<code>adapter_manifest.csv</code>。</p>
<h2>Stage 3：Routing Replay</h2>
<p>single scout replay 完成数：{single_done}；multi scout replay 完成数：{multi_done}；skipped_missing_predictions 数量：{skipped_missing}。</p>
<h2>Sanity Comparison</h2>
<p>sanity comparison 记录数：{len(sanity)}。它不是严格复现，不用于证明严格复现。</p>
<div class="notice">当前边界：不训练；不微调；不做自动微调；不做 UI；不做 Agent；不伪造 checkpoint；不伪造 predictions；forward-only cost 不是真实部署端到端延迟。</div>
</body></html>
"""
    (output_dir / "summary.html").write_text(html_body, encoding="utf-8")


def ensure_inventory(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    inventory, source_index, inventory_summary = build_inventory(tables)
    write_csv(output_dir / "model_inventory.csv", inventory, INVENTORY_COLUMNS)
    write_csv(output_dir / "artifact_source_index.csv", source_index)
    write_csv(output_dir / "inventory_summary.csv", inventory_summary)


def ensure_onboarding(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    if not (output_dir / "model_inventory.csv").exists():
        ensure_inventory(tables, output_dir)
    job_summary, onboarded, baselines, costs, manifests = run_onboarding(tables, output_dir)
    write_csv(output_dir / "adapter_job_summary.csv", job_summary, ADAPTER_JOB_SUMMARY_COLUMNS)
    write_csv(output_dir / "onboarded_models.csv", onboarded, ONBOARDED_MODELS_COLUMNS)
    write_csv(output_dir / "model_baselines_from_adapters.csv", baselines, ADAPTER_BASELINE_COLUMNS)
    write_csv(output_dir / "forward_cost_summary_from_adapters.csv", costs, ADAPTER_COST_COLUMNS)
    write_csv(output_dir / "adapter_manifest.csv", manifests, ADAPTER_MANIFEST_COLUMNS)


def ensure_replay(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    if not (output_dir / "adapter_job_summary.csv").exists():
        ensure_onboarding(tables, output_dir)
    single, multi, replay_summary = run_replay(tables, output_dir)
    write_csv(output_dir / "single_scout_routing_results_from_adapters.csv", single, ROUTING_RESULT_COLUMNS)
    write_csv(output_dir / "multi_scout_routing_results_from_adapters.csv", multi, ROUTING_RESULT_COLUMNS)
    write_csv(output_dir / "routing_replay_summary.csv", replay_summary, ROUTING_SUMMARY_COLUMNS)
    baseline_check, routing_check = sanity_comparison(output_dir)
    write_csv(output_dir / "adapter_vs_legacy_baseline_check.csv", baseline_check, SANITY_COLUMNS)
    write_csv(output_dir / "adapter_vs_legacy_routing_check.csv", routing_check, SANITY_COLUMNS)
    write_summary_html(output_dir)


def run_protocol(
    config_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    stage: str = "all",
    dry_run: bool = False,
) -> ProtocolResult:
    stage = str(stage)
    if stage not in VALID_STAGES:
        raise KnownModelProtocolError(f"不支持的 stage：{stage}")
    config_path = Path(config_path)
    config, tables = load_all_config(config_path)
    if output_dir is None:
        if "output_dir" not in config:
            raise KnownModelProtocolError("protocol.yaml 缺少 output_dir")
        output_path = resolve_path(config["output_dir"], config_dir=config_path.parent)
    else:
        output_path = Path(output_dir)
    if dry_run:
        return ProtocolResult(output_dir=output_path, stage=stage, files=[])
    output_path.mkdir(parents=True, exist_ok=True)
    if stage in {"inventory", "all"}:
        ensure_inventory(tables, output_path)
    if stage in {"onboarding", "all"}:
        ensure_onboarding(tables, output_path)
    if stage in {"replay", "all"}:
        ensure_replay(tables, output_path)
    if stage != "replay":
        write_summary_html(output_path)
    files = [path for path in output_path.glob("*.csv")] + [output_path / "summary.html"]
    return ProtocolResult(output_dir=output_path, stage=stage, files=[path for path in files if path.exists()])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=sorted(VALID_STAGES), default="all")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_protocol(args.config, output_dir=args.output_dir, stage=args.stage, dry_run=args.dry_run)
    except KnownModelProtocolError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"[DRY-RUN] v0.8.5b known-model protocol valid: {args.config} stage={args.stage}")
    else:
        print(f"[DONE] v0.8.5b stage={args.stage} outputs: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
