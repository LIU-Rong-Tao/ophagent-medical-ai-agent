"""v0.8.6 Model Hub Demo 的纯数据读取与交互预览函数。"""

from __future__ import annotations

import json
from itertools import combinations
from functools import lru_cache
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from app.model_providers import (
    OphBenchProvider,
    ProviderHealth,
    TimmProvider,
    build_provider_catalog,
)
from scripts.routing.model_metadata import canonical_timm_artifact_id, timm_pretraining_source
from scripts.routing.timm_adapter_runtime import normalize_prediction_frame


OUTPUT_FILES = {
    "models": "model_hub_snapshot.csv",
    "pairings": "pairing_results.csv",
    "traces": "case_routing_trace.csv",
    "config": "run_config.yaml",
    "manifest": "artifact_manifest.csv",
    "report": "report.html",
}
DEFAULT_MODEL_HUB_ROOT = Path(__file__).resolve().parents[1] / "experiments/model_hub"

DR_RISK_EVENTS = {
    "large_undergrading": {"true_min": 4, "pred_max": 2},
    "referable_miss": {"true_min": 2, "pred_max": 1},
    "severe_pdr_miss": {"true_min": 3, "pred_max": 2},
}

ONLINE_CASE_COLUMNS = [
    "task_id",
    "image_key",
    "image_path",
    "primary_scout_artifact_id",
    "primary_scout_pred_label",
    "scout_pred_labels",
    "scout_confidences",
    "scout_margins",
    "scout_entropies",
    "scout_probabilities",
    "scout_disagreement",
    "routing_policy",
    "routing_score",
    "routing_cutoff",
    "requested_budget",
    "realized_budget",
    "is_reviewed_by_expert",
    "expert_artifact_id",
    "expert_artifact_ids",
    "expert_pred_label",
    "expert_probabilities",
    "final_pred_label",
    "final_source",
]


def build_online_case_view(detail: pd.DataFrame) -> pd.DataFrame:
    required = {"task_id", "image_key"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"在线病例缺少连接字段：{sorted(missing)}")
    columns = [column for column in ONLINE_CASE_COLUMNS if column in detail.columns]
    online = detail[columns].copy()
    online.attrs["display_scope"] = "online_only"
    return online


def available_routing_policies(n_routes: int, n_experts: int) -> list[str]:
    if int(n_routes) <= 0 or int(n_experts) <= 0:
        return []
    if int(n_routes) == 1:
        return ["low_confidence", "low_margin", "high_entropy"]
    return ["disagreement_then_uncertainty", "mean_uncertainty"]


def attach_retrospective_evidence(
    online: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    join_keys = ["task_id", "image_key"]
    if "primary_scout_artifact_id" in online.columns and "primary_scout_artifact_id" in evidence.columns:
        join_keys.append("primary_scout_artifact_id")
    missing = [key for key in join_keys if key not in online.columns or key not in evidence.columns]
    if missing:
        raise ValueError(f"回顾性证据缺少连接字段：{missing}")
    if online.duplicated(join_keys).any() or evidence.duplicated(join_keys).any():
        raise ValueError("回顾性证据连接键存在重复记录")

    evidence_columns = [
        column
        for column in evidence.columns
        if column in join_keys
        or column == "true_label"
        or column.startswith("was_")
        or column.startswith("dr_")
    ]
    merged = online.merge(
        evidence[evidence_columns],
        on=join_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if merged["_merge"].ne("both").any():
        missing_keys = merged.loc[merged["_merge"].ne("both"), join_keys].to_dict("records")
        raise ValueError(f"回顾性证据缺少病例记录：{missing_keys[:3]}")
    merged = merged.drop(columns="_merge")
    merged.attrs["display_scope"] = "research_only"
    return merged


def enrich_cost_curve(curve: pd.DataFrame, *, metric_column: str = "accuracy") -> pd.DataFrame:
    enriched = curve.copy()
    if metric_column not in enriched.columns:
        raise ValueError(f"成本曲线缺少任务主指标：{metric_column}")
    cost_column = "estimated_total_compute_ms_per_image"
    valid_costs = pd.to_numeric(enriched[cost_column], errors="coerce")
    if "cost_status" in enriched.columns:
        valid_costs = valid_costs.mask(enriched["cost_status"].astype(str).eq("unmeasured"))
    positive = valid_costs.loc[valid_costs > 0]
    if positive.empty:
        enriched["relative_cost"] = np.nan
        enriched["is_pareto"] = False
        return enriched
    baseline = float(positive.min())
    enriched["relative_cost"] = valid_costs / baseline
    enriched = enriched.sort_values(["relative_cost", metric_column], ascending=[True, False])
    best_metric = -np.inf
    pareto: list[bool] = []
    metric_values = pd.to_numeric(enriched[metric_column], errors="coerce")
    for metric_value, cost_value in zip(metric_values, enriched["relative_cost"], strict=True):
        is_better = bool(
            np.isfinite(cost_value)
            and np.isfinite(metric_value)
            and metric_value > best_metric
        )
        pareto.append(is_better)
        if is_better:
            best_metric = float(metric_value)
    enriched["is_pareto"] = pareto
    return enriched


def select_operating_points(
    curve: pd.DataFrame,
    *,
    metric_column: str = "accuracy",
) -> dict[str, pd.Series]:
    valid = curve.dropna(subset=[metric_column, "relative_cost"]).copy()
    if valid.empty:
        return {}
    efficient = valid.sort_values(["relative_cost", metric_column], ascending=[True, False]).iloc[0]
    performance = valid.sort_values([metric_column, "relative_cost"], ascending=[False, True]).iloc[0]
    valid["utility"] = valid[metric_column].astype(float) - 0.01 * valid["relative_cost"].astype(float)
    balanced = valid.sort_values(["utility", "relative_cost"], ascending=[False, True]).iloc[0]
    return {"efficient": efficient, "balanced": balanced, "performance": performance}


def task_metric_profile(task_registry: pd.DataFrame, task_id: str) -> dict[str, Any]:
    required = {"task_id", "primary_metric", "display_metrics"}
    missing = required - set(task_registry.columns)
    if missing:
        raise ValueError(f"任务注册表缺少指标字段：{sorted(missing)}")
    rows = task_registry.loc[task_registry["task_id"].astype(str).eq(str(task_id))]
    if rows.empty:
        raise ValueError(f"任务注册表中找不到任务：{task_id}")
    row = rows.iloc[0]
    display_metrics = [value for value in str(row["display_metrics"]).split("|") if value]
    return {
        "primary_metric": str(row["primary_metric"]),
        "display_metrics": display_metrics,
    }


def task_evaluation_summary(
    task_id: str,
    metrics: dict[str, Any],
    detail: pd.DataFrame,
) -> dict[str, Any]:
    if metrics.get("risk_semantics") == "label_based_safety_proxy_not_clinical_gold_standard":
        labels = {
            "large_undergrading": "大跨度低估",
            "referable_miss": "可转诊漏检",
            "severe_pdr_miss": "重症漏检",
        }
        rows = []
        for name in DR_RISK_EVENTS:
            rows.append(
                {
                    "事件": labels[name],
                    "总事件": int(metrics.get(f"dr_{name}_event_total", 0)),
                    "送专家": int(metrics.get(f"dr_{name}_selected_n", 0)),
                    "纠正": int(metrics.get(f"dr_{name}_resolved_n", 0)),
                    "残余": int(metrics.get(f"dr_{name}_residual_n", 0)),
                }
            )
        return {"profile": "disease_proxy", "task_id": task_id, "rows": pd.DataFrame(rows)}

    if not {"true_label", "final_pred_label"}.issubset(detail.columns):
        return {"profile": "unavailable", "task_id": task_id, "rows": pd.DataFrame()}
    truth = detail["true_label"].astype(int)
    prediction = detail["final_pred_label"].astype(int)
    rows = []
    for label in sorted(truth.unique()):
        mask = truth.eq(label)
        support = int(mask.sum())
        rows.append(
            {
                "类别": int(label),
                "样本数": support,
                "召回率": float(prediction.loc[mask].eq(label).mean()) if support else np.nan,
            }
        )
    return {"profile": "generic_multiclass", "task_id": task_id, "rows": pd.DataFrame(rows)}


def load_registered_training_models(model_hub_root: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    required = {
        "model_id",
        "task_id",
        "dataset_id",
        "artifact_id",
        "model_family",
        "architecture",
        "label_space",
        "n_classes",
        "prediction_source",
        "prediction_path",
        "adapter_status",
        "compatibility_status",
    }
    registration_paths = [
        *model_hub_root.glob("runs/training/*/*/*/registration_record.csv"),
        *model_hub_root.glob("runs/inference/*/*/*/registration_record.csv"),
    ]
    for path in sorted(registration_paths):
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"训练注册记录缺少字段 {sorted(missing)}：{path}")
        frame = frame.copy()
        if "pretraining_source" in frame.columns:
            timm_rows = frame["pretraining_source"].astype(str).isin(
                {
                    "timm_pretrained",
                    "imagenet1k",
                    "imagenet12k_ft_imagenet1k",
                    "imagenet21k_ft_imagenet1k",
                }
            )
            for index in frame.index[timm_rows]:
                artifact_id = canonical_timm_artifact_id(
                    frame.at[index, "architecture"],
                    frame.at[index, "task_id"],
                )
                frame.at[index, "artifact_id"] = artifact_id
                frame.at[index, "model_id"] = f"{frame.at[index, 'task_id']}::{artifact_id}"
                if str(frame.at[index, "pretraining_source"]) == "timm_pretrained":
                    frame.at[index, "pretraining_source"] = timm_pretraining_source(
                        frame.at[index, "architecture"]
                    )
        frame["registration_file"] = str(path.resolve())
        frame["registration_mtime_ns"] = path.stat().st_mtime_ns
        records.append(frame)
    if not records:
        return pd.DataFrame()
    combined = pd.concat(records, ignore_index=True, sort=False).sort_values("registration_mtime_ns")
    return combined.drop_duplicates("model_id", keep="last").reset_index(drop=True)


def load_model_hub_outputs(
    output_dir: Path,
    *,
    model_hub_root: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"output_dir": output_dir, "missing": []}
    for key, name in OUTPUT_FILES.items():
        path = output_dir / name
        if not path.exists():
            result["missing"].append(name)
            result[key] = pd.DataFrame() if path.suffix == ".csv" else path
        elif path.suffix == ".csv":
            result[key] = pd.read_csv(path)
        else:
            result[key] = path
    dynamic_models = load_registered_training_models(model_hub_root or DEFAULT_MODEL_HUB_ROOT)
    if not dynamic_models.empty:
        combined = pd.concat([result["models"], dynamic_models], ignore_index=True, sort=False)
        if "model_id" not in combined.columns:
            combined["model_id"] = combined["task_id"].astype(str) + "::" + combined["artifact_id"].astype(str)
        result["models"] = combined.drop_duplicates("model_id", keep="last").reset_index(drop=True)
    return result


def split_task_models(models: pd.DataFrame, task_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_models = models.loc[models["task_id"].astype(str) == str(task_id)].copy()
    ready = (
        task_models["prediction_source"].astype(str).ne("missing")
        & task_models["compatibility_status"].astype(str).eq("ready_for_pairing")
    )
    return (
        task_models.loc[ready].reset_index(drop=True),
        task_models.loc[~ready].reset_index(drop=True),
    )


def _recipe_is_enabled(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_global_model_catalog(
    models: pd.DataFrame,
    *,
    target_task_id: str,
    recipes: pd.DataFrame,
) -> pd.DataFrame:
    """保留全局模型，并标明其相对目标任务的可用状态。"""

    target_rows = models.loc[models["task_id"].astype(str).eq(str(target_task_id))]
    if target_rows.empty:
        raise ValueError(f"找不到目标任务 {target_task_id} 的标签空间定义")
    target_label_spaces = target_rows["label_space"].dropna().astype(str).unique().tolist()
    target_class_counts = pd.to_numeric(target_rows["n_classes"], errors="coerce").dropna().astype(int).unique().tolist()
    if len(target_label_spaces) != 1 or len(target_class_counts) != 1:
        raise ValueError(f"目标任务 {target_task_id} 的标签空间或类别数不一致")
    target_label_space = target_label_spaces[0]
    target_n_classes = target_class_counts[0]

    if recipes.empty:
        enabled_recipes = recipes
    else:
        generic_registry = "supported_model_families" in recipes.columns
        required = {"enabled", "supported_model_families"} if generic_registry else {"model_family", "architecture", "enabled"}
        if not required.issubset(recipes.columns):
            raise ValueError(f"训练 recipe registry 缺少字段：{sorted(required - set(recipes.columns))}")
        enabled_recipes = recipes.loc[recipes["enabled"].map(_recipe_is_enabled)].copy()

    def has_training_adapter(row: pd.Series) -> bool:
        if enabled_recipes.empty:
            return False
        if "supported_model_families" in enabled_recipes.columns:
            family = str(row.get("model_family", ""))
            return any(
                family in value if isinstance(value, list) else family in str(value).split("|")
                for value in enabled_recipes["supported_model_families"]
            )
        return bool(
            (
                enabled_recipes["model_family"].astype(str).eq(str(row.get("model_family", "")))
                & enabled_recipes["architecture"].astype(str).eq(str(row.get("architecture", "")))
            ).any()
        )

    def classify(row: pd.Series) -> tuple[str, str]:
        same_task = str(row.get("task_id", "")) == str(target_task_id)
        ready = str(row.get("compatibility_status", "")) == "ready_for_pairing"
        source = str(row.get("prediction_source", ""))
        adapter_complete = str(row.get("adapter_status", "")) == "completed"
        if same_task and ready and source == "adapter" and adapter_complete:
            return "direct_inference", "当前任务在线推理链已验证"
        if same_task and ready and source == "legacy":
            return "offline_replay", "当前任务仅有冻结 prediction，在线加载链待验证"
        if has_training_adapter(row):
            source_classes = int(row.get("n_classes", 0) or 0)
            if same_task:
                return "adaptable", "已有训练 Adapter，可创建新的受控适配任务"
            return (
                "adaptable",
                f"当前权重为 {source_classes} 类，目标任务为 {target_n_classes} 类；"
                "可创建当前任务适配任务，不可直接推理",
            )
        return "blocked", "当前任务不可直接使用，且缺少已注册训练 Adapter / recipe"

    catalog = models.copy()
    classifications = catalog.apply(classify, axis=1)
    catalog["target_task_id"] = str(target_task_id)
    catalog["target_label_space"] = target_label_space
    catalog["target_n_classes"] = target_n_classes
    catalog["target_task_status"] = classifications.map(lambda value: value[0])
    catalog["target_task_reason"] = classifications.map(lambda value: value[1])
    return catalog


def _default_external_providers(models: pd.DataFrame):
    inventory = []
    for architecture, rows in models.groupby("architecture", dropna=True):
        architecture = str(architecture).strip()
        if architecture:
            inventory.append(
                {
                    "model_id": architecture,
                    "display_name": architecture,
                    "family_id": str(rows.iloc[0].get("model_family", architecture)),
                }
            )
    return [TimmProvider(inventory), OphBenchProvider()]


def build_unified_model_catalog(
    models: pd.DataFrame,
    *,
    target_task_id: str,
    recipes: pd.DataFrame,
    providers=None,
) -> pd.DataFrame:
    """聚合本地任务产物与外部基础模型，但仅允许任务 checkpoint 进入路由。"""

    local = build_global_model_catalog(
        models, target_task_id=target_task_id, recipes=recipes
    ).copy()
    local["provider_id"] = "local_artifact"
    local["source_model_id"] = local["model_family"].astype(str)
    local["source_checkpoint_id"] = local["artifact_id"].astype(str)
    local["model_id"] = "local_artifact::" + local["artifact_id"].astype(str)
    local["unified_model_id"] = local["model_id"]
    local["source_access_status"] = "open"
    local["base_adapter_status"] = local["adapter_status"].map(
        lambda value: "smoke_test_passed" if str(value) == "completed" else "not_implemented"
    )
    local["base_adapter_ready"] = local["adapter_status"].astype(str).eq("completed")
    declared_task_checkpoint = (
        local["task_checkpoint"].map(lambda value: True if pd.isna(value) else bool(value))
        if "task_checkpoint" in local
        else pd.Series(True, index=local.index)
    )
    declared_inference_ready = (
        local["task_inference_ready"].map(lambda value: True if pd.isna(value) else bool(value))
        if "task_inference_ready" in local
        else pd.Series(True, index=local.index)
    )
    declared_route_eligible = (
        local["route_eligible"].map(lambda value: True if pd.isna(value) else bool(value))
        if "route_eligible" in local
        else pd.Series(True, index=local.index)
    )
    local["task_checkpoint"] = declared_task_checkpoint
    local["task_inference_ready"] = declared_inference_ready & local["target_task_status"].isin(
        {"direct_inference", "offline_replay"}
    )
    local["route_eligible"] = (
        declared_route_eligible
        & local["task_checkpoint"]
        & local["task_inference_ready"]
    )
    local["task_compatibility_status"] = local["target_task_status"]

    provider_catalog = build_provider_catalog(
        providers if providers is not None else _default_external_providers(models)
    )
    external_rows = []
    for record in provider_catalog.records:
        adaptable = record.base_adapter_ready
        external_rows.append(
            {
                "model_id": record.unified_model_id,
                "unified_model_id": record.unified_model_id,
                "provider_id": record.provider_id,
                "source_model_id": record.source_model_id,
                "source_checkpoint_id": record.source_checkpoint_id or "",
                "task_id": "",
                "dataset_id": "",
                "dataset_display_name": "",
                "dataset_source": record.provider_id,
                "artifact_id": record.source_checkpoint_id or record.source_model_id,
                "model_family": record.family_id,
                "architecture": record.architecture,
                "label_space": "",
                "n_classes": 0,
                "prediction_source": "missing",
                "adapter_status": record.base_adapter_status.value,
                "compatibility_status": "blocked",
                "role_candidates": "",
                "pretraining_source": record.provider_id,
                "target_task_id": str(target_task_id),
                "target_label_space": local.iloc[0]["target_label_space"],
                "target_n_classes": local.iloc[0]["target_n_classes"],
                "target_task_status": "adaptable" if adaptable else "blocked",
                "target_task_reason": (
                    "基础模型 Adapter 已验证；需创建当前任务 checkpoint 后方可推理与路由"
                    if adaptable
                    else "仅提供注册信息；基础 Adapter 与当前任务 checkpoint 尚未就绪"
                ),
                "source_access_status": record.source_access_status.value,
                "base_adapter_status": record.base_adapter_status.value,
                "task_compatibility_status": record.task_compatibility_status.value,
                "base_adapter_ready": record.base_adapter_ready,
                "task_inference_ready": False,
                "route_eligible": False,
                "task_checkpoint": False,
                "model_name": record.model_name,
                "year": record.year,
                "venue": record.venue,
                "model_category": record.model_category,
                "modalities": "|".join(record.modalities),
                "capabilities": "|".join(record.capabilities),
                "pretraining_data_summary": record.pretraining_data_summary,
                "pretraining_strategy": record.pretraining_strategy,
                "reported_summary": record.reported_summary,
                "paper_url": record.paper_url,
                "code_url": record.code_url,
                "runtime_phase": record.runtime_phase,
                "verification_status": record.verification_status,
                "license": record.license,
                "license_verified": record.license_verified,
                "checkpoint_id": record.source_checkpoint_id or "",
                "checkpoint_name": record.checkpoint_name,
                "checkpoint_provider": record.checkpoint_provider,
                "weight_url": record.weight_url,
                "access_type": record.access_type,
                "requires_auth": record.requires_auth,
                "framework": record.framework,
                "input_size": record.input_size,
                "normalization": record.normalization,
                "embedding_dim": record.embedding_dim,
                "sha256": record.sha256,
                "last_verified": record.last_verified,
                "checkpoint_verification_status": record.checkpoint_verification_status,
            }
        )
    external = pd.DataFrame(external_rows)
    catalog = pd.concat([local, external], ignore_index=True, sort=False)
    if catalog["model_id"].duplicated().any():
        raise ValueError("Unified Model Hub catalog contains duplicate model IDs")
    catalog.attrs["provider_health"] = (
        ProviderHealth("local_artifact", True, "available", "local artifact provider available"),
        *provider_catalog.health,
    )
    return catalog


def route_eligible_model_ids(catalog: pd.DataFrame) -> list[str]:
    """返回经过任务 checkpoint 与推理就绪双重门控的路由候选。"""

    eligible = (
        catalog["route_eligible"].astype(bool)
        & catalog["task_checkpoint"].astype(bool)
        & catalog["task_inference_ready"].astype(bool)
    )
    return sorted(catalog.loc[eligible, "model_id"].astype(str).tolist())


def dr_risk_summary(
    detail: pd.DataFrame,
    *,
    task_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    enriched = detail.copy()
    if task_id != "aptos_dr_5class" or "primary_scout_pred_label" not in enriched:
        return {"risk_semantics": "not_applicable"}, enriched
    if (
        "primary_scout_artifact_id" not in enriched
        or enriched["primary_scout_artifact_id"].fillna("").astype(str).str.strip().eq("").all()
    ):
        return {"risk_semantics": "not_applicable"}, enriched

    truth = enriched["true_label"].astype(int).to_numpy()
    scout_prediction = enriched["primary_scout_pred_label"].astype(int).to_numpy()
    final_prediction = enriched["final_pred_label"].astype(int).to_numpy()
    selected = enriched["is_reviewed_by_expert"].astype(bool).to_numpy()
    metrics: dict[str, Any] = {
        "risk_semantics": "label_based_safety_proxy_not_clinical_gold_standard"
    }
    for name, spec in DR_RISK_EVENTS.items():
        scout_event = (truth >= spec["true_min"]) & (scout_prediction <= spec["pred_max"])
        final_residual = (truth >= spec["true_min"]) & (final_prediction <= spec["pred_max"])
        event_total = int(scout_event.sum())
        selected_n = int((scout_event & selected).sum())
        residual_n = int((scout_event & final_residual).sum())
        resolved_n = int((scout_event & ~final_residual).sum())
        enriched[f"dr_{name}_scout_event"] = scout_event
        enriched[f"dr_{name}_selected"] = scout_event & selected
        enriched[f"dr_{name}_final_residual"] = scout_event & final_residual
        metrics[f"dr_{name}_event_total"] = event_total
        metrics[f"dr_{name}_selected_n"] = selected_n
        metrics[f"dr_{name}_capture_rate"] = selected_n / event_total if event_total else np.nan
        metrics[f"dr_{name}_resolved_n"] = resolved_n
        metrics[f"dr_{name}_residual_n"] = residual_n
    enriched["dr_any_scout_risk_event"] = enriched[
        [f"dr_{name}_scout_event" for name in DR_RISK_EVENTS]
    ].any(axis=1)
    return metrics, enriched


def _model_row(models: pd.DataFrame, task_id: str, artifact_id: str) -> pd.Series:
    rows = models.loc[
        (models["task_id"].astype(str) == str(task_id))
        & (models["artifact_id"].astype(str) == str(artifact_id))
    ]
    if rows.empty:
        raise ValueError(f"任务 {task_id} 中找不到模型 {artifact_id}")
    row = rows.iloc[0]
    if str(row.get("compatibility_status", "")) != "ready_for_pairing":
        raise ValueError(f"模型 {artifact_id} 当前不可参与组合")
    return row


@lru_cache(maxsize=64)
def _load_prediction_file(path_value: str, n_classes: int) -> pd.DataFrame:
    path = Path(path_value)
    return normalize_prediction_frame(path, num_classes=n_classes).sort_values("image_key").reset_index(drop=True)


def _imagefolder_relative_key(path_value: object) -> str:
    text = str(path_value or "").replace("\\", "/").strip()
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return text


def _repair_duplicate_image_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if "image_key" not in frame.columns or not frame["image_key"].duplicated().any():
        return frame
    if "image_path" not in frame.columns:
        return frame
    if frame["image_path"].astype(str).duplicated().any():
        return frame
    repaired = frame.copy()
    repaired["image_key"] = repaired["image_path"].map(_imagefolder_relative_key)
    return repaired.sort_values("image_key").reset_index(drop=True)


def _load_model_prediction(row: pd.Series) -> pd.DataFrame:
    path = Path(str(row.get("prediction_path", "")))
    if not path.is_file():
        raise ValueError(f"模型 {row.get('artifact_id')} 的 prediction 文件不存在：{path}")
    frame = _load_prediction_file(str(path.resolve()), int(row["n_classes"])).copy()
    frame = _repair_duplicate_image_keys(frame)
    if frame["image_key"].duplicated().any():
        raise ValueError(f"模型 {row.get('artifact_id')} 的 image_key 重复")
    return frame


def _candidate_subsets(values: list[str], max_size: int) -> list[list[str]]:
    unique = list(dict.fromkeys(str(value) for value in values if str(value)))
    upper = min(max(1, int(max_size)), len(unique))
    subsets: list[list[str]] = []
    for size in range(1, upper + 1):
        subsets.extend([list(items) for items in combinations(unique, size)])
    return subsets


def estimate_global_composition_count(
    *,
    n_scouts: int,
    n_experts: int,
    max_scouts: int,
    max_experts: int,
    n_budgets: int,
) -> int:
    """Estimate how many routing points the interactive global scan would evaluate."""
    scout_count = max(0, int(n_scouts))
    expert_count = max(0, int(n_experts))
    budget_count = max(0, int(n_budgets))
    max_scout_count = min(max(0, int(max_scouts)), scout_count)
    max_expert_count = min(max(0, int(max_experts)), expert_count)
    if scout_count == 0 or expert_count == 0 or budget_count == 0:
        return 0

    single_scout_subsets = scout_count if max_scout_count >= 1 else 0
    multi_scout_subsets = sum(comb(scout_count, size) for size in range(2, max_scout_count + 1))
    # Single-route policies: low confidence / margin / entropy.
    # Multi-route policies: disagreement-first and mean uncertainty.
    routing_policy_points = single_scout_subsets * 3 + multi_scout_subsets * 2
    expert_subsets = sum(comb(expert_count, size) for size in range(1, max_expert_count + 1))
    return routing_policy_points * expert_subsets * budget_count


def scan_global_composition_candidates(
    models: pd.DataFrame,
    *,
    task_id: str,
    scout_ids: list[str],
    expert_ids: list[str],
    budgets: list[float],
    max_scouts: int = 2,
    max_experts: int = 2,
    primary_metric: str = "accuracy",
) -> pd.DataFrame:
    """Explore the current registered model pool without publishing a frozen protocol."""
    rows: list[dict[str, Any]] = []
    budget_values = sorted({float(value) for value in budgets})
    for scout_combo in _candidate_subsets(scout_ids, max_scouts):
        for expert_combo in _candidate_subsets(expert_ids, max_experts):
            policies = available_routing_policies(len(scout_combo), len(expert_combo))
            if not policies:
                continue
            if len(expert_combo) == 1:
                handoff_modes = [("fixed_expert", expert_combo[0])]
            else:
                handoff_modes = [("mean_probability_pool", None)]
            for handoff_mode, fixed_expert_id in handoff_modes:
                for policy in policies:
                    for budget in budget_values:
                        base = {
                            "task_id": task_id,
                            "scout_ids": "|".join(scout_combo),
                            "primary_scout_id": scout_combo[0],
                            "configured_expert_ids": "|".join(expert_combo),
                            "expert_handoff_mode": handoff_mode,
                            "fixed_expert_id": fixed_expert_id or "",
                            "routing_policy": policy,
                            "requested_budget": budget,
                            "scan_status": "completed",
                            "global_scan_semantics": "exploratory_current_model_pool",
                        }
                        try:
                            metrics, _ = evaluate_exploratory_composition(
                                models,
                                task_id=task_id,
                                scout_ids=scout_combo,
                                primary_scout_id=scout_combo[0],
                                expert_ids=expert_combo,
                                expert_handoff_mode=handoff_mode,
                                fixed_expert_id=fixed_expert_id,
                                policy=policy,
                                requested_budget=budget,
                            )
                        except ValueError as exc:
                            rows.append({**base, "scan_status": "failed", "error_message": str(exc)})
                            continue
                        rows.append({**base, **metrics})

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["is_pareto"] = False
    frame["relative_cost"] = np.nan
    frame["global_rank_primary"] = pd.NA
    frame["global_utility"] = np.nan

    completed = frame.loc[frame["scan_status"].eq("completed")].copy()
    if completed.empty or primary_metric not in completed.columns:
        return frame

    try:
        completed = enrich_cost_curve(completed, metric_column=primary_metric)
    except ValueError:
        completed["relative_cost"] = np.nan
        completed["is_pareto"] = False
    completed = completed.sort_values(
        [primary_metric, "estimated_total_compute_ms_per_image"],
        ascending=[False, True],
        na_position="last",
    ).copy()
    completed["global_rank_primary"] = np.arange(1, len(completed) + 1)
    metric_values = pd.to_numeric(completed[primary_metric], errors="coerce")
    relative_cost = pd.to_numeric(completed["relative_cost"], errors="coerce")
    completed["global_utility"] = metric_values - 0.01 * relative_cost

    for column in ["is_pareto", "relative_cost", "global_rank_primary", "global_utility"]:
        frame.loc[completed.index, column] = completed[column]
    return frame


def _aligned_predictions(
    models: pd.DataFrame,
    *,
    task_id: str,
    artifact_ids: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    rows = {artifact_id: _model_row(models, task_id, artifact_id) for artifact_id in artifact_ids}
    frames = {artifact_id: _load_model_prediction(row) for artifact_id, row in rows.items()}
    key_sets = {artifact_id: set(frame["image_key"].astype(str)) for artifact_id, frame in frames.items()}
    reference_keys = next(iter(key_sets.values()))
    if any(keys != reference_keys for keys in key_sets.values()):
        raise ValueError("所选模型的 image_key 集合不一致，禁止静默丢弃病例")
    ordered_keys = sorted(reference_keys)
    aligned = {
        artifact_id: frame.set_index("image_key").loc[ordered_keys].reset_index()
        for artifact_id, frame in frames.items()
    }
    reference_truth = next(iter(aligned.values()))["true_label"].astype(int).to_numpy()
    if any(
        not np.array_equal(reference_truth, frame["true_label"].astype(int).to_numpy())
        for frame in aligned.values()
    ):
        raise ValueError("所选模型的真实标签记录不一致")
    return aligned, rows


def _routing_ranking(
    scouts: dict[str, pd.DataFrame],
    *,
    primary_scout_id: str,
    policy: str,
) -> pd.DataFrame:
    primary = scouts[primary_scout_id]
    predictions = [frame["pred_label"].astype(int).to_numpy() for frame in scouts.values()]
    uncertainties = [1.0 - frame["confidence"].astype(float).to_numpy() for frame in scouts.values()]
    disagreement = np.ptp(np.column_stack(predictions), axis=1) > 0
    if len(scouts) == 1:
        if policy == "low_confidence":
            score = 1.0 - primary["confidence"].astype(float).to_numpy()
        elif policy == "low_margin":
            score = 1.0 - primary["margin"].astype(float).to_numpy()
        elif policy == "high_entropy":
            score = primary["entropy"].astype(float).to_numpy()
        else:
            raise ValueError(f"单 Scout 不支持路由策略：{policy}")
    else:
        mean_uncertainty = np.column_stack(uncertainties).mean(axis=1)
        if policy == "mean_uncertainty":
            score = mean_uncertainty
        elif policy == "disagreement_then_uncertainty":
            score = disagreement.astype(float) * 2.0 + mean_uncertainty
        else:
            raise ValueError(f"多 Scout 不支持路由策略：{policy}")
    ranking = pd.DataFrame(
        {
            "image_key": primary["image_key"].astype(str),
            "routing_score": score,
            "scout_disagreement": disagreement,
        }
    ).sort_values(["routing_score", "image_key"], ascending=[False, True], kind="mergesort")
    ranking = ranking.reset_index(drop=True)
    ranking["review_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def _mean_probability_prediction(frames: list[pd.DataFrame], n_classes: int) -> np.ndarray:
    probability_columns = [f"prob_{index}" for index in range(n_classes)]
    probabilities = np.stack(
        [frame[probability_columns].astype(float).to_numpy() for frame in frames], axis=0
    ).mean(axis=0)
    return probabilities.argmax(axis=1)


def _numeric_cost(row: pd.Series) -> float:
    try:
        value = float(row.get("forward_cost_ms_per_image", np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) and value > 0 else float("nan")


def evaluate_exploratory_composition(
    models: pd.DataFrame,
    *,
    task_id: str,
    scout_ids: list[str],
    primary_scout_id: str | None,
    expert_ids: list[str],
    policy: str,
    expert_handoff_mode: str = "mean_probability_pool",
    fixed_expert_id: str | None = None,
    requested_budget: float | None = None,
    top_n: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scout_ids = list(dict.fromkeys(map(str, scout_ids)))
    configured_expert_ids = list(dict.fromkeys(map(str, expert_ids)))
    if not configured_expert_ids:
        active_expert_ids: list[str] = []
        expert_handoff_mode = "none"
    elif expert_handoff_mode == "fixed_expert":
        selected_expert_id = str(fixed_expert_id or configured_expert_ids[0])
        if selected_expert_id not in configured_expert_ids:
            raise ValueError("固定专家必须包含在已选专家模型中")
        active_expert_ids = [selected_expert_id]
    elif expert_handoff_mode == "mean_probability_pool":
        active_expert_ids = configured_expert_ids
    else:
        raise ValueError(f"不支持的专家接管方式：{expert_handoff_mode}")
    artifact_ids = list(dict.fromkeys([*scout_ids, *active_expert_ids]))
    if not artifact_ids:
        raise ValueError("至少选择一个 Scout 或 Expert")
    if scout_ids and primary_scout_id not in scout_ids:
        raise ValueError("主 Scout 必须包含在 Scout 选择中")
    aligned, rows = _aligned_predictions(models, task_id=task_id, artifact_ids=artifact_ids)
    reference = aligned[artifact_ids[0]]
    truth = reference["true_label"].astype(int).to_numpy()
    n_cases = len(reference)
    n_classes = int(rows[artifact_ids[0]]["n_classes"])

    expert_prediction: np.ndarray | None = None
    if active_expert_ids:
        expert_prediction = _mean_probability_prediction(
            [aligned[artifact_id] for artifact_id in active_expert_ids], n_classes
        )

    if scout_ids:
        primary = aligned[str(primary_scout_id)]
        ranking = _routing_ranking(
            {artifact_id: aligned[artifact_id] for artifact_id in scout_ids},
            primary_scout_id=str(primary_scout_id),
            policy=policy,
        )
        primary_prediction = primary["pred_label"].astype(int).to_numpy()
        if active_expert_ids:
            selected_n = (
                min(n_cases, max(0, int(top_n)))
                if top_n is not None
                else min(n_cases, max(0, int(round(n_cases * float(requested_budget or 0.0)))))
            )
            selected_keys = set(ranking.head(selected_n)["image_key"].astype(str))
            selected = reference["image_key"].astype(str).isin(selected_keys).to_numpy()
            final_prediction = np.where(selected, expert_prediction, primary_prediction)
            composition_mode = "scout_to_expert_pool" if expert_handoff_mode == "mean_probability_pool" and len(active_expert_ids) > 1 else "scout_to_expert"
        else:
            selected_n = 0
            selected = np.zeros(n_cases, dtype=bool)
            final_prediction = primary_prediction
            composition_mode = "scout_only"
    else:
        primary = None
        primary_prediction = expert_prediction.copy()
        ranking = pd.DataFrame(
            {
                "image_key": reference["image_key"].astype(str),
                "routing_score": np.nan,
                "scout_disagreement": False,
                "review_rank": np.arange(1, n_cases + 1),
            }
        )
        selected_n = n_cases
        selected = np.ones(n_cases, dtype=bool)
        final_prediction = expert_prediction
        composition_mode = "expert_pool_only" if expert_handoff_mode == "mean_probability_pool" and len(active_expert_ids) > 1 else "expert_only"

    ranking_by_key = ranking.set_index("image_key")
    image_keys = reference["image_key"].astype(str)
    selected_keys_for_cutoff = set(image_keys.loc[selected].tolist())
    selected_scores = ranking.loc[ranking["image_key"].astype(str).isin(selected_keys_for_cutoff), "routing_score"]
    routing_cutoff = float(selected_scores.min()) if not selected_scores.dropna().empty else np.nan
    scout_predictions = {
        artifact_id: aligned[artifact_id]["pred_label"].astype(int).tolist()
        for artifact_id in scout_ids
    }
    scout_confidences = {
        artifact_id: aligned[artifact_id]["confidence"].astype(float).tolist()
        for artifact_id in scout_ids
    }
    scout_margins = {
        artifact_id: aligned[artifact_id]["margin"].astype(float).tolist()
        for artifact_id in scout_ids
    }
    scout_entropies = {
        artifact_id: aligned[artifact_id]["entropy"].astype(float).tolist()
        for artifact_id in scout_ids
    }
    probability_columns = [f"prob_{index}" for index in range(n_classes)]
    scout_probabilities = {
        artifact_id: aligned[artifact_id][probability_columns].astype(float).to_numpy().tolist()
        for artifact_id in scout_ids
    }
    expert_probabilities = {
        artifact_id: aligned[artifact_id][probability_columns].astype(float).to_numpy().tolist()
        for artifact_id in active_expert_ids
    }
    detail = pd.DataFrame(
        {
            "task_id": task_id,
            "image_key": image_keys,
            "image_path": reference.get("image_path", pd.Series([""] * n_cases)),
            "true_label": truth,
            "primary_scout_artifact_id": primary_scout_id or "",
            "primary_scout_pred_label": primary_prediction,
            "scout_pred_labels": [
                json.dumps({artifact: values[index] for artifact, values in scout_predictions.items()})
                for index in range(n_cases)
            ],
            "scout_confidences": [
                json.dumps({artifact: values[index] for artifact, values in scout_confidences.items()})
                for index in range(n_cases)
            ],
            "scout_margins": [
                json.dumps({artifact: values[index] for artifact, values in scout_margins.items()})
                for index in range(n_cases)
            ],
            "scout_entropies": [
                json.dumps({artifact: values[index] for artifact, values in scout_entropies.items()})
                for index in range(n_cases)
            ],
            "scout_probabilities": [
                json.dumps({artifact: values[index] for artifact, values in scout_probabilities.items()})
                for index in range(n_cases)
            ],
            "expert_artifact_id": active_expert_ids[0] if len(active_expert_ids) == 1 else "",
            "expert_artifact_ids": "|".join(active_expert_ids),
            "expert_handoff_mode": expert_handoff_mode,
            "expert_pred_label": expert_prediction if expert_prediction is not None else np.nan,
            "expert_probabilities": [
                json.dumps({artifact: values[index] for artifact, values in expert_probabilities.items()})
                for index in range(n_cases)
            ],
            "routing_score": image_keys.map(ranking_by_key["routing_score"]),
            "routing_cutoff": routing_cutoff,
            "review_rank": image_keys.map(ranking_by_key["review_rank"]).astype(int),
            "scout_disagreement": image_keys.map(ranking_by_key["scout_disagreement"]).astype(bool),
            "is_reviewed_by_expert": selected,
            "final_pred_label": final_prediction,
            "final_source": np.where(selected, "expert", "scout"),
            "was_scout_correct": primary_prediction == truth,
            "was_expert_correct": expert_prediction == truth if expert_prediction is not None else False,
            "was_final_correct": final_prediction == truth,
        }
    )

    scout_costs = [_numeric_cost(rows[artifact_id]) for artifact_id in scout_ids]
    expert_costs = [_numeric_cost(rows[artifact_id]) for artifact_id in active_expert_ids]
    realized_budget = selected_n / n_cases if n_cases else 0.0
    costs_available = all(np.isfinite(value) for value in [*scout_costs, *expert_costs])
    total_cost = (
        float(sum(scout_costs) + realized_budget * sum(expert_costs))
        if costs_available
        else np.nan
    )
    parallel_cost = (
        float((max(scout_costs) if scout_costs else 0.0) + realized_budget * (max(expert_costs) if expert_costs else 0.0))
        if costs_available
        else np.nan
    )
    metrics: dict[str, Any] = {
        "task_id": task_id,
        "composition_mode": composition_mode,
        "scout_ids": "|".join(scout_ids),
        "primary_scout_id": primary_scout_id or "",
        "configured_expert_ids": "|".join(configured_expert_ids),
        "active_expert_ids": "|".join(active_expert_ids),
        "expert_ids": "|".join(active_expert_ids),
        "n_scout": len(scout_ids),
        "n_expert": len(active_expert_ids),
        "expert_handoff_mode": expert_handoff_mode,
        "expert_aggregation": "mean_probability" if len(active_expert_ids) > 1 else "single",
        "routing_policy": policy,
        "requested_budget": selected_n / n_cases if top_n is not None and n_cases else float(requested_budget or 0.0),
        "selected_n": selected_n,
        "realized_budget": realized_budget,
        "accuracy": float(accuracy_score(truth, final_prediction)),
        "macro_f1": float(f1_score(truth, final_prediction, average="macro", zero_division=0)),
        "qwk": (
            float(cohen_kappa_score(truth, final_prediction, weights="quadratic"))
            if task_id == "aptos_dr_5class"
            else np.nan
        ),
        "estimated_total_compute_ms_per_image": total_cost,
        "estimated_parallel_latency_ms_per_image": parallel_cost,
        "preview_semantics": "exploratory_preview_not_published",
    }
    risk_metrics, detail = dr_risk_summary(detail, task_id=task_id)
    metrics.update(risk_metrics)
    return metrics, detail


def pairing_trace_base(
    traces: pd.DataFrame,
    *,
    pairing_id: str,
    policy: str,
) -> pd.DataFrame:
    selected = traces.loc[
        (traces["pairing_id"].astype(str) == pairing_id)
        & (traces["routing_policy"].astype(str) == policy)
    ].copy()
    if selected.empty:
        return selected
    return selected.sort_values(["requested_budget", "review_rank"]).drop_duplicates(
        "image_key", keep="first"
    )


def derive_budget_preview(
    traces: pd.DataFrame,
    pairings: pd.DataFrame,
    *,
    pairing_id: str,
    policy: str,
    requested_budget: float | None = None,
    top_n: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base = pairing_trace_base(traces, pairing_id=pairing_id, policy=policy)
    if base.empty:
        return {}, base
    n_cases = len(base)
    if top_n is not None:
        selected_n = min(n_cases, max(0, int(top_n)))
        requested = selected_n / n_cases
    else:
        requested = float(requested_budget or 0.0)
        selected_n = min(n_cases, max(0, int(round(n_cases * requested))))
    selected = base["review_rank"].astype(int) <= selected_n
    final_prediction = np.where(
        selected,
        base["expert_pred_label"].astype(int),
        base["scout_pred_labels"].map(
            lambda value: int(json.loads(value)[str(base["primary_scout_artifact_id"].iloc[0])])
        ),
    )
    truth = base["true_label"].astype(int).to_numpy()
    reference = pairings.loc[
        (pairings["pairing_id"].astype(str) == pairing_id)
        & (pairings["routing_policy"].astype(str) == policy)
        & (pairings["status"].astype(str) == "completed")
    ]
    qwk_enabled = not reference.empty and str(reference.iloc[0].get("qwk_status", "")) == "computed"
    scout_sum = float(reference.iloc[0].get("scout_cost_sum_ms_per_image", np.nan)) if not reference.empty else np.nan
    scout_parallel = float(reference.iloc[0].get("scout_parallel_scenario_ms_per_image", np.nan)) if not reference.empty else np.nan
    expert_cost = float(reference.iloc[0].get("expert_cost_ms_per_image", np.nan)) if not reference.empty else np.nan
    realized = selected_n / n_cases
    detail = base.copy()
    detail["is_reviewed_by_expert"] = selected
    selected_scores = detail.loc[selected, "routing_score"]
    detail["routing_cutoff"] = float(selected_scores.min()) if not selected_scores.dropna().empty else np.nan
    detail["primary_scout_pred_label"] = base["scout_pred_labels"].map(
        lambda value: int(json.loads(value)[str(base["primary_scout_artifact_id"].iloc[0])])
    )
    detail["final_pred_label"] = final_prediction
    detail["final_source"] = np.where(selected, "expert", "scout")
    detail["was_final_correct"] = final_prediction == truth
    metrics = {
        "requested_budget": requested,
        "selected_n": selected_n,
        "realized_budget": realized,
        "accuracy": float(accuracy_score(truth, final_prediction)),
        "macro_f1": float(f1_score(truth, final_prediction, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(truth, final_prediction, weights="quadratic")) if qwk_enabled else np.nan,
        "estimated_total_compute_ms_per_image": scout_sum + realized * expert_cost,
        "estimated_parallel_latency_ms_per_image": scout_parallel + realized * expert_cost,
        "preview_semantics": "exploratory_preview_not_published",
    }
    task_id = str(base["task_id"].iloc[0]) if "task_id" in base else ""
    risk_metrics, detail = dr_risk_summary(detail, task_id=task_id)
    metrics.update(risk_metrics)
    return metrics, detail
