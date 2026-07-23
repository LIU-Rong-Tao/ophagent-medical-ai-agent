#!/usr/bin/env python3
"""任务无关的单 Scout / 单 Expert 确定性评测器。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_COLUMNS = {
    "artifact_id",
    "task_id",
    "dataset_id",
    "modality",
    "label_schema_id",
    "model_family",
    "prediction_csv",
    "cost_csv",
    "checkpoint_path",
    "split",
    "enabled",
}
RISK_COLUMNS = [
    "protocol_id",
    "task_id",
    "method_kind",
    "scout_artifact",
    "expert_artifact",
    "event_id",
    "event_name",
    "budget",
    "policy",
    "event_total",
    "selected_event_n",
    "corrected_event_n",
    "introduced_event_n",
    "residual_event_n",
    "event_recall",
    "event_precision",
    "event_lift_vs_budget",
    "event_scope",
]


class EvaluationError(RuntimeError):
    """输入或评测协议不满足确定性评测要求。"""


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = (base or REPO_ROOT) / path
    return candidate.resolve()


def load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise EvaluationError(f"模型产物登记表不存在：{path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REGISTRY_COLUMNS - set(frame.columns))
    if missing:
        raise EvaluationError(f"模型产物登记表缺少字段：{', '.join(missing)}")
    enabled = frame["enabled"].str.strip().str.lower().isin({"1", "true", "yes"})
    frame = frame.loc[enabled].copy()
    if frame.empty:
        raise EvaluationError("模型产物登记表中没有启用的记录")
    if frame["artifact_id"].duplicated().any():
        duplicated = frame.loc[frame["artifact_id"].duplicated(), "artifact_id"].tolist()
        raise EvaluationError(f"artifact_id 重复：{duplicated}")
    return frame


def load_protocol(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EvaluationError(f"评测协议不存在：{path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise EvaluationError("读取 YAML 协议需要安装 PyYAML") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise EvaluationError("评测协议根节点必须是对象")

    required = {
        "protocol_id",
        "task_id",
        "task_type",
        "mode",
        "selection_split",
        "evaluation_split",
        "metric_profile",
        "scouts",
        "experts",
        "budgets",
        "policies",
        "random_trials",
        "seed",
        "risk_events",
        "case_export_points",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise EvaluationError(f"评测协议缺少字段：{', '.join(missing)}")
    if payload["metric_profile"] != "classification_standard_v1":
        raise EvaluationError("v0.8.4 仅支持 classification_standard_v1 统一分类指标")
    if len(payload["scouts"]) != 1 or len(payload["experts"]) != 1:
        raise EvaluationError("当前确定性评测器仅接受一个 Scout 和一个 Expert")
    if payload["mode"] == "final" and payload["selection_split"] == payload["evaluation_split"]:
        raise EvaluationError("final 模式下 selection_split 与 evaluation_split 必须分离")
    validate_risk_events(payload["risk_events"])
    return payload


def validate_risk_events(events: object) -> None:
    if not isinstance(events, list):
        raise EvaluationError("risk_events 必须是列表")
    supported = {
        "true_label_eq",
        "true_label_gte",
        "true_label_lte",
        "pred_label_eq",
        "pred_label_gte",
        "pred_label_lte",
        "pred_label_lt",
        "pred_label_gt",
    }
    for event in events:
        if not isinstance(event, dict):
            raise EvaluationError("risk_events 的每项必须是对象")
        missing = {"event_id", "event_name"} - set(event)
        if missing:
            raise EvaluationError(f"风险事件缺少字段：{sorted(missing)}")
        configured = set(event) & supported
        if not configured:
            raise EvaluationError(
                f"风险事件 {event['event_id']} 未声明标签条件"
            )
        invalid = set(event) - (supported | {"event_id", "event_name", "scope"})
        if invalid:
            raise EvaluationError(
                f"风险事件 {event['event_id']} 包含不支持字段：{sorted(invalid)}"
            )
        for field in configured:
            try:
                int(event[field])
            except (TypeError, ValueError) as exc:
                raise EvaluationError(
                    f"风险事件 {event['event_id']} 的 {field} 必须是整数"
                ) from exc

def probability_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for column in frame.columns:
        if not column.startswith("prob_"):
            continue
        suffix = column.removeprefix("prob_")
        if suffix.isdigit():
            indexed.append((int(suffix), column))
    indexed.sort()
    if len(indexed) < 2:
        raise EvaluationError("prediction CSV 至少需要两列连续概率列 prob_0...prob_n")
    expected = list(range(len(indexed)))
    observed = [index for index, _ in indexed]
    if observed != expected:
        raise EvaluationError(f"概率列必须从 prob_0 连续编号，当前编号为：{observed}")
    return [column for _, column in indexed]


def validate_prediction_frame(
    frame: pd.DataFrame,
    probability_columns: list[str] | None = None,
) -> None:
    probability_columns = probability_columns or probability_columns_from_frame(frame)
    required = {
        "image_key",
        "true_label",
        "pred_label",
        "model_name",
        "split",
        *probability_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        if any(column.startswith("prob_") for column in missing):
            raise EvaluationError(f"prediction CSV 缺少概率列：{', '.join(missing)}")
        raise EvaluationError(f"prediction CSV 缺少字段：{', '.join(missing)}")
    if frame.empty:
        raise EvaluationError("prediction CSV 没有记录")
    if frame["image_key"].isna().any() or (frame["image_key"].astype(str).str.strip() == "").any():
        raise EvaluationError("image_key 不能为空")
    if frame["image_key"].duplicated().any():
        duplicated = frame.loc[frame["image_key"].duplicated(), "image_key"].astype(str).tolist()
        raise EvaluationError(f"image_key 存在重复记录：{duplicated[:5]}")

    try:
        probabilities = frame[probability_columns].astype(float).to_numpy()
        true_labels = frame["true_label"].astype(int).to_numpy()
    except (TypeError, ValueError) as exc:
        raise EvaluationError("概率或标签字段不是合法数值") from exc
    if not np.isfinite(probabilities).all():
        raise EvaluationError("概率列包含 NaN 或无穷值")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise EvaluationError("概率值必须位于 [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise EvaluationError("每条记录的概率和必须接近 1")
    if ((true_labels < 0) | (true_labels >= len(probability_columns))).any():
        raise EvaluationError("真实标签超出概率列定义的类别范围")


def recompute_probability_signals(
    frame: pd.DataFrame,
    probability_columns: list[str] | None = None,
) -> pd.DataFrame:
    probability_columns = probability_columns or probability_columns_from_frame(frame)
    validate_prediction_frame(frame, probability_columns)
    output = frame.copy()
    probabilities = output[probability_columns].astype(float).to_numpy()
    sorted_probabilities = np.sort(probabilities, axis=1)
    output[probability_columns] = probabilities
    output["true_label"] = output["true_label"].astype(int)
    output["pred_label"] = probabilities.argmax(axis=1).astype(int)
    output["confidence"] = sorted_probabilities[:, -1]
    output["margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    safe_probabilities = np.clip(probabilities, np.finfo(float).tiny, 1.0)
    entropy = -(probabilities * np.log(safe_probabilities)).sum(axis=1)
    output["entropy"] = entropy
    output["normalized_entropy"] = entropy / math.log(len(probability_columns))
    return output


def load_prediction_artifact(
    registry_row: pd.Series,
    protocol: dict[str, Any],
    *,
    registry_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    prediction_path = resolve_path(
        registry_row["prediction_csv"],
        base=registry_path.parent,
    )
    if not prediction_path.exists():
        prediction_path = resolve_path(registry_row["prediction_csv"])
    if not prediction_path.exists():
        raise EvaluationError(
            f"找不到 {registry_row['artifact_id']} 的 prediction CSV：{prediction_path}"
        )
    frame = pd.read_csv(prediction_path)
    probability_columns = probability_columns_from_frame(frame)
    frame = recompute_probability_signals(frame, probability_columns)
    expected_split = str(protocol["evaluation_split"])
    observed_splits = set(frame["split"].astype(str))
    if observed_splits != {expected_split}:
        raise EvaluationError(
            f"{registry_row['artifact_id']} 的 split={sorted(observed_splits)}，"
            f"与 evaluation_split={expected_split!r} 不一致"
        )
    if str(registry_row["task_id"]) != str(protocol["task_id"]):
        raise EvaluationError(f"{registry_row['artifact_id']} 的 task_id 与协议不一致")
    return frame, probability_columns


def validate_artifact_compatibility(
    scout_row: pd.Series,
    expert_row: pd.Series,
    protocol: dict[str, Any],
) -> None:
    for field in ("task_id", "dataset_id", "modality", "label_schema_id", "split"):
        scout_value = str(scout_row.get(field, ""))
        expert_value = str(expert_row.get(field, ""))
        if scout_value != expert_value:
            raise EvaluationError(
                f"Scout 与 Expert 的 {field} 不一致：{scout_value!r} != {expert_value!r}"
            )
    if str(scout_row["task_id"]) != str(protocol["task_id"]):
        raise EvaluationError("registry 的 task_id 与评测协议不一致")
    if str(scout_row["split"]) != str(protocol["evaluation_split"]):
        raise EvaluationError("registry 的 split 与 evaluation_split 不一致")


def merge_scout_expert(scout: pd.DataFrame, expert: pd.DataFrame) -> pd.DataFrame:
    scout_probabilities = probability_columns_from_frame(scout)
    expert_probabilities = probability_columns_from_frame(expert)
    if len(scout_probabilities) != len(expert_probabilities):
        raise EvaluationError("Scout 与 Expert 的类别数量不一致")
    if set(scout["image_key"].astype(str)) != set(expert["image_key"].astype(str)):
        raise EvaluationError("Scout 与 Expert 的 image_key 集合不一致")

    scout_columns = [
        "image_key",
        "true_label",
        "split",
        "pred_label",
        "confidence",
        "margin",
        "entropy",
        "normalized_entropy",
        *scout_probabilities,
    ]
    expert_columns = ["image_key", "true_label", "pred_label", *expert_probabilities]
    scout_view = scout[scout_columns].rename(
        columns={
            "pred_label": "scout_pred_label",
            "confidence": "scout_confidence",
            "margin": "scout_margin",
            "entropy": "scout_entropy",
            "normalized_entropy": "scout_normalized_entropy",
            **{column: f"scout_{column}" for column in scout_probabilities},
        }
    )
    expert_view = expert[expert_columns].rename(
        columns={
            "true_label": "expert_true_label",
            "pred_label": "expert_pred_label",
            **{column: f"expert_{column}" for column in expert_probabilities},
        }
    )
    merged = scout_view.merge(expert_view, on="image_key", how="inner", validate="one_to_one")
    if not (merged["true_label"].astype(int) == merged["expert_true_label"].astype(int)).all():
        raise EvaluationError("Scout 与 Expert 的真实标签不一致")
    merged = merged.drop(columns="expert_true_label").sort_values("image_key").reset_index(drop=True)
    return merged


def compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    truth = np.asarray(list(y_true), dtype=int)
    prediction = np.asarray(list(y_pred), dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(truth):
        raise EvaluationError("概率矩阵形状与标签数量不一致")
    n_classes = probabilities.shape[1]
    if n_classes < 2:
        raise EvaluationError("分类任务至少需要两个类别")

    if n_classes == 2:
        auroc = roc_auc_score(truth, probabilities[:, 1])
        aupr = average_precision_score(truth, probabilities[:, 1])
    else:
        aurocs: list[float] = []
        auprs: list[float] = []
        for class_index in range(n_classes):
            binary_truth = (truth == class_index).astype(int)
            if binary_truth.min() == binary_truth.max():
                continue
            aurocs.append(roc_auc_score(binary_truth, probabilities[:, class_index]))
            auprs.append(average_precision_score(binary_truth, probabilities[:, class_index]))
        auroc = float(np.mean(aurocs)) if aurocs else float("nan")
        aupr = float(np.mean(auprs)) if auprs else float("nan")

    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "macro_auroc_ovr": float(auroc),
        "macro_aupr_ovr": float(aupr),
        "n_error": int(np.sum(truth != prediction)),
    }


def selected_n_for_budget(n_cases: int, budget: float) -> int:
    if not 0 <= float(budget) <= 1:
        raise EvaluationError(f"budget 必须位于 [0, 1]，当前为 {budget}")
    return min(n_cases, max(0, int(round(n_cases * float(budget)))))


def policy_priority(frame: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "low_confidence":
        return 1.0 - frame["scout_confidence"].astype(float)
    if policy == "low_margin":
        return 1.0 - frame["scout_margin"].astype(float)
    if policy == "high_entropy":
        return frame["scout_normalized_entropy"].astype(float)
    raise EvaluationError(f"不支持的路由策略：{policy}")


def ranked_for_expert(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    ranked = pd.DataFrame(
        {
            "image_key": frame["image_key"].astype(str),
            "routing_score": policy_priority(frame, policy),
        }
    )
    return ranked.sort_values(
        ["routing_score", "image_key"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_for_expert(frame: pd.DataFrame, policy: str, selected_n: int) -> list[str]:
    if not 0 <= selected_n <= len(frame):
        raise EvaluationError("selected_n 超出病例数量范围")
    return ranked_for_expert(frame, policy).head(selected_n)["image_key"].tolist()


def route_predictions(frame: pd.DataFrame, selected_keys: set[str]) -> np.ndarray:
    selected = frame["image_key"].astype(str).isin(selected_keys).to_numpy()
    return np.where(
        selected,
        frame["expert_pred_label"].astype(int).to_numpy(),
        frame["scout_pred_label"].astype(int).to_numpy(),
    )


def routed_probabilities(frame: pd.DataFrame, selected_keys: set[str]) -> np.ndarray | None:
    scout_columns = probability_columns_with_prefix(frame, "scout_prob_")
    expert_columns = probability_columns_with_prefix(frame, "expert_prob_")
    if not scout_columns or len(scout_columns) != len(expert_columns):
        return None
    selected = frame["image_key"].astype(str).isin(selected_keys).to_numpy()
    scout_probabilities = frame[scout_columns].astype(float).to_numpy()
    expert_probabilities = frame[expert_columns].astype(float).to_numpy()
    return np.where(selected[:, None], expert_probabilities, scout_probabilities)


def probability_columns_with_prefix(frame: pd.DataFrame, prefix: str) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for column in frame.columns:
        if column.startswith(prefix) and column[len(prefix) :].isdigit():
            indexed.append((int(column[len(prefix) :]), column))
    return [column for _, column in sorted(indexed)]


def routing_metrics(frame: pd.DataFrame, selected_keys: set[str]) -> dict[str, float | int]:
    prediction = route_predictions(frame, selected_keys)
    probabilities = routed_probabilities(frame, selected_keys)
    if probabilities is None:
        return {
            "accuracy": float(accuracy_score(frame["true_label"], prediction)),
            "n_error": int(np.sum(frame["true_label"].to_numpy() != prediction)),
        }
    return compute_metrics(frame["true_label"], prediction, probabilities)

def event_mask(
    true_labels: np.ndarray, predictions: np.ndarray, event: dict[str, Any]
) -> np.ndarray:
    mask = np.ones(len(true_labels), dtype=bool)
    comparisons = {
        "true_label_eq": (true_labels, np.equal),
        "true_label_gte": (true_labels, np.greater_equal),
        "true_label_lte": (true_labels, np.less_equal),
        "pred_label_eq": (predictions, np.equal),
        "pred_label_gte": (predictions, np.greater_equal),
        "pred_label_lte": (predictions, np.less_equal),
        "pred_label_lt": (predictions, np.less),
        "pred_label_gt": (predictions, np.greater),
    }
    for field, (values, operator) in comparisons.items():
        if field in event:
            mask &= operator(values, int(event[field]))
    return mask


def risk_event_rows(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    scout_id: str,
    expert_id: str,
    *,
    method_kind: str,
    budget: float,
    policy: str,
    selected_keys: set[str],
) -> list[dict[str, Any]]:
    if not protocol["risk_events"]:
        return []
    truth = frame["true_label"].astype(int).to_numpy()
    scout_prediction = frame["scout_pred_label"].astype(int).to_numpy()
    routed_prediction = route_predictions(frame, selected_keys)
    selected = frame["image_key"].astype(str).isin(selected_keys).to_numpy()
    selected_n = int(selected.sum())
    rows: list[dict[str, Any]] = []
    for event in protocol["risk_events"]:
        base_event = event_mask(truth, scout_prediction, event)
        routed_event = event_mask(truth, routed_prediction, event)
        event_total = int(base_event.sum())
        selected_event_n = int((base_event & selected).sum())
        residual_event_n = int(routed_event.sum())
        corrected_event_n = int((base_event & ~routed_event).sum())
        introduced_event_n = int((~base_event & routed_event).sum())
        rows.append(
            {
                "protocol_id": protocol["protocol_id"],
                "task_id": protocol["task_id"],
                "method_kind": method_kind,
                "scout_artifact": scout_id,
                "expert_artifact": expert_id,
                "event_id": event["event_id"],
                "event_name": event["event_name"],
                "budget": float(budget),
                "policy": policy,
                "event_total": event_total,
                "selected_event_n": selected_event_n,
                "corrected_event_n": corrected_event_n,
                "introduced_event_n": introduced_event_n,
                "residual_event_n": residual_event_n,
                "event_recall": (
                    selected_event_n / event_total if event_total else float("nan")
                ),
                "event_precision": (
                    selected_event_n / selected_n if selected_n else float("nan")
                ),
                "event_lift_vs_budget": (
                    (selected_event_n / event_total) / budget
                    if event_total and budget > 0
                    else float("nan")
                ),
                "event_scope": event.get(
                    "scope", "label_defined_error_event_not_clinical_outcome"
                ),
            }
        )
    return rows

def random_same_budget(
    frame: pd.DataFrame,
    budgets: Iterable[float],
    trials: int,
    seed: int,
) -> pd.DataFrame:
    if trials <= 0:
        raise EvaluationError("random_trials 必须大于 0")
    rng = np.random.default_rng(seed)
    keys = frame["image_key"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        selected_n = selected_n_for_budget(len(frame), float(budget))
        samples: list[dict[str, float | int]] = []
        for _ in range(trials):
            selected = set(rng.choice(keys, size=selected_n, replace=False).tolist())
            samples.append(routing_metrics(frame, selected))
        sample_frame = pd.DataFrame(samples)
        row: dict[str, Any] = {
            "budget": float(budget),
            "selected_n": selected_n,
            "expert_call_rate": selected_n / len(frame),
            "n_trials": trials,
        }
        for metric in sample_frame.columns:
            values = sample_frame[metric].astype(float)
            row[metric] = float(values.mean())
            row[f"{metric}_ci_low"] = float(values.quantile(0.025))
            row[f"{metric}_ci_high"] = float(values.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_order(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    truth = frame["true_label"].astype(int)
    scout_correct = frame["scout_pred_label"].astype(int) == truth
    expert_correct = frame["expert_pred_label"].astype(int) == truth
    beneficial = sorted(frame.loc[~scout_correct & expert_correct, "image_key"].astype(str))
    harmful = sorted(frame.loc[scout_correct & ~expert_correct, "image_key"].astype(str))
    neutral = sorted(
        frame.loc[(scout_correct == expert_correct), "image_key"].astype(str)
    )
    return beneficial, neutral, harmful


def oracle_exact_k_curve(frame: pd.DataFrame, budgets: Iterable[float]) -> pd.DataFrame:
    beneficial, neutral, harmful = oracle_order(frame)
    order = beneficial + neutral + harmful
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        selected_n = selected_n_for_budget(len(frame), float(budget))
        metrics = routing_metrics(frame, set(order[:selected_n]))
        rows.append(
            {
                "budget": float(budget),
                "selected_n": selected_n,
                "expert_call_rate": selected_n / len(frame),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def oracle_up_to_k_curve(frame: pd.DataFrame, budgets: Iterable[float]) -> pd.DataFrame:
    beneficial, _, _ = oracle_order(frame)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        maximum_n = selected_n_for_budget(len(frame), float(budget))
        selected = beneficial[:maximum_n]
        metrics = routing_metrics(frame, set(selected))
        rows.append(
            {
                "budget": float(budget),
                "selected_n": len(selected),
                "expert_call_rate": len(selected) / len(frame),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def artifact_cost(registry_row: pd.Series, registry_path: Path) -> dict[str, Any]:
    cost_value = str(registry_row.get("cost_csv", "")).strip()
    if not cost_value:
        return {"cost_status": "missing"}
    cost_path = resolve_path(cost_value, base=registry_path.parent)
    if not cost_path.exists():
        cost_path = resolve_path(cost_value)
    if not cost_path.exists():
        return {"cost_status": "missing"}
    cost_frame = pd.read_csv(cost_path)
    artifact_id = str(registry_row["artifact_id"])
    cost_profile_id = str(registry_row.get("cost_profile_id", "")).strip()
    if "artifact_id" in cost_frame.columns:
        cost_frame = cost_frame.loc[
            cost_frame["artifact_id"].astype(str) == artifact_id
        ]
        if "cost_profile_id" in cost_frame.columns:
            if not cost_profile_id:
                raise EvaluationError(
                    f"{artifact_id} 已配置共享成本表，但 registry 缺少 cost_profile_id"
                )
            cost_frame = cost_frame.loc[
                cost_frame["cost_profile_id"].astype(str) == cost_profile_id
            ]
    elif "model_name" in cost_frame.columns:
        cost_frame = cost_frame.loc[
            cost_frame["model_name"].astype(str) == artifact_id
        ]
    if cost_frame.empty:
        return {"cost_status": "missing"}
    if len(cost_frame) != 1:
        raise EvaluationError(
            f"{artifact_id} / {cost_profile_id or '<未指定>'} 的成本记录不唯一"
        )
    row = cost_frame.iloc[0]
    if str(row.get("cost_status", "measured")) != "measured":
        return {"cost_status": "missing"}
    estimate = row.get("estimated_forward_ms_per_image", row.get("mean_ms_per_image"))
    try:
        estimate = float(estimate)
    except (TypeError, ValueError):
        return {"cost_status": "missing"}
    return {
        "cost_status": "measured",
        "estimated_forward_ms_per_image": estimate,
        "images_per_second": float(row.get("images_per_second", 1000.0 / estimate)),
        "checkpoint_mb": row.get("checkpoint_mb", ""),
        "cost_profile_id": row.get("cost_profile_id", cost_profile_id),
        "mean_ms_per_image": row.get("mean_ms_per_image", ""),
        "median_ms_per_image": row.get("median_ms_per_image", estimate),
        "std_ms_per_image": row.get("std_ms_per_image", ""),
        "cv_ms_per_image": row.get("cv_ms_per_image", ""),
        "n_repeats": row.get("n_repeats", ""),
        "peak_allocated_memory_mb": row.get("peak_allocated_memory_mb", ""),
        "timing_scope": row.get("timing_scope", "forward_only"),
        "timing_source": str(cost_path),
    }


def baseline_row(
    artifact_id: str,
    role: str,
    frame: pd.DataFrame,
    probability_columns: list[str],
    registry_row: pd.Series,
    protocol: dict[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    metrics = compute_metrics(
        frame["true_label"],
        frame["pred_label"],
        frame[probability_columns].to_numpy(),
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "task_id": protocol["task_id"],
        "artifact_id": artifact_id,
        "role": role,
        "model_family": registry_row["model_family"],
        "split": protocol["evaluation_split"],
        "n_images": len(frame),
        **metrics,
        **artifact_cost(registry_row, registry_path),
    }


def cost_fields(
    scout_cost: dict[str, Any],
    expert_cost: dict[str, Any],
    expert_call_rate: float,
) -> dict[str, Any]:
    if scout_cost.get("cost_status") != "measured" or expert_cost.get("cost_status") != "measured":
        return {
            "cost_status": "missing",
            "estimated_forward_ms_per_image": np.nan,
            "relative_forward_cost_vs_expert_only": np.nan,
            "forward_cost_reduction_vs_expert_only": np.nan,
        }
    scout_ms = float(scout_cost["estimated_forward_ms_per_image"])
    expert_ms = float(expert_cost["estimated_forward_ms_per_image"])
    estimate = scout_ms + expert_call_rate * expert_ms
    relative = estimate / expert_ms
    return {
        "cost_status": "estimated_from_measured_models",
        "estimated_forward_ms_per_image": estimate,
        "relative_forward_cost_vs_expert_only": relative,
        "forward_cost_reduction_vs_expert_only": 1.0 - relative,
    }


def routing_row(
    protocol: dict[str, Any],
    scout_id: str,
    expert_id: str,
    method_kind: str,
    policy: str,
    values: dict[str, Any],
    scout_cost: dict[str, Any],
    expert_cost: dict[str, Any],
) -> dict[str, Any]:
    expert_call_rate = float(values["expert_call_rate"])
    return {
        "protocol_id": protocol["protocol_id"],
        "task_id": protocol["task_id"],
        "split": protocol["evaluation_split"],
        "method_kind": method_kind,
        "protocol_name": f"{scout_id}_to_{expert_id}",
        "scout_artifact": scout_id,
        "expert_artifact": expert_id,
        "budget": float(values["budget"]),
        "policy": policy,
        "selected_n": int(values["selected_n"]),
        "expert_call_rate": expert_call_rate,
        "non_deployable": method_kind == "oracle",
        **{
            key: values.get(key, np.nan)
            for key in (
                "accuracy",
                "macro_f1",
                "macro_auroc_ovr",
                "macro_aupr_ovr",
                "n_error",
                "accuracy_ci_low",
                "accuracy_ci_high",
            )
        },
        **cost_fields(scout_cost, expert_cost, expert_call_rate),
    }


def case_audit_rows(
    frame: pd.DataFrame,
    protocol: dict[str, Any],
    scout_id: str,
    expert_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in protocol["case_export_points"]:
        budget = float(point["budget"])
        policy = str(point["policy"])
        selected_n = selected_n_for_budget(len(frame), budget)
        ranked = ranked_for_expert(frame, policy)
        rank_by_key = {
            key: rank
            for rank, key in enumerate(ranked["image_key"].tolist(), start=1)
        }
        score_by_key = dict(zip(ranked["image_key"], ranked["routing_score"]))
        selected = set(ranked.head(selected_n)["image_key"])
        for record in frame.to_dict(orient="records"):
            key = str(record["image_key"])
            scout_correct = int(record["scout_pred_label"]) == int(record["true_label"])
            expert_correct = int(record["expert_pred_label"]) == int(record["true_label"])
            is_selected = key in selected
            routed_prediction = (
                int(record["expert_pred_label"])
                if is_selected
                else int(record["scout_pred_label"])
            )
            if not scout_correct and expert_correct:
                case_type = "expert_correctable"
            elif scout_correct and not expert_correct:
                case_type = "expert_harmful"
            elif scout_correct and expert_correct:
                case_type = "both_correct"
            else:
                case_type = "both_wrong"
            rows.append(
                {
                    "protocol_id": protocol["protocol_id"],
                    "task_id": protocol["task_id"],
                    "budget": budget,
                    "policy": policy,
                    "image_key": key,
                    "true_label": int(record["true_label"]),
                    "scout_pred_label": int(record["scout_pred_label"]),
                    "expert_pred_label": int(record["expert_pred_label"]),
                    "routed_pred_label": routed_prediction,
                    "selected_for_expert": is_selected,
                    "routing_score": float(score_by_key[key]),
                    "routing_rank": int(rank_by_key[key]),
                    "scout_correct": scout_correct,
                    "expert_correct": expert_correct,
                    "routed_correct": routed_prediction == int(record["true_label"]),
                    "expert_corrected": is_selected and (not scout_correct) and expert_correct,
                    "expert_induced_error": is_selected and scout_correct and (not expert_correct),
                    "case_type": case_type,
                    "scout_artifact": scout_id,
                    "expert_artifact": expert_id,
                }
            )
    return rows


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_evaluation(registry_path: Path, protocol_path: Path, work_dir: Path) -> None:
    registry_path = Path(registry_path)
    protocol_path = Path(protocol_path)
    work_dir = Path(work_dir)
    registry = load_registry(registry_path)
    protocol = load_protocol(protocol_path)
    scout_id = str(protocol["scouts"][0])
    expert_id = str(protocol["experts"][0])

    indexed_registry = registry.set_index("artifact_id", drop=False)
    missing_artifacts = [
        artifact for artifact in (scout_id, expert_id) if artifact not in indexed_registry.index
    ]
    if missing_artifacts:
        raise EvaluationError(f"登记表缺少协议所需产物：{missing_artifacts}")
    scout_row = indexed_registry.loc[scout_id]
    expert_row = indexed_registry.loc[expert_id]
    validate_artifact_compatibility(scout_row, expert_row, protocol)
    scout, scout_probabilities = load_prediction_artifact(
        scout_row, protocol, registry_path=registry_path
    )
    expert, expert_probabilities = load_prediction_artifact(
        expert_row, protocol, registry_path=registry_path
    )
    merged = merge_scout_expert(scout, expert)

    scout_cost = artifact_cost(scout_row, registry_path)
    expert_cost = artifact_cost(expert_row, registry_path)
    baseline_rows = [
        baseline_row(
            scout_id,
            "scout",
            scout,
            scout_probabilities,
            scout_row,
            protocol,
            registry_path,
        ),
        baseline_row(
            expert_id,
            "expert",
            expert,
            expert_probabilities,
            expert_row,
            protocol,
            registry_path,
        ),
    ]

    routing_rows: list[dict[str, Any]] = []
    risk_rows = risk_event_rows(
        merged,
        protocol,
        scout_id,
        expert_id,
        method_kind="baseline",
        budget=0.0,
        policy="scout_only",
        selected_keys=set(),
    )
    for policy in protocol["policies"]:
        for budget in protocol["budgets"]:
            selected_n = selected_n_for_budget(len(merged), float(budget))
            selected = set(select_for_expert(merged, str(policy), selected_n))
            values = {
                "budget": float(budget),
                "selected_n": selected_n,
                "expert_call_rate": selected_n / len(merged),
                **routing_metrics(merged, selected),
            }
            routing_rows.append(
                routing_row(
                    protocol,
                    scout_id,
                    expert_id,
                    "uncertainty",
                    str(policy),
                    values,
                    scout_cost,
                    expert_cost,
                )
            )

    random_rows = random_same_budget(
        merged,
        protocol["budgets"],
        int(protocol["random_trials"]),
        int(protocol["seed"]),
    )
    for values in random_rows.to_dict(orient="records"):
        routing_rows.append(
            routing_row(
                protocol,
                scout_id,
                expert_id,
                "random",
                "random_same_budget",
                values,
                scout_cost,
                expert_cost,
            )
        )
    for policy, curve in (
        ("oracle_exact_k", oracle_exact_k_curve(merged, protocol["budgets"])),
        ("oracle_up_to_k", oracle_up_to_k_curve(merged, protocol["budgets"])),
    ):
        for values in curve.to_dict(orient="records"):
            routing_rows.append(
                routing_row(
                    protocol,
                    scout_id,
                    expert_id,
                    "oracle",
                    policy,
                    values,
                    scout_cost,
                    expert_cost,
                )
            )

    for policy in protocol["policies"]:
        for budget in protocol["budgets"]:
            selected_n = selected_n_for_budget(len(merged), float(budget))
            selected = set(select_for_expert(merged, str(policy), selected_n))
            risk_rows.extend(
                risk_event_rows(
                    merged,
                    protocol,
                    scout_id,
                    expert_id,
                    method_kind="uncertainty",
                    budget=float(budget),
                    policy=str(policy),
                    selected_keys=selected,
                )
            )
    write_csv(pd.DataFrame(baseline_rows), work_dir / "model_baselines.csv")
    write_csv(pd.DataFrame(routing_rows), work_dir / "routing_results.csv")
    write_csv(pd.DataFrame(risk_rows, columns=RISK_COLUMNS), work_dir / "risk_results.csv")
    write_csv(
        pd.DataFrame(case_audit_rows(merged, protocol, scout_id, expert_id)),
        work_dir / "case_audit.csv",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="任务无关的 Scout-to-Expert 分类评测器")
    parser.add_argument("--registry", type=Path, required=True, help="模型产物登记表 CSV")
    parser.add_argument("--protocol", type=Path, required=True, help="评测协议 YAML/JSON")
    parser.add_argument("--work-dir", type=Path, required=True, help="四张原始结果表输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_evaluation(args.registry, args.protocol, args.work_dir)
    except EvaluationError as exc:
        raise SystemExit(f"评测失败：{exc}") from exc
    print(f"[完成] 已生成任务无关评测原始表：{args.work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
