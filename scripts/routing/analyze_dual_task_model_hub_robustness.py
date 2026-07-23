#!/usr/bin/env python3
"""Summarize dual-task Model Hub robustness without running model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.routing import run_interactive_model_hub as model_hub  # noqa: E402

DEFAULT_OUTPUT = Path(
    "experiments/opening_risk_routing_closure/dual_task_robustness.csv"
)
DEFAULT_COST_OUTPUT = Path(
    "experiments/opening_risk_routing_closure/h100_cost_evidence.csv"
)
DEFAULT_PROTOCOL_OUTPUT = Path(
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "dual_task_independent_confirmatory_protocol.json"
)

TASKS = {
    "aptos_dr_5class": {
        "display_name": "APTOS2019 DR five-class",
        "n_classes": 5,
        "qwk_enabled": True,
        "proxy_threshold": 3,
        "config": Path(
            "experiments/opening_risk_routing_closure/configs/protocols/"
            "aptos_h100_ten_model_locked_test.yaml"
        ),
        "test_trace": Path(
            "experiments/opening_risk_routing_closure/outputs/"
            "model_hub_test_locked_ten_model/case_routing_trace.csv"
        ),
        "primary_pairing": "aptos_locked_performance_primary",
        "baseline_model": "swin_tiny",
        "primary_metric": "qwk",
    },
    "glaucoma_3class": {
        "display_name": "Glaucoma three-class",
        "n_classes": 3,
        "qwk_enabled": False,
        "proxy_threshold": 2,
        "config": Path(
            "experiments/model_hub/tasks/glaucoma_3class/configs/"
            "glaucoma_h100_locked_test.yaml"
        ),
        "test_trace": Path(
            "experiments/model_hub/tasks/glaucoma_3class/outputs/"
            "test_locked/case_routing_trace.csv"
        ),
        "primary_pairing": "glaucoma_locked_multi",
        "baseline_model": "glaucoma_retfound_dinov2",
        "primary_metric": "macro_f1",
    },
}

APTOS_COST_FILES = {
    "convnext_tiny": Path(
        "experiments/opening_risk_routing_closure/replays/convnext_tiny/"
        "full_replay_20260722/costs/forward_cost.json"
    ),
    "eyeclip_cfp": Path(
        "experiments/opening_risk_routing_closure/replays/eyeclip_cfp/"
        "20260723T082659Z/costs/forward_cost.json"
    ),
    "flair": Path(
        "experiments/opening_risk_routing_closure/replays/flair/"
        "full_replay_20260722/costs/forward_cost.json"
    ),
    "keepfit_cfp": Path(
        "experiments/opening_risk_routing_closure/replays/keepfit_cfp/"
        "20260723T082655Z/costs/forward_cost.json"
    ),
    "ret_clip": Path(
        "experiments/opening_risk_routing_closure/replays/ret_clip/"
        "20260723T082658Z/costs/forward_cost.json"
    ),
    "retfound_green": Path(
        "experiments/opening_risk_routing_closure/replays/retfound_green/"
        "full_replay_20260722/costs/forward_cost.json"
    ),
    "retizero": Path(
        "experiments/opening_risk_routing_closure/replays/retizero/"
        "20260723T082829Z/costs/forward_cost.json"
    ),
    "swin_tiny": Path(
        "experiments/opening_risk_routing_closure/replays/swin_tiny/"
        "full_replay_20260722/costs/forward_cost.json"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: str, base_seed: int) -> int:
    token = ":".join(parts)
    return base_seed ^ int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)


def metric_values(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    qwk_enabled: bool,
) -> dict[str, float]:
    values = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
    }
    if qwk_enabled:
        values["qwk"] = float(
            cohen_kappa_score(truth, prediction, weights="quadratic")
        )
    return values


def percentile_interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if finite.size == 0:
        return np.nan, np.nan
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def bootstrap_metric_rows(
    *,
    task_id: str,
    split: str,
    candidate_id: str,
    truth: np.ndarray,
    scout: np.ndarray,
    final: np.ndarray,
    qwk_enabled: bool,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    point_scout = metric_values(truth, scout, qwk_enabled=qwk_enabled)
    point_final = metric_values(truth, final, qwk_enabled=qwk_enabled)
    samples: dict[str, list[float]] = {
        f"scout_{name}": [] for name in point_scout
    }
    samples.update({f"final_{name}": [] for name in point_final})
    samples.update({f"delta_{name}": [] for name in point_final})
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_truth = truth[indices]
        sampled_scout = scout[indices]
        sampled_final = final[indices]
        scout_values = metric_values(
            sampled_truth, sampled_scout, qwk_enabled=qwk_enabled
        )
        final_values = metric_values(
            sampled_truth, sampled_final, qwk_enabled=qwk_enabled
        )
        for name in final_values:
            samples[f"scout_{name}"].append(scout_values[name])
            samples[f"final_{name}"].append(final_values[name])
            samples[f"delta_{name}"].append(final_values[name] - scout_values[name])

    rows: list[dict[str, Any]] = []
    for name in point_final:
        for estimate, point in (
            ("scout", point_scout[name]),
            ("route", point_final[name]),
            ("route_minus_scout", point_final[name] - point_scout[name]),
        ):
            sample_prefix = {
                "scout": "scout",
                "route": "final",
                "route_minus_scout": "delta",
            }[estimate]
            key = f"{sample_prefix}_{name}"
            lower, upper = percentile_interval(samples[key])
            rows.append(
                {
                    "analysis_section": "route_metric_ci",
                    "task_id": task_id,
                    "split": split,
                    "candidate_id": candidate_id,
                    "estimate": estimate,
                    "metric": name,
                    "class_id": "",
                    "point": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n": len(truth),
                    "bootstrap_repeats": repeats,
                    "notes": "case-level paired bootstrap",
                }
            )
    return rows


def proxy_masks(
    truth: np.ndarray,
    scout: np.ndarray,
    final: np.ndarray,
    *,
    threshold: int,
) -> dict[str, np.ndarray]:
    scout_event = (truth >= threshold) & (scout < threshold)
    final_event = (truth >= threshold) & (final < threshold)
    return {
        "scout_proxy_event": scout_event,
        "final_proxy_event": final_event,
        "corrected": scout_event & ~final_event,
        "introduced": ~scout_event & final_event,
        "net_reduction": (scout_event.astype(int) - final_event.astype(int)),
    }


def bootstrap_proxy_rows(
    *,
    task_id: str,
    split: str,
    candidate_id: str,
    truth: np.ndarray,
    scout: np.ndarray,
    final: np.ndarray,
    threshold: int,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    masks = proxy_masks(truth, scout, final, threshold=threshold)
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {name: [] for name in masks}
    for _ in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_masks = proxy_masks(
            truth[indices], scout[indices], final[indices], threshold=threshold
        )
        for name, values in sampled_masks.items():
            sampled[name].append(float(values.mean()))
    rows = []
    for name, values in masks.items():
        lower, upper = percentile_interval(sampled[name])
        rows.append(
            {
                "analysis_section": "proxy_error_variation",
                "task_id": task_id,
                "split": split,
                "candidate_id": candidate_id,
                "estimate": "rate",
                "metric": name,
                "class_id": "",
                "point": float(values.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(truth),
                "bootstrap_repeats": repeats,
                "notes": (
                    f"label-order proxy threshold={threshold}; "
                    "not a clinical-outcome event"
                ),
            }
        )
    return rows


def bootstrap_class_rows(
    *,
    task_id: str,
    split: str,
    candidate_id: str,
    truth: np.ndarray,
    scout: np.ndarray,
    final: np.ndarray,
    n_classes: int,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sampled: dict[tuple[str, str, int], list[float]] = {}
    for estimate in ("scout", "route"):
        for metric in ("recall", "f1"):
            for class_id in range(n_classes):
                sampled[(estimate, metric, class_id)] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_truth = truth[indices]
        for estimate, predictions in (("scout", scout), ("route", final)):
            sampled_prediction = predictions[indices]
            recalls = recall_score(
                sampled_truth,
                sampled_prediction,
                labels=list(range(n_classes)),
                average=None,
                zero_division=0,
            )
            f1_values = f1_score(
                sampled_truth,
                sampled_prediction,
                labels=list(range(n_classes)),
                average=None,
                zero_division=0,
            )
            for class_id in range(n_classes):
                sampled[(estimate, "recall", class_id)].append(recalls[class_id])
                sampled[(estimate, "f1", class_id)].append(f1_values[class_id])

    rows = []
    for estimate, predictions in (("scout", scout), ("route", final)):
        point_values = {
            "recall": recall_score(
                truth,
                predictions,
                labels=list(range(n_classes)),
                average=None,
                zero_division=0,
            ),
            "f1": f1_score(
                truth,
                predictions,
                labels=list(range(n_classes)),
                average=None,
                zero_division=0,
            ),
        }
        for metric, values in point_values.items():
            for class_id, point in enumerate(values):
                lower, upper = percentile_interval(
                    sampled[(estimate, metric, class_id)]
                )
                rows.append(
                    {
                        "analysis_section": "class_level_ci",
                        "task_id": task_id,
                        "split": split,
                        "candidate_id": candidate_id,
                        "estimate": estimate,
                        "metric": metric,
                        "class_id": class_id,
                        "point": float(point),
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n": int((truth == class_id).sum()),
                        "bootstrap_repeats": repeats,
                        "notes": "case-level bootstrap; class support shown in n",
                    }
                )
    return rows


def complementarity_rows(
    *,
    task_id: str,
    split: str,
    candidate_id: str,
    truth: np.ndarray,
    scout: np.ndarray,
    expert: np.ndarray,
) -> list[dict[str, Any]]:
    states = {
        "both_correct": (scout == truth) & (expert == truth),
        "scout_wrong_expert_right": (scout != truth) & (expert == truth),
        "scout_right_expert_wrong": (scout == truth) & (expert != truth),
        "both_wrong": (scout != truth) & (expert != truth),
    }
    return [
        {
            "analysis_section": "error_complementarity",
            "task_id": task_id,
            "split": split,
            "candidate_id": candidate_id,
            "estimate": "rate",
            "metric": name,
            "class_id": "",
            "point": float(mask.mean()),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n": int(mask.sum()),
            "bootstrap_repeats": 0,
            "notes": "primary Scout versus registered Expert",
        }
        for name, mask in states.items()
    ]


def primary_scout_predictions(trace: pd.DataFrame) -> np.ndarray:
    artifact_id = str(trace["primary_scout_artifact_id"].iloc[0])
    return np.asarray(
        [int(json.loads(value)[artifact_id]) for value in trace["scout_pred_labels"]]
    )


def reconstruct_validation(
    config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = model_hub.load_yaml_or_json(config_path)
    pairings = model_hub.read_csv(
        config["pairing_protocols"], config_dir=config_path.parent
    )
    budgets = sorted(
        {
            budget
            for value in pairings["budget_grid"]
            for budget in model_hub.parse_budgets(value)
        }
    )
    config["selection_split"] = "val"
    config["evaluation_split"] = "val"
    config["case_trace_budgets"] = budgets
    config["pairing_expansion"] = {"enabled": False}
    hub = model_hub.build_prediction_asset_hub(config, config_path=config_path)
    results, traces = model_hub.evaluate_pairings(
        config, config_path=config_path, hub=hub
    )
    return hub, results, traces


def selection_stability_rows(
    *,
    task_id: str,
    traces: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    primary_metric: str,
    qwk_enabled: bool,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidate_predictions: dict[str, pd.Series] = {}
    truth_by_key: pd.Series | None = None
    for candidate_id, group in traces.groupby("pairing_id", sort=True):
        indexed = group.sort_values("image_key").set_index("image_key")
        candidate_predictions[str(candidate_id)] = indexed["final_pred_label"].astype(int)
        current_truth = indexed["true_label"].astype(int)
        if truth_by_key is None:
            truth_by_key = current_truth
        elif not truth_by_key.equals(current_truth):
            raise ValueError(f"{task_id} validation traces do not align")
    baseline = baseline_frame.sort_values("image_key").set_index("image_key")
    candidate_predictions[f"baseline::{baseline_frame.attrs['artifact_id']}"] = (
        baseline["pred_label"].astype(int)
    )
    if truth_by_key is None:
        raise ValueError(f"{task_id} has no validation trace")
    if not truth_by_key.equals(baseline["true_label"].astype(int)):
        raise ValueError(f"{task_id} baseline and route labels do not align")

    keys = truth_by_key.index
    truth = truth_by_key.to_numpy()
    arrays = {
        candidate_id: predictions.loc[keys].to_numpy()
        for candidate_id, predictions in candidate_predictions.items()
    }
    wins = {candidate_id: 0 for candidate_id in arrays}
    ranks = {candidate_id: [] for candidate_id in arrays}
    rng = np.random.default_rng(seed)
    candidate_order = sorted(arrays)
    for _ in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        values = {}
        for candidate_id in candidate_order:
            metrics = metric_values(
                truth[indices], arrays[candidate_id][indices], qwk_enabled=qwk_enabled
            )
            values[candidate_id] = metrics[primary_metric]
        ranked = sorted(candidate_order, key=lambda item: (-values[item], item))
        wins[ranked[0]] += 1
        for rank, candidate_id in enumerate(ranked, start=1):
            ranks[candidate_id].append(rank)

    rows = []
    for candidate_id in candidate_order:
        rank_array = np.asarray(ranks[candidate_id])
        rows.append(
            {
                "analysis_section": "validation_selection_stability",
                "task_id": task_id,
                "split": "val",
                "candidate_id": candidate_id,
                "estimate": "bootstrap_selection_frequency",
                "metric": primary_metric,
                "class_id": "",
                "point": wins[candidate_id] / repeats,
                "ci_lower": float(np.percentile(rank_array, 2.5)),
                "ci_upper": float(np.percentile(rank_array, 97.5)),
                "n": len(truth),
                "bootstrap_repeats": repeats,
                "notes": (
                    f"median_rank={float(np.median(rank_array)):.1f}; "
                    "comparison limited to frozen candidates plus strong baseline"
                ),
            }
        )
    return rows


def prediction_frame(
    hub: pd.DataFrame,
    artifact_id: str,
    *,
    n_classes: int,
) -> pd.DataFrame:
    row = hub.loc[hub["artifact_id"].eq(artifact_id)].iloc[0]
    exclusions = set(filter(None, str(row["excluded_image_keys"]).split("|")))
    return model_hub.normalize_and_validate_prediction(
        Path(row["prediction_path"]),
        n_classes=n_classes,
        excluded_image_keys=exclusions,
    )


def analyze_task(
    task_id: str,
    spec: dict[str, Any],
    *,
    repeats: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    config_path = REPO_ROOT / spec["config"]
    validation_hub, _, validation_traces = reconstruct_validation(config_path)
    test_traces = pd.read_csv(REPO_ROOT / spec["test_trace"], low_memory=False)
    rows: list[dict[str, Any]] = []
    for split, traces in (("val", validation_traces), ("test", test_traces)):
        for candidate_id, group in traces.groupby("pairing_id", sort=True):
            group = group.sort_values("image_key").reset_index(drop=True)
            truth = group["true_label"].astype(int).to_numpy()
            scout = primary_scout_predictions(group)
            final = group["final_pred_label"].astype(int).to_numpy()
            expert = group["expert_pred_label"].astype(int).to_numpy()
            common = {
                "task_id": task_id,
                "split": split,
                "candidate_id": str(candidate_id),
                "truth": truth,
                "scout": scout,
                "final": final,
                "repeats": repeats,
            }
            rows.extend(
                bootstrap_metric_rows(
                    **common,
                    qwk_enabled=spec["qwk_enabled"],
                    seed=stable_seed(
                        task_id, split, str(candidate_id), "metrics", base_seed=base_seed
                    ),
                )
            )
            rows.extend(
                bootstrap_proxy_rows(
                    **common,
                    threshold=spec["proxy_threshold"],
                    seed=stable_seed(
                        task_id, split, str(candidate_id), "proxy", base_seed=base_seed
                    ),
                )
            )
            rows.extend(
                bootstrap_class_rows(
                    **common,
                    n_classes=spec["n_classes"],
                    seed=stable_seed(
                        task_id, split, str(candidate_id), "class", base_seed=base_seed
                    ),
                )
            )
            rows.extend(
                complementarity_rows(
                    task_id=task_id,
                    split=split,
                    candidate_id=str(candidate_id),
                    truth=truth,
                    scout=scout,
                    expert=expert,
                )
            )

    baseline = prediction_frame(
        validation_hub,
        spec["baseline_model"],
        n_classes=spec["n_classes"],
    )
    baseline.attrs["artifact_id"] = spec["baseline_model"]
    rows.extend(
        selection_stability_rows(
            task_id=task_id,
            traces=validation_traces,
            baseline_frame=baseline,
            primary_metric=spec["primary_metric"],
            qwk_enabled=spec["qwk_enabled"],
            repeats=repeats,
            seed=stable_seed(task_id, "selection", base_seed=base_seed),
        )
    )
    return rows, validation_hub


def parse_cost_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" in payload:
        batch_1 = payload["results"]["batch_1"]
        batch_16 = payload["results"]["batch_16"]
        return {
            "hardware": payload.get("device", "cuda:0 (H100 run)"),
            "dtype": payload.get("dtype", ""),
            "batch1_ms_per_image": batch_1["median_ms_per_image"],
            "batch16_ms_per_image": batch_16["median_ms_per_image"],
            "batch16_images_per_second": batch_16["throughput_images_per_second"],
            "peak_memory_mb": max(
                batch_1["peak_memory_mb"], batch_16["peak_memory_mb"]
            ),
            "warmup_runs": payload.get("warmup"),
            "measured_runs": payload.get("repeats"),
        }
    batch_1 = payload["batch_1"]
    batch_16 = payload["batch_16"]
    return {
        "hardware": payload.get("device", ""),
        "dtype": batch_16.get("dtype", batch_1.get("dtype", "")),
        "batch1_ms_per_image": batch_1["median_per_image_ms"],
        "batch16_ms_per_image": batch_16["median_per_image_ms"],
        "batch16_images_per_second": batch_16["images_per_second"],
        "peak_memory_mb": max(
            batch_1["peak_memory_mb"], batch_16["peak_memory_mb"]
        ),
        "warmup_runs": batch_16.get("warmup_runs"),
        "measured_runs": batch_16.get("measured_runs"),
    }


def collect_cost_evidence(
    aptos_hub: pd.DataFrame,
    glaucoma_hub: pd.DataFrame,
) -> pd.DataFrame:
    aptos_summary = pd.read_csv(
        REPO_ROOT
        / "experiments/opening_risk_routing_closure/aptos_ten_model_summary.csv"
    )
    rows: list[dict[str, Any]] = []
    for _, model in aptos_summary.iterrows():
        model_id = str(model["model_id"])
        evidence: dict[str, Any] = {
            "hardware": "NVIDIA H100 80GB HBM3",
            "dtype": "fp32",
            "batch1_ms_per_image": np.nan,
            "batch16_ms_per_image": float(model["h100_batch16_ms_per_image"]),
            "batch16_images_per_second": np.nan,
            "peak_memory_mb": np.nan,
            "warmup_runs": np.nan,
            "measured_runs": np.nan,
        }
        evidence_path: Path | None = None
        if model_id in APTOS_COST_FILES:
            evidence_path = REPO_ROOT / APTOS_COST_FILES[model_id]
            evidence.update(parse_cost_json(evidence_path))
        elif model_id == "preti":
            evidence_path = REPO_ROOT / (
                "experiments/opening_risk_routing_closure/replays/preti/"
                "seed42_20260722/run_summary.json"
            )
            latency = json.loads(
                evidence_path.read_text(encoding="utf-8")
            )["forward_latency"]
            evidence.update(
                {
                    "hardware": "NVIDIA H100 80GB HBM3",
                    "dtype": "fp32",
                    "batch16_ms_per_image": latency["per_image_ms_median"],
                    "batch16_images_per_second": (
                        1000.0 / latency["per_image_ms_median"]
                    ),
                    "peak_memory_mb": latency["peak_memory_mb"],
                    "warmup_runs": latency["warmup_runs"],
                    "measured_runs": latency["measured_runs"],
                }
            )
        elif model_id == "retfound_cfp":
            evidence_path = REPO_ROOT / (
                "experiments/opening_risk_routing_closure/replays/retfound_cfp/"
                "bicubic_evalcrop_20260722/forward_cost.csv"
            )
            costs = pd.read_csv(evidence_path)
            for batch_size in (1, 16):
                item = costs.loc[costs["batch_size"].eq(batch_size)].iloc[0]
                evidence[f"batch{batch_size}_ms_per_image"] = float(
                    item["median_per_image_ms"]
                )
                if batch_size == 16:
                    evidence["batch16_images_per_second"] = float(
                        item["throughput_images_per_second"]
                    )
            evidence["peak_memory_mb"] = float(costs["peak_memory_mb"].max())
            evidence["warmup_runs"] = int(costs["warmup_runs"].max())
            evidence["measured_runs"] = int(costs["measured_runs"].max())
        hub_row = aptos_hub.loc[aptos_hub["artifact_id"].eq(model_id)].iloc[0]
        rows.append(
            {
                "task_id": "aptos_dr_5class",
                "model_id": model_id,
                **evidence,
                "scope": str(hub_row["cost_scope"]),
                "comparability": "comparable_h100_forward_only",
                "evidence_path": str(evidence_path.relative_to(REPO_ROOT)),
                "evidence_sha256": sha256_file(evidence_path),
                "notes": str(hub_row.get("cpu_postprocess_status", "")),
            }
        )
    glaucoma_cost_path = REPO_ROOT / (
        "experiments/model_hub/tasks/glaucoma_3class/outputs/h100_cost/"
        "glaucoma_h100_forward_cost_summary.csv"
    )
    glaucoma_costs = (
        pd.read_csv(glaucoma_cost_path)
        if glaucoma_cost_path.exists()
        else pd.DataFrame()
    )
    for _, model in glaucoma_hub.iterrows():
        artifact_id = str(model["artifact_id"])
        measured = glaucoma_costs.loc[
            glaucoma_costs["artifact_id"].eq(artifact_id)
        ]
        if measured.empty:
            rows.append(
                {
                    "task_id": "glaucoma_3class",
                    "model_id": artifact_id,
                    "hardware": "historical RTX4090 (registry declaration)",
                    "dtype": "",
                    "batch1_ms_per_image": np.nan,
                    "batch16_ms_per_image": np.nan,
                    "batch16_images_per_second": np.nan,
                    "peak_memory_mb": np.nan,
                    "warmup_runs": np.nan,
                    "measured_runs": np.nan,
                    "scope": model["cost_scope"],
                    "comparability": "historical_hardware_not_comparable",
                    "evidence_path": (
                        "experiments/model_hub/tasks/glaucoma_3class/configs/"
                        "glaucoma_h100_prediction_assets.csv"
                    ),
                    "evidence_sha256": sha256_file(
                        REPO_ROOT
                        / "experiments/model_hub/tasks/glaucoma_3class/configs/"
                        "glaucoma_h100_prediction_assets.csv"
                    ),
                    "notes": "No formal H100 cost evidence",
                }
            )
            continue
        batch_1 = measured.loc[measured["batch_size"].eq(1)].iloc[0]
        batch_16 = measured.loc[measured["batch_size"].eq(16)].iloc[0]
        rows.append(
            {
                "task_id": "glaucoma_3class",
                "model_id": artifact_id,
                "hardware": batch_16["device"],
                "dtype": batch_16["precision"],
                "batch1_ms_per_image": batch_1["median_ms_per_image"],
                "batch16_ms_per_image": batch_16["median_ms_per_image"],
                "batch16_images_per_second": batch_16["images_per_second"],
                "peak_memory_mb": measured[
                    "peak_allocated_memory_mb"
                ].max(),
                "warmup_runs": batch_16["warmup_runs"],
                "measured_runs": batch_16["timed_runs"],
                "scope": "H100 GPU forward-only batch1/batch16",
                "comparability": "comparable_h100_forward_only",
                "evidence_path": str(
                    glaucoma_cost_path.relative_to(REPO_ROOT)
                ),
                "evidence_sha256": sha256_file(glaucoma_cost_path),
                "notes": "fixed-batch FP32; image I/O and preprocessing excluded",
            }
        )
    return pd.DataFrame(rows)


def prediction_asset_hashes(config_path: Path) -> list[dict[str, str]]:
    config = model_hub.load_yaml_or_json(config_path)
    registry_path = model_hub.resolve_path(
        config["prediction_asset_registry"], config_dir=config_path.parent
    )
    registry = pd.read_csv(registry_path)
    rows = []
    for _, item in registry.iterrows():
        for split, column in (
            ("validation", "validation_prediction_path"),
            ("test", "test_prediction_path"),
        ):
            path = model_hub.resolve_path(
                item[column], config_dir=config_path.parent
            )
            rows.append(
                {
                    "task_id": str(item["task_id"]),
                    "model_id": str(item["artifact_id"]),
                    "split": split,
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(path),
                }
            )
    return rows


def write_confirmatory_protocol(
    output_path: Path,
    *,
    source_commit: str,
) -> None:
    aptos_config = REPO_ROOT / TASKS["aptos_dr_5class"]["config"]
    glaucoma_config = REPO_ROOT / TASKS["glaucoma_3class"]["config"]
    protocol = {
        "protocol_id": "ophagent_dual_task_independent_confirmatory_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_independent_data_access",
        "source_commit": source_commit,
        "evaluation_design": "independent_unexposed_external_confirmation",
        "data_access_state": "not_accessed",
        "tasks": {
            "aptos_dr_5class": {
                "primary_candidate": {
                    "pairing_id": "aptos_locked_performance_primary",
                    "scouts": ["flair", "ret_clip"],
                    "expert": "retfound_cfp",
                    "policy": "disagreement_then_uncertainty",
                    "budget": 0.20,
                },
                "proxy_guardrail_candidate": {
                    "pairing_id": "aptos_locked_zero_introduced",
                    "scouts": ["retfound_cfp", "retizero"],
                    "expert": "flair",
                    "policy": "disagreement_then_uncertainty",
                    "budget": 0.20,
                },
                "primary_endpoint": "paired QWK difference versus primary Scout",
                "secondary_endpoints": [
                    "paired Macro-F1 difference",
                    "paired Accuracy difference",
                    "label-order proxy corrected/introduced/net rates",
                ],
                "class_order": [0, 1, 2, 3, 4],
            },
            "glaucoma_3class": {
                "primary_candidate": {
                    "pairing_id": "glaucoma_locked_multi",
                    "scouts": [
                        "glaucoma_retfound_dinov2",
                        "glaucoma_vit_b",
                    ],
                    "expert": "glaucoma_swin_tiny",
                    "policy": "disagreement_then_uncertainty",
                    "budget": 0.10,
                },
                "single_scout_reference": {
                    "pairing_id": "glaucoma_locked_single",
                    "scouts": ["glaucoma_retfound_dinov2"],
                    "expert": "glaucoma_swin_tiny",
                    "policy": "high_entropy",
                    "budget": 0.05,
                },
                "primary_endpoint": (
                    "paired Macro-F1 difference versus primary Scout"
                ),
                "secondary_endpoints": [
                    "paired Accuracy difference",
                    "label-order proxy corrected/introduced/net rates",
                ],
                "class_order": [0, 1, 2],
            },
        },
        "analysis": {
            "unit": "patient",
            "confidence_interval": (
                "patient-clustered paired bootstrap, 5000 resamples, "
                "two-sided percentile 95% CI"
            ),
            "random_seed": 20260723,
            "primary_success_gate": (
                "lower 95% CI of the task primary paired endpoint is > 0"
            ),
            "accuracy_guardrail": (
                "lower 95% CI of paired Accuracy difference is > -0.01"
            ),
            "proxy_guardrail": (
                "upper 95% CI of introduced label-order proxy-event rate "
                "is <= 0.01"
            ),
            "multiplicity": (
                "hierarchical task-specific gates; secondary endpoints are "
                "descriptive unless the primary gate passes"
            ),
        },
        "data_requirements": {
            "independent_from_current_train_validation_test": True,
            "stable_case_id": True,
            "patient_id_required": True,
            "patient_overlap_prohibited": True,
            "labels_frozen_before_inference": True,
            "minimum_sample_size": (
                "determined prospectively from the primary paired endpoint; "
                "not from confirmatory outcomes"
            ),
        },
        "execution_locks": {
            "no_model_training": True,
            "no_recalibration": True,
            "no_threshold_or_budget_selection": True,
            "no_candidate_promotion": True,
            "no_test_driven_preprocessing_change": True,
            "route_eligible_before_gate_completion": False,
        },
        "cost_protocol": {
            "hardware": "NVIDIA H100 80GB HBM3",
            "batch_sizes": [1, 16],
            "scope": "forward-only, excluding image I/O and preprocessing",
            "warmup_runs": 10,
            "measured_runs": 30,
            "same_host_required": True,
            "glaucoma_cost_measurement_required_before_cost_claim": True,
        },
        "frozen_source_protocols": [
            {
                "path": str(aptos_config.relative_to(REPO_ROOT)),
                "sha256": sha256_file(aptos_config),
            },
            {
                "path": str(glaucoma_config.relative_to(REPO_ROOT)),
                "sha256": sha256_file(glaucoma_config),
            },
        ],
        "prediction_assets": [
            *prediction_asset_hashes(aptos_config),
            *prediction_asset_hashes(glaucoma_config),
        ],
        "interpretation_boundary": (
            "Label-order proxy events are model-output audit proxies, not "
            "clinical outcomes or treatment recommendations."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_COST_OUTPUT)
    parser.add_argument(
        "--confirmatory-protocol-output",
        type=Path,
        default=DEFAULT_PROTOCOL_OUTPUT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")
    all_rows: list[dict[str, Any]] = []
    hubs: dict[str, pd.DataFrame] = {}
    for task_id, spec in TASKS.items():
        rows, hub = analyze_task(
            task_id,
            spec,
            repeats=args.bootstrap_repeats,
            base_seed=args.seed,
        )
        all_rows.extend(rows)
        hubs[task_id] = hub

    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    robustness = pd.DataFrame(all_rows)
    robustness.to_csv(output_path, index=False)

    cost_path = REPO_ROOT / args.cost_output
    costs = collect_cost_evidence(
        hubs["aptos_dr_5class"], hubs["glaucoma_3class"]
    )
    costs.to_csv(cost_path, index=False)

    source_commit = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
        .strip()
    )
    write_confirmatory_protocol(
        REPO_ROOT / args.confirmatory_protocol_output,
        source_commit=source_commit,
    )
    print(
        json.dumps(
            {
                "robustness_rows": len(robustness),
                "cost_rows": len(costs),
                "bootstrap_repeats": args.bootstrap_repeats,
                "output": str(output_path.relative_to(REPO_ROOT)),
                "cost_output": str(cost_path.relative_to(REPO_ROOT)),
                "confirmatory_protocol": str(
                    args.confirmatory_protocol_output
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
