"""Read-only Route Qualification Benchmark v1.1.

The benchmark consumes explicit Validation/frozen asset identities.  It never
trains a model, reruns routing, changes thresholds from held-out tasks, or
combines absolute costs across different cost protocols.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from app.model_hub_index import build_model_hub_index
from app.route_qualification import (
    OUTPUT_RELATIVE_DIR,
    V1_1_CONTRACT_RELATIVE_PATH,
    V1_1_OUTPUT_RELATIVE_DIR,
    V1_1_RULE_FAMILIES,
    _finite,
    _git_commit,
    _scout_metric,
    evaluate_route_qualification,
    file_sha256,
    route_qualification_request_from_row,
)


EVIDENCE_SOURCES_RELATIVE_PATH = (
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "route_qualification_evidence_sources_v1_1.json"
)
ROBUSTNESS_RELATIVE_PATH = (
    "experiments/opening_risk_routing_closure/dual_task_robustness.csv"
)
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260728


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _assert_sha(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if expected and actual != expected:
        raise ValueError(
            f"asset SHA mismatch: {path}; expected={expected}; actual={actual}"
        )
    return actual


def _route_signature(row: pd.Series) -> tuple[str, str, str, float]:
    scouts = "|".join(
        sorted(
            value
            for value in str(row.get("scout_artifact_ids", "")).split("|")
            if value
        )
    )
    return (
        scouts,
        str(row.get("expert_artifact_id", "")),
        str(row.get("routing_policy", "")),
        round(float(row.get("requested_budget", 0.0)), 6),
    )


def _route_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if "evaluation_kind" in rows:
        rows = rows.loc[rows["evaluation_kind"].astype(str).eq("routed")]
    if "status" in rows:
        rows = rows.loc[rows["status"].astype(str).eq("completed")]
    if "result_semantics" in rows:
        rows = rows.loc[
            ~rows["result_semantics"]
            .astype(str)
            .str.contains("protocol_invalid", case=False, na=False)
        ]
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _match_route(
    frame: pd.DataFrame,
    ledger_row: pd.Series,
) -> pd.Series | None:
    rows = _route_rows(frame)
    pairing_id = str(ledger_row["pairing_id"])
    if "pairing_id" in rows:
        exact = rows.loc[rows["pairing_id"].astype(str).eq(pairing_id)]
        if len(exact) == 1:
            return exact.iloc[0]
    signature = _route_signature(ledger_row)
    matches = rows.loc[
        rows.apply(_route_signature, axis=1).map(
            lambda value: value == signature
        )
    ]
    return matches.iloc[0] if len(matches) == 1 else None


def _metric_policy(
    contract: dict[str, Any],
    task_id: str,
    route: pd.Series | None,
) -> tuple[str, bool]:
    policy = dict(contract["task_policies"][task_id])
    primary = str(policy["primary_metric"])
    if route is not None and _finite(route.get(primary)) is not None:
        return primary, True
    return str(policy.get("fallback_metric", "")), False


def _metric_delta(
    frame: pd.DataFrame,
    route: pd.Series | None,
    metric: str,
) -> tuple[float | None, float | None, float | None]:
    if route is None or not metric:
        return None, None, None
    value = _finite(route.get(metric))
    scout = _scout_metric(frame, route, metric)
    single_mask = (
        frame.get(
            "evaluation_kind",
            pd.Series("", index=frame.index),
        )
        .astype(str)
        .eq("scout_only")
    )
    for column in (
        "protocol_valid",
        "completed",
        "result_complete",
    ):
        if column in frame:
            single_mask &= frame[column].map(_as_bool)
    for column in (
        "protocol_invalid",
        "test_used_for_selection",
        "selection_used_test",
    ):
        if column in frame:
            single_mask &= ~frame[column].map(_as_bool)
    for column in ("status", "result_status", "protocol_status"):
        if column in frame:
            single_mask &= ~frame[column].astype(str).str.lower().isin(
                {
                    "failed",
                    "error",
                    "incomplete",
                    "protocol_invalid",
                    "invalid",
                }
            )
    singles = frame.loc[single_mask]
    single_values = pd.to_numeric(
        singles.get(metric, pd.Series(dtype=float)),
        errors="coerce",
    )
    best = float(single_values.max()) if single_values.notna().any() else None
    return (
        value,
        value - scout if value is not None and scout is not None else None,
        value - best if value is not None and best is not None else None,
    )


def _risk_values(
    route: pd.Series | None,
) -> tuple[float | None, float | None, float | None]:
    if route is None:
        return None, None, None
    return (
        _finite(route.get("dangerous_corrected")),
        _finite(route.get("dangerous_introduced")),
        _finite(route.get("net_dangerous_reduction")),
    )


def _task_asset_rows(
    task_assets: pd.DataFrame,
    *,
    task_id: str,
    artifact_ids: tuple[str, ...],
) -> pd.DataFrame:
    return task_assets.loc[
        task_assets["task_id"].astype(str).eq(task_id)
        & task_assets["artifact_id"].astype(str).isin(artifact_ids)
    ].copy()


def _cost_evidence(
    task_assets: pd.DataFrame,
    ledger_row: pd.Series,
    source: dict[str, Any],
) -> dict[str, Any]:
    scouts = tuple(
        value
        for value in str(ledger_row["scout_artifact_ids"]).split("|")
        if value
    )
    expert = str(ledger_row["expert_artifact_id"])
    rows = _task_asset_rows(
        task_assets,
        task_id=str(ledger_row["task_id"]),
        artifact_ids=(*scouts, expert),
    )
    route_override = dict(
        source.get("route_cost_protocols", {}).get(
            str(ledger_row["pairing_id"]),
            {},
        )
    )
    protocol_id = str(
        route_override.get(
            "cost_protocol_id",
            source.get("cost_protocol_id", ""),
        )
    )
    comparable = bool(
        route_override.get(
            "cost_protocol_match",
            source.get("cost_protocol_match", False),
        )
    )
    costs = {
        str(row.get("artifact_id", "")): _finite(
            row.get("forward_cost_ms_per_image")
        )
        for _, row in rows.iterrows()
    }
    all_found = len(costs) == len(set((*scouts, expert)))
    all_measured = all(
        costs.get(artifact_id) is not None
        for artifact_id in (*scouts, expert)
    )
    expected_cost = None
    if all_found and all_measured:
        expected_cost = sum(float(costs[value]) for value in scouts)
        expected_cost += float(ledger_row["requested_budget"]) * float(
            costs[expert]
        )
    elif _finite(ledger_row.get("expected_cost_ms_per_image")) is not None:
        expected_cost = _finite(
            ledger_row.get("expected_cost_ms_per_image")
        )
    fingerprint_rows = [
        {
            "artifact_id": str(row.get("artifact_id", "")),
            "cost_ms": _finite(row.get("forward_cost_ms_per_image")),
            "cost_scope": str(row.get("cost_scope", "")),
            "cost_status": str(row.get("cost_status", "")),
        }
        for _, row in rows.sort_values("artifact_id").iterrows()
    ]
    return {
        "cost_protocol_id": protocol_id,
        "cost_protocol_comparable": comparable,
        "cost_protocol_complete": bool(
            comparable and all_found and all_measured
        ),
        "cost_post_hoc_derived": bool(
            source.get("cost_post_hoc_derived", False)
        ),
        "expected_cost_ms_per_image": expected_cost,
        "cost_evidence_sha256": _sha256_json(fingerprint_rows),
    }


def _capability_evidence(
    model_capabilities: pd.DataFrame,
    ledger_row: pd.Series,
) -> dict[str, Any]:
    scouts = tuple(
        value
        for value in str(ledger_row["scout_artifact_ids"]).split("|")
        if value
    )
    expert = str(ledger_row["expert_artifact_id"])
    rows = _task_asset_rows(
        model_capabilities,
        task_id=str(ledger_row["task_id"]),
        artifact_ids=(*scouts, expert),
    )
    found = len(rows["artifact_id"].unique()) == len(set((*scouts, expert)))
    adapters = {
        str(row["artifact_id"]): str(row.get("adapter_type", ""))
        for _, row in rows.iterrows()
    }
    prediction_complete = bool(
        found
        and rows["prediction_asset_available"].fillna(False).astype(bool).all()
    )
    return {
        "scout_adapter_types": "|".join(
            adapters.get(value, "unregistered") for value in scouts
        ),
        "expert_adapter_type": adapters.get(expert, "unregistered"),
        "prediction_asset_complete": prediction_complete,
        "historical_replay_eligible": prediction_complete,
        "offline_batch_eligible": bool(
            found
            and rows["offline_batch_inference_ready"]
            .fillna(False)
            .astype(bool)
            .all()
        ),
        "single_case_original_ready": bool(
            found
            and rows["online_case_inference_ready"]
            .fillna(False)
            .astype(bool)
            .all()
        ),
    }


def _robustness_maps(
    project_root: Path,
) -> tuple[dict[tuple[str, str, str], tuple[float, float]], dict[tuple[str, str], float]]:
    path = project_root / ROBUSTNESS_RELATIVE_PATH
    if not path.is_file():
        return {}, {}
    frame = pd.read_csv(path)
    intervals: dict[
        tuple[str, str, str],
        tuple[float, float],
    ] = {}
    selection: dict[tuple[str, str], float] = {}
    route_rows = frame.loc[
        frame["analysis_section"].astype(str).eq("route_metric_ci")
        & frame["split"].astype(str).eq("val")
        & frame["estimate"].astype(str).eq("route_minus_scout")
    ]
    for _, row in route_rows.iterrows():
        intervals[
            (
                str(row["task_id"]),
                str(row["candidate_id"]),
                str(row["metric"]),
            )
        ] = (
            float(row["ci_lower"]),
            float(row["ci_upper"]),
        )
    selection_rows = frame.loc[
        frame["analysis_section"]
        .astype(str)
        .eq("validation_selection_stability")
        & frame["split"].astype(str).eq("val")
        & frame["estimate"]
        .astype(str)
        .eq("bootstrap_selection_frequency")
    ]
    for _, row in selection_rows.iterrows():
        selection[
            (str(row["task_id"]), str(row["candidate_id"]))
        ] = float(row["point"])
    return intervals, selection


def _trace_path(result_path: Path) -> Path:
    return result_path.parent / "case_routing_trace.csv"


def _metric_function(metric: str, labels: list[int]):
    if metric == "qwk":
        return lambda truth, pred: float(
            cohen_kappa_score(truth, pred, labels=labels, weights="quadratic")
        )
    if metric == "macro_f1":
        return lambda truth, pred: float(
            f1_score(
                truth,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    return None


def _bootstrap_fixed_trace(
    trace_path: Path,
    *,
    pairing_id: str,
    metric: str,
    labels: list[int],
    seed: int,
) -> tuple[float | None, float | None]:
    if not trace_path.is_file():
        return None, None
    trace = pd.read_csv(trace_path, low_memory=False)
    if "pairing_id" in trace:
        trace = trace.loc[trace["pairing_id"].astype(str).eq(pairing_id)]
    truth_column = next(
        (
            name
            for name in ("true_label", "y_true", "target")
            if name in trace
        ),
        "",
    )
    scout_column = next(
        (
            name
            for name in (
                "primary_scout_pred_label",
                "scout_pred_label",
            )
            if name in trace
        ),
        "",
    )
    if (
        not truth_column
        or not scout_column
        or "final_pred_label" not in trace
        or trace.empty
    ):
        return None, None
    metric_fn = _metric_function(metric, labels)
    if metric_fn is None:
        return None, None
    truth = trace[truth_column].to_numpy(dtype=int)
    scout = trace[scout_column].to_numpy(dtype=int)
    final = trace["final_pred_label"].to_numpy(dtype=int)
    generator = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPEATS, dtype=float)
    for index in range(BOOTSTRAP_REPEATS):
        sample = generator.integers(0, len(trace), len(trace))
        deltas[index] = metric_fn(
            truth[sample],
            final[sample],
        ) - metric_fn(
            truth[sample],
            scout[sample],
        )
    finite = deltas[np.isfinite(deltas)]
    if finite.size < BOOTSTRAP_REPEATS * 0.95:
        return None, None
    return float(np.quantile(finite, 0.025)), float(
        np.quantile(finite, 0.975)
    )


def _descriptive_candidate_ranks(matrix: pd.DataFrame) -> pd.DataFrame:
    result = matrix.copy()
    result["candidate_rank"] = np.nan
    result["candidate_count"] = np.nan
    comparable = result.loc[
        result["cost_protocol_comparable"].fillna(False).astype(bool)
        & result["validation_main_metric"].notna()
    ]
    for _, group in comparable.groupby(
        ["task_id", "cost_protocol_id"],
        dropna=False,
    ):
        ranks = group["validation_main_metric"].rank(
            method="min",
            ascending=False,
        )
        result.loc[group.index, "candidate_rank"] = ranks
        result.loc[group.index, "candidate_count"] = len(group)
    result["candidate_rank_stability_verified"] = False
    return result


def build_route_qualification_evidence_matrix(
    project_root: Path,
) -> pd.DataFrame:
    """Build the 16-route v1.1 matrix from explicit read-only assets."""

    legacy_path = (
        project_root
        / OUTPUT_RELATIVE_DIR
        / "route_qualification_matrix.csv"
    )
    legacy = pd.read_csv(legacy_path)
    if len(legacy) != 16 or legacy["task_id"].nunique() != 6:
        raise ValueError("formal frozen route ledger is not the expected 16x6 set")
    contract_path = project_root / V1_1_CONTRACT_RELATIVE_PATH
    sources_path = project_root / EVIDENCE_SOURCES_RELATIVE_PATH
    contract = _read_json(contract_path)
    sources = _read_json(sources_path)["tasks"]
    index = build_model_hub_index(project_root)
    task_assets = index["task_assets"]
    capabilities = index["model_capabilities"]
    intervals, selection_frequencies = _robustness_maps(project_root)
    rows: list[dict[str, Any]] = []

    for _, ledger_row in legacy.iterrows():
        task_id = str(ledger_row["task_id"])
        pairing_id = str(ledger_row["pairing_id"])
        policy = dict(contract["task_policies"][task_id])
        source = dict(sources[task_id])
        validation_path = project_root / source["validation_result_path"]
        frozen_path = project_root / source["frozen_result_path"]
        validation_sha = _assert_sha(
            validation_path,
            str(source["validation_result_sha256"]),
        )
        frozen_sha = _assert_sha(
            frozen_path,
            str(source["frozen_result_sha256"]),
        )
        validation_frame = pd.read_csv(validation_path, low_memory=False)
        frozen_frame = pd.read_csv(frozen_path, low_memory=False)
        validation_route = _match_route(validation_frame, ledger_row)
        frozen_route = _match_route(frozen_frame, ledger_row)
        if frozen_route is None:
            raise ValueError(f"frozen route identity is not unique: {pairing_id}")
        metric, primary_available = _metric_policy(
            contract,
            task_id,
            validation_route,
        )
        validation_metric, validation_delta, best_delta = _metric_delta(
            validation_frame,
            validation_route,
            metric,
        )
        frozen_metric, frozen_delta, _ = _metric_delta(
            frozen_frame,
            frozen_route,
            metric,
        )
        validation_risk = _risk_values(validation_route)
        frozen_risk = _risk_values(frozen_route)
        risk_evidence_available = True
        risk_source = "pairing_results_task_proxy"
        risk_cohort_n = _finite(frozen_route.get("n_samples"))
        metric_cohort_n = _finite(frozen_route.get("n_samples"))
        if (
            "validation_risk_values" in source
            or "frozen_risk_values" in source
        ):
            validation_values = dict(
                source.get("validation_risk_values", {}).get(
                    pairing_id,
                    {},
                )
            )
            frozen_values = dict(
                source.get("frozen_risk_values", {}).get(
                    pairing_id,
                    {},
                )
            )
            validation_risk = (
                _finite(validation_values.get("corrected")),
                _finite(validation_values.get("introduced")),
                _finite(validation_values.get("net")),
            )
            frozen_risk = (
                _finite(frozen_values.get("corrected")),
                _finite(frozen_values.get("introduced")),
                _finite(frozen_values.get("net")),
            )
            risk_evidence_available = all(
                value is not None for value in validation_risk
            )
            risk_source = str(
                validation_values.get(
                    "source",
                    "validation_task_proxy_missing",
                )
            )
            risk_cohort_n = _finite(source.get("frozen_risk_cohort_n"))
            metric_cohort_n = _finite(source.get("frozen_metric_cohort_n"))

        target_validation_missing = bool(
            source.get("target_validation_missing", False)
        )
        source_validation_metric = validation_metric
        source_validation_delta = validation_delta
        source_best_delta = best_delta
        if target_validation_missing:
            validation_metric = None
            validation_delta = None
            best_delta = None
            validation_risk = (None, None, None)
            risk_evidence_available = False

        interval = intervals.get((task_id, pairing_id, metric), (None, None))
        stability_source = (
            ROBUSTNESS_RELATIVE_PATH
            if interval[0] is not None
            else "missing"
        )
        if interval[0] is None and not target_validation_missing:
            seed = BOOTSTRAP_SEED + int(
                hashlib.sha256(
                    f"{task_id}:{pairing_id}".encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            interval = _bootstrap_fixed_trace(
                _trace_path(validation_path),
                pairing_id=pairing_id,
                metric=metric,
                labels=list(range(int(policy["n_classes"]))),
                seed=seed,
            )
            if interval[0] is not None:
                stability_source = (
                    "v1_1_fixed_validation_trace_paired_bootstrap"
                )
        capability = _capability_evidence(capabilities, ledger_row)
        cost = _cost_evidence(task_assets, ledger_row, source)
        scouts = tuple(
            value
            for value in str(ledger_row["scout_artifact_ids"]).split("|")
            if value
        )
        expert = str(ledger_row["expert_artifact_id"])
        assets = _task_asset_rows(
            task_assets,
            task_id=task_id,
            artifact_ids=(*scouts, expert),
        )
        asset_fingerprint = _sha256_json(
            [
                {
                    "artifact_id": str(row.get("artifact_id", "")),
                    "checkpoint_sha256": str(
                        row.get("checkpoint_sha256", "")
                    ),
                    "validation_prediction_path": str(
                        row.get("validation_prediction_path", "")
                    ),
                    "test_prediction_path": str(
                        row.get("test_prediction_path", "")
                    ),
                }
                for _, row in assets.sort_values("artifact_id").iterrows()
            ]
        )
        frozen_favorable = bool(
            frozen_delta is not None
            and frozen_delta > 0
            and (
                frozen_risk[2] is None
                or frozen_risk[2] >= 0
            )
        )
        row = {
            "task_id": task_id,
            "dataset_id": str(policy["dataset_id"]),
            "pairing_id": pairing_id,
            "selection_source_task_id": str(
                source["selection_source_task_id"]
            ),
            "selection_split": str(source.get("selection_split", "")),
            "protocol_frozen": _as_bool(
                source.get("protocol_frozen", False)
            ),
            "test_used_for_selection": _as_bool(
                source.get("test_used_for_selection", True)
            ),
            "target_validation_missing": target_validation_missing,
            "scout_artifact_ids": "|".join(scouts),
            "expert_artifact_id": expert,
            "scout_adapter_types": capability["scout_adapter_types"],
            "expert_adapter_type": capability["expert_adapter_type"],
            "adaptation_type": str(policy["adaptation_type"]),
            "routing_policy": str(ledger_row["routing_policy"]),
            "requested_budget": float(ledger_row["requested_budget"]),
            "expert_budget": float(ledger_row["requested_budget"]),
            "primary_metric": str(policy["primary_metric"]),
            "metric_used": metric,
            "primary_metric_available": primary_available,
            "validation_main_metric": validation_metric,
            "validation_delta_vs_scout": validation_delta,
            "validation_delta_vs_best_single": best_delta,
            "source_validation_main_metric": source_validation_metric,
            "source_validation_delta_vs_scout": source_validation_delta,
            "source_validation_delta_vs_best_single": source_best_delta,
            "validation_corrected": validation_risk[0],
            "validation_introduced": validation_risk[1],
            "validation_net": validation_risk[2],
            "risk_proxy_semantics": str(policy["proxy_semantics"]),
            "risk_evidence_available": risk_evidence_available,
            "risk_evidence_source": risk_source,
            "stability_ci_lower": interval[0],
            "stability_ci_upper": interval[1],
            "candidate_selection_frequency": selection_frequencies.get(
                (task_id, pairing_id)
            ),
            "stability_source": stability_source,
            "stability_bootstrap_repeats": (
                BOOTSTRAP_REPEATS if interval[0] is not None else None
            ),
            "frozen_main_metric": frozen_metric,
            "frozen_delta_vs_scout": frozen_delta,
            "frozen_corrected": frozen_risk[0],
            "frozen_introduced": frozen_risk[1],
            "frozen_net": frozen_risk[2],
            "frozen_favorable": frozen_favorable,
            "risk_cohort_n": risk_cohort_n,
            "metric_cohort_n": metric_cohort_n,
            "domain_shift_status": str(source["domain_shift_status"]),
            "task_adapter_compatible": True,
            "prediction_asset_complete": capability[
                "prediction_asset_complete"
            ],
            "prediction_assets_valid": capability[
                "prediction_asset_complete"
            ],
            "historical_replay_eligible": capability[
                "historical_replay_eligible"
            ],
            "offline_batch_eligible": capability[
                "offline_batch_eligible"
            ],
            "single_case_original_ready": capability[
                "single_case_original_ready"
            ],
            "all_models_online_case_ready": capability[
                "single_case_original_ready"
            ],
            **cost,
            "validation_result_asset": str(
                source["validation_result_path"]
            ),
            "frozen_result_asset": str(source["frozen_result_path"]),
            "validation_protocol_id": str(
                source["validation_protocol_id"]
            ),
            "frozen_protocol_id": str(source["frozen_protocol_id"]),
            "protocol_sha256": str(source["frozen_protocol_sha256"]),
            "validation_protocol_sha256": str(
                source["validation_protocol_sha256"]
            ),
            "validation_result_sha256": validation_sha,
            "frozen_result_sha256": frozen_sha,
            "unique_protocol_identity": bool(
                source.get("frozen_protocol_id")
                and source.get("frozen_protocol_sha256")
            ),
            "input_asset_fingerprint": asset_fingerprint,
            "source_commit_sha": str(
                ledger_row.get("source_commit_sha", "not_registered")
            ),
            "evidence_build_commit_sha": _git_commit(project_root),
            "clinical_route_eligible": False,
        }
        rows.append(row)

    matrix = _descriptive_candidate_ranks(pd.DataFrame(rows))
    decisions = []
    for _, row in matrix.iterrows():
        request = route_qualification_request_from_row(row)
        decision = evaluate_route_qualification(
            request,
            contract=contract,
            contract_sha256=file_sha256(contract_path),
        )
        decisions.append(decision)
    matrix["evidence_label"] = [
        decision.evidence_label for decision in decisions
    ]
    matrix["execution_level"] = [
        decision.execution_level for decision in decisions
    ]
    matrix["allow_cached_replay"] = [
        decision.allow_cached_replay for decision in decisions
    ]
    matrix["allow_case_simulation"] = [
        decision.allow_case_simulation for decision in decisions
    ]
    matrix["allow_new_case_route"] = [
        decision.allow_new_case_route for decision in decisions
    ]
    matrix["error_codes"] = [
        "|".join(decision.error_codes) for decision in decisions
    ]
    matrix["qualification_contract_sha256"] = file_sha256(contract_path)
    return matrix.sort_values(
        ["task_id", "pairing_id"],
        ignore_index=True,
    )


def _wilson_interval(
    numerator: int,
    denominator: int,
) -> tuple[float | None, float | None]:
    if denominator <= 0:
        return None, None
    z = 1.959963984540054
    proportion = numerator / denominator
    denominator_term = 1 + z**2 / denominator
    center = (proportion + z**2 / (2 * denominator)) / denominator_term
    half = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / denominator
            + z**2 / (4 * denominator**2)
        )
        / denominator_term
    )
    return float(center - half), float(center + half)


def _rate_record(
    name: str,
    numerator: int,
    denominator: int,
) -> dict[str, Any]:
    lower, upper = _wilson_interval(numerator, denominator)
    return {
        name: numerator / denominator if denominator else None,
        f"{name}_numerator": numerator,
        f"{name}_denominator": denominator,
        f"{name}_ci_lower": lower,
        f"{name}_ci_upper": upper,
    }


def _evaluate_rows(
    matrix: pd.DataFrame,
    contract: dict[str, Any],
    *,
    rule_family: str,
) -> tuple[pd.Series, list[str]]:
    granted: list[bool] = []
    codes: list[str] = []
    for _, row in matrix.iterrows():
        decision = evaluate_route_qualification(
            route_qualification_request_from_row(row),
            contract=contract,
            rule_family=rule_family,
        )
        granted.append(decision.allow_case_simulation)
        codes.append("|".join(decision.error_codes))
    return pd.Series(granted, index=matrix.index), codes


def benchmark_metrics(
    frame: pd.DataFrame,
    granted: pd.Series,
) -> dict[str, Any]:
    favorable = frame["frozen_favorable"].fillna(False).astype(bool)
    ineffective = ~favorable
    reversals = (
        pd.to_numeric(
            frame["frozen_delta_vs_scout"],
            errors="coerce",
        ).le(0)
        | (
            pd.to_numeric(
                frame["validation_net"],
                errors="coerce",
            ).fillna(0).ge(0)
            & pd.to_numeric(
                frame["frozen_net"],
                errors="coerce",
            ).fillna(0).lt(0)
        )
    )
    risk_tradeoff = (
        pd.to_numeric(
            frame["validation_introduced"],
            errors="coerce",
        ).fillna(0).gt(0)
        | pd.to_numeric(
            frame["validation_net"],
            errors="coerce",
        ).fillna(0).lt(0)
    )
    result: dict[str, Any] = {
        "formal_routes": len(frame),
        "favorable_routes": int(favorable.sum()),
        "granted_routes": int(granted.sum()),
    }
    result.update(
        _rate_record(
            "beneficial_route_retention_rate",
            int((favorable & granted).sum()),
            int(favorable.sum()),
        )
    )
    result.update(
        _rate_record(
            "ineffective_route_interception_rate",
            int((ineffective & ~granted).sum()),
            int(ineffective.sum()),
        )
    )
    result.update(
        _rate_record(
            "false_grant_rate",
            int((ineffective & granted).sum()),
            int(granted.sum()),
        )
    )
    result.update(
        _rate_record(
            "false_rejection_rate",
            int((favorable & ~granted).sum()),
            int(favorable.sum()),
        )
    )
    result.update(
        _rate_record(
            "test_reversal_interception_rate",
            int((reversals & ~granted).sum()),
            int(reversals.sum()),
        )
    )
    result.update(
        _rate_record(
            "risk_tradeoff_limitation_rate",
            int((risk_tradeoff & ~granted).sum()),
            int(risk_tradeoff.sum()),
        )
    )
    result.update(
        _rate_record(
            "executable_coverage_rate",
            int(granted.sum()),
            len(frame),
        )
    )
    costs = pd.to_numeric(
        frame["expected_cost_ms_per_image"],
        errors="coerce",
    )
    comparable = frame["cost_protocol_comparable"].fillna(False).astype(bool)
    protocol_costs: dict[str, dict[str, float]] = {}
    comparable_frame = frame.loc[comparable]
    for protocol_id, group in comparable_frame.groupby(
        "cost_protocol_id"
    ):
        group_costs = costs.loc[group.index]
        group_granted = granted.loc[group.index]
        ungated = float(group_costs.sum())
        gated = float(group_costs.loc[group_granted].sum())
        protocol_costs[str(protocol_id)] = {
            "ungated_ms_per_image_sum": ungated,
            "gated_ms_per_image_sum": gated,
            "change_ms_per_image": gated - ungated,
        }
    result["cost_protocol_group_count"] = len(protocol_costs)
    result["comparable_cost_by_protocol_json"] = _stable_json(
        protocol_costs
    )
    if len(protocol_costs) == 1:
        only = next(iter(protocol_costs.values()))
        result["ungated_comparable_cost_ms_per_image"] = only[
            "ungated_ms_per_image_sum"
        ]
        result["gated_comparable_cost_ms_per_image"] = only[
            "gated_ms_per_image_sum"
        ]
        result["comparable_cost_change_ms_per_image"] = only[
            "change_ms_per_image"
        ]
    else:
        result["ungated_comparable_cost_ms_per_image"] = None
        result["gated_comparable_cost_ms_per_image"] = None
        result["comparable_cost_change_ms_per_image"] = None
    budgets = pd.to_numeric(frame["expert_budget"], errors="coerce")
    result["ungated_expert_budget_sum"] = float(budgets.sum())
    result["gated_expert_budget_sum"] = float(budgets.loc[granted].sum())
    result["expert_budget_change"] = (
        result["gated_expert_budget_sum"]
        - result["ungated_expert_budget_sum"]
    )
    return result


def _misclassification_summary(
    frame: pd.DataFrame,
    granted: pd.Series,
    codes: list[str],
) -> str:
    records = []
    for pairing, favorable, allowed, error_codes in zip(
        frame["pairing_id"],
        frame["frozen_favorable"].fillna(False).astype(bool),
        granted,
        codes,
        strict=True,
    ):
        if bool(allowed) == bool(favorable):
            continue
        outcome = "false_grant" if allowed else "false_rejection"
        records.append(f"{pairing}:{outcome}:{error_codes}")
    return "|".join(records)


def build_ablation_results(
    matrix: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for rule_family in V1_1_RULE_FAMILIES:
        granted, _ = _evaluate_rows(
            matrix,
            contract,
            rule_family=rule_family,
        )
        rows.append(
            {
                "rule_family": rule_family,
                **benchmark_metrics(matrix, granted),
            }
        )
    return pd.DataFrame(rows)


def _derive_training_thresholds(
    training: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    thresholds = deepcopy(contract["thresholds"])
    ceilings: dict[str, float] = {}
    comparable = training.loc[
        training["cost_protocol_comparable"].fillna(False).astype(bool)
        & training["expected_cost_ms_per_image"].notna()
    ]
    for protocol_id, group in comparable.groupby("cost_protocol_id"):
        ceilings[str(protocol_id)] = round(
            float(group["expected_cost_ms_per_image"].max()) * 1.25,
            6,
        )
    thresholds["cost_ceiling_ms_per_image_by_protocol"] = ceilings
    budgets = pd.to_numeric(training["expert_budget"], errors="coerce")
    if budgets.notna().any():
        thresholds["max_expert_budget"] = float(budgets.max())
    return thresholds


def build_leave_one_task_out_results(
    matrix: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    validation_only_decisions: list[pd.Series] = []
    overlay_decisions: list[pd.Series] = []
    frozen_outcome_columns = (
        "frozen_main_metric",
        "frozen_delta_vs_scout",
        "frozen_corrected",
        "frozen_introduced",
        "frozen_net",
    )
    for held_out in sorted(matrix["task_id"].unique()):
        task_excluded = matrix["task_id"].eq(held_out)
        selection_dependency = matrix[
            "selection_source_task_id"
        ].eq(held_out)
        training = matrix.loc[
            ~(task_excluded | selection_dependency)
        ]
        evaluation = matrix.loc[matrix["task_id"].eq(held_out)]
        validation_only = evaluation.copy()
        validation_only.loc[
            :,
            list(frozen_outcome_columns),
        ] = np.nan
        fold_contract = deepcopy(contract)
        fold_contract["thresholds"] = _derive_training_thresholds(
            training,
            contract,
        )
        validation_granted, validation_codes = _evaluate_rows(
            validation_only,
            fold_contract,
            rule_family="complete_layered_gate",
        )
        overlay_granted, overlay_codes = _evaluate_rows(
            evaluation,
            fold_contract,
            rule_family="complete_layered_gate",
        )
        validation_only_decisions.append(validation_granted)
        overlay_decisions.append(overlay_granted)
        validation_failures = _misclassification_summary(
            evaluation,
            validation_granted,
            validation_codes,
        )
        overlay_failures = _misclassification_summary(
            evaluation,
            overlay_granted,
            overlay_codes,
        )
        shared = {
            "held_out_task_id": held_out,
            "training_task_count": training["task_id"].nunique(),
            "training_route_count": len(training),
            "selection_dependency_rows_excluded": int(
                (selection_dependency & ~task_excluded).sum()
            ),
            "thresholds_json": _stable_json(
                fold_contract["thresholds"]
            ),
            "held_out_not_used_for_thresholds": True,
        }
        rows.append(
            {
                "record_type": "held_out_task_validation_only",
                "prediction_scope": (
                    "validation_only_no_held_out_frozen_outcomes"
                ),
                "held_out_frozen_outcomes_used_for_decision": False,
                "failure_cases": validation_failures,
                **shared,
                **benchmark_metrics(
                    evaluation,
                    validation_granted,
                ),
            }
        )
        rows.append(
            {
                "record_type": "held_out_task_post_freeze_overlay",
                "prediction_scope": (
                    "validation_gate_plus_post_freeze_safety_overlay"
                ),
                "held_out_frozen_outcomes_used_for_decision": True,
                "failure_cases": overlay_failures,
                **shared,
                **benchmark_metrics(evaluation, overlay_granted),
            }
        )
    combined_validation = pd.concat(
        validation_only_decisions
    ).sort_index()
    combined_overlay = pd.concat(overlay_decisions).sort_index()
    rows.append(
        {
            "record_type": "all_out_of_task_validation_only",
            "held_out_task_id": "all_tasks",
            "training_task_count": matrix["task_id"].nunique() - 1,
            "training_route_count": None,
            "selection_dependency_rows_excluded": None,
            "thresholds_json": "per_fold_training_only",
            "held_out_not_used_for_thresholds": True,
            "prediction_scope": (
                "validation_only_no_held_out_frozen_outcomes"
            ),
            "held_out_frozen_outcomes_used_for_decision": False,
            "failure_cases": "",
            **benchmark_metrics(matrix, combined_validation),
        }
    )
    rows.append(
        {
            "record_type": "post_freeze_safety_overlay",
            "held_out_task_id": "all_tasks",
            "training_task_count": matrix["task_id"].nunique() - 1,
            "training_route_count": None,
            "selection_dependency_rows_excluded": None,
            "thresholds_json": "per_fold_training_only",
            "held_out_not_used_for_thresholds": True,
            "prediction_scope": (
                "validation_gate_plus_post_freeze_safety_overlay"
            ),
            "held_out_frozen_outcomes_used_for_decision": True,
            "failure_cases": "",
            **benchmark_metrics(matrix, combined_overlay),
        }
    )
    return pd.DataFrame(rows)


def build_sensitivity_results(
    matrix: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    variants = []
    for max_budget in (0.10, 0.20, 0.30):
        variants.append(("max_expert_budget", max_budget))
    for frequency in (0.10, 0.20, 0.30, 0.50):
        variants.append(("min_candidate_selection_frequency", frequency))
    for best_delta in (0.0, 0.001, 0.003, 0.005):
        variants.append(("min_validation_delta_vs_best_single", best_delta))
    rows = []
    for parameter, value in variants:
        candidate = deepcopy(contract)
        candidate["thresholds"][parameter] = value
        granted, _ = _evaluate_rows(
            matrix,
            candidate,
            rule_family="complete_layered_gate",
        )
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "one_parameter_at_a_time": True,
                **benchmark_metrics(matrix, granted),
            }
        )
    return pd.DataFrame(rows)


def build_failure_case_audit(
    matrix: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_rows(
        frame: pd.DataFrame,
        granted: pd.Series,
        codes: list[str],
        *,
        prediction_scope: str,
        held_out_task_id: str,
        frozen_outcomes_used_for_decision: bool,
    ) -> None:
        for (_, row), allowed, error_codes in zip(
            frame.iterrows(),
            granted,
            codes,
            strict=True,
        ):
            favorable = bool(row["frozen_favorable"])
            if bool(allowed) == favorable:
                outcome = (
                    "correct_grant" if allowed else "correct_rejection"
                )
            else:
                outcome = (
                    "false_grant" if allowed else "false_rejection"
                )
            rows.append(
                {
                    "prediction_scope": prediction_scope,
                    "held_out_task_id": held_out_task_id,
                    "frozen_outcomes_used_for_decision": (
                        frozen_outcomes_used_for_decision
                    ),
                    "task_id": row["task_id"],
                    "pairing_id": row["pairing_id"],
                    "benchmark_outcome": outcome,
                    "is_misclassification": outcome.startswith("false_"),
                    "frozen_favorable": favorable,
                    "v1_1_granted": bool(allowed),
                    "error_codes": error_codes,
                    "target_validation_missing": row[
                        "target_validation_missing"
                    ],
                    "stability_source": row["stability_source"],
                    "cost_protocol_id": row["cost_protocol_id"],
                    "cost_protocol_comparable": row[
                        "cost_protocol_comparable"
                    ],
                    "domain_shift_status": row["domain_shift_status"],
                    "risk_proxy_semantics": row["risk_proxy_semantics"],
                }
            )

    full_granted, full_codes = _evaluate_rows(
        matrix,
        contract,
        rule_family="complete_layered_gate",
    )
    append_rows(
        matrix,
        full_granted,
        full_codes,
        prediction_scope="full_matrix_post_freeze_gate",
        held_out_task_id="none",
        frozen_outcomes_used_for_decision=True,
    )

    frozen_outcome_columns = (
        "frozen_main_metric",
        "frozen_delta_vs_scout",
        "frozen_corrected",
        "frozen_introduced",
        "frozen_net",
    )
    for held_out in sorted(matrix["task_id"].unique()):
        task_excluded = matrix["task_id"].eq(held_out)
        selection_dependency = matrix[
            "selection_source_task_id"
        ].eq(held_out)
        training = matrix.loc[
            ~(task_excluded | selection_dependency)
        ]
        evaluation = matrix.loc[task_excluded]
        validation_only = evaluation.copy()
        validation_only.loc[
            :,
            list(frozen_outcome_columns),
        ] = np.nan
        fold_contract = deepcopy(contract)
        fold_contract["thresholds"] = _derive_training_thresholds(
            training,
            contract,
        )
        validation_granted, validation_codes = _evaluate_rows(
            validation_only,
            fold_contract,
            rule_family="complete_layered_gate",
        )
        overlay_granted, overlay_codes = _evaluate_rows(
            evaluation,
            fold_contract,
            rule_family="complete_layered_gate",
        )
        append_rows(
            evaluation,
            validation_granted,
            validation_codes,
            prediction_scope="held_out_task_validation_only",
            held_out_task_id=str(held_out),
            frozen_outcomes_used_for_decision=False,
        )
        append_rows(
            evaluation,
            overlay_granted,
            overlay_codes,
            prediction_scope="held_out_task_post_freeze_overlay",
            held_out_task_id=str(held_out),
            frozen_outcomes_used_for_decision=True,
        )
    return pd.DataFrame(rows)


def _legacy_summary(project_root: Path) -> dict[str, Any]:
    comparison = pd.read_csv(
        project_root
        / OUTPUT_RELATIVE_DIR
        / "gate_retrospective_comparison.csv"
    )
    row = comparison.loc[comparison["task_id"].eq("all_tasks")].iloc[0]
    return {
        "beneficial_route_retention_rate": _finite(
            row.get("beneficial_route_retention_rate")
        ),
        "false_grant_rate": _finite(row.get("gated_false_grant_rate")),
        "false_rejection_rate": _finite(
            row.get("gated_false_rejection_rate")
        ),
        "ineffective_route_interception_rate": _finite(
            row.get("ineffective_route_interception_rate")
        ),
        "test_reversal_interception_rate": _finite(
            row.get("test_reversal_interception_rate")
        ),
        "executable_coverage_rate": _finite(
            row.get("research_action_coverage")
        ),
    }


def write_route_qualification_benchmark_artifacts(
    project_root: Path,
) -> dict[str, Path]:
    output_dir = project_root / V1_1_OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = project_root / V1_1_CONTRACT_RELATIVE_PATH
    sources_path = project_root / EVIDENCE_SOURCES_RELATIVE_PATH
    contract = _read_json(contract_path)
    matrix = build_route_qualification_evidence_matrix(project_root)
    ablation = build_ablation_results(matrix, contract)
    loto = build_leave_one_task_out_results(matrix, contract)
    sensitivity = build_sensitivity_results(matrix, contract)
    failure_audit = build_failure_case_audit(matrix, contract)

    paths = {
        "evidence_matrix": (
            output_dir / "route_qualification_evidence_matrix.csv"
        ),
        "leave_one_task_out": (
            output_dir / "leave_one_task_out_results.csv"
        ),
        "ablation": output_dir / "ablation_results.csv",
        "sensitivity": output_dir / "sensitivity_results.csv",
        "failure_case_audit": output_dir / "failure_case_audit.csv",
        "benchmark_summary": output_dir / "benchmark_summary.json",
    }
    matrix.to_csv(paths["evidence_matrix"], index=False)
    loto.to_csv(paths["leave_one_task_out"], index=False)
    ablation.to_csv(paths["ablation"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    failure_audit.to_csv(paths["failure_case_audit"], index=False)
    v1_1_all = loto.loc[
        loto["record_type"].eq("all_out_of_task_validation_only")
    ].iloc[0]
    post_freeze_overlay = loto.loc[
        loto["record_type"].eq("post_freeze_safety_overlay")
    ].iloc[0]
    v1_summary = _legacy_summary(project_root)
    v1_retention = _finite(
        v1_summary.get("beneficial_route_retention_rate")
    )
    v1_false_grant = _finite(v1_summary.get("false_grant_rate"))
    v1_1_retention = _finite(
        v1_1_all.get("beneficial_route_retention_rate")
    )
    v1_1_false_grant = _finite(v1_1_all.get("false_grant_rate"))
    retention_change = (
        v1_1_retention - v1_retention
        if v1_1_retention is not None and v1_retention is not None
        else None
    )
    false_grant_change = (
        v1_1_false_grant - v1_false_grant
        if v1_1_false_grant is not None and v1_false_grant is not None
        else None
    )
    if retention_change is None or false_grant_change is None:
        improvement_judgement = "insufficient_comparable_metrics"
    elif retention_change >= 0 and false_grant_change <= 0:
        improvement_judgement = "improved_or_equal_on_retention_and_false_grant"
    elif false_grant_change < 0:
        improvement_judgement = "safety_improved_with_retention_tradeoff"
    elif retention_change > 0:
        improvement_judgement = "coverage_improved_with_false_grant_tradeoff"
    else:
        improvement_judgement = "not_improved_on_primary_loto_metrics"
    summary = {
        "schema_version": "ophagent.route_qualification_benchmark.v1_1",
        "protocol_id": contract["protocol_id"],
        "formal_route_count": len(matrix),
        "task_count": int(matrix["task_id"].nunique()),
        "v1": v1_summary,
        "v1_1_leave_one_task_out": {
            key: _json_scalar(v1_1_all.get(key))
            for key in (
                "beneficial_route_retention_rate",
                "false_grant_rate",
                "false_rejection_rate",
                "ineffective_route_interception_rate",
                "test_reversal_interception_rate",
                "risk_tradeoff_limitation_rate",
                "executable_coverage_rate",
                "cost_protocol_group_count",
                "comparable_cost_by_protocol_json",
                "comparable_cost_change_ms_per_image",
                "expert_budget_change",
            )
        },
        "v1_1_post_freeze_safety_overlay": {
            key: _json_scalar(post_freeze_overlay.get(key))
            for key in (
                "beneficial_route_retention_rate",
                "false_grant_rate",
                "false_rejection_rate",
                "ineffective_route_interception_rate",
                "test_reversal_interception_rate",
                "risk_tradeoff_limitation_rate",
                "executable_coverage_rate",
            )
        },
        "loto_leakage_control": {
            "held_out_frozen_outcomes_used_for_primary_prediction": False,
            "selection_source_dependencies_excluded": True,
            "post_freeze_overlay_reported_separately": True,
        },
        "v1_to_v1_1_change": {
            "beneficial_route_retention_rate_change": retention_change,
            "false_grant_rate_change": false_grant_change,
            "judgement": improvement_judgement,
            "comparison_scope": (
                "v1 retrospective frozen-gate metrics versus v1.1 "
                "validation-only leave-one-task-out predictions; "
                "post-freeze safety overlay is reported separately"
            ),
        },
        "v1_failure_explanation": {
            "false_grants": [
                "aptos_locked_performance_primary",
                "aptos_locked_single_reference",
            ],
            "cause": (
                "v1允许risk_tradeoff进入研究病例模拟，且负稳定性下界只会"
                "降级beneficial，不会阻断risk_tradeoff"
            ),
            "false_rejections": [
                "aptos_locked_zero_introduced",
                "deepdrid_frozen_low_budget",
                "deepdrid_frozen_single_reference",
                "deepdrid_frozen_six_model_primary",
                "glaucoma_locked_single",
            ],
        },
        "interpretation_boundary": (
            "retrospective research evidence; not independent deployment or "
            "clinical validation"
        ),
        "clinical_route_eligible": False,
        "source_commit_sha": _git_commit(project_root),
    }
    paths["benchmark_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "ophagent.route_qualification_artifacts.v1_1",
        "contract_path": V1_1_CONTRACT_RELATIVE_PATH,
        "contract_sha256": file_sha256(contract_path),
        "evidence_sources_path": EVIDENCE_SOURCES_RELATIVE_PATH,
        "evidence_sources_sha256": file_sha256(sources_path),
        "source_commit_sha": _git_commit(project_root),
        "frozen_inputs_modified": False,
        "clinical_route_eligible": False,
        "artifacts": {
            name: {
                "path": _relative(project_root, path),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths
