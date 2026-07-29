#!/usr/bin/env python3
"""Run the frozen pre-consultation selective consultation study v0.1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.help_or_harm_benchmark import (  # noqa: E402
    ConsultationPolicyBaselineV1_1,
)
from app.selective_consultation import (  # noqa: E402
    LEARNED_POLICY_NAMES,
    METHOD_FEATURE_COLUMNS,
    fit_full_development_and_predict,
    model_discrimination_rows,
    nested_group_oof_predictions,
    paired_cluster_bootstrap_difference,
    select_consultations,
)


PROTOCOL_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "selective_consultation_method_v0_1.json"
)
BENCHMARK_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "selective_consultation_method_v0_1"
)
FIXED_BUDGETS = (0.05, 0.10, 0.20, 0.30)
OPERATING_BUDGETS = (0.10, 0.20, 0.30)
RISK_CAPS = (0.05, 0.10, 0.15, 0.20)
CURVE_POLICIES = (
    "entropy",
    "margin",
    "help_only_logistic",
    "harm_only_logistic",
    "dual_logistic_harm_screened_help",
    "oracle",
)
FIXED_POLICIES = (
    "entropy",
    "margin",
    "help_only_logistic",
    "harm_only_logistic",
    "dual_logistic_harm_screened_help",
    "oracle",
)
RISK_LOCK_POLICIES = (
    "entropy",
    "margin",
    "help_only_logistic",
    "harm_only_logistic",
    "dual_logistic_harm_screened_help",
)
PRIMARY_TASK_ID = "deepdrid_dr_5class_native"
PRIMARY_ROUTE_ID = (
    "deepdrid_dr_5class_native::keepfit_cfp__to__flair"
)
SENSITIVITY_TASK_ID = "aptos_dr_5class"
FORMAL_OUTPUT_NAMES = (
    "core_results.csv",
    "risk_budget_curve.csv",
    "failure_case_audit.csv",
    "research_report.md",
)


@dataclass
class RouteContext:
    task_id: str
    route_id: str
    scout_id: str
    expert_id: str
    route: pd.Series
    development: pd.DataFrame
    evaluation: pd.DataFrame
    development_predictions: pd.DataFrame
    evaluation_predictions: pd.DataFrame
    evaluation_split: str
    cohort: str
    formal_v1_1: ConsultationPolicyBaselineV1_1 | None


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def current_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_timestamp(project_root: Path) -> str:
    return subprocess.run(
        ["git", "show", "-s", "--format=%cI", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.10g",
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def rounded_budget(value: float) -> float:
    return round(float(value), 6)


def verify_benchmark_inputs(
    *,
    benchmark_dir: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = benchmark_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = set(protocol["input_benchmark"]["decision_required"])
    if str(manifest["decision"]) not in allowed:
        raise ValueError(
            f"Input benchmark decision {manifest['decision']} is not permitted."
        )
    expected_protocol_sha = str(
        protocol["input_benchmark"]["protocol_sha256"]
    )
    if str(manifest["protocol_sha256"]) != expected_protocol_sha:
        raise ValueError("Input benchmark protocol SHA256 changed.")
    outputs = {
        Path(str(item["uri"])).name: str(item["sha256"])
        for item in manifest["outputs"]
    }
    for name in ("case_level_benchmark.csv.gz", "candidate_route_inventory.csv"):
        path = benchmark_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = file_sha256(path)
        if observed != outputs[name]:
            raise ValueError(f"Frozen benchmark asset SHA256 changed: {name}")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": file_sha256(manifest_path),
        "case_table_sha256": outputs["case_level_benchmark.csv.gz"],
        "route_inventory_sha256": outputs["candidate_route_inventory.csv"],
        "benchmark_source_commit": str(manifest["source_commit_sha"]),
        "benchmark_decision": str(manifest["decision"]),
    }


def policy_baseline_from_route(
    route: pd.Series,
) -> ConsultationPolicyBaselineV1_1 | None:
    if not as_bool(route["v1_1_case_baseline_available"]):
        return None
    return ConsultationPolicyBaselineV1_1(
        route_id=str(route["formal_v1_1_route_id"]),
        routing_policy=str(route["formal_v1_1_policy"]),
        budget=float(route["formal_v1_1_budget"]),
        qualification_level=str(route["formal_v1_1_qualification_level"]),
    )


def evaluation_row(
    frame: pd.DataFrame,
    *,
    selected: np.ndarray,
    route: pd.Series,
    policy: str,
    budget: float,
    analysis_split: str,
    cohort: str,
    comparison_axis: str,
    risk_cap: float | None = None,
    budget_source: str = "fixed_grid",
) -> dict[str, Any]:
    n_cases = len(frame)
    selected_n = int(selected.sum())
    corrected = frame["corrected"].astype(bool).to_numpy()
    introduced = frame["introduced"].astype(bool).to_numpy()
    dangerous = frame["dangerous_introduced"].astype(bool).to_numpy()
    scout = frame["scout_pred"].astype(int).to_numpy()
    expert = frame["expert_pred"].astype(int).to_numpy()
    truth = frame["y_true"].astype(int).to_numpy()
    final = np.where(selected, expert, scout)
    corrected_selected = int((selected & corrected).sum())
    introduced_selected = int((selected & introduced).sum())
    realized_budget = selected_n / n_cases if n_cases else 0.0
    scout_cost = pd.to_numeric(
        pd.Series([route["scout_cost_ms_per_image"]]),
        errors="coerce",
    ).iloc[0]
    expert_cost = pd.to_numeric(
        pd.Series([route["expert_cost_ms_per_image"]]),
        errors="coerce",
    ).iloc[0]
    cost_comparable = as_bool(route["cost_comparable"])
    component_cost = (
        float(scout_cost + realized_budget * expert_cost)
        if cost_comparable and pd.notna(scout_cost) and pd.notna(expert_cost)
        else np.nan
    )
    harm_rate = (
        introduced_selected / selected_n if selected_n else np.nan
    )
    result = {
        "record_type": "policy_performance",
        "task_id": str(frame["task_id"].iloc[0]),
        "dataset_id": str(frame["dataset_id"].iloc[0]),
        "evaluation_design": str(frame["evaluation_design"].iloc[0]),
        "route_id": str(frame["route_id"].iloc[0]),
        "scout_id": str(frame["scout_id"].iloc[0]),
        "expert_id": str(frame["expert_id"].iloc[0]),
        "benchmark_split": str(frame["benchmark_split"].iloc[0]),
        "analysis_split": analysis_split,
        "cohort": cohort,
        "comparison_axis": comparison_axis,
        "policy": policy,
        "requested_budget": rounded_budget(budget),
        "budget_source": budget_source,
        "risk_cap": risk_cap,
        "risk_cap_met": (
            bool(harm_rate <= risk_cap)
            if risk_cap is not None and selected_n
            else (True if risk_cap is not None and not selected_n else np.nan)
        ),
        "n_cases": n_cases,
        "selected_n": selected_n,
        "realized_budget": realized_budget,
        "corrected_total": int(corrected.sum()),
        "introduced_total": int(introduced.sum()),
        "corrected_selected": corrected_selected,
        "introduced_selected": introduced_selected,
        "dangerous_introduced_selected": int((selected & dangerous).sum()),
        "net_selected": corrected_selected - introduced_selected,
        "corrected_capture_rate": (
            corrected_selected / corrected.sum()
            if corrected.sum()
            else np.nan
        ),
        "help_rate_among_consulted": (
            corrected_selected / selected_n if selected_n else np.nan
        ),
        "harm_rate_among_consulted": harm_rate,
        "introduced_burden_per_case": (
            introduced_selected / n_cases if n_cases else np.nan
        ),
        "scout_accuracy": float(np.mean(scout == truth)),
        "final_accuracy": float(np.mean(final == truth)),
        "cost_protocol_id": str(route["cost_protocol_id"]),
        "cost_comparable": cost_comparable,
        "estimated_component_cost_ms_per_case": component_cost,
        "formal_v1_1_route": bool(
            as_bool(route["v1_1_case_baseline_available"])
        ),
        "policy_may_grant_eligibility": False,
        "safety_eligibility_gate_required": True,
        "dataset_id_used_as_predictor": False,
        "current_case_expert_output_used_for_ranking": policy == "oracle",
        "test_used_for_fit_threshold_budget_or_route_selection": False,
        "retrospective_only": analysis_split != "development_oof",
    }
    return result


def discrimination_core_rows(
    context: RouteContext,
    *,
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    analysis_split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in model_discrimination_rows(frame, predictions):
        rows.append(
            {
                "record_type": "model_discrimination",
                "task_id": context.task_id,
                "dataset_id": str(frame["dataset_id"].iloc[0]),
                "evaluation_design": str(
                    frame["evaluation_design"].iloc[0]
                ),
                "route_id": context.route_id,
                "scout_id": context.scout_id,
                "expert_id": context.expert_id,
                "benchmark_split": str(frame["benchmark_split"].iloc[0]),
                "analysis_split": analysis_split,
                "cohort": context.cohort,
                "model_component": f"{item['outcome']}_logistic",
                "evaluation_cohort": item["evaluation_cohort"],
                "n_cases": item["n_cases"],
                "events": item["events"],
                "prevalence": item["prevalence"],
                "auroc": item["auroc"],
                "auprc": item["auprc"],
                "feature_columns": "|".join(METHOD_FEATURE_COLUMNS),
                "dataset_id_used_as_predictor": False,
                "current_case_expert_output_used_for_ranking": False,
                "test_used_for_fit_threshold_budget_or_route_selection": False,
                "retrospective_only": analysis_split != "development_oof",
            }
        )
    return rows


def select_for_policy(
    context: RouteContext,
    *,
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    policy: str,
    budget: float,
    safe_pool_multiplier: float,
) -> np.ndarray:
    return select_consultations(
        frame,
        policy=policy,
        budget=budget,
        predictions=predictions if policy in LEARNED_POLICY_NAMES else None,
        safe_pool_multiplier=safe_pool_multiplier,
        v1_1_baseline=(
            context.formal_v1_1
            if policy == "consultation_policy_baseline_v1_1"
            else None
        ),
    )


def strongest_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("No baseline candidate was available.")
    return sorted(
        values,
        key=lambda row: (
            -int(row["net_selected"]),
            -int(row["corrected_selected"]),
            int(row["introduced_selected"]),
            str(row["policy"]),
        ),
    )[0]


def annotate_comparators(
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, float], str]]:
    result = core.copy()
    comparator_map: dict[tuple[str, str, float], str] = {}
    performance = result["record_type"].eq("policy_performance")
    for route_id in sorted(result.loc[performance, "route_id"].dropna().unique()):
        for budget in FIXED_BUDGETS:
            development = result.loc[
                performance
                & result["route_id"].eq(route_id)
                & result["analysis_split"].eq("development_oof")
                & result["comparison_axis"].eq("same_budget")
                & result["requested_budget"].eq(budget)
                & result["policy"].isin(
                    [
                        "entropy",
                        "margin",
                        "consultation_policy_baseline_v1_1",
                    ]
                )
            ]
            if development.empty:
                continue
            comparator = strongest_row(development.to_dict("records"))
            policy = str(comparator["policy"])
            comparator_map[(route_id, "same_budget", budget)] = policy
            for split in ("development_oof", "retrospective_evaluation"):
                baseline_rows = result.loc[
                    performance
                    & result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["comparison_axis"].eq("same_budget")
                    & result["requested_budget"].eq(budget)
                    & result["policy"].eq(policy)
                ]
                if baseline_rows.empty:
                    continue
                baseline = baseline_rows.iloc[0]
                mask = (
                    performance
                    & result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["comparison_axis"].eq("same_budget")
                    & result["requested_budget"].eq(budget)
                )
                result.loc[mask, "locked_comparator_policy"] = policy
                for metric in (
                    "corrected_selected",
                    "introduced_selected",
                    "net_selected",
                    "selected_n",
                ):
                    result.loc[mask, f"comparator_{metric}"] = baseline[metric]
                    result.loc[mask, f"delta_{metric}"] = (
                        result.loc[mask, metric].astype(float)
                        - float(baseline[metric])
                    )
        for risk_cap in RISK_CAPS:
            development = result.loc[
                performance
                & result["route_id"].eq(route_id)
                & result["analysis_split"].eq("development_oof")
                & result["comparison_axis"].eq(
                    "locked_introduced_risk_cap"
                )
                & result["risk_cap"].eq(risk_cap)
                & result["policy"].isin(["entropy", "margin"])
            ]
            if development.empty:
                continue
            comparator = strongest_row(development.to_dict("records"))
            policy = str(comparator["policy"])
            comparator_map[(route_id, "risk_cap", risk_cap)] = policy
            for split in ("development_oof", "retrospective_evaluation"):
                baseline_rows = result.loc[
                    performance
                    & result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["comparison_axis"].eq(
                        "locked_introduced_risk_cap"
                    )
                    & result["risk_cap"].eq(risk_cap)
                    & result["policy"].eq(policy)
                ]
                if baseline_rows.empty:
                    continue
                baseline = baseline_rows.iloc[0]
                mask = (
                    performance
                    & result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["comparison_axis"].eq(
                        "locked_introduced_risk_cap"
                    )
                    & result["risk_cap"].eq(risk_cap)
                )
                result.loc[mask, "locked_comparator_policy"] = policy
                for metric in (
                    "corrected_selected",
                    "introduced_selected",
                    "net_selected",
                    "selected_n",
                ):
                    result.loc[mask, f"comparator_{metric}"] = baseline[metric]
                    result.loc[mask, f"delta_{metric}"] = (
                        result.loc[mask, metric].astype(float)
                        - float(baseline[metric])
                    )
    return result, comparator_map


def choose_locked_risk_budget(
    development_curve: pd.DataFrame,
    *,
    risk_cap: float,
) -> float:
    candidates = development_curve.loc[
        development_curve["requested_budget"].le(0.30)
        & development_curve["selected_n"].gt(0)
        & development_curve["harm_rate_among_consulted"].le(risk_cap)
    ].copy()
    if candidates.empty:
        return 0.0
    chosen = sorted(
        candidates.to_dict("records"),
        key=lambda row: (
            -int(row["corrected_selected"]),
            -int(row["net_selected"]),
            int(row["selected_n"]),
            float(row["requested_budget"]),
        ),
    )[0]
    return rounded_budget(float(chosen["requested_budget"]))


def add_bootstrap_intervals(
    core: pd.DataFrame,
    *,
    contexts: Mapping[str, RouteContext],
    comparator_map: Mapping[tuple[str, str, float], str],
    selection_cache: Mapping[tuple[str, str, str, float], np.ndarray],
    replicates: int,
) -> pd.DataFrame:
    result = core.copy()
    for route_id, context in contexts.items():
        if context.formal_v1_1 is None:
            continue
        for budget in FIXED_BUDGETS:
            comparator = comparator_map[(route_id, "same_budget", budget)]
            method_key = (
                route_id,
                "retrospective_evaluation",
                "dual_logistic_harm_screened_help",
                budget,
            )
            baseline_key = (
                route_id,
                "retrospective_evaluation",
                comparator,
                budget,
            )
            interval = paired_cluster_bootstrap_difference(
                context.evaluation,
                method_selected=selection_cache[method_key],
                baseline_selected=selection_cache[baseline_key],
                replicates=replicates,
                seed=stable_seed(f"{route_id}:{budget}:paired-bootstrap-v0.1"),
            )
            mask = (
                result["record_type"].eq("policy_performance")
                & result["route_id"].eq(route_id)
                & result["analysis_split"].eq("retrospective_evaluation")
                & result["comparison_axis"].eq("same_budget")
                & result["policy"].eq(
                    "dual_logistic_harm_screened_help"
                )
                & result["requested_budget"].eq(budget)
            )
            for column, value in interval.items():
                result.loc[mask, column] = value
    return result


def dominant_budgets(
    core: pd.DataFrame,
    *,
    route_id: str,
    analysis_split: str,
    policy: str,
) -> list[float]:
    rows = core.loc[
        core["record_type"].eq("policy_performance")
        & core["route_id"].eq(route_id)
        & core["analysis_split"].eq(analysis_split)
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq(policy)
        & core["requested_budget"].isin(OPERATING_BUDGETS)
    ].copy()
    dominant = rows.loc[
        rows["delta_corrected_selected"].ge(0)
        & rows["delta_introduced_selected"].le(0)
        & rows["delta_net_selected"].gt(0)
    ]
    return sorted(float(value) for value in dominant["requested_budget"])


def route_qualifies(dominant: Iterable[float]) -> bool:
    budgets = set(rounded_budget(value) for value in dominant)
    return len(budgets) >= 2 and 0.30 in budgets


def discrimination_value(
    core: pd.DataFrame,
    *,
    route_id: str,
    analysis_split: str,
    model_component: str,
    evaluation_cohort: str,
) -> float:
    rows = core.loc[
        core["record_type"].eq("model_discrimination")
        & core["route_id"].eq(route_id)
        & core["analysis_split"].eq(analysis_split)
        & core["model_component"].eq(model_component)
        & core["evaluation_cohort"].eq(evaluation_cohort)
    ]
    return float(rows.iloc[0]["auroc"]) if not rows.empty else float("nan")


def decide_result(
    core: pd.DataFrame,
    *,
    contexts: Mapping[str, RouteContext],
) -> tuple[str, dict[str, Any]]:
    deep_routes = sorted(
        route_id
        for route_id, context in contexts.items()
        if context.task_id == PRIMARY_TASK_ID
    )
    development_prequalified = [
        route_id
        for route_id in deep_routes
        if route_qualifies(
            dominant_budgets(
                core,
                route_id=route_id,
                analysis_split="development_oof",
                policy="dual_logistic_harm_screened_help",
            )
        )
    ]
    reproduced = [
        route_id
        for route_id in development_prequalified
        if route_qualifies(
            dominant_budgets(
                core,
                route_id=route_id,
                analysis_split="retrospective_evaluation",
                policy="dual_logistic_harm_screened_help",
            )
        )
    ]
    primary_dominant = dominant_budgets(
        core,
        route_id=PRIMARY_ROUTE_ID,
        analysis_split="retrospective_evaluation",
        policy="dual_logistic_harm_screened_help",
    )
    primary_budget_30 = core.loc[
        core["record_type"].eq("policy_performance")
        & core["route_id"].eq(PRIMARY_ROUTE_ID)
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq("dual_logistic_harm_screened_help")
        & core["requested_budget"].eq(0.30)
    ].iloc[0]
    ci_lower = float(primary_budget_30.get("net_difference_ci_lower", np.nan))
    method_go = route_qualifies(primary_dominant) and ci_lower >= 0.0

    harm_rows = core.loc[
        core["record_type"].eq("policy_performance")
        & core["route_id"].eq(PRIMARY_ROUTE_ID)
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq("harm_only_logistic")
        & core["requested_budget"].isin(OPERATING_BUDGETS)
    ].copy()
    harm_rows["harm_condition"] = (
        harm_rows["delta_introduced_selected"].lt(0)
        & harm_rows["delta_net_selected"].ge(0)
        & harm_rows["corrected_selected"].ge(
            0.8 * harm_rows["comparator_corrected_selected"]
        )
    )
    harm_budgets = sorted(
        float(value)
        for value in harm_rows.loc[
            harm_rows["harm_condition"], "requested_budget"
        ]
    )
    development_harm_auroc = discrimination_value(
        core,
        route_id=PRIMARY_ROUTE_ID,
        analysis_split="development_oof",
        model_component="introduced_logistic",
        evaluation_cohort="scout_correct_only",
    )
    evaluation_harm_auroc = discrimination_value(
        core,
        route_id=PRIMARY_ROUTE_ID,
        analysis_split="retrospective_evaluation",
        model_component="introduced_logistic",
        evaluation_cohort="scout_correct_only",
    )
    harm_only_go = (
        not method_go
        and route_qualifies(harm_budgets)
        and development_harm_auroc >= 0.60
        and evaluation_harm_auroc >= 0.60
    )

    reproduced_contexts = [contexts[route_id] for route_id in reproduced]
    distinct_scouts = {context.scout_id for context in reproduced_contexts}
    distinct_experts = {context.expert_id for context in reproduced_contexts}
    route_specific_go = (
        not method_go
        and not harm_only_go
        and len(reproduced) >= 3
        and len(distinct_scouts) >= 2
        and len(distinct_experts) >= 2
    )
    if method_go:
        decision = "METHOD_GO"
    elif harm_only_go:
        decision = "HARM_ONLY_GO"
    elif route_specific_go:
        decision = "ROUTE_SPECIFIC_GO"
    else:
        decision = "NO_IMPROVEMENT"
    evidence = {
        "primary_route_id": PRIMARY_ROUTE_ID,
        "primary_dominant_budgets": primary_dominant,
        "primary_budget_0_3_net_difference_ci_lower": ci_lower,
        "primary_harm_only_condition_budgets": harm_budgets,
        "primary_conditional_harm_auroc_development_oof": development_harm_auroc,
        "primary_conditional_harm_auroc_retrospective": evaluation_harm_auroc,
        "deepdrid_native_route_count": len(deep_routes),
        "development_prequalified_route_count": len(development_prequalified),
        "retrospectively_reproduced_route_count": len(reproduced),
        "retrospectively_reproduced_route_ids": reproduced,
        "reproduced_distinct_scout_count": len(distinct_scouts),
        "reproduced_distinct_expert_count": len(distinct_experts),
    }
    return decision, evidence


def build_failure_audit(
    *,
    contexts: Mapping[str, RouteContext],
    comparator_map: Mapping[tuple[str, str, float], str],
    selection_cache: Mapping[tuple[str, str, str, float], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route_id, context in sorted(contexts.items()):
        if context.formal_v1_1 is None:
            continue
        budget = rounded_budget(context.formal_v1_1.budget)
        comparator = comparator_map[(route_id, "same_budget", budget)]
        method = selection_cache[
            (
                route_id,
                "retrospective_evaluation",
                "dual_logistic_harm_screened_help",
                budget,
            )
        ]
        baseline = selection_cache[
            (
                route_id,
                "retrospective_evaluation",
                comparator,
                budget,
            )
        ]
        frame = context.evaluation.reset_index(drop=True)
        predictions = context.evaluation_predictions.reset_index(drop=True)
        masks = {
            "method_selected_introduced": (
                method & frame["introduced"].astype(bool).to_numpy()
            ),
            "method_missed_corrected": (
                (~method) & frame["corrected"].astype(bool).to_numpy()
            ),
            "method_only_introduced": (
                method
                & (~baseline)
                & frame["introduced"].astype(bool).to_numpy()
            ),
            "baseline_only_corrected": (
                baseline
                & (~method)
                & frame["corrected"].astype(bool).to_numpy()
            ),
            "method_selected_both_wrong": (
                method & frame["both_wrong"].astype(bool).to_numpy()
            ),
        }
        for category, mask in masks.items():
            indices = np.flatnonzero(mask)
            if "introduced" in category:
                order = np.argsort(
                    -predictions.loc[
                        indices, "predicted_introduced_probability"
                    ].to_numpy(dtype=float),
                    kind="stable",
                )
            else:
                order = np.argsort(
                    -predictions.loc[
                        indices, "predicted_corrected_probability"
                    ].to_numpy(dtype=float),
                    kind="stable",
                )
            for index in indices[order[:20]]:
                case = frame.iloc[int(index)]
                prediction = predictions.iloc[int(index)]
                rows.append(
                    {
                        "task_id": context.task_id,
                        "route_id": route_id,
                        "scout_id": context.scout_id,
                        "expert_id": context.expert_id,
                        "benchmark_split": context.evaluation_split,
                        "cohort": context.cohort,
                        "failure_category": category,
                        "case_audit_id": hashlib.sha256(
                            (
                                f"{context.task_id}\0{route_id}\0"
                                f"{case['case_id']}"
                            ).encode("utf-8")
                        ).hexdigest()[:20],
                        "y_true": int(case["y_true"]),
                        "scout_pred": int(case["scout_pred"]),
                        "expert_pred": int(case["expert_pred"]),
                        "corrected": bool(case["corrected"]),
                        "introduced": bool(case["introduced"]),
                        "both_wrong": bool(case["both_wrong"]),
                        "dangerous_introduced": bool(
                            case["dangerous_introduced"]
                        ),
                        "predicted_corrected_probability": float(
                            prediction["predicted_corrected_probability"]
                        ),
                        "predicted_introduced_probability": float(
                            prediction["predicted_introduced_probability"]
                        ),
                        "scout_entropy": float(case["scout_entropy"]),
                        "scout_margin": float(case["scout_margin"]),
                        "budget": budget,
                        "locked_comparator_policy": comparator,
                        "method_selected": bool(method[int(index)]),
                        "comparator_selected": bool(baseline[int(index)]),
                        "expert_output_use": "posthoc_audit_only",
                        "patient_identity_included": False,
                        "private_path_included": False,
                    }
                )
    return pd.DataFrame(rows)


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(
    *,
    core: pd.DataFrame,
    decision: str,
    evidence: Mapping[str, Any],
    source_commit: str,
    protocol_sha256: str,
    benchmark_audit: Mapping[str, Any],
) -> str:
    primary_rows = core.loc[
        core["record_type"].eq("policy_performance")
        & core["route_id"].eq(PRIMARY_ROUTE_ID)
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["comparison_axis"].eq("same_budget")
        & core["policy"].eq("dual_logistic_harm_screened_help")
        & core["requested_budget"].isin(OPERATING_BUDGETS)
    ].sort_values("requested_budget")
    primary_table: list[list[Any]] = []
    for _, row in primary_rows.iterrows():
        primary_table.append(
            [
                format_metric(row["requested_budget"], 2),
                row["locked_comparator_policy"],
                (
                    f"{int(row['corrected_selected'])}/"
                    f"{int(row['comparator_corrected_selected'])}"
                ),
                (
                    f"{int(row['introduced_selected'])}/"
                    f"{int(row['comparator_introduced_selected'])}"
                ),
                (
                    f"{int(row['net_selected'])}/"
                    f"{int(row['comparator_net_selected'])}"
                ),
                int(row["delta_net_selected"]),
            ]
        )

    discrimination_table: list[list[Any]] = []
    for split in ("development_oof", "retrospective_evaluation"):
        for component, cohort in (
            ("corrected_logistic", "scout_wrong_only"),
            ("introduced_logistic", "scout_correct_only"),
        ):
            row = core.loc[
                core["record_type"].eq("model_discrimination")
                & core["route_id"].eq(PRIMARY_ROUTE_ID)
                & core["analysis_split"].eq(split)
                & core["model_component"].eq(component)
                & core["evaluation_cohort"].eq(cohort)
            ].iloc[0]
            discrimination_table.append(
                [
                    split,
                    component,
                    cohort,
                    int(row["events"]),
                    format_metric(row["auroc"]),
                    format_metric(row["auprc"]),
                ]
            )

    risk_table: list[list[Any]] = []
    for risk_cap in RISK_CAPS:
        method = core.loc[
            core["record_type"].eq("policy_performance")
            & core["route_id"].eq(PRIMARY_ROUTE_ID)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["comparison_axis"].eq("locked_introduced_risk_cap")
            & core["risk_cap"].eq(risk_cap)
            & core["policy"].eq("dual_logistic_harm_screened_help")
        ].iloc[0]
        baseline = core.loc[
            core["record_type"].eq("policy_performance")
            & core["route_id"].eq(PRIMARY_ROUTE_ID)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["comparison_axis"].eq("locked_introduced_risk_cap")
            & core["risk_cap"].eq(risk_cap)
            & core["policy"].eq(method["locked_comparator_policy"])
        ].iloc[0]
        risk_table.append(
            [
                format_metric(risk_cap, 2),
                (
                    f"{format_metric(method['requested_budget'], 2)}/"
                    f"{format_metric(baseline['requested_budget'], 2)}"
                ),
                (
                    f"{int(method['corrected_selected'])}/"
                    f"{int(baseline['corrected_selected'])}"
                ),
                (
                    f"{int(method['introduced_selected'])}/"
                    f"{int(baseline['introduced_selected'])}"
                ),
                (
                    f"{format_metric(method['harm_rate_among_consulted'])}/"
                    f"{format_metric(baseline['harm_rate_among_consulted'])}"
                ),
                (
                    f"{bool(method['risk_cap_met'])}/"
                    f"{bool(baseline['risk_cap_met'])}"
                ),
            ]
        )

    aptos_routes = core.loc[
        core["record_type"].eq("policy_performance")
        & core["task_id"].eq(SENSITIVITY_TASK_ID),
        "route_id",
    ].dropna().unique()
    aptos_prequalified = 0
    aptos_reproduced = 0
    for route_id in aptos_routes:
        dev = dominant_budgets(
            core,
            route_id=str(route_id),
            analysis_split="development_oof",
            policy="dual_logistic_harm_screened_help",
        )
        if route_qualifies(dev):
            aptos_prequalified += 1
            evaluated = dominant_budgets(
                core,
                route_id=str(route_id),
                analysis_split="retrospective_evaluation",
                policy="dual_logistic_harm_screened_help",
            )
            if route_qualifies(evaluated):
                aptos_reproduced += 1

    decision_meanings = {
        "METHOD_GO": (
            "固定 DeepDRiD 主路线达到预声明的双模型支配条件；仅建议进入独立"
            "未暴露确认，不代表临床或部署有效。"
        ),
        "HARM_ONLY_GO": (
            "当前稳定证据只支持调用前的 introduced 风险识别；尚不支持声称能"
            "更好保留 corrected。"
        ),
        "ROUTE_SPECIFIC_GO": (
            "改进只在开发阶段预筛出的部分路线复现，不能作为跨路线通用方法。"
        ),
        "NO_IMPROVEMENT": (
            "相对既有简单基线未达到预声明改进条件，不建议以当前方法进入独立"
            "确认或更复杂训练。"
        ),
    }
    return "\n".join(
        [
            "# OphAgent 预咨询选择性会诊方法研究 v0.1",
            "",
            "## Material Passport",
            "",
            "- Origin Skill: experiment-agent",
            "- Origin Mode: run",
            "- Verification Status: UNVERIFIED（需以同一提交确定性复跑核验）",
            "- Version Label: selective_consultation_method_v0_1",
            "",
            "## 结论",
            "",
            f"**{decision}**",
            "",
            decision_meanings[decision],
            "",
            "本结论属于已暴露冻结结果上的回顾性研究证据；`SafetyEligibilityGate` "
            "仍不可绕过，方法本身不能授予路线资格或触发 Expert。",
            "",
            "## 研究设计",
            "",
            "- 主分析固定为 DeepDRiD 原生 `keepfit_cfp→flair`，按患者分组；"
            "- 其余 DeepDRiD 原生路线只做开发预筛后的异质性分析；"
            "- APTOS 仅在确认重复排除后的图像级队列做敏感性分析；"
            "- DeepDRiD 外部迁移因缺少同域开发折而排除，未在冻结结果上拟合；"
            "- 两个 L2 逻辑回归分别预测 corrected 与 introduced；无调参搜索；"
            "- 双模型策略先取预测 introduced 风险最低的 2×预算安全池，再按"
            " predicted corrected 排序；没有事后加权综合分数；"
            "- 开发预测采用嵌套分组交叉拟合；回顾性预测只使用完整开发折拟合。"
            "",
            "## 固定主路线：相同 Expert 预算",
            "",
            markdown_table(
                primary_table,
                [
                    "预算",
                    "开发锁定基线",
                    "corrected 方法/基线",
                    "introduced 方法/基线",
                    "net 方法/基线",
                    "Δnet",
                ],
            ),
            "",
            "预算 0.30 的患者配对 bootstrap Δnet 95% 区间："
            f"[{format_metric(primary_rows.loc[primary_rows['requested_budget'].eq(0.30), 'net_difference_ci_lower'].iloc[0])}, "
            f"{format_metric(primary_rows.loc[primary_rows['requested_budget'].eq(0.30), 'net_difference_ci_upper'].iloc[0])}]。",
            "",
            "## 固定主路线：模型可识别性",
            "",
            markdown_table(
                discrimination_table,
                ["数据", "模型", "条件队列", "事件", "AUROC", "AUPRC"],
            ),
            "",
            "条件 AUROC 用于区分真正的 Expert 特异信号与一般 Scout 错误检测，"
            "不作为临床安全终点。",
            "",
            "## 固定主路线：由开发 OOF 锁定的 introduced 风险上限",
            "",
            markdown_table(
                risk_table,
                [
                    "风险上限",
                    "预算 方法/基线",
                    "corrected 方法/基线",
                    "introduced 方法/基线",
                    "实际风险 方法/基线",
                    "回顾性达标 方法/基线",
                ],
            ),
            "",
            "风险上限只在开发 OOF 上选择预算；回顾性队列不会重新选阈值。若"
            "回顾性实际风险超限，该行视为未外推成功。",
            "",
            "## 路线异质性与敏感性",
            "",
            f"- DeepDRiD 原生路线：90；开发预筛通过 "
            f"{evidence['development_prequalified_route_count']}；回顾性复现 "
            f"{evidence['retrospectively_reproduced_route_count']}；",
            f"- 回顾性复现覆盖 Scout {evidence['reproduced_distinct_scout_count']} "
            f"种、Expert {evidence['reproduced_distinct_expert_count']} 种；",
            f"- APTOS 图像级敏感性：开发预筛通过 {aptos_prequalified}/90，"
            f"回顾性复现 {aptos_reproduced}/{aptos_prequalified or 0}；",
            "- APTOS 无患者/眼别标识，不能把图像级稳定性解释为患者级泛化。"
            "",
            "## 失败边界",
            "",
            "- 冻结回顾性结果此前已暴露，不能充当独立确认；",
            "- corrected/introduced 是标签定义的模型错误代理，不是临床伤害、"
            "治疗获益或最终诊断；",
            "- 路线共享病例和模型，90 条路线不能当作 90 个独立临床样本；",
            "- 当前特征没有真实图像质量、多模态或临床资料；",
            "- 更复杂模型只有在独立患者级确认集与足够事件数就绪后才有意义。"
            "",
            "## 追溯",
            "",
            f"- 实现提交：`{source_commit}`",
            f"- 方法协议 SHA256：`{protocol_sha256}`",
            f"- 输入 Benchmark manifest SHA256："
            f"`{benchmark_audit['manifest_sha256']}`",
            f"- 输入病例表 SHA256：`{benchmark_audit['case_table_sha256']}`",
            "- 眼底模型训练/推理：未执行；冻结预测资产：未修改。"
            "",
        ]
    )


def ensure_no_sensitive_output(frame: pd.DataFrame) -> None:
    forbidden = {
        "patient_group_id",
        "patient_id",
        "image_path",
        "private_path",
        "image_sha256",
    }
    present = sorted(forbidden.intersection(frame.columns))
    if present:
        raise ValueError(f"Sensitive fields entered a formal output: {present}")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    benchmark_dir = (
        args.benchmark_dir.resolve()
        if args.benchmark_dir
        else project_root / BENCHMARK_RELATIVE_PATH
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / OUTPUT_RELATIVE_PATH
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = project_root / PROTOCOL_RELATIVE_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = file_sha256(protocol_path)
    benchmark_audit = verify_benchmark_inputs(
        benchmark_dir=benchmark_dir,
        protocol=protocol,
    )
    source_commit = current_commit(project_root)
    generated_at = commit_timestamp(project_root)
    safe_pool_multiplier = float(
        protocol["policy_contract"]["safe_pool_multiplier"]
    )

    cases = pd.read_csv(
        benchmark_dir / "case_level_benchmark.csv.gz",
        low_memory=False,
    )
    inventory = pd.read_csv(
        benchmark_dir / "candidate_route_inventory.csv",
        low_memory=False,
    )
    included_tasks = {PRIMARY_TASK_ID, SENSITIVITY_TASK_ID}
    cases = cases.loc[
        cases["task_id"].isin(included_tasks)
        & cases["primary_cohort_eligible"].astype(bool)
    ].copy()
    inventory = inventory.loc[inventory["task_id"].isin(included_tasks)].copy()
    if len(inventory) != 180:
        raise ValueError(f"Expected 180 in-scope routes, observed {len(inventory)}.")

    core_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    selection_cache: dict[tuple[str, str, str, float], np.ndarray] = {}
    contexts: dict[str, RouteContext] = {}
    curve_budgets = tuple(
        [0.0]
        + [rounded_budget(index / 100) for index in range(1, 31)]
        + [1.0]
    )

    for _, route in inventory.sort_values("route_id").iterrows():
        route_id = str(route["route_id"])
        task_id = str(route["task_id"])
        route_cases = cases.loc[cases["route_id"].eq(route_id)].copy()
        development = route_cases.loc[
            route_cases["benchmark_split"].eq("development")
        ].sort_values("case_id").reset_index(drop=True)
        evaluation_splits = sorted(
            set(route_cases["benchmark_split"]) - {"development"}
        )
        if len(evaluation_splits) != 1:
            raise ValueError(f"{route_id}: expected one retrospective split.")
        evaluation_split = evaluation_splits[0]
        evaluation = route_cases.loc[
            route_cases["benchmark_split"].eq(evaluation_split)
        ].sort_values("case_id").reset_index(drop=True)
        if development.empty or evaluation.empty:
            raise ValueError(f"{route_id}: missing development or evaluation cases.")
        salt = f"{task_id}:{route_id}:selective-consultation-v0.1"
        development_predictions = nested_group_oof_predictions(
            development,
            n_folds=5,
            minimum_route_events=10,
            salt=salt,
        ).reset_index(drop=True)
        _, evaluation_predictions = fit_full_development_and_predict(
            development,
            evaluation,
            n_folds=5,
            minimum_route_events=10,
            salt=salt,
        )
        evaluation_predictions = evaluation_predictions.reset_index(drop=True)
        cohort = (
            "deepdrid_patient_grouped_primary"
            if task_id == PRIMARY_TASK_ID
            else "aptos_exact_duplicate_excluded_image_sensitivity"
        )
        context = RouteContext(
            task_id=task_id,
            route_id=route_id,
            scout_id=str(route["scout_id"]),
            expert_id=str(route["expert_id"]),
            route=route,
            development=development,
            evaluation=evaluation,
            development_predictions=development_predictions,
            evaluation_predictions=evaluation_predictions,
            evaluation_split=evaluation_split,
            cohort=cohort,
            formal_v1_1=policy_baseline_from_route(route),
        )
        contexts[route_id] = context
        core_rows.extend(
            discrimination_core_rows(
                context,
                frame=development,
                predictions=development_predictions,
                analysis_split="development_oof",
            )
        )
        core_rows.extend(
            discrimination_core_rows(
                context,
                frame=evaluation,
                predictions=evaluation_predictions,
                analysis_split="retrospective_evaluation",
            )
        )

        split_inputs = (
            (
                "development_oof",
                development,
                development_predictions,
            ),
            (
                "retrospective_evaluation",
                evaluation,
                evaluation_predictions,
            ),
        )
        for analysis_split, frame, predictions in split_inputs:
            for policy, budget in (
                ("scout_only", 0.0),
                ("always_expert", 1.0),
            ):
                selected = select_for_policy(
                    context,
                    frame=frame,
                    predictions=predictions,
                    policy=policy,
                    budget=budget,
                    safe_pool_multiplier=safe_pool_multiplier,
                )
                core_rows.append(
                    evaluation_row(
                        frame,
                        selected=selected,
                        route=route,
                        policy=policy,
                        budget=budget,
                        analysis_split=analysis_split,
                        cohort=cohort,
                        comparison_axis="reference_endpoint",
                    )
                )
                selection_cache[
                    (route_id, analysis_split, policy, rounded_budget(budget))
                ] = selected
            for budget in FIXED_BUDGETS:
                for policy in FIXED_POLICIES:
                    selected = select_for_policy(
                        context,
                        frame=frame,
                        predictions=predictions,
                        policy=policy,
                        budget=budget,
                        safe_pool_multiplier=safe_pool_multiplier,
                    )
                    core_rows.append(
                        evaluation_row(
                            frame,
                            selected=selected,
                            route=route,
                            policy=policy,
                            budget=budget,
                            analysis_split=analysis_split,
                            cohort=cohort,
                            comparison_axis="same_budget",
                        )
                    )
                    selection_cache[
                        (
                            route_id,
                            analysis_split,
                            policy,
                            rounded_budget(budget),
                        )
                    ] = selected
            if context.formal_v1_1 is not None:
                budget = rounded_budget(context.formal_v1_1.budget)
                policy = "consultation_policy_baseline_v1_1"
                selected = select_for_policy(
                    context,
                    frame=frame,
                    predictions=predictions,
                    policy=policy,
                    budget=budget,
                    safe_pool_multiplier=safe_pool_multiplier,
                )
                core_rows.append(
                    evaluation_row(
                        frame,
                        selected=selected,
                        route=route,
                        policy=policy,
                        budget=budget,
                        analysis_split=analysis_split,
                        cohort=cohort,
                        comparison_axis="same_budget",
                        budget_source="frozen_v1_1_route_budget",
                    )
                )
                selection_cache[
                    (route_id, analysis_split, policy, budget)
                ] = selected

            for policy in CURVE_POLICIES:
                for budget in curve_budgets:
                    selected = select_for_policy(
                        context,
                        frame=frame,
                        predictions=predictions,
                        policy=policy,
                        budget=budget,
                        safe_pool_multiplier=safe_pool_multiplier,
                    )
                    curve_rows.append(
                        evaluation_row(
                            frame,
                            selected=selected,
                            route=route,
                            policy=policy,
                            budget=budget,
                            analysis_split=analysis_split,
                            cohort=cohort,
                            comparison_axis="risk_budget_curve",
                            budget_source="predeclared_curve_grid",
                        )
                    )

    curve = pd.DataFrame(curve_rows)
    for route_id, context in contexts.items():
        for policy in RISK_LOCK_POLICIES:
            development_curve = curve.loc[
                curve["route_id"].eq(route_id)
                & curve["analysis_split"].eq("development_oof")
                & curve["policy"].eq(policy)
            ]
            for risk_cap in RISK_CAPS:
                budget = choose_locked_risk_budget(
                    development_curve,
                    risk_cap=risk_cap,
                )
                for analysis_split in (
                    "development_oof",
                    "retrospective_evaluation",
                ):
                    source = curve.loc[
                        curve["route_id"].eq(route_id)
                        & curve["analysis_split"].eq(analysis_split)
                        & curve["policy"].eq(policy)
                        & curve["requested_budget"].eq(budget)
                    ].iloc[0].to_dict()
                    source["comparison_axis"] = "locked_introduced_risk_cap"
                    source["risk_cap"] = risk_cap
                    source["risk_cap_met"] = (
                        bool(
                            float(source["harm_rate_among_consulted"])
                            <= risk_cap
                        )
                        if int(source["selected_n"]) > 0
                        else True
                    )
                    source["budget_source"] = (
                        "development_oof_locked_for_risk_cap"
                    )
                    core_rows.append(source)

    core = pd.DataFrame(core_rows)
    core, comparator_map = annotate_comparators(core)
    core = add_bootstrap_intervals(
        core,
        contexts=contexts,
        comparator_map=comparator_map,
        selection_cache=selection_cache,
        replicates=args.bootstrap_replicates,
    )
    decision, decision_evidence = decide_result(core, contexts=contexts)
    failure_audit = build_failure_audit(
        contexts=contexts,
        comparator_map=comparator_map,
        selection_cache=selection_cache,
    )
    ensure_no_sensitive_output(core)
    ensure_no_sensitive_output(curve)
    ensure_no_sensitive_output(failure_audit)
    core = core.sort_values(
        [
            "record_type",
            "task_id",
            "route_id",
            "analysis_split",
            "comparison_axis",
            "risk_cap",
            "requested_budget",
            "policy",
            "model_component",
            "evaluation_cohort",
        ],
        na_position="last",
    ).reset_index(drop=True)
    curve = curve.sort_values(
        [
            "task_id",
            "route_id",
            "analysis_split",
            "policy",
            "requested_budget",
        ]
    ).reset_index(drop=True)
    failure_audit = failure_audit.sort_values(
        [
            "task_id",
            "route_id",
            "failure_category",
            "case_audit_id",
        ]
    ).reset_index(drop=True)
    report = build_report(
        core=core,
        decision=decision,
        evidence=decision_evidence,
        source_commit=source_commit,
        protocol_sha256=protocol_sha256,
        benchmark_audit=benchmark_audit,
    )

    write_csv(output_dir / "core_results.csv", core)
    write_csv(output_dir / "risk_budget_curve.csv", curve)
    write_csv(output_dir / "failure_case_audit.csv", failure_audit)
    (output_dir / "research_report.md").write_text(
        report,
        encoding="utf-8",
    )
    output_assets = []
    for name in FORMAL_OUTPUT_NAMES:
        path = output_dir / name
        output_assets.append(
            {
                "uri": f"repo://{path.relative_to(project_root).as_posix()}",
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": (
            "ophagent.selective_consultation_artifact_manifest.v0_1"
        ),
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_uri": f"repo://{PROTOCOL_RELATIVE_PATH.as_posix()}",
        "protocol_sha256": protocol_sha256,
        "source_commit_sha": source_commit,
        "generated_at_utc": generated_at,
        "decision": decision,
        "decision_evidence": decision_evidence,
        "input_benchmark": benchmark_audit,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "frozen_benchmark_modified": False,
        "frozen_prediction_assets_modified": False,
        "ophthalmic_model_training_performed": False,
        "ophthalmic_model_inference_performed": False,
        "statistical_control_model_fit_performed": True,
        "external_api_used": False,
        "test_used_for_fit_threshold_budget_or_route_selection": False,
        "retrospective_results_used_for_research_decision": True,
        "independent_confirmation_required": True,
        "outputs": output_assets,
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "decision": decision,
                "source_commit_sha": source_commit,
                "protocol_sha256": protocol_sha256,
                "core_rows": len(core),
                "curve_rows": len(curve),
                "failure_rows": len(failure_audit),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
