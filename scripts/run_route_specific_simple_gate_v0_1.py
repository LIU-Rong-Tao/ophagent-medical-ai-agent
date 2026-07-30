#!/usr/bin/env python3
"""Revalidate the simple gate on development-qualified native routes only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.route_specific_simple_gate import (  # noqa: E402
    RouteScreeningCriteria,
    screen_high_quality_routes,
)
from app.selective_consultation import (  # noqa: E402
    fit_full_development_and_predict,
    nested_group_oof_predictions,
    paired_cluster_bootstrap_difference,
)
from scripts.run_selective_consultation_method_v0_1 import (  # noqa: E402
    RouteContext,
    commit_timestamp,
    current_commit,
    evaluation_row,
    file_sha256,
    policy_baseline_from_route,
    rounded_budget,
    select_for_policy,
    stable_seed,
    strongest_row,
    verify_benchmark_inputs,
    write_csv,
)


PROTOCOL_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "route_specific_simple_gate_v0_1.json"
)
BENCHMARK_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "route_specific_simple_gate_v0_1"
)
TASK_ID = "deepdrid_dr_5class_native"
OPERATING_BUDGETS = (0.10, 0.20, 0.30)
RANKING_POLICIES = (
    "entropy",
    "margin",
    "dual_logistic_harm_screened_help",
)
METHOD_POLICY = "dual_logistic_harm_screened_help"
FORMAL_OUTPUT_NAMES = (
    "qualified_routes.csv",
    "core_comparison_results.csv",
    "research_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser.parse_args()


def _annotate_screening_thresholds(
    qualified: pd.DataFrame,
    criteria: RouteScreeningCriteria,
) -> pd.DataFrame:
    result = qualified.copy()
    values = {
        "criterion_minimum_scout_accuracy": criteria.minimum_scout_accuracy,
        "criterion_minimum_expert_accuracy": criteria.minimum_expert_accuracy,
        "criterion_minimum_expert_accuracy_gain": (
            criteria.minimum_expert_accuracy_gain
        ),
        "criterion_minimum_corrected_to_introduced_ratio": (
            criteria.minimum_corrected_to_introduced_ratio
        ),
        "criterion_minimum_corrected_events": criteria.minimum_corrected_events,
        "criterion_minimum_introduced_events": (
            criteria.minimum_introduced_events
        ),
        "criterion_minimum_net_events": criteria.minimum_net_events,
        "criterion_minimum_positive_gain_folds": (
            criteria.minimum_positive_gain_folds
        ),
        "criterion_minimum_nonnegative_net_folds": (
            criteria.minimum_nonnegative_net_folds
        ),
        "criterion_maximum_fold_gain_standard_deviation": (
            criteria.maximum_fold_gain_standard_deviation
        ),
    }
    for column, value in values.items():
        result[column] = value
    result["evaluation_used_for_screening"] = False
    return result


def _unavailable_v1_1_row(
    context: RouteContext,
    *,
    analysis_split: str,
) -> dict[str, Any]:
    return {
        "record_type": "policy_availability",
        "task_id": context.task_id,
        "dataset_id": str(context.route["dataset_id"]),
        "evaluation_design": str(context.route["evaluation_design"]),
        "route_id": context.route_id,
        "scout_id": context.scout_id,
        "expert_id": context.expert_id,
        "benchmark_split": (
            "development"
            if analysis_split == "development_oof"
            else context.evaluation_split
        ),
        "analysis_split": analysis_split,
        "cohort": context.cohort,
        "comparison_axis": "frozen_v1_1_availability",
        "policy": "consultation_policy_baseline_v1_1",
        "requested_budget": np.nan,
        "selected_n": np.nan,
        "corrected_selected": np.nan,
        "introduced_selected": np.nan,
        "net_selected": np.nan,
        "comparison_status": "not_applicable_no_frozen_route_identity",
        "v1_1_unavailable_reason": str(
            context.route["v1_1_unavailable_reason"]
        ),
        "policy_may_grant_eligibility": False,
        "safety_eligibility_gate_required": True,
        "current_case_expert_output_used_for_ranking": False,
        "test_used_for_fit_threshold_budget_or_route_selection": False,
        "route_selected_from_development_only": True,
    }


def _build_context(
    *,
    route: pd.Series,
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> RouteContext:
    route_id = str(route["route_id"])
    salt = f"{TASK_ID}:{route_id}:selective-consultation-v0.1"
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
    return RouteContext(
        task_id=TASK_ID,
        route_id=route_id,
        scout_id=str(route["scout_id"]),
        expert_id=str(route["expert_id"]),
        route=route,
        development=development,
        evaluation=evaluation,
        development_predictions=development_predictions,
        evaluation_predictions=evaluation_predictions.reset_index(drop=True),
        evaluation_split=str(evaluation["benchmark_split"].iloc[0]),
        cohort="deepdrid_patient_grouped_high_quality_route",
        formal_v1_1=policy_baseline_from_route(route),
    )


def _generate_policy_rows(
    contexts: Mapping[str, RouteContext],
    *,
    safe_pool_multiplier: float,
) -> tuple[
    pd.DataFrame,
    dict[tuple[str, str, str, float], np.ndarray],
]:
    rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, str, str, float], np.ndarray] = {}
    for route_id, context in sorted(contexts.items()):
        split_inputs = (
            (
                "development_oof",
                context.development,
                context.development_predictions,
            ),
            (
                "retrospective_evaluation",
                context.evaluation,
                context.evaluation_predictions,
            ),
        )
        for analysis_split, frame, predictions in split_inputs:
            for policy, budget in (("scout_only", 0.0), ("always_expert", 1.0)):
                selected = select_for_policy(
                    context,
                    frame=frame,
                    predictions=predictions,
                    policy=policy,
                    budget=budget,
                    safe_pool_multiplier=safe_pool_multiplier,
                )
                row = evaluation_row(
                    frame,
                    selected=selected,
                    route=context.route,
                    policy=policy,
                    budget=budget,
                    analysis_split=analysis_split,
                    cohort=context.cohort,
                    comparison_axis="reference_endpoint",
                )
                row["comparison_status"] = "available"
                row["route_selected_from_development_only"] = True
                rows.append(row)
                selections[
                    (route_id, analysis_split, policy, rounded_budget(budget))
                ] = selected
            for budget in OPERATING_BUDGETS:
                for policy in RANKING_POLICIES:
                    selected = select_for_policy(
                        context,
                        frame=frame,
                        predictions=predictions,
                        policy=policy,
                        budget=budget,
                        safe_pool_multiplier=safe_pool_multiplier,
                    )
                    row = evaluation_row(
                        frame,
                        selected=selected,
                        route=context.route,
                        policy=policy,
                        budget=budget,
                        analysis_split=analysis_split,
                        cohort=context.cohort,
                        comparison_axis="same_budget",
                    )
                    row["comparison_status"] = "available"
                    row["route_selected_from_development_only"] = True
                    rows.append(row)
                    selections[
                        (route_id, analysis_split, policy, rounded_budget(budget))
                    ] = selected
            if context.formal_v1_1 is None:
                rows.append(
                    _unavailable_v1_1_row(
                        context,
                        analysis_split=analysis_split,
                    )
                )
            else:
                budget = rounded_budget(context.formal_v1_1.budget)
                selected = select_for_policy(
                    context,
                    frame=frame,
                    predictions=predictions,
                    policy="consultation_policy_baseline_v1_1",
                    budget=budget,
                    safe_pool_multiplier=safe_pool_multiplier,
                )
                row = evaluation_row(
                    frame,
                    selected=selected,
                    route=context.route,
                    policy="consultation_policy_baseline_v1_1",
                    budget=budget,
                    analysis_split=analysis_split,
                    cohort=context.cohort,
                    comparison_axis="same_budget",
                    budget_source="frozen_v1_1_route_budget",
                )
                row["comparison_status"] = "available"
                row["route_selected_from_development_only"] = True
                rows.append(row)
                selections[
                    (
                        route_id,
                        analysis_split,
                        "consultation_policy_baseline_v1_1",
                        budget,
                    )
                ] = selected
    return pd.DataFrame(rows), selections


def _annotate_locked_comparators(
    core: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, float], str]]:
    result = core.copy()
    comparator_map: dict[tuple[str, float], str] = {}
    result["locked_comparator_policy"] = pd.Series(
        index=result.index,
        dtype="object",
    )
    for column in (
        "comparator_corrected_selected",
        "comparator_introduced_selected",
        "comparator_net_selected",
        "delta_corrected_selected",
        "delta_introduced_selected",
        "delta_net_selected",
    ):
        result[column] = np.nan
    for route_id in sorted(result["route_id"].dropna().unique()):
        for budget in OPERATING_BUDGETS:
            development = result.loc[
                result["record_type"].eq("policy_performance")
                & result["route_id"].eq(route_id)
                & result["analysis_split"].eq("development_oof")
                & result["comparison_axis"].eq("same_budget")
                & result["requested_budget"].eq(budget)
                & result["policy"].isin(["entropy", "margin"])
            ]
            comparator = strongest_row(development.to_dict("records"))
            policy = str(comparator["policy"])
            comparator_map[(route_id, budget)] = policy
            for analysis_split in (
                "development_oof",
                "retrospective_evaluation",
            ):
                baseline = result.loc[
                    result["record_type"].eq("policy_performance")
                    & result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(analysis_split)
                    & result["comparison_axis"].eq("same_budget")
                    & result["requested_budget"].eq(budget)
                    & result["policy"].eq(policy)
                ].iloc[0]
                mask = (
                    result["record_type"].eq("policy_performance")
                    & result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(analysis_split)
                    & result["comparison_axis"].eq("same_budget")
                    & result["requested_budget"].eq(budget)
                )
                result.loc[mask, "locked_comparator_policy"] = policy
                for metric in (
                    "corrected_selected",
                    "introduced_selected",
                    "net_selected",
                ):
                    result.loc[mask, f"comparator_{metric}"] = baseline[metric]
                    result.loc[mask, f"delta_{metric}"] = (
                        result.loc[mask, metric].astype(float)
                        - float(baseline[metric])
                    )
    return result, comparator_map


def _add_bootstrap_intervals(
    core: pd.DataFrame,
    *,
    contexts: Mapping[str, RouteContext],
    comparator_map: Mapping[tuple[str, float], str],
    selections: Mapping[tuple[str, str, str, float], np.ndarray],
    replicates: int,
) -> pd.DataFrame:
    result = core.copy()
    for route_id, context in contexts.items():
        for budget in OPERATING_BUDGETS:
            comparator = comparator_map[(route_id, budget)]
            interval = paired_cluster_bootstrap_difference(
                context.evaluation,
                method_selected=selections[
                    (
                        route_id,
                        "retrospective_evaluation",
                        METHOD_POLICY,
                        budget,
                    )
                ],
                baseline_selected=selections[
                    (
                        route_id,
                        "retrospective_evaluation",
                        comparator,
                        budget,
                    )
                ],
                replicates=replicates,
                seed=stable_seed(
                    f"{route_id}:{budget}:route-specific-bootstrap-v0.1"
                ),
            )
            mask = (
                result["record_type"].eq("policy_performance")
                & result["route_id"].eq(route_id)
                & result["analysis_split"].eq("retrospective_evaluation")
                & result["comparison_axis"].eq("same_budget")
                & result["policy"].eq(METHOD_POLICY)
                & result["requested_budget"].eq(budget)
            )
            for column, value in interval.items():
                result.loc[mask, column] = value
    return result


def route_success(
    method_rows: pd.DataFrame,
    *,
    minimum_dominant_budgets: int,
    required_budget: float,
) -> tuple[bool, list[float]]:
    """Apply the predeclared route-level retrospective success rule."""

    operating = method_rows.loc[
        method_rows["requested_budget"].isin(OPERATING_BUDGETS)
    ].copy()
    dominant = operating.loc[
        operating["delta_corrected_selected"].ge(0)
        & operating["delta_introduced_selected"].le(0)
        & operating["delta_net_selected"].gt(0)
    ]
    dominant_budgets = sorted(
        float(value) for value in dominant["requested_budget"]
    )
    budget_row = operating.loc[
        operating["requested_budget"].eq(required_budget)
    ]
    ci_lower = (
        float(budget_row.iloc[0]["net_difference_ci_lower"])
        if not budget_row.empty
        else float("nan")
    )
    success = (
        len(dominant_budgets) >= minimum_dominant_budgets
        and required_budget in dominant_budgets
        and np.isfinite(ci_lower)
        and ci_lower >= 0.0
    )
    return success, dominant_budgets


def _decide(
    core: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    qualified_route_ids: list[str],
) -> tuple[str, dict[str, Any]]:
    contract = protocol["decision_contract"]
    route_results: list[dict[str, Any]] = []
    for route_id in qualified_route_ids:
        rows = core.loc[
            core["record_type"].eq("policy_performance")
            & core["route_id"].eq(route_id)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["comparison_axis"].eq("same_budget")
            & core["policy"].eq(METHOD_POLICY)
        ]
        success, dominant = route_success(
            rows,
            minimum_dominant_budgets=int(
                contract["minimum_dominant_operating_budgets_per_route"]
            ),
            required_budget=float(contract["must_include_budget"]),
        )
        budget_30 = rows.loc[rows["requested_budget"].eq(0.30)].iloc[0]
        route_results.append(
            {
                "route_id": route_id,
                "success": success,
                "dominant_budgets": dominant,
                "budget_0_3_net_difference_ci_lower": float(
                    budget_30["net_difference_ci_lower"]
                ),
                "budget_0_3_net_difference_ci_upper": float(
                    budget_30["net_difference_ci_upper"]
                ),
            }
        )
    successful = [item for item in route_results if item["success"]]
    route_specific_go = (
        len(qualified_route_ids) >= int(contract["minimum_qualified_routes"])
        and len(successful) >= int(contract["minimum_successful_routes"])
    )
    return (
        "ROUTE_SPECIFIC_GO" if route_specific_go else "NO_IMPROVEMENT",
        {
            "qualified_route_count": len(qualified_route_ids),
            "successful_route_count": len(successful),
            "successful_route_ids": [
                str(item["route_id"]) for item in successful
            ],
            "route_results": route_results,
        },
    )


def _format(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _build_report(
    *,
    qualified: pd.DataFrame,
    core: pd.DataFrame,
    decision: str,
    evidence: Mapping[str, Any],
    source_commit: str,
    protocol_sha256: str,
    benchmark_audit: Mapping[str, Any],
    qualified_sha256: str,
    core_sha256: str,
) -> str:
    screening_rows = [
        [
            row.scout_id,
            row.expert_id,
            row.scout_accuracy,
            row.expert_accuracy,
            row.expert_accuracy_gain,
            f"{int(row.corrected_events)}/{int(row.introduced_events)}",
            row.corrected_to_introduced_ratio,
            f"{int(row.positive_gain_folds)}/5",
            f"{int(row.nonnegative_net_folds)}/5",
        ]
        for row in qualified.itertuples()
    ]
    result_rows: list[list[Any]] = []
    for route_id in sorted(qualified["route_id"]):
        rows = core.loc[
            core["route_id"].eq(route_id)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["comparison_axis"].eq("same_budget")
            & core["policy"].eq(METHOD_POLICY)
        ].sort_values("requested_budget")
        for row in rows.itertuples():
            result_rows.append(
                [
                    str(row.scout_id) + "→" + str(row.expert_id),
                    row.requested_budget,
                    row.locked_comparator_policy,
                    f"{int(row.corrected_selected)}/"
                    f"{int(row.comparator_corrected_selected)}",
                    f"{int(row.introduced_selected)}/"
                    f"{int(row.comparator_introduced_selected)}",
                    f"{int(row.net_selected)}/"
                    f"{int(row.comparator_net_selected)}",
                    row.delta_net_selected,
                    f"[{_format(row.net_difference_ci_lower)}, "
                    f"{_format(row.net_difference_ci_upper)}]",
                ]
            )
    return (
        "# OphAgent 高质量路线简单门控复核 v0.1\n\n"
        "## Material Passport\n\n"
        "- Origin Skill: experiment-agent\n"
        "- Origin Mode: run\n"
        "- Verification Status: UNVERIFIED（需确定性复跑核验）\n"
        "- Version Label: route_specific_simple_gate_v0_1\n\n"
        "## 结论\n\n"
        f"**{decision}**\n\n"
        f"开发集从 90 条 DeepDRiD 原生路线中筛出 "
        f"{len(qualified)} 条；冻结回顾性评价中满足预声明路线成功规则 "
        f"{evidence['successful_route_count']} 条。\n\n"
        "路线筛选没有读取冻结评估结果。该结论仍是回顾性错误代理研究，"
        "不能授予路线资格或替代 `SafetyEligibilityGate`。\n\n"
        "## 开发集路线筛选\n\n"
        + _markdown_table(
            [
                "Scout",
                "Expert",
                "Scout acc",
                "Expert acc",
                "增量",
                "corrected/introduced",
                "比值",
                "正增益折",
                "非负 net 折",
            ],
            screening_rows,
        )
        + "\n\n筛选阈值固定为：两个单模型 accuracy ≥0.60、Expert 增量 "
        "≥0.05、corrected/introduced ≥1.5、两类事件各 ≥15、"
        "net ≥10、固定患者折中至少 4/5 为正增益且 4/5 为非负 net，"
        "折间增量标准差 ≤0.12。\n\n"
        "## 冻结回顾性相同预算比较\n\n"
        + _markdown_table(
            [
                "路线",
                "预算",
                "开发锁定基线",
                "corrected 方法/基线",
                "introduced 方法/基线",
                "net 方法/基线",
                "Δnet",
                "患者配对 95% CI",
            ],
            result_rows,
        )
        + "\n\n每个预算的 entropy/margin 强基线只由 development OOF "
        "锁定。Scout-only 与 Always-Expert 作为两端参考保留在核心结果表。"
        "两条合格路线均没有唯一冻结 v1.1 路由身份，因此 v1.1 如实记为 "
        "not applicable；未移植其他路线的 v1.1 策略。\n\n"
        "## 决策规则与下一步\n\n"
        "单条路线必须在至少两个预声明预算（含 30%）同时做到 corrected "
        "不低、introduced 不高且 net 严格更高，并且 30% 预算患者配对 "
        "Δnet 95% CI 下界非负。至少两条开发合格路线满足，才授予 "
        "`ROUTE_SPECIFIC_GO`。\n\n"
        + (
            "当前仅支持特定高质量路线上的后续独立确认，不支持扩大到其他"
            "路线。\n\n"
            if decision == "ROUTE_SPECIFIC_GO"
            else "缩小范围后仍无稳定收益；停止继续调整简单门控，不上 RL "
            "或复杂模型。下一步应研究获得第二意见后的 "
            "KEEP_SCOUT / ADOPT_SECOND_OPINION / HUMAN_REVIEW 采纳机制。\n\n"
        )
        + "## 追溯\n\n"
        f"- 实现提交：`{source_commit}`\n"
        f"- 协议 SHA256：`{protocol_sha256}`\n"
        f"- Benchmark manifest SHA256："
        f"`{benchmark_audit['manifest_sha256']}`\n"
        f"- 病例表 SHA256：`{benchmark_audit['case_table_sha256']}`\n"
        f"- qualified_routes.csv SHA256：`{qualified_sha256}`\n"
        f"- core_comparison_results.csv SHA256：`{core_sha256}`\n"
        "- 眼底模型训练/推理：未执行；冻结 Benchmark、Test 和预测资产："
        "未修改。\n"
    )


def _ensure_safe_output(frame: pd.DataFrame) -> None:
    forbidden = {
        "patient_id",
        "patient_group_id",
        "resampling_group_id",
        "case_id",
        "image_id",
        "image_sha256",
        "image_path",
        "private_path",
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

    cases = pd.read_csv(
        benchmark_dir / "case_level_benchmark.csv.gz",
        low_memory=False,
    )
    inventory = pd.read_csv(
        benchmark_dir / "candidate_route_inventory.csv",
        low_memory=False,
    )
    inventory = inventory.loc[inventory["task_id"].eq(TASK_ID)].copy()
    if len(inventory) != 90:
        raise ValueError(f"Expected 90 native routes, observed {len(inventory)}.")
    development = cases.loc[
        cases["task_id"].eq(TASK_ID)
        & cases["benchmark_split"].eq("development")
        & cases["primary_cohort_eligible"].astype(bool)
    ].copy()
    criteria = RouteScreeningCriteria.from_mapping(protocol["route_screening"])

    # This call is deliberately completed before any evaluation rows are read.
    screening = screen_high_quality_routes(
        development,
        criteria=criteria,
    )
    qualified = screening.loc[screening["qualified"]].copy()
    qualified_route_ids = sorted(qualified["route_id"].astype(str))
    if len(qualified_route_ids) < int(
        protocol["decision_contract"]["minimum_qualified_routes"]
    ):
        raise ValueError(
            "Development screening produced too few routes for the "
            "predeclared route-specific comparison."
        )
    qualified = _annotate_screening_thresholds(qualified, criteria)

    contexts: dict[str, RouteContext] = {}
    for route_id in qualified_route_ids:
        route = inventory.loc[inventory["route_id"].eq(route_id)].iloc[0]
        route_development = development.loc[
            development["route_id"].eq(route_id)
        ].sort_values("case_id").reset_index(drop=True)
        evaluation = cases.loc[
            cases["route_id"].eq(route_id)
            & cases["benchmark_split"].eq(
                protocol["analysis_scope"]["evaluation_split"]
            )
            & cases["primary_cohort_eligible"].astype(bool)
        ].sort_values("case_id").reset_index(drop=True)
        if route_development.empty or evaluation.empty:
            raise ValueError(f"{route_id}: missing development or evaluation.")
        contexts[route_id] = _build_context(
            route=route,
            development=route_development,
            evaluation=evaluation,
        )

    core, selections = _generate_policy_rows(
        contexts,
        safe_pool_multiplier=float(
            protocol["method_contract"]["safe_pool_multiplier"]
        ),
    )
    core, comparator_map = _annotate_locked_comparators(core)
    core = _add_bootstrap_intervals(
        core,
        contexts=contexts,
        comparator_map=comparator_map,
        selections=selections,
        replicates=args.bootstrap_replicates,
    )
    decision, evidence = _decide(
        core,
        protocol=protocol,
        qualified_route_ids=qualified_route_ids,
    )
    core["study_decision"] = decision
    core["source_commit_sha"] = source_commit
    core["protocol_sha256"] = protocol_sha256
    core["generated_at_utc"] = generated_at
    qualified["study_decision"] = decision
    qualified["source_commit_sha"] = source_commit
    qualified["protocol_sha256"] = protocol_sha256
    qualified["generated_at_utc"] = generated_at
    core = core.sort_values(
        [
            "route_id",
            "analysis_split",
            "comparison_axis",
            "requested_budget",
            "policy",
        ],
        na_position="last",
    ).reset_index(drop=True)
    qualified = qualified.sort_values("route_id").reset_index(drop=True)
    _ensure_safe_output(qualified)
    _ensure_safe_output(core)

    write_csv(output_dir / "qualified_routes.csv", qualified)
    write_csv(output_dir / "core_comparison_results.csv", core)
    report = _build_report(
        qualified=qualified,
        core=core,
        decision=decision,
        evidence=evidence,
        source_commit=source_commit,
        protocol_sha256=protocol_sha256,
        benchmark_audit=benchmark_audit,
        qualified_sha256=file_sha256(output_dir / "qualified_routes.csv"),
        core_sha256=file_sha256(output_dir / "core_comparison_results.csv"),
    )
    (output_dir / "research_report.md").write_text(report, encoding="utf-8")
    observed = {path.name for path in output_dir.iterdir() if path.is_file()}
    if observed != set(FORMAL_OUTPUT_NAMES):
        raise ValueError(
            f"Formal output set changed: expected {FORMAL_OUTPUT_NAMES}, "
            f"observed {sorted(observed)}."
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "decision": decision,
                "qualified_routes": qualified_route_ids,
                "decision_evidence": evidence,
                "protocol_sha256": protocol_sha256,
                "source_commit_sha": source_commit,
                "outputs": {
                    name: file_sha256(output_dir / name)
                    for name in FORMAL_OUTPUT_NAMES
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
