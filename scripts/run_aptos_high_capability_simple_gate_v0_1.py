#!/usr/bin/env python3
"""Test the simple gate on development-qualified high-capability APTOS routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

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
)
from scripts.run_route_specific_simple_gate_v0_1 import (  # noqa: E402
    METHOD_POLICY,
    _add_bootstrap_intervals,
    _annotate_locked_comparators,
    _annotate_screening_thresholds,
    _decide,
    _ensure_safe_output,
    _format,
    _generate_policy_rows,
    _markdown_table,
)
from scripts.run_selective_consultation_method_v0_1 import (  # noqa: E402
    RouteContext,
    commit_timestamp,
    current_commit,
    file_sha256,
    policy_baseline_from_route,
    verify_benchmark_inputs,
    write_csv,
)


PROTOCOL_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "aptos_high_capability_simple_gate_v0_1.json"
)
BENCHMARK_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "aptos_high_capability_simple_gate_v0_1"
)
TASK_ID = "aptos_dr_5class"
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
        cohort="aptos_exact_duplicate_excluded_high_capability_image",
        formal_v1_1=policy_baseline_from_route(route),
    )


def _screening_table(qualified: pd.DataFrame) -> str:
    rows = [
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
    return _markdown_table(
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
        rows,
    )


def _result_table(qualified: pd.DataFrame, core: pd.DataFrame) -> str:
    rows: list[list[Any]] = []
    for route_id in sorted(qualified["route_id"]):
        route_rows = core.loc[
            core["route_id"].eq(route_id)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["comparison_axis"].eq("same_budget")
            & core["policy"].eq(METHOD_POLICY)
        ].sort_values("requested_budget")
        for row in route_rows.itertuples():
            rows.append(
                [
                    f"{row.scout_id}→{row.expert_id}",
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
    return _markdown_table(
        [
            "路线",
            "预算",
            "开发锁定基线",
            "corrected 方法/基线",
            "introduced 方法/基线",
            "net 方法/基线",
            "Δnet",
            "图像组配对 95% CI",
        ],
        rows,
    )


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
    successful = int(evidence["successful_route_count"])
    return (
        "# OphAgent APTOS 高能力路线简单门控复核 v0.1\n\n"
        "## Material Passport\n\n"
        "- Origin Skill: experiment-agent\n"
        "- Origin Mode: run\n"
        "- Verification Status: UNVERIFIED（需确定性复跑核验）\n"
        "- Version Label: aptos_high_capability_simple_gate_v0_1\n\n"
        "## 结论\n\n"
        f"**{decision}**\n\n"
        f"仅用 APTOS development 从 90 条路线中筛出 "
        f"{len(qualified)} 条高能力同任务路线；冻结回顾性评价中满足"
        f"预声明路线成功规则 {successful} 条。\n\n"
        "本实验把问题限定为：Scout 与 Expert 已有较高同任务能力和开发集"
        "互补性时，简单预咨询门控能否锦上添花。它不是 APTOS→DeepDRiD "
        "跨数据集迁移实验。\n\n"
        "## 开发集筛选\n\n"
        + _screening_table(qualified)
        + "\n\n固定阈值：Scout accuracy ≥0.80、Expert accuracy ≥0.85、"
        "Expert 增量 ≥0.05、corrected/introduced ≥1.7、corrected "
        "≥30、introduced ≥20、net ≥20；固定图像组折中至少 4/5 "
        "为正增益且 4/5 为非负 net，折间增量标准差 ≤0.12。冻结评估"
        "没有参与路线筛选。\n\n"
        "## 冻结回顾性相同预算比较\n\n"
        + _result_table(qualified, core)
        + "\n\n每个预算的 entropy/margin 强基线只由 development OOF "
        "锁定。Scout-only 与 Always-Expert 保留为端点参考。若合格路线"
        "没有唯一冻结 v1.1 身份，v1.1 只记录为 not applicable，不移植"
        "其他路线策略。\n\n"
        "## 证据边界\n\n"
        "- development 每路线 485 个去重后图像分析单位；冻结回顾性"
        "评价每路线 1036 个；\n"
        "- APTOS 缺少患者、眼别和检查标识，不能把图像组 bootstrap "
        "解释为患者级泛化；\n"
        "- APTOS Test 指标在严格冻结协议建立前已存在，本轮只作回顾性"
        "评价，仍需独立未暴露患者级确认；\n"
        "- corrected/introduced 是标签定义的错误代理，不是临床获益"
        "或伤害；`SafetyEligibilityGate` 仍不可绕过。\n\n"
        "## 决策与下一步\n\n"
        "单条路线必须在至少两个预算（含 30%）同时做到 corrected 不低、"
        "introduced 不高且 net 严格更高，并且 30% 的配对 Δnet 95% CI "
        "下界非负；至少两条路线满足才授予 `ROUTE_SPECIFIC_GO`。\n\n"
        + (
            "结果支持仅在这些高能力 APTOS 路线上开展独立患者级确认，"
            "不外推到其他任务或路线。\n\n"
            if decision == "ROUTE_SPECIFIC_GO"
            else "若仍为负，则停止调整简单预咨询门控；下一步转向第二意见"
            "到达后的 KEEP_SCOUT / ADOPT_SECOND_OPINION / "
            "HUMAN_REVIEW 采纳机制。\n\n"
        )
        + "## 追溯\n\n"
        f"- 实现提交：`{source_commit}`\n"
        f"- 协议 SHA256：`{protocol_sha256}`\n"
        f"- Benchmark manifest SHA256："
        f"`{benchmark_audit['manifest_sha256']}`\n"
        f"- 病例表 SHA256：`{benchmark_audit['case_table_sha256']}`\n"
        f"- qualified_routes.csv SHA256：`{qualified_sha256}`\n"
        f"- core_comparison_results.csv SHA256：`{core_sha256}`\n"
        "- 眼底模型训练/推理：未执行；冻结 Benchmark、Test、预测资产"
        "及既有 DeepDRiD 结论：未修改。\n"
    )


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
        raise ValueError(f"Expected 90 APTOS routes, observed {len(inventory)}.")
    development = cases.loc[
        cases["task_id"].eq(TASK_ID)
        & cases["benchmark_split"].eq("development")
        & cases["primary_cohort_eligible"].astype(bool)
    ].copy()
    criteria = RouteScreeningCriteria.from_mapping(protocol["route_screening"])

    # Route identities are fixed before any frozen evaluation rows are read.
    screening = screen_high_quality_routes(
        development,
        criteria=criteria,
    )
    qualified = screening.loc[screening["qualified"]].copy()
    qualified_route_ids = sorted(qualified["route_id"].astype(str))
    if len(qualified_route_ids) < int(
        protocol["decision_contract"]["minimum_qualified_routes"]
    ):
        raise ValueError("Too few development-qualified APTOS routes.")
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
        if len(route_development) != 485 or len(evaluation) != 1036:
            raise ValueError(
                f"{route_id}: unexpected duplicate-excluded cohort size."
            )
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
            f"APTOS formal output set changed: observed {sorted(observed)}."
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
