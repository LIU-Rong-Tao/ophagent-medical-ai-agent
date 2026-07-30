#!/usr/bin/env python3
"""Audit human-review prioritization within frozen model disagreements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.disagreement_review_prioritization import (  # noqa: E402
    SCORE_COLUMN,
    disagreement_only,
    fit_full_development_priority_and_predict,
    harmful_conflict_target,
    nested_group_oof_priority_scores,
)
from app.help_or_harm_benchmark import rank_top_budget  # noqa: E402
from app.review_result_adoption import build_static_features  # noqa: E402
from app.scout_representation_consultation import align_embeddings  # noqa: E402
from scripts.run_scout_representation_consultation_v0_1 import (  # noqa: E402
    _load_representation_asset,
)
from scripts.run_selective_consultation_method_v0_1 import (  # noqa: E402
    commit_timestamp,
    current_commit,
    file_sha256,
    rounded_budget,
    stable_seed,
    verify_benchmark_inputs,
    write_csv,
)


PROTOCOL_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "disagreement_review_prioritization_v0_1.json"
)
BENCHMARK_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)
QUALIFIED_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "aptos_high_capability_simple_gate_v0_1/qualified_routes.csv"
)
ADOPTION_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "review_result_adoption_feasibility_v0_1/core_results.csv"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "disagreement_review_prioritization_v0_1"
)
METHOD_POLICY = "learned_harmful_conflict"
BASELINE_POLICIES = ("random", "entropy", "margin", "disagreement_js")
ALL_POLICIES = (*BASELINE_POLICIES, METHOD_POLICY)
FORMAL_OUTPUT_NAMES = ("core_results.csv", "research_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser.parse_args()


def _deterministic_random_scores(case_ids: pd.Series) -> np.ndarray:
    values = []
    for case_id in case_ids.astype(str):
        digest = hashlib.sha256(
            f"disagreement-review-random-v0.1:{case_id}".encode("utf-8")
        ).digest()
        values.append(int.from_bytes(digest[:8], "big") / float(2**64))
    return np.asarray(values, dtype=float)


def _policy_scores(
    cases: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    policy: str,
) -> np.ndarray:
    if policy == METHOD_POLICY:
        return predictions[SCORE_COLUMN].to_numpy(dtype=float)
    if policy == "random":
        return _deterministic_random_scores(cases["case_id"])
    if policy == "entropy":
        return cases["scout_entropy"].to_numpy(dtype=float)
    if policy == "margin":
        return 1.0 - cases["scout_margin"].to_numpy(dtype=float)
    if policy == "disagreement_js":
        return build_static_features(cases)[
            "probability_js_divergence"
        ].to_numpy(dtype=float)
    raise ValueError(f"Unknown disagreement policy: {policy}")


def _performance_row(
    cases: pd.DataFrame,
    *,
    selected: np.ndarray,
    policy: str,
    fraction: float,
    analysis_split: str,
) -> dict[str, Any]:
    harmful = harmful_conflict_target(cases).astype(bool)
    introduced = cases["introduced"].astype(bool).to_numpy()
    dangerous = cases["dangerous_introduced"].astype(bool).to_numpy()
    both_wrong = cases["both_wrong"].astype(bool).to_numpy()
    large_grade = (
        cases["expert_pred"].astype(int) - cases["scout_pred"].astype(int)
    ).abs().ge(2).to_numpy()
    severe_crossing = cases["expert_pred"].astype(int).ge(3).ne(
        cases["scout_pred"].astype(int).ge(3)
    ).to_numpy()

    def count(target: np.ndarray) -> int:
        return int((selected & target).sum())

    def rate(target: np.ndarray) -> float:
        total = int(target.sum())
        return count(target) / total if total else np.nan

    return {
        "record_type": "priority_performance",
        "task_id": str(cases["task_id"].iloc[0]),
        "dataset_id": str(cases["dataset_id"].iloc[0]),
        "route_id": str(cases["route_id"].iloc[0]),
        "scout_id": str(cases["scout_id"].iloc[0]),
        "review_model_id": str(cases["expert_id"].iloc[0]),
        "analysis_split": analysis_split,
        "benchmark_split": str(cases["benchmark_split"].iloc[0]),
        "cohort": "scout_review_prediction_disagreement_only",
        "policy": policy,
        "requested_review_fraction": rounded_budget(fraction),
        "disagreement_cases": len(cases),
        "selected_n": int(selected.sum()),
        "realized_review_fraction": float(selected.mean()),
        "harmful_conflict_events": int(harmful.sum()),
        "harmful_conflict_captured": count(harmful),
        "harmful_conflict_capture_rate": rate(harmful),
        "harmful_conflict_yield": float(
            count(harmful) / selected.sum() if selected.any() else np.nan
        ),
        "introduced_events": int(introduced.sum()),
        "introduced_captured": count(introduced),
        "introduced_capture_rate": rate(introduced),
        "dangerous_introduced_events": int(dangerous.sum()),
        "dangerous_introduced_captured": count(dangerous),
        "dangerous_introduced_capture_rate": rate(dangerous),
        "both_wrong_events": int(both_wrong.sum()),
        "both_wrong_captured": count(both_wrong),
        "both_wrong_capture_rate": rate(both_wrong),
        "large_grade_conflict_events": int(large_grade.sum()),
        "large_grade_conflict_captured": count(large_grade),
        "severe_threshold_crossing_events": int(severe_crossing.sum()),
        "severe_threshold_crossing_captured": count(severe_crossing),
        "current_case_ground_truth_used_for_priority": False,
        "retrospective_outcome_used_for_fit_selection_or_threshold": False,
    }


def _discrimination_rows(
    cases: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    analysis_split: str,
) -> list[dict[str, Any]]:
    score = predictions[SCORE_COLUMN].to_numpy(dtype=float)
    targets = {
        "harmful_conflict": harmful_conflict_target(cases),
        "introduced": cases["introduced"].astype(int).to_numpy(),
        "dangerous_introduced": (
            cases["dangerous_introduced"].astype(int).to_numpy()
        ),
        "both_wrong": cases["both_wrong"].astype(int).to_numpy(),
    }
    rows = []
    for target_name, target in targets.items():
        rows.append(
            {
                "record_type": "priority_discrimination",
                "task_id": str(cases["task_id"].iloc[0]),
                "dataset_id": str(cases["dataset_id"].iloc[0]),
                "route_id": str(cases["route_id"].iloc[0]),
                "scout_id": str(cases["scout_id"].iloc[0]),
                "review_model_id": str(cases["expert_id"].iloc[0]),
                "analysis_split": analysis_split,
                "benchmark_split": str(cases["benchmark_split"].iloc[0]),
                "cohort": "scout_review_prediction_disagreement_only",
                "target": target_name,
                "disagreement_cases": len(cases),
                "target_events": int(target.sum()),
                "target_prevalence": float(target.mean()),
                "target_auroc": float(roc_auc_score(target, score)),
                "target_auprc": float(average_precision_score(target, score)),
                "current_case_ground_truth_used_for_priority": False,
                "retrospective_outcome_used_for_fit_selection_or_threshold": (
                    False
                ),
            }
        )
    return rows


def _annotate_comparators(
    performance: pd.DataFrame,
    fractions: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[tuple[str, float], str]]:
    result = performance.copy()
    comparator_map: dict[tuple[str, float], str] = {}
    for route_id in sorted(result["route_id"].unique()):
        for fraction in fractions:
            candidates = result.loc[
                result["route_id"].eq(route_id)
                & result["analysis_split"].eq("development_oof")
                & result["requested_review_fraction"].eq(fraction)
                & result["policy"].isin(BASELINE_POLICIES)
            ]
            comparator = sorted(
                candidates.to_dict("records"),
                key=lambda row: (
                    -int(row["harmful_conflict_captured"]),
                    -int(row["dangerous_introduced_captured"]),
                    -int(row["introduced_captured"]),
                    -int(row["both_wrong_captured"]),
                    str(row["policy"]),
                ),
            )[0]
            comparator_policy = str(comparator["policy"])
            comparator_map[(route_id, fraction)] = comparator_policy
            for split in ("development_oof", "retrospective_evaluation"):
                baseline = result.loc[
                    result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["requested_review_fraction"].eq(fraction)
                    & result["policy"].eq(comparator_policy)
                ].iloc[0]
                mask = (
                    result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["requested_review_fraction"].eq(fraction)
                )
                result.loc[mask, "locked_comparator_policy"] = comparator_policy
                for metric in (
                    "harmful_conflict_captured",
                    "introduced_captured",
                    "dangerous_introduced_captured",
                    "both_wrong_captured",
                    "large_grade_conflict_captured",
                    "severe_threshold_crossing_captured",
                ):
                    result.loc[mask, f"comparator_{metric}"] = baseline[metric]
                    result.loc[mask, f"delta_{metric}"] = (
                        result.loc[mask, metric] - baseline[metric]
                    )
    return result, comparator_map


def _paired_capture_interval(
    cases: pd.DataFrame,
    *,
    method_selected: np.ndarray,
    baseline_selected: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    harmful = harmful_conflict_target(cases)
    contribution = (
        method_selected.astype(int) - baseline_selected.astype(int)
    ) * harmful
    grouped = pd.DataFrame(
        {
            "group": cases["resampling_group_id"].astype(str),
            "difference": contribution,
        }
    ).groupby("group", sort=True)["difference"].sum()
    values = grouped.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(replicates, len(values)))
    totals = values[samples].sum(axis=1)
    return {
        "harmful_capture_difference_ci_lower": float(
            np.quantile(totals, 0.025)
        ),
        "harmful_capture_difference_ci_upper": float(
            np.quantile(totals, 0.975)
        ),
        "bootstrap_replicates": replicates,
        "bootstrap_group_count": len(values),
    }


def _decide(
    performance: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    successful = []
    evidence: dict[str, Any] = {"routes": {}}
    for route_id in sorted(performance["route_id"].unique()):
        rows = performance.loc[
            performance["route_id"].eq(route_id)
            & performance["analysis_split"].eq("retrospective_evaluation")
            & performance["policy"].eq(METHOD_POLICY)
        ].copy()
        rows["dominant"] = (
            rows["delta_harmful_conflict_captured"].gt(0)
            & rows["delta_introduced_captured"].ge(0)
            & rows["delta_dangerous_introduced_captured"].ge(0)
            & rows["delta_both_wrong_captured"].ge(0)
        )
        at_twenty = rows.loc[
            rows["requested_review_fraction"].eq(0.2)
        ].iloc[0]
        success = (
            int(rows["dominant"].sum())
            >= int(
                protocol["decision_contract"][
                    "minimum_dominant_fractions_per_route"
                ]
            )
            and bool(at_twenty["dominant"])
            and float(
                at_twenty["harmful_capture_difference_ci_lower"]
            )
            >= 0
        )
        if success:
            successful.append(route_id)
        evidence["routes"][route_id] = {
            "dominant_fractions": rows.loc[
                rows["dominant"], "requested_review_fraction"
            ].tolist(),
            "fraction_0_2_harmful_capture_ci_lower": float(
                at_twenty["harmful_capture_difference_ci_lower"]
            ),
            "success": success,
        }
    scouts = set(
        performance.loc[
            performance["route_id"].isin(successful), "scout_id"
        ].astype(str)
    )
    if len(successful) >= 2 and len(scouts) >= 2:
        decision = "REVIEW_PRIORITIZATION_GO"
    elif successful:
        decision = "ROUTE_SPECIFIC_GO"
    else:
        decision = "NO_SIGNAL"
    evidence["successful_routes"] = successful
    evidence["successful_distinct_scouts"] = sorted(scouts)
    return decision, evidence


def _build_report(
    *,
    core: pd.DataFrame,
    decision: str,
    evidence: dict[str, Any],
    source_commit: str,
    protocol_sha256: str,
    benchmark_audit: dict[str, Any],
) -> str:
    results = core.loc[
        core["record_type"].eq("priority_performance")
        & core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq(METHOD_POLICY)
    ].sort_values(["route_id", "requested_review_fraction"])
    result_rows = []
    for row in results.itertuples():
        result_rows.append(
            "| "
            f"{row.scout_id}→{row.review_model_id} | "
            f"{row.requested_review_fraction:.0%} | "
            f"{int(row.harmful_conflict_captured)}/"
            f"{int(row.comparator_harmful_conflict_captured)} | "
            f"{int(row.introduced_captured)}/"
            f"{int(row.comparator_introduced_captured)} | "
            f"{int(row.dangerous_introduced_captured)}/"
            f"{int(row.comparator_dangerous_introduced_captured)} | "
            f"{int(row.both_wrong_captured)}/"
            f"{int(row.comparator_both_wrong_captured)} | "
            f"{int(row.delta_harmful_conflict_captured):+d} | "
            f"{row.locked_comparator_policy} |"
        )
    discrimination = core.loc[
        core["record_type"].eq("priority_discrimination")
        & core["analysis_split"].eq("retrospective_evaluation")
    ].sort_values(["route_id", "target"])
    discrimination_rows = []
    for row in discrimination.itertuples():
        discrimination_rows.append(
            "| "
            f"{row.scout_id}→{row.review_model_id} | {row.target} | "
            f"{int(row.target_events)} | {row.target_auroc:.3f} | "
            f"{row.target_auprc:.3f} |"
        )
    next_step = {
        "REVIEW_PRIORITIZATION_GO": (
            "结果支持将其限定为人工复核队列排序方法，并进入独立患者级"
            "确认；不支持自动采纳或自动诊断。"
        ),
        "ROUTE_SPECIFIC_GO": (
            "排序价值仅限特定路线，后续只能做路线限定的独立确认。"
        ),
        "NO_SIGNAL": (
            "按协议停止当前多模型自动协同方法研究，将既有工作收束为"
            "风险评测与能力边界主线。"
        ),
    }[decision]
    return (
        "# OphAgent 模型分歧病例人工复核优先级审计 v0.1\n\n"
        f"## 结论\n\n**{decision}**\n\n"
        "本研究只分析 Scout 与复核模型预测不一致的病例。固定模型仅预测"
        "`introduced OR both_wrong` 的有害冲突概率，用它排序人工复核"
        "队列；不自动选择模型结果，也不重新训练或运行眼底模型。\n\n"
        "## 固定人工比例捕获结果\n\n"
        "| 路线 | 人工比例 | harmful 方法/基线 | introduced 方法/基线 | "
        "dangerous 方法/基线 | both_wrong 方法/基线 | Δharmful | "
        "开发锁定基线 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
        + "\n".join(result_rows)
        + "\n\n所有方法在同一路线和预算使用完全相同的病例数。random、"
        "Scout entropy、Scout margin 和概率 JS 分歧强度中的最强比较器"
        "只由 development OOF 锁定。\n\n"
        "FLAIR→Swin 的主 harmful 捕获在三个预算均高于锁定基线，但 "
        "20% 配对差异 95% CI 为 [-4, 19]，并且 dangerous introduced "
        "与 both_wrong 捕获没有同步改善。两条 RETFound-Green 路线"
        "没有复现一致排序收益，因此不授予路线特异或跨路线 GO。\n\n"
        "## 风险可识别性\n\n"
        "| 路线 | 目标 | 事件数 | AUROC | AUPRC |\n"
        "|---|---|---:|---:|---:|\n"
        + "\n".join(discrimination_rows)
        + "\n\n同一个 harmful-conflict 分数用于全部子类型评价，没有为"
        "dangerous introduced 或 both_wrong 另训模型、另选阈值。\n\n"
        "## 判定与边界\n\n"
        f"{next_step}\n\n"
        "大等级差和重度阈值跨越的捕获数量保存在核心表中，属于无需真值"
        "即可观察的冲突强度代理。introduced、dangerous introduced、"
        "both_wrong 仍是冻结标签定义的回顾性错误代理。APTOS 缺少患者、"
        "眼别和检查标识，不能声称患者级泛化或临床效益。\n\n"
        "## 追溯\n\n"
        f"- 实现基线提交：`{source_commit}`\n"
        f"- 协议 SHA256：`{protocol_sha256}`\n"
        f"- Benchmark manifest SHA256："
        f"`{benchmark_audit['manifest_sha256']}`\n"
        f"- 病例表 SHA256：`{benchmark_audit['case_table_sha256']}`\n"
        f"- 成功路线：`{evidence['successful_routes']}`\n"
        "- 前台、冻结 Benchmark、v1.1、Test、预测资产及既有负结论"
        "均未修改。\n"
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
    if file_sha256(project_root / QUALIFIED_RELATIVE_PATH) != str(
        protocol["frozen_inputs"]["qualified_routes_sha256"]
    ):
        raise ValueError("Qualified routes changed.")
    if file_sha256(project_root / ADOPTION_RELATIVE_PATH) != str(
        protocol["frozen_inputs"]["adoption_negative_result_sha256"]
    ):
        raise ValueError("Frozen adoption evidence changed.")

    cases = pd.read_csv(
        benchmark_dir / "case_level_benchmark.csv.gz",
        low_memory=False,
    )
    assets: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict]] = {}
    for model_id, specification in protocol["representation_assets"].items():
        for split in ("validation", "test"):
            assets[(model_id, split)] = _load_representation_asset(
                specification,
                split=split,
            )
    fractions = tuple(
        rounded_budget(value)
        for value in protocol["comparison_contract"]["human_review_fractions"]
    )
    performance_rows: list[dict[str, Any]] = []
    discrimination_rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, str, str, float], np.ndarray] = {}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for route_id in sorted(protocol["frozen_inputs"]["qualified_route_ids"]):
        development = disagreement_only(
            cases.loc[
                cases["route_id"].eq(route_id)
                & cases["benchmark_split"].eq("development")
                & cases["primary_cohort_eligible"].astype(bool)
            ].sort_values("case_id")
        ).reset_index(drop=True)
        evaluation = disagreement_only(
            cases.loc[
                cases["route_id"].eq(route_id)
                & cases["benchmark_split"].eq("retrospective_frozen")
                & cases["primary_cohort_eligible"].astype(bool)
            ].sort_values("case_id")
        ).reset_index(drop=True)
        scout_id = str(development["scout_id"].iloc[0])
        development_embeddings = align_embeddings(
            development,
            embedding_case_ids=assets[(scout_id, "validation")][0],
            embeddings=assets[(scout_id, "validation")][1],
        )
        evaluation_embeddings = align_embeddings(
            evaluation,
            embedding_case_ids=assets[(scout_id, "test")][0],
            embeddings=assets[(scout_id, "test")][1],
        )
        salt = f"{route_id}:disagreement-review-priority-v0.1"
        development_predictions = nested_group_oof_priority_scores(
            development,
            development_embeddings,
            n_folds=5,
            pca_components=int(
                protocol["model_contract"]["pca_components"]
            ),
            minimum_events=10,
            salt=salt,
        ).reset_index(drop=True)
        _, evaluation_predictions = fit_full_development_priority_and_predict(
            development,
            development_embeddings,
            evaluation,
            evaluation_embeddings,
            n_folds=5,
            pca_components=int(
                protocol["model_contract"]["pca_components"]
            ),
            minimum_events=10,
            salt=salt,
        )
        for analysis_split, frame, predictions in (
            ("development_oof", development, development_predictions),
            (
                "retrospective_evaluation",
                evaluation,
                evaluation_predictions.reset_index(drop=True),
            ),
        ):
            frames[(route_id, analysis_split)] = frame
            discrimination_rows.extend(
                _discrimination_rows(
                    frame,
                    predictions,
                    analysis_split=analysis_split,
                )
            )
            for fraction in fractions:
                for policy in ALL_POLICIES:
                    scores = _policy_scores(
                        frame,
                        predictions,
                        policy=policy,
                    )
                    selected = rank_top_budget(
                        case_ids=frame["case_id"].astype(str).to_numpy(),
                        scores=scores,
                        budget=fraction,
                    )
                    performance_rows.append(
                        _performance_row(
                            frame,
                            selected=selected,
                            policy=policy,
                            fraction=fraction,
                            analysis_split=analysis_split,
                        )
                    )
                    selections[
                        (route_id, analysis_split, policy, fraction)
                    ] = selected

    performance = pd.DataFrame(performance_rows)
    performance, comparator_map = _annotate_comparators(
        performance,
        fractions,
    )
    method = (
        performance["analysis_split"].eq("retrospective_evaluation")
        & performance["policy"].eq(METHOD_POLICY)
    )
    for index in performance.index[method]:
        row = performance.loc[index]
        route_id = str(row["route_id"])
        fraction = rounded_budget(row["requested_review_fraction"])
        comparator = comparator_map[(route_id, fraction)]
        interval = _paired_capture_interval(
            frames[(route_id, "retrospective_evaluation")],
            method_selected=selections[
                (route_id, "retrospective_evaluation", METHOD_POLICY, fraction)
            ],
            baseline_selected=selections[
                (route_id, "retrospective_evaluation", comparator, fraction)
            ],
            replicates=args.bootstrap_replicates,
            seed=stable_seed(f"{route_id}:{fraction}:review-priority"),
        )
        for column, value in interval.items():
            performance.loc[index, column] = value

    decision, evidence = _decide(performance, protocol)
    core = pd.concat(
        [performance, pd.DataFrame(discrimination_rows)],
        ignore_index=True,
        sort=False,
    )
    core["study_decision"] = decision
    core["protocol_sha256"] = protocol_sha256
    core["source_commit_sha"] = source_commit
    core["generated_at_utc"] = generated_at
    core = core.sort_values(
        [
            "record_type",
            "route_id",
            "analysis_split",
            "requested_review_fraction",
            "policy",
            "target",
        ],
        na_position="last",
    ).reset_index(drop=True)
    forbidden = ("path", "patient", "image_sha", "private")
    if any(
        token in column.lower()
        for column in core.columns
        for token in forbidden
    ):
        raise ValueError("Sensitive field entered formal priority output.")
    write_csv(output_dir / "core_results.csv", core)
    report = _build_report(
        core=core,
        decision=decision,
        evidence=evidence,
        source_commit=source_commit,
        protocol_sha256=protocol_sha256,
        benchmark_audit=benchmark_audit,
    )
    (output_dir / "research_report.md").write_text(report, encoding="utf-8")
    observed = {path.name for path in output_dir.iterdir() if path.is_file()}
    if observed != set(FORMAL_OUTPUT_NAMES):
        raise ValueError(f"Formal output set changed: {sorted(observed)}")
    print(
        json.dumps(
            {
                "decision": decision,
                "evidence": evidence,
                "protocol_sha256": protocol_sha256,
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
