#!/usr/bin/env python3
"""Evaluate safe adoption after a frozen review-model result is available."""

from __future__ import annotations

import argparse
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

from app.review_result_adoption import (  # noqa: E402
    baseline_actions,
    fit_full_development_adoption_and_predict,
    learned_adoption_actions,
    nested_group_oof_adoption_predictions,
)
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
    "review_result_adoption_feasibility_v0_1.json"
)
BENCHMARK_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)
QUALIFIED_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "aptos_high_capability_simple_gate_v0_1/qualified_routes.csv"
)
PRECONSULTATION_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "scout_representation_consultation_v0_1/core_results.csv"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "review_result_adoption_feasibility_v0_1"
)
TASK_ID = "aptos_dr_5class"
METHOD_POLICY = "learned_safe_adoption"
BASELINE_POLICIES = (
    "keep_scout",
    "always_adopt_review",
    "higher_confidence",
    "soft_vote",
)
DECISION_COMPARATORS = ("higher_confidence", "soft_vote")
FORMAL_OUTPUT_NAMES = ("core_results.csv", "research_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser.parse_args()


def _final_prediction_from_actions(
    cases: pd.DataFrame,
    actions: np.ndarray,
) -> np.ndarray:
    return np.where(
        actions == "ADOPT_REVIEW_RESULT",
        cases["expert_pred"].to_numpy(dtype=int),
        cases["scout_pred"].to_numpy(dtype=int),
    )


def _performance_row(
    cases: pd.DataFrame,
    *,
    actions: np.ndarray,
    final_prediction: np.ndarray,
    policy: str,
    human_review_fraction: float,
    analysis_split: str,
) -> dict[str, Any]:
    review = actions == "HUMAN_REVIEW"
    automatic = ~review
    truth = cases["y_true"].to_numpy(dtype=int)
    scout = cases["scout_pred"].to_numpy(dtype=int)
    final = np.asarray(final_prediction, dtype=int)
    scout_correct = scout == truth
    final_correct = final == truth
    corrected_retained = automatic & (~scout_correct) & final_correct
    introduced = automatic & scout_correct & (~final_correct)
    scout_dangerous = (truth >= 3) & (scout < 3)
    final_dangerous = (truth >= 3) & (final < 3)
    dangerous = automatic & (~scout_dangerous) & final_dangerous
    both_wrong_auto = automatic & cases["both_wrong"].astype(bool).to_numpy()
    review_corrected = review & cases["corrected"].astype(bool).to_numpy()
    review_introduced = review & cases["introduced"].astype(bool).to_numpy()
    review_both_wrong = review & cases["both_wrong"].astype(bool).to_numpy()
    review_dangerous = (
        review & cases["dangerous_introduced"].astype(bool).to_numpy()
    )
    return {
        "record_type": "policy_performance",
        "task_id": str(cases["task_id"].iloc[0]),
        "dataset_id": str(cases["dataset_id"].iloc[0]),
        "route_id": str(cases["route_id"].iloc[0]),
        "scout_id": str(cases["scout_id"].iloc[0]),
        "review_model_id": str(cases["expert_id"].iloc[0]),
        "analysis_split": analysis_split,
        "benchmark_split": str(cases["benchmark_split"].iloc[0]),
        "policy": policy,
        "requested_human_review_fraction": rounded_budget(
            human_review_fraction
        ),
        "n_cases": len(cases),
        "human_review_n": int(review.sum()),
        "human_review_fraction": float(review.mean()),
        "automatic_n": int(automatic.sum()),
        "automatic_coverage": float(automatic.mean()),
        "corrected_retained": int(corrected_retained.sum()),
        "introduced_auto": int(introduced.sum()),
        "dangerous_introduced_auto": int(dangerous.sum()),
        "both_wrong_auto_closed": int(both_wrong_auto.sum()),
        "net_retained": int(
            corrected_retained.sum() - introduced.sum()
        ),
        "automatic_correct_n": int((automatic & final_correct).sum()),
        "automatic_error_n": int((automatic & (~final_correct)).sum()),
        "automatic_accuracy": float(
            final_correct[automatic].mean() if automatic.any() else np.nan
        ),
        "review_captured_corrected": int(review_corrected.sum()),
        "review_captured_introduced": int(review_introduced.sum()),
        "review_captured_dangerous_introduced": int(
            review_dangerous.sum()
        ),
        "review_captured_both_wrong": int(review_both_wrong.sum()),
        "human_review_resolution_assumed_correct": False,
        "current_case_ground_truth_used_for_action": False,
        "retrospective_outcome_used_for_fit_selection_or_threshold": False,
    }


def _annotate_comparators(
    core: pd.DataFrame,
    fractions: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[tuple[str, float], str]]:
    result = core.copy()
    comparator_map: dict[tuple[str, float], str] = {}
    for route_id in sorted(result["route_id"].unique()):
        for fraction in fractions:
            candidates = result.loc[
                result["route_id"].eq(route_id)
                & result["analysis_split"].eq("development_oof")
                & result["requested_human_review_fraction"].eq(fraction)
                & result["policy"].isin(DECISION_COMPARATORS)
            ].copy()
            comparator = sorted(
                candidates.to_dict("records"),
                key=lambda row: (
                    -int(row["net_retained"]),
                    -int(row["corrected_retained"]),
                    int(row["introduced_auto"]),
                    int(row["dangerous_introduced_auto"]),
                    str(row["policy"]),
                ),
            )[0]
            comparator_policy = str(comparator["policy"])
            comparator_map[(route_id, fraction)] = comparator_policy
            for split in ("development_oof", "retrospective_evaluation"):
                baseline = result.loc[
                    result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["requested_human_review_fraction"].eq(fraction)
                    & result["policy"].eq(comparator_policy)
                ].iloc[0]
                mask = (
                    result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["requested_human_review_fraction"].eq(fraction)
                )
                result.loc[mask, "locked_comparator_policy"] = comparator_policy
                for metric in (
                    "corrected_retained",
                    "introduced_auto",
                    "dangerous_introduced_auto",
                    "both_wrong_auto_closed",
                    "net_retained",
                    "automatic_error_n",
                ):
                    result.loc[mask, f"comparator_{metric}"] = baseline[metric]
                    result.loc[mask, f"delta_{metric}"] = (
                        result.loc[mask, metric] - baseline[metric]
                    )
    return result, comparator_map


def _discrimination_rows(
    cases: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    analysis_split: str,
) -> list[dict[str, Any]]:
    rows = []
    for outcome in ("corrected", "introduced", "both_wrong"):
        target = cases[outcome].astype(int).to_numpy()
        score = predictions[f"probability_{outcome}"].to_numpy(dtype=float)
        rows.append(
            {
                "record_type": "outcome_discrimination",
                "task_id": str(cases["task_id"].iloc[0]),
                "dataset_id": str(cases["dataset_id"].iloc[0]),
                "route_id": str(cases["route_id"].iloc[0]),
                "scout_id": str(cases["scout_id"].iloc[0]),
                "review_model_id": str(cases["expert_id"].iloc[0]),
                "analysis_split": analysis_split,
                "benchmark_split": str(cases["benchmark_split"].iloc[0]),
                "outcome": outcome,
                "n_cases": len(cases),
                "outcome_events": int(target.sum()),
                "outcome_prevalence": float(target.mean()),
                "outcome_auroc": float(roc_auc_score(target, score)),
                "outcome_auprc": float(average_precision_score(target, score)),
                "retrospective_outcome_used_for_fit_selection_or_threshold": (
                    False
                ),
            }
        )
    return rows


def _paired_net_interval(
    cases: pd.DataFrame,
    *,
    method_actions: np.ndarray,
    method_final: np.ndarray,
    baseline_actions: np.ndarray,
    baseline_final: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    truth = cases["y_true"].to_numpy(dtype=int)
    scout = cases["scout_pred"].to_numpy(dtype=int)
    scout_correct = scout == truth

    def contribution(actions: np.ndarray, final: np.ndarray) -> np.ndarray:
        automatic = actions != "HUMAN_REVIEW"
        final_correct = final == truth
        corrected = automatic & (~scout_correct) & final_correct
        introduced = automatic & scout_correct & (~final_correct)
        return corrected.astype(int) - introduced.astype(int)

    difference = contribution(method_actions, method_final) - contribution(
        baseline_actions,
        baseline_final,
    )
    grouped = pd.DataFrame(
        {
            "group": cases["resampling_group_id"].astype(str),
            "difference": difference,
        }
    ).groupby("group", sort=True)["difference"].sum()
    values = grouped.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(replicates, len(values)))
    totals = values[samples].sum(axis=1)
    return {
        "net_difference_ci_lower": float(np.quantile(totals, 0.025)),
        "net_difference_ci_upper": float(np.quantile(totals, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_group_count": len(values),
    }


def _decide(
    core: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    fractions = {
        rounded_budget(value)
        for value in protocol["comparison_contract"]["decision_fractions"]
    }
    successful: list[str] = []
    evidence: dict[str, Any] = {"routes": {}}
    for route_id in sorted(core["route_id"].unique()):
        rows = core.loc[
            core["route_id"].eq(route_id)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["policy"].eq(METHOD_POLICY)
            & core["requested_human_review_fraction"].isin(fractions)
        ].copy()
        rows["dominant"] = (
            rows["delta_corrected_retained"].ge(0)
            & rows["delta_introduced_auto"].le(0)
            & rows["delta_dangerous_introduced_auto"].le(0)
            & rows["delta_net_retained"].gt(0)
        )
        at_twenty = rows.loc[
            rows["requested_human_review_fraction"].eq(0.2)
        ].iloc[0]
        success = (
            int(rows["dominant"].sum())
            >= int(
                protocol["decision_contract"][
                    "minimum_dominant_fractions_per_route"
                ]
            )
            and bool(at_twenty["dominant"])
            and float(at_twenty["net_difference_ci_lower"]) >= 0
        )
        if success:
            successful.append(route_id)
        evidence["routes"][route_id] = {
            "dominant_fractions": rows.loc[
                rows["dominant"], "requested_human_review_fraction"
            ].tolist(),
            "fraction_0_2_net_ci_lower": float(
                at_twenty["net_difference_ci_lower"]
            ),
            "success": success,
        }
    scouts = set(
        core.loc[core["route_id"].isin(successful), "scout_id"].astype(str)
    )
    if len(successful) >= 2 and len(scouts) >= 2:
        decision = "ADOPTION_GO"
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
    rows = core.loc[
        core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq(METHOD_POLICY)
        & core["requested_human_review_fraction"].isin([0.1, 0.2, 0.3])
    ].sort_values(["route_id", "requested_human_review_fraction"])
    table = []
    for row in rows.itertuples():
        table.append(
            "| "
            f"{row.scout_id}→{row.review_model_id} | "
            f"{row.requested_human_review_fraction:.0%} | "
            f"{int(row.corrected_retained)}/"
            f"{int(row.comparator_corrected_retained)} | "
            f"{int(row.introduced_auto)}/"
            f"{int(row.comparator_introduced_auto)} | "
            f"{int(row.dangerous_introduced_auto)}/"
            f"{int(row.comparator_dangerous_introduced_auto)} | "
            f"{int(row.delta_net_retained):+d} | "
            f"{row.locked_comparator_policy} |"
        )
    discrimination = core.loc[
        core["record_type"].eq("outcome_discrimination")
        & core["analysis_split"].eq("retrospective_evaluation")
    ].sort_values(["route_id", "outcome"])
    discrimination_table = []
    for row in discrimination.itertuples():
        discrimination_table.append(
            "| "
            f"{row.scout_id}→{row.review_model_id} | {row.outcome} | "
            f"{int(row.outcome_events)} | {row.outcome_auroc:.3f} | "
            f"{row.outcome_auprc:.3f} |"
        )
    discrimination_ranges = (
        discrimination.groupby("outcome")[["outcome_auroc", "outcome_auprc"]]
        .agg(["min", "max"])
        .round(3)
    )
    corrected_range = discrimination_ranges.loc["corrected"]
    introduced_range = discrimination_ranges.loc["introduced"]
    both_wrong_range = discrimination_ranges.loc["both_wrong"]
    next_step = {
        "ADOPTION_GO": (
            "结果支持把安全采纳作为后续独立患者级确认主线；在取得真实"
            "人工复核结果前，仍不能估计完整临床闭环净获益。"
        ),
        "ROUTE_SPECIFIC_GO": (
            "信号只在特定路线成立，只能开展路线限定的独立患者级确认，"
            "不能宣称通用复核采纳能力。"
        ),
        "NO_SIGNAL": (
            "按协议停止继续开发模型采纳方法，并重新评估开题主线；不得"
            "通过更复杂模型或假设人工全对来掩盖负结果。"
        ),
    }[decision]
    return (
        "# OphAgent 复核结果安全采纳可行性审计 v0.1\n\n"
        f"## 结论\n\n**{decision}**\n\n"
        "本研究发生在复核模型已经运行之后。模型只读取两模型完整概率、"
        "置信度/分歧、等级变化、开发折内转移画像和既有 Scout 表征，"
        "输出 KEEP_SCOUT、ADOPT_REVIEW_RESULT 或 HUMAN_REVIEW。"
        "未重新训练或运行任何眼底模型。\n\n"
        "## 核心比较\n\n"
        "| 路线 | 人工比例 | corrected 方法/基线 | introduced 方法/基线 | "
        "dangerous 方法/基线 | Δnet | 开发锁定基线 |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        + "\n".join(table)
        + "\n\n每个预算的正式比较器只在 development OOF 中从"
        "置信度更高者和软投票锁定。保留 Scout 与始终采用复核结果作为"
        "安全/效能端点完整保留，但不用于不可能的联合支配判定。\n\n"
        "## 状态可识别性\n\n"
        "| 路线 | 状态 | 事件数 | AUROC | AUPRC |\n"
        "|---|---|---:|---:|---:|\n"
        + "\n".join(discrimination_table)
        + "\n\n这些指标使用同一开发折锁定模型在冻结回顾集上的输出，"
        "只描述标签状态的回顾性可分性，不等同于部署时已知病例真值。"
        f"corrected AUROC 为 {corrected_range[('outcome_auroc', 'min')]:.3f}"
        f"–{corrected_range[('outcome_auroc', 'max')]:.3f}，introduced 为 "
        f"{introduced_range[('outcome_auroc', 'min')]:.3f}–"
        f"{introduced_range[('outcome_auroc', 'max')]:.3f}，说明谁可能纠错"
        "或引错具有排序信息；但 both_wrong AUROC 仅 "
        f"{both_wrong_range[('outcome_auroc', 'min')]:.3f}–"
        f"{both_wrong_range[('outcome_auroc', 'max')]:.3f}，AUPRC 仅 "
        f"{both_wrong_range[('outcome_auprc', 'min')]:.3f}–"
        f"{both_wrong_range[('outcome_auprc', 'max')]:.3f}。\n\n"
        "`NO_SIGNAL` 在本协议中特指没有形成满足预声明联合安全改善的"
        "可执行三动作策略，不表示所有输入特征都没有统计判别信息。\n\n"
        "## HUMAN_REVIEW 解释\n\n"
        "现有资产没有真实人工复核结论，因此 HUMAN_REVIEW 只表示延期"
        "裁决：主结果不把这些病例计为已纠正，也不假定人工一定正确。"
        "相同比例比较使用完全相同的复核病例数；报告同时记录人工队列"
        "捕获的 corrected、introduced、dangerous introduced 与 "
        "both_wrong 负担。\n\n"
        "## 判定与边界\n\n"
        f"{next_step}\n\n"
        "APTOS 缺少患者、眼别和检查标识，结论仅属于确认完全重复剔除"
        "后的图像级回顾证据。corrected/introduced/dangerous "
        "introduced 是标签错误代理，不是临床结局；SafetyEligibilityGate "
        "与人工最终责任均不可绕过。\n\n"
        "## 追溯\n\n"
        f"- 实现基线提交：`{source_commit}`\n"
        f"- 协议 SHA256：`{protocol_sha256}`\n"
        f"- Benchmark manifest SHA256："
        f"`{benchmark_audit['manifest_sha256']}`\n"
        f"- 病例表 SHA256：`{benchmark_audit['case_table_sha256']}`\n"
        f"- 成功路线：`{evidence['successful_routes']}`\n"
        "- 冻结前台、Benchmark、v1.1、Test、预测资产及既有负结论"
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
        raise ValueError("Qualified route evidence changed.")
    if file_sha256(project_root / PRECONSULTATION_RELATIVE_PATH) != str(
        protocol["frozen_inputs"]["preconsultation_negative_result_sha256"]
    ):
        raise ValueError("Preconsultation negative evidence changed.")

    cases = pd.read_csv(
        benchmark_dir / "case_level_benchmark.csv.gz",
        low_memory=False,
    )
    route_ids = sorted(protocol["frozen_inputs"]["qualified_route_ids"])
    asset_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict]] = {}
    for model_id, specification in protocol["representation_assets"].items():
        for split in ("validation", "test"):
            asset_cache[(model_id, split)] = _load_representation_asset(
                specification,
                split=split,
            )

    fractions = tuple(
        rounded_budget(value)
        for value in protocol["comparison_contract"]["human_review_fractions"]
    )
    rows: list[dict[str, Any]] = []
    discrimination: list[dict[str, Any]] = []
    selections: dict[tuple[str, str, str, float], tuple[np.ndarray, np.ndarray]] = {}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for route_id in route_ids:
        development = cases.loc[
            cases["route_id"].eq(route_id)
            & cases["benchmark_split"].eq("development")
            & cases["primary_cohort_eligible"].astype(bool)
        ].sort_values("case_id").reset_index(drop=True)
        evaluation = cases.loc[
            cases["route_id"].eq(route_id)
            & cases["benchmark_split"].eq("retrospective_frozen")
            & cases["primary_cohort_eligible"].astype(bool)
        ].sort_values("case_id").reset_index(drop=True)
        if len(development) != 485 or len(evaluation) != 1036:
            raise ValueError(f"{route_id}: frozen cohort size changed.")
        scout_id = str(development["scout_id"].iloc[0])
        development_embeddings = align_embeddings(
            development,
            embedding_case_ids=asset_cache[(scout_id, "validation")][0],
            embeddings=asset_cache[(scout_id, "validation")][1],
        )
        evaluation_embeddings = align_embeddings(
            evaluation,
            embedding_case_ids=asset_cache[(scout_id, "test")][0],
            embeddings=asset_cache[(scout_id, "test")][1],
        )
        salt = f"{route_id}:review-result-adoption-v0.1"
        development_predictions = nested_group_oof_adoption_predictions(
            development,
            development_embeddings,
            n_folds=5,
            pca_components=int(
                protocol["model_contract"]["pca_components"]
            ),
            minimum_events=10,
            salt=salt,
        ).reset_index(drop=True)
        _, evaluation_predictions = fit_full_development_adoption_and_predict(
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
            discrimination.extend(
                _discrimination_rows(
                    frame,
                    predictions,
                    analysis_split=analysis_split,
                )
            )
            for fraction in fractions:
                learned_actions = learned_adoption_actions(
                    frame,
                    predictions,
                    human_review_fraction=fraction,
                )
                learned_final = _final_prediction_from_actions(
                    frame,
                    learned_actions,
                )
                rows.append(
                    _performance_row(
                        frame,
                        actions=learned_actions,
                        final_prediction=learned_final,
                        policy=METHOD_POLICY,
                        human_review_fraction=fraction,
                        analysis_split=analysis_split,
                    )
                )
                selections[
                    (route_id, analysis_split, METHOD_POLICY, fraction)
                ] = (learned_actions, learned_final)
                for policy in BASELINE_POLICIES:
                    actions, final = baseline_actions(
                        frame,
                        policy=policy,
                        human_review_fraction=fraction,
                    )
                    rows.append(
                        _performance_row(
                            frame,
                            actions=actions,
                            final_prediction=final,
                            policy=policy,
                            human_review_fraction=fraction,
                            analysis_split=analysis_split,
                        )
                    )
                    selections[
                        (route_id, analysis_split, policy, fraction)
                    ] = (actions, final)

    core = pd.DataFrame(rows)
    core, comparator_map = _annotate_comparators(core, fractions)
    decision_fractions = {
        rounded_budget(value)
        for value in protocol["comparison_contract"]["decision_fractions"]
    }
    method_rows = (
        core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq(METHOD_POLICY)
        & core["requested_human_review_fraction"].isin(decision_fractions)
    )
    for index in core.index[method_rows]:
        row = core.loc[index]
        route_id = str(row["route_id"])
        fraction = rounded_budget(row["requested_human_review_fraction"])
        comparator = comparator_map[(route_id, fraction)]
        method_actions, method_final = selections[
            (route_id, "retrospective_evaluation", METHOD_POLICY, fraction)
        ]
        baseline_actions_value, baseline_final = selections[
            (route_id, "retrospective_evaluation", comparator, fraction)
        ]
        interval = _paired_net_interval(
            frames[(route_id, "retrospective_evaluation")],
            method_actions=method_actions,
            method_final=method_final,
            baseline_actions=baseline_actions_value,
            baseline_final=baseline_final,
            replicates=args.bootstrap_replicates,
            seed=stable_seed(f"{route_id}:{fraction}:adoption"),
        )
        for column, value in interval.items():
            core.loc[index, column] = value

    decision, evidence = _decide(core, protocol)
    core = pd.concat(
        [core, pd.DataFrame(discrimination)],
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
            "requested_human_review_fraction",
            "policy",
        ]
    ).reset_index(drop=True)
    forbidden = ("path", "patient", "image_sha", "private")
    if any(
        token in column.lower()
        for column in core.columns
        for token in forbidden
    ):
        raise ValueError("Sensitive field entered formal adoption output.")
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
