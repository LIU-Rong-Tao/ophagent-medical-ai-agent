#!/usr/bin/env python3
"""Evaluate same-forward Scout representations for pre-consultation routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.scout_representation_consultation import (  # noqa: E402
    align_embeddings,
    fit_full_development_representation_and_predict,
    nested_group_oof_representation_predictions,
)
from app.selective_consultation import (  # noqa: E402
    fit_full_development_and_predict,
    nested_group_oof_predictions,
    paired_cluster_bootstrap_difference,
    select_consultations,
)
from scripts.run_selective_consultation_method_v0_1 import (  # noqa: E402
    commit_timestamp,
    current_commit,
    evaluation_row,
    file_sha256,
    rounded_budget,
    stable_seed,
    strongest_row,
    verify_benchmark_inputs,
    write_csv,
)


PROTOCOL_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/configs/protocols/"
    "scout_representation_consultation_v0_1.json"
)
BENCHMARK_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "help_or_harm_benchmark_v0_1"
)
PRIOR_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "aptos_high_capability_simple_gate_v0_1"
)
OUTPUT_RELATIVE_PATH = Path(
    "experiments/opening_risk_routing_closure/outputs/"
    "scout_representation_consultation_v0_1"
)
TASK_ID = "aptos_dr_5class"
METHOD_POLICY = "scout_representation"
COMPARATOR_POLICIES = ("entropy", "margin", "prior_simple_gate")
CURVE_POLICIES = (*COMPARATOR_POLICIES, METHOD_POLICY)
FORMAL_OUTPUT_NAMES = (
    "core_results.csv",
    "risk_budget_comparison.png",
    "research_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser.parse_args()


def _load_representation_asset(
    specification: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(specification[f"{split}_path"])
    expected_sha256 = str(specification[f"{split}_sha256"])
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Representation asset changed: {path.name}")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            case_ids = payload["case_ids"].astype(str)
            embeddings = payload["embeddings"].astype(np.float32)
            metadata = json.loads(str(payload["metadata_json"].item()))
        metadata["mean_inference_ms_per_image"] = float(
            metadata["one_time_retrospective_forward_ms_per_image"]
        )
    else:
        frame = pd.read_csv(path, low_memory=False)
        embedding_columns = [
            column for column in frame if column.startswith("emb_")
        ]
        case_ids = frame["image_key"].astype(str).to_numpy()
        embeddings = frame[embedding_columns].to_numpy(dtype=np.float32)
        metadata = {
            "model_id": "retfound_green",
            "embedding_dim": len(embedding_columns),
            "case_count": len(frame),
            "mean_inference_ms_per_image": float(
                frame["inference_ms_per_image"].mean()
            ),
            "incremental_online_encoder_forward_ms_per_image": 0.0,
        }
    if embeddings.shape != (
        int(metadata["case_count"]),
        int(metadata["embedding_dim"]),
    ):
        raise ValueError(f"Invalid representation shape in {path.name}.")
    metadata.update(
        {
            "asset_sha256": observed_sha256,
            "asset_size_bytes": path.stat().st_size,
            "asset_name": path.name,
        }
    )
    return case_ids, embeddings, metadata


def _select(
    frame: pd.DataFrame,
    *,
    policy: str,
    budget: float,
    simple_predictions: pd.DataFrame,
    representation_predictions: pd.DataFrame,
    safe_pool_multiplier: float,
) -> np.ndarray:
    if policy in {"entropy", "margin", "scout_only", "always_expert"}:
        return select_consultations(frame, policy=policy, budget=budget)
    predictions = (
        simple_predictions
        if policy == "prior_simple_gate"
        else representation_predictions
    )
    return select_consultations(
        frame,
        policy="dual_logistic_harm_screened_help",
        budget=budget,
        predictions=predictions,
        safe_pool_multiplier=safe_pool_multiplier,
    )


def _annotate_comparators(
    core: pd.DataFrame,
    budgets: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[tuple[str, float], str]]:
    result = core.copy()
    comparator_map: dict[tuple[str, float], str] = {}
    for route_id in sorted(result["route_id"].unique()):
        for budget in budgets:
            candidates = result.loc[
                result["route_id"].eq(route_id)
                & result["analysis_split"].eq("development_oof")
                & result["requested_budget"].eq(budget)
                & result["policy"].isin(COMPARATOR_POLICIES)
            ]
            comparator = strongest_row(candidates.to_dict("records"))
            comparator_policy = str(comparator["policy"])
            comparator_map[(route_id, budget)] = comparator_policy
            for split in ("development_oof", "retrospective_evaluation"):
                baseline = result.loc[
                    result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["requested_budget"].eq(budget)
                    & result["policy"].eq(comparator_policy)
                ].iloc[0]
                mask = (
                    result["route_id"].eq(route_id)
                    & result["analysis_split"].eq(split)
                    & result["requested_budget"].eq(budget)
                )
                result.loc[mask, "locked_comparator_policy"] = comparator_policy
                for metric in (
                    "selected_n",
                    "corrected_selected",
                    "introduced_selected",
                    "net_selected",
                ):
                    result.loc[mask, f"comparator_{metric}"] = baseline[metric]
                    result.loc[mask, f"delta_{metric}"] = (
                        result.loc[mask, metric] - baseline[metric]
                    )
    return result, comparator_map


def _make_figure(core: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "entropy": "#0072B2",
        "margin": "#E69F00",
        "prior_simple_gate": "#009E73",
        "scout_representation": "#D55E00",
    }
    labels = {
        "entropy": "Entropy",
        "margin": "Margin",
        "prior_simple_gate": "Prior simple gate",
        "scout_representation": "Scout representation",
    }
    styles = {
        "entropy": ("--", "o"),
        "margin": (":", "s"),
        "prior_simple_gate": ("-.", "^"),
        "scout_representation": ("-", "D"),
    }
    evaluation = core.loc[
        core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].isin(CURVE_POLICIES)
    ]
    route_ids = sorted(evaluation["route_id"].unique())
    figure, axes = plt.subplots(2, len(route_ids), figsize=(12.2, 6.4))
    for column, route_id in enumerate(route_ids):
        route = evaluation.loc[evaluation["route_id"].eq(route_id)]
        title = route_id.split("::", 1)[1].replace("__to__", " → ")
        for policy in CURVE_POLICIES:
            values = route.loc[route["policy"].eq(policy)].sort_values(
                "requested_budget"
            )
            line, marker = styles[policy]
            for row, metric in enumerate(
                ("corrected_selected", "introduced_selected")
            ):
                axes[row, column].plot(
                    values["requested_budget"] * 100,
                    values[metric],
                    color=colors[policy],
                    linestyle=line,
                    marker=marker,
                    linewidth=2,
                    markersize=4,
                    label=labels[policy],
                )
        axes[0, column].set_title(title, fontsize=10)
        axes[1, column].set_xlabel("Expert call budget (%)")
        axes[0, column].grid(alpha=0.22)
        axes[1, column].grid(alpha=0.22)
    axes[0, 0].set_ylabel("Corrected retained (count)")
    axes[1, 0].set_ylabel("Introduced (count)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "APTOS retrospective risk–budget comparison",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(figure)


def _decide(
    core: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    decision_budgets = {
        rounded_budget(value)
        for value in protocol["comparison_contract"]["decision_budgets"]
    }
    evidence: dict[str, Any] = {"routes": {}}
    successful: list[str] = []
    for route_id in sorted(core["route_id"].unique()):
        rows = core.loc[
            core["route_id"].eq(route_id)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["policy"].eq(METHOD_POLICY)
            & core["requested_budget"].isin(decision_budgets)
        ].copy()
        rows["dominant"] = (
            rows["delta_corrected_selected"].ge(0)
            & rows["delta_introduced_selected"].le(0)
            & rows["delta_net_selected"].gt(0)
        )
        at_thirty = rows.loc[rows["requested_budget"].eq(0.3)].iloc[0]
        success = (
            int(rows["dominant"].sum())
            >= int(
                protocol["decision_contract"][
                    "minimum_dominant_decision_budgets_per_route"
                ]
            )
            and bool(at_thirty["dominant"])
            and float(at_thirty["net_difference_ci_lower"]) >= 0
        )
        if success:
            successful.append(route_id)
        evidence["routes"][route_id] = {
            "dominant_budgets": rows.loc[
                rows["dominant"], "requested_budget"
            ].tolist(),
            "budget_0_3_net_ci_lower": float(
                at_thirty["net_difference_ci_lower"]
            ),
            "success": success,
        }
    successful_scouts = set(
        core.loc[core["route_id"].isin(successful), "scout_id"].astype(str)
    )
    if len(successful) >= 2 and len(successful_scouts) >= 2:
        decision = "REPRESENTATION_GO"
    elif successful:
        decision = "ROUTE_SPECIFIC_GO"
    else:
        decision = "NO_IMPROVEMENT"
    evidence["successful_routes"] = successful
    evidence["successful_distinct_scouts"] = sorted(successful_scouts)
    return decision, evidence


def _build_report(
    *,
    core: pd.DataFrame,
    decision: str,
    evidence: dict[str, Any],
    protocol_sha256: str,
    source_commit: str,
    benchmark_audit: dict[str, Any],
    asset_metadata: dict[str, dict[str, Any]],
) -> str:
    method = core.loc[
        core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq(METHOD_POLICY)
        & core["requested_budget"].isin([0.1, 0.2, 0.3])
    ]
    rows = []
    for row in method.itertuples():
        rows.append(
            "| "
            f"{row.scout_id}→{row.expert_id} | {row.requested_budget:.0%} | "
            f"{int(row.corrected_selected)}/{int(row.comparator_corrected_selected)} | "
            f"{int(row.introduced_selected)}/{int(row.comparator_introduced_selected)} | "
            f"{int(row.delta_net_selected):+d} | {row.locked_comparator_policy} |"
        )
    prior_comparisons = []
    for route_id in sorted(method["route_id"].unique()):
        representation = core.loc[
            core["route_id"].eq(route_id)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["policy"].eq(METHOD_POLICY)
            & core["requested_budget"].eq(0.3)
        ].iloc[0]
        prior = core.loc[
            core["route_id"].eq(route_id)
            & core["analysis_split"].eq("retrospective_evaluation")
            & core["policy"].eq("prior_simple_gate")
            & core["requested_budget"].eq(0.3)
        ].iloc[0]
        prior_comparisons.append(
            f"- {representation['scout_id']}→{representation['expert_id']}："
            f"corrected {int(representation['corrected_selected'])}/"
            f"{int(prior['corrected_selected'])}，introduced "
            f"{int(representation['introduced_selected'])}/"
            f"{int(prior['introduced_selected'])}"
            "（表征/上一版）。"
        )
    asset_lines = []
    for model_id, metadata in sorted(asset_metadata.items()):
        asset_lines.append(
            f"- {model_id}：{metadata['embedding_dim']} 维，"
            f"开发/回顾资产合计 "
            f"{metadata['total_asset_size_bytes'] / 1024 / 1024:.2f} MiB；"
            f"同一次在线 Scout 前向的额外编码器调用为 0。"
        )
    next_step = {
        "REPRESENTATION_GO": (
            "结果支持进入独立、未暴露且具患者标识的确认研究；当前仍不能"
            "授予部署或临床路由资格。"
        ),
        "ROUTE_SPECIFIC_GO": (
            "信号仅在特定路线成立，后续只能做路线限定的独立患者级确认，"
            "不得概括为通用调用前门控。"
        ),
        "NO_IMPROVEMENT": (
            "按预声明停止增加调用前模型复杂度；下一阶段应评估 Expert "
            "输出到达后的 KEEP_SCOUT / ADOPT_SECOND_OPINION / "
            "HUMAN_REVIEW 采纳控制。"
        ),
    }[decision]
    return (
        "# OphAgent Scout 视觉表征预咨询研究 v0.1\n\n"
        f"## 结论\n\n**{decision}**\n\n"
        "本研究只使用冻结 Scout 同一次前向的分类头前视觉表征、Scout "
        "概率与开发折内路线画像。未读取当前病例 Expert 输出或表征，"
        "未训练 Scout/Expert，未用冻结回顾结果选择路线、特征、阈值"
        "或模型。\n\n"
        "## 核心结果\n\n"
        "| 路线 | 预算 | corrected 表征/基线 | introduced 表征/基线 | "
        "Δnet | 开发集锁定最强基线 |\n"
        "|---|---:|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n\n最强基线在每条路线、每个预算上仅由 development OOF "
        "在 entropy、margin 与上一版简单门控中锁定；冻结回顾集只评价。"
        "完整 5%–30% 风险—预算关系见 `risk_budget_comparison.png`。\n\n"
        "30% 预算下，表征相对上一版简单门控的直接比较为：\n\n"
        + "\n".join(prior_comparisons)
        + "\n\n这说明视觉表征能改善部分旧门控排序，但仍明显落后于"
        "开发集锁定的简单不确定性基线，故不能据此授予 GO。\n\n"
        "## 表征与成本\n\n"
        + "\n".join(asset_lines)
        + "\n\nFLAIR 旧运行未保存表征，故本研究做了一次冻结前向回提取；"
        "这属于回顾性研究成本。未来在线实现直接保留分类头输入，不产生"
        "第二次 Scout 编码。RETFound-Green 复用既有表征资产。\n\n"
        "## 判定与边界\n\n"
        f"{next_step}\n\n"
        "APTOS 缺少患者、眼别和检查标识；分析单位是确认像素级重复剔除"
        "后的图像组，不能声称患者级泛化。corrected/introduced 仍是"
        "标签定义的错误代理，不是临床获益/伤害。SafetyEligibilityGate "
        "始终保留。\n\n"
        "## 追溯\n\n"
        f"- 实现基线提交：`{source_commit}`\n"
        f"- 协议 SHA256：`{protocol_sha256}`\n"
        f"- Benchmark manifest SHA256："
        f"`{benchmark_audit['manifest_sha256']}`\n"
        f"- 病例表 SHA256：`{benchmark_audit['case_table_sha256']}`\n"
        f"- 成功路线：`{evidence['successful_routes']}`\n"
        "- 冻结 Benchmark、v1.1、Test、预测资产与既有负结论均未修改。\n"
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

    prior_dir = project_root / PRIOR_RELATIVE_PATH
    for name, expected in (
        ("qualified_routes.csv", protocol["input_contract"]["qualified_routes_sha256"]),
        (
            "core_comparison_results.csv",
            protocol["input_contract"]["prior_simple_gate_results_sha256"],
        ),
    ):
        if file_sha256(prior_dir / name) != expected:
            raise ValueError(f"Prior frozen evidence changed: {name}")
    qualified = pd.read_csv(prior_dir / "qualified_routes.csv")
    route_ids = sorted(qualified["route_id"].astype(str))
    if route_ids != sorted(protocol["input_contract"]["qualified_route_ids"]):
        raise ValueError("Qualified route identities changed.")

    cases = pd.read_csv(
        benchmark_dir / "case_level_benchmark.csv.gz",
        low_memory=False,
    )
    inventory = pd.read_csv(
        benchmark_dir / "candidate_route_inventory.csv",
        low_memory=False,
    )
    asset_cache: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]
    ] = {}
    asset_summary: dict[str, dict[str, Any]] = {}
    for model_id, specification in protocol["representation_assets"].items():
        total_size = 0
        for split in ("validation", "test"):
            loaded = _load_representation_asset(specification, split=split)
            asset_cache[(model_id, split)] = loaded
            total_size += int(loaded[2]["asset_size_bytes"])
        asset_summary[model_id] = {
            **asset_cache[(model_id, "validation")][2],
            "total_asset_size_bytes": total_size,
        }

    budgets = tuple(
        rounded_budget(value)
        for value in protocol["comparison_contract"]["curve_budgets"]
    )
    safe_pool_multiplier = float(
        protocol["model_contract"]["safe_pool_multiplier"]
    )
    rows: list[dict[str, Any]] = []
    selections: dict[tuple[str, str, str, float], np.ndarray] = {}
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for route_id in route_ids:
        route = inventory.loc[inventory["route_id"].eq(route_id)].iloc[0]
        scout_id = str(route["scout_id"])
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
            raise ValueError(f"{route_id}: cohort size changed.")
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
        salt = f"{TASK_ID}:{route_id}:scout-representation-v0.1"
        simple_development = nested_group_oof_predictions(
            development,
            n_folds=5,
            minimum_route_events=10,
            salt=f"{TASK_ID}:{route_id}:selective-consultation-v0.1",
        ).reset_index(drop=True)
        _, simple_evaluation = fit_full_development_and_predict(
            development,
            evaluation,
            n_folds=5,
            minimum_route_events=10,
            salt=f"{TASK_ID}:{route_id}:selective-consultation-v0.1",
        )
        representation_development = (
            nested_group_oof_representation_predictions(
                development,
                development_embeddings,
                n_folds=5,
                pca_components=int(
                    protocol["model_contract"]["pca_components"]
                ),
                minimum_route_events=10,
                salt=salt,
            ).reset_index(drop=True)
        )
        _, representation_evaluation = (
            fit_full_development_representation_and_predict(
                development,
                development_embeddings,
                evaluation,
                evaluation_embeddings,
                n_folds=5,
                pca_components=int(
                    protocol["model_contract"]["pca_components"]
                ),
                minimum_route_events=10,
                salt=salt,
            )
        )
        split_data = (
            (
                "development_oof",
                development,
                simple_development,
                representation_development,
            ),
            (
                "retrospective_evaluation",
                evaluation,
                simple_evaluation.reset_index(drop=True),
                representation_evaluation.reset_index(drop=True),
            ),
        )
        for (
            analysis_split,
            frame,
            simple_predictions,
            representation_predictions,
        ) in split_data:
            frames[(route_id, analysis_split)] = frame
            for policy, budget in (
                ("scout_only", 0.0),
                ("always_expert", 1.0),
            ):
                selected = _select(
                    frame,
                    policy=policy,
                    budget=budget,
                    simple_predictions=simple_predictions,
                    representation_predictions=representation_predictions,
                    safe_pool_multiplier=safe_pool_multiplier,
                )
                row = evaluation_row(
                    frame,
                    selected=selected,
                    route=route,
                    policy=policy,
                    budget=budget,
                    analysis_split=analysis_split,
                    cohort="aptos_duplicate_excluded_image",
                    comparison_axis="reference_endpoint",
                )
                rows.append(row)
            for budget in budgets:
                for policy in CURVE_POLICIES:
                    selected = _select(
                        frame,
                        policy=policy,
                        budget=budget,
                        simple_predictions=simple_predictions,
                        representation_predictions=representation_predictions,
                        safe_pool_multiplier=safe_pool_multiplier,
                    )
                    row = evaluation_row(
                        frame,
                        selected=selected,
                        route=route,
                        policy=policy,
                        budget=budget,
                        analysis_split=analysis_split,
                        cohort="aptos_duplicate_excluded_image",
                        comparison_axis="same_budget",
                    )
                    row["representation_dimension"] = int(
                        asset_summary[scout_id]["embedding_dim"]
                    )
                    row["representation_asset_sha256"] = str(
                        asset_summary[scout_id]["asset_sha256"]
                    )
                    row["incremental_online_encoder_forward_calls"] = 0
                    row["current_case_expert_output_used_for_ranking"] = False
                    row["test_used_for_fit_selection_or_threshold"] = False
                    rows.append(row)
                    selections[
                        (route_id, analysis_split, policy, budget)
                    ] = selected

    core = pd.DataFrame(rows)
    core, comparator_map = _annotate_comparators(core, budgets)
    decision_budgets = {
        rounded_budget(value)
        for value in protocol["comparison_contract"]["decision_budgets"]
    }
    method_rows = (
        core["analysis_split"].eq("retrospective_evaluation")
        & core["policy"].eq(METHOD_POLICY)
        & core["requested_budget"].isin(decision_budgets)
    )
    for index in core.index[method_rows]:
        row = core.loc[index]
        route_id = str(row["route_id"])
        budget = rounded_budget(row["requested_budget"])
        comparator_policy = comparator_map[(route_id, budget)]
        interval = paired_cluster_bootstrap_difference(
            frames[(route_id, "retrospective_evaluation")],
            method_selected=selections[
                (route_id, "retrospective_evaluation", METHOD_POLICY, budget)
            ],
            baseline_selected=selections[
                (
                    route_id,
                    "retrospective_evaluation",
                    comparator_policy,
                    budget,
                )
            ],
            replicates=args.bootstrap_replicates,
            seed=stable_seed(f"{route_id}:{budget}:representation"),
        )
        for column, value in interval.items():
            core.loc[index, column] = value

    decision, evidence = _decide(core, protocol)
    core["study_decision"] = decision
    core["protocol_sha256"] = protocol_sha256
    core["source_commit_sha"] = source_commit
    core["generated_at_utc"] = generated_at
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
    if any(
        "path" in column.lower()
        or "patient" in column.lower()
        or "image_sha" in column.lower()
        for column in core.columns
    ):
        raise ValueError("Sensitive or private fields entered formal output.")

    write_csv(output_dir / "core_results.csv", core)
    _make_figure(core, output_dir / "risk_budget_comparison.png")
    report = _build_report(
        core=core,
        decision=decision,
        evidence=evidence,
        protocol_sha256=protocol_sha256,
        source_commit=source_commit,
        benchmark_audit=benchmark_audit,
        asset_metadata=asset_summary,
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
