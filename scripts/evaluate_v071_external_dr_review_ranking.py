#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v0.7.1 外部 DR review ranking 评估脚本。

输入：
- experiments/summary/v0_7_1/external_dr_direct_inference_predictions.csv

用途：
- 基于 frozen checkpoint direct external inference 结果；
- 按 v0.7.0 冻结口径构造危险事件；
- 评估 Top10% / Top20% / Top30% 复核预算下的危险样本富集能力。

事件定义：
- large_undergrading = true_grade - pred_grade >= 2
- vision_threatening_dr_miss = true_grade >= 3 and pred_grade < 3
- dangerous_undergrading = large_undergrading OR vision_threatening_dr_miss

解释边界：
- 本脚本不是重新训练，不调参；
- 不根据外部结果重新选择 primary signal；
- 结果用于 external frozen checkpoint failure enrichment / residual risk analysis。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGETS = [
    "large_undergrading",
    "vision_threatening_dr_miss",
    "dangerous_undergrading",
]

BUDGETS = [0.10, 0.20, 0.30]

RANKING_METHODS: dict[str, dict[str, Any]] = {
    "expected_gap_only": {
        "score_col": "expected_gap",
        "ascending": False,
        "description": "expected_grade - pred_grade，越大表示模型概率质量更偏向更重等级",
    },
    "gated_severe_prob_mass_only": {
        "score_col": "gated_severe_prob_mass",
        "ascending": False,
        "description": "pred_grade <= 2 时的 P(3)+P(4)，用于重症漏检风险",
    },
    "severe_prob_mass_only": {
        "score_col": "severe_prob_mass",
        "ascending": False,
        "description": "P(3)+P(4)",
    },
    "entropy_only": {
        "score_col": "entropy",
        "ascending": False,
        "description": "预测分布熵，越大表示越不确定",
    },
    "low_margin_only": {
        "score_col": "margin",
        "ascending": True,
        "description": "top1-top2 margin，越小表示越不确定",
    },
    "low_confidence_only": {
        "score_col": "confidence",
        "ascending": True,
        "description": "top1 confidence，越小表示越不确定",
    },
}

PRIMARY_PROTOCOL = {
    "large_undergrading": "expected_gap_only",
    "vision_threatening_dr_miss": "gated_severe_prob_mass_only",
}

SECONDARY_PROTOCOL = {
    "dangerous_undergrading": [
        "expected_gap_only",
        "gated_severe_prob_mass_only",
    ]
}


def df_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    """避免依赖 tabulate 的极简 markdown 表格。"""
    if df.empty:
        return "_Empty table._"

    tmp = df.copy()

    def fmt(x):
        if isinstance(x, float):
            if np.isnan(x):
                return ""
            return f"{x:.{float_digits}f}"
        return str(x)

    headers = list(tmp.columns)
    rows = [[fmt(v) for v in row] for row in tmp.to_numpy()]

    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def add_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["true_grade"] = out["true_grade"].astype(int)
    out["pred_grade"] = out["pred_grade"].astype(int)

    out["large_undergrading"] = (out["true_grade"] - out["pred_grade"]) >= 2
    out["vision_threatening_dr_miss"] = (out["true_grade"] >= 3) & (out["pred_grade"] < 3)
    out["dangerous_undergrading"] = out["large_undergrading"] | out["vision_threatening_dr_miss"]

    return out


def rank_one_group(
    *,
    g: pd.DataFrame,
    target: str,
    method: str,
    budget: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    method_cfg = RANKING_METHODS[method]
    score_col = method_cfg["score_col"]
    ascending = method_cfg["ascending"]

    n = len(g)
    top_k = max(1, int(np.ceil(n * budget)))

    ranked = g.sort_values(
        by=[score_col, "relative_image_path", "backbone"],
        ascending=[ascending, True, True],
        kind="mergesort",
    ).copy()

    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["review_flag"] = ranked["rank"] <= top_k
    ranked["target_event"] = ranked[target].astype(bool)
    ranked["ranking_target"] = target
    ranked["ranking_method"] = method
    ranked["ranking_score_col"] = score_col
    ranked["ranking_score"] = ranked[score_col]
    ranked["budget"] = budget
    ranked["top_k"] = top_k

    flagged = ranked[ranked["review_flag"]]
    unflagged = ranked[~ranked["review_flag"]]

    total_event = int(ranked["target_event"].sum())
    captured_event = int(flagged["target_event"].sum())
    residual_event = int(total_event - captured_event)

    flagged_n = int(len(flagged))
    unflagged_n = int(len(unflagged))

    base_event_rate = total_event / n if n > 0 else np.nan
    flagged_event_rate = captured_event / flagged_n if flagged_n > 0 else np.nan
    event_recall = captured_event / total_event if total_event > 0 else np.nan
    enrichment_ratio = (
        flagged_event_rate / base_event_rate
        if base_event_rate and base_event_rate > 0
        else np.nan
    )
    low_risk_npv = (
        1.0 - (residual_event / unflagged_n)
        if unflagged_n > 0
        else np.nan
    )

    metrics = {
        "dataset": str(g["dataset"].iloc[0]),
        "backbone": str(g["backbone"].iloc[0]),
        "target": target,
        "ranking_method": method,
        "score_col": score_col,
        "budget": budget,
        "n": n,
        "top_k": top_k,
        "total_event": total_event,
        "captured_event": captured_event,
        "residual_event": residual_event,
        "base_event_rate": base_event_rate,
        "flagged_event_rate": flagged_event_rate,
        "event_recall": event_recall,
        "enrichment_ratio": enrichment_ratio,
        "low_risk_npv": low_risk_npv,
    }

    table_cols = [
        "dataset",
        "backbone",
        "ranking_target",
        "ranking_method",
        "budget",
        "rank",
        "review_flag",
        "target_event",
        "true_grade",
        "pred_grade",
        "relative_image_path",
        "confidence",
        "margin",
        "entropy",
        "expected_grade",
        "severe_prob_mass",
        "expected_gap",
        "gated_severe_prob_mass",
        "ranking_score",
    ]

    # 只保存 Top-K flagged 和 target_event residual 的关键行，避免表过大。
    compact = ranked[
        ranked["review_flag"] | ranked["target_event"]
    ][table_cols].copy()

    return metrics, compact


def evaluate(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = add_event_columns(pred)

    metric_rows: list[dict[str, Any]] = []
    table_parts: list[pd.DataFrame] = []

    for (dataset, backbone), g in pred.groupby(["dataset", "backbone"], sort=True):
        for target in TARGETS:
            for method in RANKING_METHODS:
                for budget in BUDGETS:
                    metrics, table = rank_one_group(
                        g=g,
                        target=target,
                        method=method,
                        budget=budget,
                    )
                    metric_rows.append(metrics)
                    table_parts.append(table)

    metrics_df = pd.DataFrame(metric_rows)
    table_df = pd.concat(table_parts, ignore_index=True)

    return metrics_df, table_df


def write_summary(out_dir: Path, metrics: pd.DataFrame, pred: pd.DataFrame) -> None:
    lines: list[str] = []

    lines.append("# v0.7.1 External DR Review Ranking Summary")
    lines.append("")
    lines.append("## 版本定位")
    lines.append("")
    lines.append(
        "本结果基于 v0.7.1 frozen checkpoint direct external inference 输出，"
        "评估外部 DR 数据上的危险错误富集与自动放行区残余风险。"
    )
    lines.append("")
    lines.append("本阶段不使用 IDRiD_data / MESSIDOR2 train 或 val 训练，不根据外部结果重新选择 primary target、ranking signal 或 review budget。")
    lines.append("")
    lines.append("## 事件定义")
    lines.append("")
    lines.append("- `large_undergrading = true_grade - pred_grade >= 2`")
    lines.append("- `vision_threatening_dr_miss = true_grade >= 3 and pred_grade < 3`")
    lines.append("- `dangerous_undergrading = large_undergrading OR vision_threatening_dr_miss`")
    lines.append("")
    lines.append("## 冻结协议")
    lines.append("")
    lines.append("- `large_undergrading`: primary ranking signal = `expected_gap_only`, primary budget = Top20%")
    lines.append("- `vision_threatening_dr_miss`: primary ranking signal = `gated_severe_prob_mass_only`, primary budget = Top20%")
    lines.append("- `dangerous_undergrading`: secondary composite target，仅作为补充分析")
    lines.append("")

    lines.append("## 事件数量")
    lines.append("")
    event_counts = []
    pred2 = add_event_columns(pred)
    for (dataset, backbone), g in pred2.groupby(["dataset", "backbone"], sort=True):
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "n": len(g),
        }
        for target in TARGETS:
            row[target] = int(g[target].sum())
            row[f"{target}_rate"] = float(g[target].mean())
        event_counts.append(row)
    event_counts_df = pd.DataFrame(event_counts)
    lines.append(df_to_markdown(event_counts_df, float_digits=4))
    lines.append("")

    lines.append("## Primary Top20% 结果")
    lines.append("")
    primary_rows = []
    for target, method in PRIMARY_PROTOCOL.items():
        x = metrics[
            (metrics["target"] == target)
            & (metrics["ranking_method"] == method)
            & (np.isclose(metrics["budget"], 0.20))
        ].copy()
        primary_rows.append(x)
    primary_df = pd.concat(primary_rows, ignore_index=True)
    primary_show = primary_df[
        [
            "dataset",
            "backbone",
            "target",
            "ranking_method",
            "n",
            "top_k",
            "total_event",
            "captured_event",
            "residual_event",
            "base_event_rate",
            "flagged_event_rate",
            "event_recall",
            "enrichment_ratio",
            "low_risk_npv",
        ]
    ].sort_values(["target", "dataset", "backbone"])
    lines.append(df_to_markdown(primary_show, float_digits=4))
    lines.append("")

    lines.append("## 解释边界")
    lines.append("")
    lines.append("- 当前外部分类迁移表现存在域迁移压力，尤其 MESSIDOR2 上多模型预测分布偏向 0 类。")
    lines.append("- 因此本结果应解释为 frozen APTOS checkpoints 在外部 DR 数据上的错误富集与 residual risk analysis。")
    lines.append("- 若分类迁移不足，不能将 ranking 结果强称为临床泛化成功验证。")
    lines.append("- Top-K 使用 `ceil(n * budget)`，因此 IDRiD_data Top20% 为 21 张，MESSIDOR2 Top20% 为 106 张。")
    lines.append("")

    (out_dir / "external_dr_review_ranking_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="experiments/summary/v0_7_1/external_dr_direct_inference_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/summary/v0_7_1",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"predictions 不存在：{pred_path}")

    pred = pd.read_csv(pred_path)

    required = [
        "dataset",
        "backbone",
        "relative_image_path",
        "true_grade",
        "pred_grade",
        "confidence",
        "margin",
        "entropy",
        "expected_grade",
        "severe_prob_mass",
        "expected_gap",
        "gated_severe_prob_mass",
    ]
    missing = [c for c in required if c not in pred.columns]
    if missing:
        raise ValueError(f"predictions 缺少必要字段：{missing}")

    metrics, ranking_table = evaluate(pred)

    metrics_path = out_dir / "external_dr_review_ranking_metrics.csv"
    table_path = out_dir / "external_dr_review_ranking_table.csv"

    metrics.to_csv(metrics_path, index=False)
    ranking_table.to_csv(table_path, index=False)

    write_summary(out_dir, metrics, pred)

    print("已保存：")
    print(metrics_path)
    print(table_path)
    print(out_dir / "external_dr_review_ranking_summary.md")


if __name__ == "__main__":
    main()
