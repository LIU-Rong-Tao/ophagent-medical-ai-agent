from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

def trapz_area(y, x):
    """Compatibility wrapper for trapezoidal integration."""
    fn = getattr(np, "trapezoid", np.trapz)
    return fn(y, x)



EVENT_DEFS = {
    "general_error": lambda df: (df["pred_grade"] != df["true_grade"]).astype(int),
    "large_undergrading": lambda df: ((df["true_grade"] - df["pred_grade"]) >= 2).astype(int),
    "vtdr_miss": lambda df: ((df["true_grade"] >= 3) & (df["pred_grade"] < 3)).astype(int),
}


def safe_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    """AUROC for event/failure detection. Higher score means higher event risk."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)

    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]

    if len(y_true) == 0 or y_true.min() == y_true.max():
        return float("nan")

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks handle tied scores correctly.
    ranks = pd.Series(score).rank(method="average", ascending=True).to_numpy()
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _normalized_partial_trapz(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Area over [lo, hi], normalized by interval width, using linear interpolation."""
    if hi <= lo:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if lo < x.min() or hi > x.max():
        return float("nan")

    inside = (x > lo) & (x < hi)
    xs = np.concatenate([[lo], x[inside], [hi]])
    ys = np.concatenate([[np.interp(lo, x, y)], y[inside], [np.interp(hi, x, y)]])
    return float(trapz_area(ys, xs) / (hi - lo))


def metric_curves(event: np.ndarray, risk_score: np.ndarray) -> Dict[str, float]:
    """
    Discrete AURC / AUGRC under review-priority semantics.

    Convention:
    - Higher risk_score means the sample should be reviewed earlier.
    - We sort samples from high risk to low risk.
    - The top-k samples are treated as entering the priority review set.
    - coverage = fraction not yet reviewed, still handled by the model.
    - selective_risk = residual events / residual sample count.
    - generalized_risk = residual events / total sample count.
    - AURC is the selective_risk-coverage curve area, matching the v0.6.6 implementation.
    - AUGRC is the generalized_risk-coverage curve area.
    - partial_AUGRC_70_90 is the normalized generalized_risk area over coverage 0.70–0.90,
      matching Top10%–Top30% review budgets.
    """
    event = np.asarray(event).astype(int)
    risk_score = np.asarray(risk_score).astype(float)
    n = len(event)
    if n == 0:
        return {
            "aurc": float("nan"),
            "augrc": float("nan"),
            "partial_augrc_70_90": float("nan"),
        }

    # Stable tie handling by original order after descending score.
    order = np.lexsort((np.arange(n), -risk_score))
    sorted_event = event[order]

    rows = []
    for reviewed_n in range(0, n + 1):
        residual = sorted_event[reviewed_n:]
        retained_n = n - reviewed_n
        retained_events = int(residual.sum()) if retained_n > 0 else 0
        coverage = retained_n / n

        selective_risk = retained_events / retained_n if retained_n > 0 else 0.0
        generalized_risk = retained_events / n

        rows.append((coverage, selective_risk, generalized_risk))

    curve = pd.DataFrame(rows, columns=["coverage", "selective_risk", "generalized_risk"])
    curve = curve.sort_values("coverage")

    x = curve["coverage"].to_numpy(dtype=float)
    selective = curve["selective_risk"].to_numpy(dtype=float)
    generalized = curve["generalized_risk"].to_numpy(dtype=float)

    return {
        "aurc": float(trapz_area(selective, x)),
        "augrc": float(trapz_area(generalized, x)),
        "partial_augrc_70_90": _normalized_partial_trapz(x, generalized, 0.70, 0.90),
    }


def topk_metrics(event: np.ndarray, risk_score: np.ndarray, budget: float = 0.20) -> Dict[str, float]:
    event = np.asarray(event).astype(int)
    risk_score = np.asarray(risk_score).astype(float)
    n = len(event)
    top_k = int(math.ceil(n * budget))
    total_events = int(event.sum())

    order = np.lexsort((np.arange(n), -risk_score))
    captured = int(event[order[:top_k]].sum())
    residual = total_events - captured

    return {
        "budget": budget,
        "top_k": top_k,
        "total_event": total_events,
        "top20_event_recall": captured / total_events if total_events > 0 else float("nan"),
        "top20_captured_event_count": captured,
        "top20_residual_event_count": residual,
        "top20_residual_event_rate": residual / n if n > 0 else float("nan"),
    }


def build_risk_scores(df: pd.DataFrame) -> Dict[str, pd.Series]:
    scores: Dict[str, pd.Series] = {}

    if "confidence" in df.columns:
        scores["confidence_only_1msp"] = 1.0 - df["confidence"].astype(float)

    if "margin" in df.columns:
        # Smaller margin = more uncertain = higher review risk.
        scores["negative_margin"] = -df["margin"].astype(float)

    if "entropy" in df.columns:
        scores["entropy"] = df["entropy"].astype(float)

    if "expected_gap" in df.columns:
        scores["expected_gap"] = df["expected_gap"].astype(float)

    if "severe_prob_mass" in df.columns:
        scores["severe_prob_mass"] = df["severe_prob_mass"].astype(float)

    if "gated_severe_prob_mass" in df.columns:
        scores["gated_severe_prob_mass_only"] = df["gated_severe_prob_mass"].astype(float)

    return scores


def rank_methods(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "backbone", "event"]

    for keys, g in table.groupby(group_cols, dropna=False):
        gg = g.copy()

        # Lower is better for AURC/AUGRC; higher is better for top20 recall and AUROC-Error.
        gg["rank_aurc"] = gg["aurc"].rank(method="min", ascending=True, na_option="bottom")
        gg["rank_augrc"] = gg["augrc"].rank(method="min", ascending=True, na_option="bottom")
        gg["rank_partial_augrc_70_90"] = gg["partial_augrc_70_90"].rank(
            method="min", ascending=True, na_option="bottom"
        )
        gg["rank_top20_event_recall"] = gg["top20_event_recall"].rank(
            method="min", ascending=False, na_option="bottom"
        )
        gg["rank_auroc_error"] = gg["auroc_error"].rank(
            method="min", ascending=False, na_option="bottom"
        )

        gg["rank_changed_aurc_vs_augrc"] = gg["rank_aurc"] != gg["rank_augrc"]
        gg["rank_changed_augrc_vs_partial"] = gg["rank_augrc"] != gg["rank_partial_augrc_70_90"]

        rows.append(gg)

    return pd.concat(rows, ignore_index=True)


def sanity_check() -> None:
    # Perfect ranking should have lower AUGRC than reversed ranking.
    event = np.array([1, 1, 0, 0, 0, 0])
    perfect = np.array([0.99, 0.98, 0.3, 0.2, 0.1, 0.0])
    reversed_score = -perfect

    p = metric_curves(event, perfect)["augrc"]
    r = metric_curves(event, reversed_score)["augrc"]
    if not p < r:
        raise RuntimeError(f"AUGRC sanity check failed: perfect={p}, reversed={r}")

    auc_p = safe_auc_binary(event, perfect)
    auc_r = safe_auc_binary(event, reversed_score)
    if not auc_p > auc_r:
        raise RuntimeError(f"AUROC sanity check failed: perfect={auc_p}, reversed={auc_r}")



def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal markdown table writer to avoid optional pandas tabulate dependency."""
    if df.empty:
        return "_No rows._"

    d = df.copy()
    if index:
        d = d.reset_index()

    cols = list(d.columns)

    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in d.iterrows():
        rows.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")

    return "\n".join([header, sep] + rows)

def write_key_findings(table: pd.DataFrame, rank_table: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# v0.7.2 Metric-Sensitivity Audit")
    lines.append("")
    lines.append("本版本是 secondary metric audit，不替代 v0.7.1b 的 primary analysis。")
    lines.append("")
    lines.append("目的：检查外部 DR review-ranking 结论是否依赖单一评价口径，尤其是 AURC、AUGRC、partial AUGRC 与固定 Top20% 工作点之间是否一致。")
    lines.append("")
    lines.append("## 指标口径")
    lines.append("")
    lines.append("- 高 risk score 表示更应优先复核。")
    lines.append("- AURC：selective risk-coverage 曲线面积。")
    lines.append("- AUGRC：generalized risk-coverage 曲线面积。")
    lines.append("- partial AUGRC 0.70–0.90：coverage 0.70–0.90 区间的归一化 generalized risk 面积，对应 Top10%–Top30% 复核预算范围。")
    lines.append("- Top20% event recall / residual event count：固定复核预算下的工作点指标。")
    lines.append("")
    lines.append("## 排名变化概览")
    lines.append("")

    summary = (
        rank_table.groupby(["dataset", "event"], dropna=False)
        .agg(
            n_rows=("method", "count"),
            n_changed_aurc_vs_augrc=("rank_changed_aurc_vs_augrc", "sum"),
            n_changed_augrc_vs_partial=("rank_changed_augrc_vs_partial", "sum"),
        )
        .reset_index()
    )
    lines.append(df_to_markdown(summary, index=False))
    lines.append("")

    lines.append("## VTDR miss 下的 Top-ranked methods")
    lines.append("")
    vt = rank_table[rank_table["event"] == "vtdr_miss"].copy()
    if not vt.empty:
        best = (
            vt.sort_values(["dataset", "backbone", "rank_partial_augrc_70_90", "rank_top20_event_recall"])
            .groupby(["dataset", "backbone"], as_index=False)
            .head(1)
        )
        cols = [
            "dataset",
            "backbone",
            "method",
            "auroc_error",
            "aurc",
            "augrc",
            "partial_augrc_70_90",
            "top20_event_recall",
            "top20_residual_event_count",
        ]
        lines.append(df_to_markdown(best[cols], index=False))
    else:
        lines.append("No VTDR miss rows found.")
    lines.append("")

    lines.append("## 解释边界")
    lines.append("")
    lines.append("- AUGRC 不是临床效用指标，也不是临床安全证明。")
    lines.append("- AURC 不是错误指标；AUGRC 是补充敏感性评价。")
    lines.append("- 跨 backbone 比较会混合基础分类准确率与 risk score 排序能力，因此主要作为 descriptive audit。")
    lines.append("- 本结果仍是 image-level retrospective audit，不是 patient-level clinical workflow validation。")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="experiments/summary/v0_7_1/external_dr_direct_inference_predictions.csv",
    )
    parser.add_argument("--out-dir", default="experiments/summary/v0_7_2")
    parser.add_argument("--budget", type=float, default=0.20)
    args = parser.parse_args()

    sanity_check()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)

    required = {"dataset", "image_key", "true_grade", "pred_grade", "backbone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []

    for (dataset, backbone), g in df.groupby(["dataset", "backbone"], dropna=False):
        g = g.copy()
        acc = float((g["pred_grade"] == g["true_grade"]).mean())
        scores = build_risk_scores(g)

        for event_name, event_fn in EVENT_DEFS.items():
            event = event_fn(g).to_numpy(dtype=int)

            for method, score_s in scores.items():
                score = score_s.to_numpy(dtype=float)

                curve = metric_curves(event, score)
                topk = topk_metrics(event, score, budget=args.budget)
                auroc_error = safe_auc_binary(event, score)

                rows.append(
                    {
                        "dataset": dataset,
                        "backbone": backbone,
                        "event": event_name,
                        "method": method,
                        "n": len(g),
                        "accuracy": acc,
                        "event_count": int(event.sum()),
                        "event_rate": float(event.mean()),
                        "auroc_error": auroc_error,
                        **curve,
                        **topk,
                    }
                )

    table = pd.DataFrame(rows)
    rank_table = rank_methods(table)

    audit_csv = out_dir / "v072_metric_sensitivity_audit_table.csv"
    rank_csv = out_dir / "v072_method_rank_comparison.csv"
    key_md = out_dir / "v072_metric_sensitivity_key_findings.md"
    readme = out_dir / "README.md"

    table.to_csv(audit_csv, index=False)
    rank_table.to_csv(rank_csv, index=False)
    write_key_findings(table, rank_table, key_md)

    readme.write_text(
        "# v0.7.2 Metric-Sensitivity Audit\n\n"
        "本目录记录 OphAgent v0.7.2 的 secondary metric audit。\n\n"
        "本版本只读取已有外部 DR direct inference predictions，不重跑模型，不替代 v0.7.1b primary conclusion。\n\n"
        "核心问题：AURC、AUGRC、partial AUGRC 0.70–0.90 与固定 Top20% 工作点是否给出一致或冲突的方法排序。\n\n"
        "输出文件：\n\n"
        "- `v072_metric_sensitivity_audit_table.csv`\n"
        "- `v072_method_rank_comparison.csv`\n"
        "- `v072_metric_sensitivity_key_findings.md`\n\n"
        "边界：AUGRC 是 failure-ranking / selective-classification 的补充评价，不是临床效用指标，不证明临床部署安全。\n",
        encoding="utf-8",
    )

    print("saved:", audit_csv)
    print("saved:", rank_csv)
    print("saved:", key_md)
    print("saved:", readme)


if __name__ == "__main__":
    main()
