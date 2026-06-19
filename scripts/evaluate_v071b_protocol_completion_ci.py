#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v0.7.1b protocol completion and clustered CI evaluation.

主目标：
VTDR miss / Top20% / gated_severe_prob_mass_only vs random gate-only.

本脚本不在外部数据上训练、拟合、重标定任何模型。
本脚本不做 CAM、SHAP、attention map 或其他 XAI 支线。
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
PRIMARY_BUDGET = 0.20
TARGET_NAME = "vtdr_miss"


def stable_seed(*items: object, base_seed: int = 42) -> int:
    text = "||".join(str(x) for x in (base_seed, *items))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v0.7.1b random gate-only baseline and image-clustered bootstrap CI."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("experiments/summary/v0_7_1/external_dr_direct_inference_predictions.csv"),
        help="v0.7.1 external direct inference predictions CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/summary/v0_7_1b"),
        help="Output directory.",
    )
    parser.add_argument("--n-random", type=int, default=2000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=DEFAULT_BUDGETS,
        help="Review budgets.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def resolve_image_column(df: pd.DataFrame) -> str:
    candidates = [
        "image_key",
        "relative_image_path",
        "image_path",
        "relative_path",
        "rel_path",
        "path",
        "filepath",
        "file_path",
        "filename",
        "image_name",
        "image_id",
        "case_id",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    image_tokens = (".png", ".jpg", ".jpeg", ".tif", ".tiff", "/")
    for c in object_cols:
        sample = df[c].dropna().astype(str).head(50).tolist()
        if sample and any(any(tok in s.lower() for tok in image_tokens) for s in sample):
            return c

    raise ValueError(
        "无法识别图像 ID 列。请确认 CSV 中存在 image_path / relative_image_path / filename / image_id 等字段。"
    )


def prepare_predictions(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    require_columns(df, ["dataset", "backbone", "true_grade", "pred_grade"])

    out = df.copy()
    out["true_grade"] = pd.to_numeric(out["true_grade"], errors="raise").astype(int)
    out["pred_grade"] = pd.to_numeric(out["pred_grade"], errors="raise").astype(int)

    image_col = resolve_image_column(out)
    out["__image_key__"] = out["dataset"].astype(str) + "::" + out[image_col].astype(str)

    if "severe_prob_mass" in out.columns:
        out["__severe_prob_mass__"] = pd.to_numeric(out["severe_prob_mass"], errors="raise")
    else:
        require_columns(out, ["prob_3", "prob_4"])
        out["__severe_prob_mass__"] = (
            pd.to_numeric(out["prob_3"], errors="raise")
            + pd.to_numeric(out["prob_4"], errors="raise")
        )

    # 重新计算 gated score，避免旧 CSV 中 gate 编码习惯不一致。
    out["__gated_severe_prob_mass__"] = np.where(
        out["pred_grade"].to_numpy() <= 2,
        out["__severe_prob_mass__"].to_numpy(),
        -np.inf,
    )

    # VTDR miss：模型输出相关事件，不使用真实标签参与排序，只用于后验验证。
    out["__target_event__"] = (out["true_grade"] >= 3) & (out["pred_grade"] < 3)

    prob_cols = [f"prob_{i}" for i in range(5)]

    if "confidence" in out.columns:
        out["__low_confidence_score__"] = -pd.to_numeric(out["confidence"], errors="coerce")
    else:
        require_columns(out, prob_cols)
        out["__low_confidence_score__"] = -out[prob_cols].max(axis=1)

    if "margin" in out.columns:
        out["__low_margin_score__"] = -pd.to_numeric(out["margin"], errors="coerce")
    else:
        require_columns(out, prob_cols)
        probs = out[prob_cols].to_numpy(dtype=float)
        sorted_probs = np.sort(probs, axis=1)
        out["__low_margin_score__"] = -(sorted_probs[:, -1] - sorted_probs[:, -2])

    if "entropy" in out.columns:
        out["__entropy_score__"] = pd.to_numeric(out["entropy"], errors="coerce")
    else:
        require_columns(out, prob_cols)
        probs = np.clip(out[prob_cols].to_numpy(dtype=float), 1e-12, 1.0)
        out["__entropy_score__"] = -(probs * np.log(probs)).sum(axis=1)

    if "expected_gap" in out.columns:
        out["__expected_gap_score__"] = pd.to_numeric(out["expected_gap"], errors="coerce")
    elif "expected_grade" in out.columns:
        out["__expected_gap_score__"] = (
            pd.to_numeric(out["expected_grade"], errors="coerce") - out["pred_grade"]
        )
    else:
        require_columns(out, prob_cols)
        probs = out[prob_cols].to_numpy(dtype=float)
        out["__expected_gap_score__"] = probs @ np.arange(5) - out["pred_grade"].to_numpy()

    # ordinal diagnostic baseline：预测等级越低越先看。
    out["__pred_grade_ascending_score__"] = -out["pred_grade"]

    out = out.sort_values(["dataset", "backbone", "__image_key__"]).reset_index(drop=True)
    return out, image_col


def top_k(n: int, budget: float) -> int:
    return int(math.ceil(n * budget)) if n > 0 else 0


def deterministic_topk_metric(group: pd.DataFrame, score_col: str, budget: float) -> Dict[str, float]:
    n = len(group)
    k = top_k(n, budget)
    total_event = int(group["__target_event__"].sum())

    ranked = group.sort_values(
        by=[score_col, "__image_key__"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected = ranked.head(k)

    captured = int(selected["__target_event__"].sum())
    residual = total_event - captured
    non_reviewed = max(n - k, 0)

    base_event_rate = total_event / n if n else np.nan
    flagged_event_rate = captured / k if k else np.nan
    event_recall = captured / total_event if total_event else np.nan
    residual_event_rate = residual / non_reviewed if non_reviewed else np.nan
    enrichment = flagged_event_rate / base_event_rate if base_event_rate and base_event_rate > 0 else np.nan

    return {
        "n": n,
        "top_k": k,
        "total_event": total_event,
        "captured_event": captured,
        "residual_event_count": residual,
        "non_reviewed_count": non_reviewed,
        "event_recall": event_recall,
        "flagged_event_rate": flagged_event_rate,
        "base_event_rate": base_event_rate,
        "residual_event_rate": residual_event_rate,
        "enrichment_ratio": enrichment,
    }


def gate_only_random_once(group: pd.DataFrame, budget: float, rng: np.random.Generator) -> Dict[str, float]:
    n = len(group)
    k = top_k(n, budget)
    total_event = int(group["__target_event__"].sum())

    eligible_idx = group.index[group["pred_grade"] <= 2].to_numpy()
    non_eligible_idx = group.index[group["pred_grade"] > 2].to_numpy()

    if len(eligible_idx) >= k:
        selected_idx = rng.choice(eligible_idx, size=k, replace=False)
    else:
        need = k - len(eligible_idx)
        extra = rng.choice(non_eligible_idx, size=need, replace=False) if need > 0 else np.array([], dtype=int)
        selected_idx = np.concatenate([eligible_idx, extra])

    selected = group.loc[selected_idx]
    captured = int(selected["__target_event__"].sum())
    residual = total_event - captured
    non_reviewed = max(n - len(selected_idx), 0)

    base_event_rate = total_event / n if n else np.nan
    flagged_event_rate = captured / len(selected_idx) if len(selected_idx) else np.nan
    event_recall = captured / total_event if total_event else np.nan
    residual_event_rate = residual / non_reviewed if non_reviewed else np.nan
    enrichment = flagged_event_rate / base_event_rate if base_event_rate and base_event_rate > 0 else np.nan

    return {
        "n": n,
        "top_k": k,
        "total_event": total_event,
        "captured_event": captured,
        "residual_event_count": residual,
        "non_reviewed_count": non_reviewed,
        "event_recall": event_recall,
        "flagged_event_rate": flagged_event_rate,
        "base_event_rate": base_event_rate,
        "residual_event_rate": residual_event_rate,
        "enrichment_ratio": enrichment,
    }


def gate_only_expected_metric(group: pd.DataFrame, budget: float) -> Dict[str, float]:
    n = len(group)
    k = top_k(n, budget)
    total_event = float(group["__target_event__"].sum())

    eligible_n = int((group["pred_grade"] <= 2).sum())

    # VTDR 事件天然在 eligible set 内。
    if total_event == 0:
        expected_captured = 0.0
    elif eligible_n >= k:
        expected_captured = k * total_event / eligible_n if eligible_n else 0.0
    else:
        expected_captured = total_event

    expected_residual = total_event - expected_captured
    non_reviewed = max(n - k, 0)

    base_event_rate = total_event / n if n else np.nan
    flagged_event_rate = expected_captured / k if k else np.nan
    event_recall = expected_captured / total_event if total_event else np.nan
    residual_event_rate = expected_residual / non_reviewed if non_reviewed else np.nan
    enrichment = flagged_event_rate / base_event_rate if base_event_rate and base_event_rate > 0 else np.nan

    return {
        "n": n,
        "top_k": k,
        "eligible_count": eligible_n,
        "total_event": total_event,
        "captured_event": expected_captured,
        "residual_event_count": expected_residual,
        "non_reviewed_count": non_reviewed,
        "event_recall": event_recall,
        "flagged_event_rate": flagged_event_rate,
        "base_event_rate": base_event_rate,
        "residual_event_rate": residual_event_rate,
        "enrichment_ratio": enrichment,
    }


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }


def compute_random_gate_baseline(
    df: pd.DataFrame,
    budgets: Sequence[float],
    n_random: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for dataset, ddf in df.groupby("dataset", sort=True):
        for backbone, bdf in ddf.groupby("backbone", sort=True):
            bdf = bdf.reset_index(drop=True)

            for budget in budgets:
                rng = np.random.default_rng(stable_seed(dataset, backbone, budget, "random_gate", base_seed=seed))
                samples = {
                    "event_recall": [],
                    "captured_event": [],
                    "residual_event_count": [],
                    "residual_event_rate": [],
                    "enrichment_ratio": [],
                }

                for _ in range(n_random):
                    m = gate_only_random_once(bdf, budget, rng)
                    for key in samples:
                        samples[key].append(float(m[key]))

                expected = gate_only_expected_metric(bdf, budget)

                row: Dict[str, object] = {
                    "dataset": dataset,
                    "backbone": backbone,
                    "target": TARGET_NAME,
                    "method": "random_gate_only",
                    "budget": budget,
                    "n_random": n_random,
                    "n": int(expected["n"]),
                    "top_k": int(expected["top_k"]),
                    "eligible_count": int(expected["eligible_count"]),
                    "total_event": int(expected["total_event"]),
                }

                for metric, vals in samples.items():
                    stats = summarize(vals)
                    row[f"{metric}_mean"] = stats["mean"]
                    row[f"{metric}_ci95_low"] = stats["ci95_low"]
                    row[f"{metric}_ci95_high"] = stats["ci95_high"]

                row["event_recall_expected"] = expected["event_recall"]
                row["captured_event_expected"] = expected["captured_event"]
                row["residual_event_count_expected"] = expected["residual_event_count"]
                row["residual_event_rate_expected"] = expected["residual_event_rate"]
                row["enrichment_ratio_expected"] = expected["enrichment_ratio"]

                rows.append(row)

    return pd.DataFrame(rows).sort_values(["dataset", "backbone", "budget"]).reset_index(drop=True)


def compute_budget_curve(
    df: pd.DataFrame,
    random_baseline: pd.DataFrame,
    budgets: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    deterministic_methods = {
        "gated_severe_prob_mass_only": "__gated_severe_prob_mass__",
        "entropy_only": "__entropy_score__",
        "low_confidence_only": "__low_confidence_score__",
        "low_margin_only": "__low_margin_score__",
        "expected_gap_only": "__expected_gap_score__",
        "pred_grade_ascending": "__pred_grade_ascending_score__",
    }

    for dataset, ddf in df.groupby("dataset", sort=True):
        for backbone, bdf in ddf.groupby("backbone", sort=True):
            bdf = bdf.reset_index(drop=True)

            for budget in budgets:
                for method, score_col in deterministic_methods.items():
                    m = deterministic_topk_metric(bdf, score_col, budget)
                    rows.append(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "target": TARGET_NAME,
                            "method": method,
                            "budget": budget,
                            "n": int(m["n"]),
                            "top_k": int(m["top_k"]),
                            "total_event": int(m["total_event"]),
                            "captured_event": float(m["captured_event"]),
                            "captured_event_ci95_low": np.nan,
                            "captured_event_ci95_high": np.nan,
                            "event_recall": float(m["event_recall"]),
                            "event_recall_ci95_low": np.nan,
                            "event_recall_ci95_high": np.nan,
                            "residual_event_count": float(m["residual_event_count"]),
                            "residual_event_count_ci95_low": np.nan,
                            "residual_event_count_ci95_high": np.nan,
                            "residual_event_rate": float(m["residual_event_rate"]),
                            "residual_event_rate_ci95_low": np.nan,
                            "residual_event_rate_ci95_high": np.nan,
                            "enrichment_ratio": float(m["enrichment_ratio"]),
                            "enrichment_ratio_ci95_low": np.nan,
                            "enrichment_ratio_ci95_high": np.nan,
                        }
                    )

                rb = random_baseline[
                    (random_baseline["dataset"].eq(dataset))
                    & (random_baseline["backbone"].eq(backbone))
                    & (np.isclose(random_baseline["budget"].astype(float), float(budget)))
                ]

                if len(rb) != 1:
                    raise RuntimeError(f"random gate-only row missing: {dataset}/{backbone}/budget={budget}")

                r = rb.iloc[0]
                rows.append(
                    {
                        "dataset": dataset,
                        "backbone": backbone,
                        "target": TARGET_NAME,
                        "method": "random_gate_only",
                        "budget": budget,
                        "n": int(r["n"]),
                        "top_k": int(r["top_k"]),
                        "total_event": int(r["total_event"]),
                        "captured_event": float(r["captured_event_mean"]),
                        "captured_event_ci95_low": float(r["captured_event_ci95_low"]),
                        "captured_event_ci95_high": float(r["captured_event_ci95_high"]),
                        "event_recall": float(r["event_recall_mean"]),
                        "event_recall_ci95_low": float(r["event_recall_ci95_low"]),
                        "event_recall_ci95_high": float(r["event_recall_ci95_high"]),
                        "residual_event_count": float(r["residual_event_count_mean"]),
                        "residual_event_count_ci95_low": float(r["residual_event_count_ci95_low"]),
                        "residual_event_count_ci95_high": float(r["residual_event_count_ci95_high"]),
                        "residual_event_rate": float(r["residual_event_rate_mean"]),
                        "residual_event_rate_ci95_low": float(r["residual_event_rate_ci95_low"]),
                        "residual_event_rate_ci95_high": float(r["residual_event_rate_ci95_high"]),
                        "enrichment_ratio": float(r["enrichment_ratio_mean"]),
                        "enrichment_ratio_ci95_low": float(r["enrichment_ratio_ci95_low"]),
                        "enrichment_ratio_ci95_high": float(r["enrichment_ratio_ci95_high"]),
                    }
                )

    return pd.DataFrame(rows).sort_values(["dataset", "backbone", "budget", "method"]).reset_index(drop=True)


def bootstrap_one_dataset(
    ddf: pd.DataFrame,
    dataset: str,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    image_keys = np.array(sorted(ddf["__image_key__"].unique()))
    by_image = {key: g.copy() for key, g in ddf.groupby("__image_key__", sort=False)}

    rng = np.random.default_rng(stable_seed(dataset, "clustered_bootstrap", base_seed=seed))

    delta_recall_samples: List[float] = []
    delta_residual_count_samples: List[float] = []
    delta_residual_rate_samples: List[float] = []

    for boot_idx in range(n_bootstrap):
        sampled_keys = rng.choice(image_keys, size=len(image_keys), replace=True)

        parts = []
        for draw_idx, key in enumerate(sampled_keys):
            g = by_image[key].copy()
            g["__image_key__"] = g["__image_key__"].astype(str) + f"__boot{boot_idx}_{draw_idx}"
            parts.append(g)

        boot = pd.concat(parts, ignore_index=True)

        per_backbone_recall_delta: List[float] = []
        per_backbone_residual_count_delta: List[float] = []
        per_backbone_residual_rate_delta: List[float] = []

        for backbone, bdf in boot.groupby("backbone", sort=True):
            bdf = bdf.reset_index(drop=True)

            gated = deterministic_topk_metric(bdf, "__gated_severe_prob_mass__", PRIMARY_BUDGET)
            gate_exp = gate_only_expected_metric(bdf, PRIMARY_BUDGET)

            if np.isnan(gated["event_recall"]) or np.isnan(gate_exp["event_recall"]):
                continue

            per_backbone_recall_delta.append(float(gated["event_recall"] - gate_exp["event_recall"]))

            # 正值表示 gated 比 random gate-only 少漏。
            per_backbone_residual_count_delta.append(
                float(gate_exp["residual_event_count"] - gated["residual_event_count"])
            )
            per_backbone_residual_rate_delta.append(
                float(gate_exp["residual_event_rate"] - gated["residual_event_rate"])
            )

        if not per_backbone_recall_delta:
            continue

        delta_recall_samples.append(float(np.mean(per_backbone_recall_delta)))
        delta_residual_count_samples.append(float(np.mean(per_backbone_residual_count_delta)))
        delta_residual_rate_samples.append(float(np.mean(per_backbone_residual_rate_delta)))

    recall = np.asarray(delta_recall_samples, dtype=float)
    residual_count = np.asarray(delta_residual_count_samples, dtype=float)
    residual_rate = np.asarray(delta_residual_rate_samples, dtype=float)

    def mean_ci(arr: np.ndarray) -> Tuple[float, float, float]:
        arr = arr[~np.isnan(arr)]
        return (
            float(np.mean(arr)),
            float(np.quantile(arr, 0.025)),
            float(np.quantile(arr, 0.975)),
        )

    mean_recall, low_recall, high_recall = mean_ci(recall)
    mean_residual_count, low_residual_count, high_residual_count = mean_ci(residual_count)
    mean_residual_rate, low_residual_rate, high_residual_rate = mean_ci(residual_rate)

    return {
        "dataset": dataset,
        "target": TARGET_NAME,
        "budget": PRIMARY_BUDGET,
        "method": "gated_severe_prob_mass_only",
        "comparator": "random_gate_only_expected",
        "n_bootstrap": int(len(recall)),
        "n_unique_images": int(len(image_keys)),
        "mean_delta_event_recall": mean_recall,
        "ci95_low_delta_event_recall": low_recall,
        "ci95_high_delta_event_recall": high_recall,
        "win_rate_delta_gt_0": float(np.mean(recall > 0)),
        "mean_residual_event_count_reduction": mean_residual_count,
        "ci95_low_residual_event_count_reduction": low_residual_count,
        "ci95_high_residual_event_count_reduction": high_residual_count,
        "mean_residual_event_rate_reduction": mean_residual_rate,
        "ci95_low_residual_event_rate_reduction": low_residual_rate,
        "ci95_high_residual_event_rate_reduction": high_residual_rate,
    }


def compute_primary_bootstrap_ci(df: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for dataset, ddf in df.groupby("dataset", sort=True):
        rows.append(bootstrap_one_dataset(ddf.reset_index(drop=True), str(dataset), n_bootstrap, seed))
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def summarize_dataset_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, ddf in df.groupby("dataset", sort=True):
        event_by_backbone = ddf.groupby("backbone")["__target_event__"].sum().astype(int)
        rows.append(
            {
                "dataset": dataset,
                "n_unique_images": int(ddf["__image_key__"].nunique()),
                "n_prediction_rows": int(len(ddf)),
                "n_backbones": int(ddf["backbone"].nunique()),
                "event_count_min_per_backbone": int(event_by_backbone.min()),
                "event_count_mean_per_backbone": float(event_by_backbone.mean()),
                "event_count_max_per_backbone": int(event_by_backbone.max()),
            }
        )

    return pd.DataFrame(rows)


def df_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    display = df.copy()

    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")

    headers = list(display.columns)
    rows = display.astype(str).values.tolist()

    widths = []
    for i, h in enumerate(headers):
        col_values = [r[i] for r in rows]
        widths.append(max(len(str(h)), *(len(v) for v in col_values)))

    def fmt_row(values: List[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    out = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    out.extend(fmt_row(row) for row in rows)
    return "\n".join(out)


def decide_go_no_go(primary_ci: pd.DataFrame) -> Tuple[str, str]:
    rows = {str(row["dataset"]): row for _, row in primary_ci.iterrows()}
    all_positive = all(float(row["mean_delta_event_recall"]) > 0 for row in rows.values())

    mess = rows.get("MESSIDOR2")
    mess_ci_positive = bool(mess is not None and float(mess["ci95_low_delta_event_recall"]) > 0)

    if all_positive and mess_ci_positive:
        return (
            "强证据",
            "两个外部数据集差值点估计均为正，MESSIDOR2 的 95% CI 不跨 0，IDRiD_data 方向一致。",
        )

    if all_positive:
        return (
            "中等证据",
            "两个外部数据集差值点估计均为正，但至少一个数据集的置信区间仍跨 0，需要按方向一致但不确定性较大表述。",
        )

    return (
        "不一致/证据不足",
        "两个数据集方向冲突，或任一数据集点估计不为正，不能声称 severe-class probability mass 提供稳定额外排序信息。",
    )


def write_summary(
    out_path: Path,
    predictions_path: Path,
    df: pd.DataFrame,
    image_col: str,
    dataset_events: pd.DataFrame,
    primary_ci: pd.DataFrame,
) -> None:
    decision, reason = decide_go_no_go(primary_ci)

    lines: List[str] = []
    lines.append("# v0.7.1b Go/No-Go Summary\n")

    lines.append("## 输入\n")
    lines.append(f"- Predictions: `{predictions_path}`")
    lines.append(f"- Prediction rows: {len(df)}")
    lines.append(f"- Image identifier column: `{image_col}`")
    lines.append("- Target: `VTDR miss = true_grade >= 3 and pred_grade < 3`")
    lines.append("- Primary budget: Top20%")
    lines.append("- Primary comparison: `gated_severe_prob_mass_only` vs `random_gate_only`\n")

    lines.append("## 数据集与事件规模\n")
    lines.append(df_to_markdown(dataset_events))
    lines.append("")

    lines.append("## Primary comparison: Top20% recall difference\n")
    show_cols = [
        "dataset",
        "mean_delta_event_recall",
        "ci95_low_delta_event_recall",
        "ci95_high_delta_event_recall",
        "win_rate_delta_gt_0",
        "mean_residual_event_count_reduction",
        "ci95_low_residual_event_count_reduction",
        "ci95_high_residual_event_count_reduction",
        "mean_residual_event_rate_reduction",
        "ci95_low_residual_event_rate_reduction",
        "ci95_high_residual_event_rate_reduction",
    ]
    lines.append(df_to_markdown(primary_ci[show_cols]))
    lines.append("")

    lines.append("## Go/No-Go 判断\n")
    lines.append(f"- 结论等级：**{decision}**")
    lines.append(f"- 判断依据：{reason}\n")

    lines.append("## 展示措辞\n")
    lines.append("- 使用：**未进入优先复核区的残余危险事件**")
    lines.append("- 不使用：自动放行区")
    lines.append("- 病例页标注：公共数据集回顾性 grade-based proxy 示例，不是患者级临床判断。\n")

    lines.append("## 边界\n")
    lines.append("- 公共数据集回顾性评估。")
    lines.append("- grade-based proxy，不是医生定义的患者级临床终点。")
    lines.append("- 外部数据未用于重新拟合、重新选特征或重新标准化。")
    lines.append("- 本结果不是临床部署验证。")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.predictions)
    df, image_col = prepare_predictions(df_raw)

    random_baseline = compute_random_gate_baseline(
        df=df,
        budgets=args.budgets,
        n_random=args.n_random,
        seed=args.seed,
    )

    budget_curve = compute_budget_curve(
        df=df,
        random_baseline=random_baseline,
        budgets=args.budgets,
    )

    primary_ci = compute_primary_bootstrap_ci(
        df=df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    dataset_events = summarize_dataset_events(df)

    random_path = args.out_dir / "v071b_random_gate_only_baseline.csv"
    ci_path = args.out_dir / "v071b_primary_bootstrap_ci.csv"
    curve_path = args.out_dir / "v071b_budget_curve.csv"
    events_path = args.out_dir / "v071b_dataset_event_summary.csv"
    summary_path = args.out_dir / "v071b_go_no_go_summary.md"

    random_baseline.to_csv(random_path, index=False)
    primary_ci.to_csv(ci_path, index=False)
    budget_curve.to_csv(curve_path, index=False)
    dataset_events.to_csv(events_path, index=False)

    write_summary(
        out_path=summary_path,
        predictions_path=args.predictions,
        df=df,
        image_col=image_col,
        dataset_events=dataset_events,
        primary_ci=primary_ci,
    )

    print("Saved:")
    for p in [random_path, ci_path, curve_path, events_path, summary_path]:
        print(f"- {p}")


if __name__ == "__main__":
    main()
