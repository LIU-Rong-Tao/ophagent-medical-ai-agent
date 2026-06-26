#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


PROB_COLS = [f"prob_{i}" for i in range(5)]


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    probs = out[PROB_COLS].to_numpy(dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    out["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["entropy"] = entropy_from_probs(probs)
    out["correct"] = out["pred_label"].astype(int) == out["true_label"].astype(int)
    return out


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "name": name,
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n_error": int((y_true != y_pred).sum()),
    }


def load_enabled_predictions(registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    enabled = registry[registry["enabled"].astype(int) == 1].copy()
    cache: dict[str, pd.DataFrame] = {}
    out: dict[str, pd.DataFrame] = {}

    for _, r in enabled.iterrows():
        model_name = r["model_name"]
        path = r["prediction_csv"]
        if path not in cache:
            cache[path] = read_csv(path)
        sub = cache[path][cache[path]["model_name"] == model_name].copy()
        if sub.empty:
            raise ValueError(f"no prediction rows for model: {model_name}")
        out[model_name] = add_derived_columns(sub.sort_values("image_key").reset_index(drop=True))

    return out


def single_model_summary(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_name, df in preds.items():
        y_true = df["true_label"].to_numpy(dtype=int)
        y_pred = df["pred_label"].to_numpy(dtype=int)
        r = metric_row(model_name, y_true, y_pred)
        r["mean_confidence"] = float(df["confidence"].mean())
        r["mean_margin"] = float(df["margin"].mean())
        r["mean_entropy"] = float(df["entropy"].mean())
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1", "qwk"], ascending=False)


def ensemble_summary(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    names = list(preds.keys())
    y_true = preds[names[0]]["true_label"].to_numpy(dtype=int)

    for k in range(2, len(names) + 1):
        for group in combinations(names, k):
            probs = np.mean([preds[m][PROB_COLS].to_numpy(dtype=float) for m in group], axis=0)
            y_pred = probs.argmax(axis=1)
            r = metric_row("+".join(group), y_true, y_pred)
            r["n_models"] = k
            rows.append(r)

    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1", "qwk"], ascending=False)


def pairwise_complementarity(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    names = list(preds.keys())

    for a, b in combinations(names, 2):
        da = preds[a]
        db = preds[b]

        a_correct = da["correct"].to_numpy(dtype=bool)
        b_correct = db["correct"].to_numpy(dtype=bool)

        a_wrong = ~a_correct
        b_wrong = ~b_correct

        a_wrong_b_correct = int((a_wrong & b_correct).sum())
        b_wrong_a_correct = int((b_wrong & a_correct).sum())
        both_wrong = int((a_wrong & b_wrong).sum())
        both_correct = int((a_correct & b_correct).sum())

        rows.append({
            "model_a": a,
            "model_b": b,
            "n_images": int(len(da)),
            "a_errors": int(a_wrong.sum()),
            "b_errors": int(b_wrong.sum()),
            "a_wrong_b_correct": a_wrong_b_correct,
            "b_wrong_a_correct": b_wrong_a_correct,
            "both_wrong": both_wrong,
            "both_correct": both_correct,
            "a_error_fixable_by_b_rate": float(a_wrong_b_correct / max(1, a_wrong.sum())),
            "b_error_fixable_by_a_rate": float(b_wrong_a_correct / max(1, b_wrong.sum())),
            "disagreement_rate": float((da["pred_label"].to_numpy() != db["pred_label"].to_numpy()).mean()),
        })

    return pd.DataFrame(rows).sort_values(
        ["a_wrong_b_correct", "b_wrong_a_correct", "disagreement_rate"],
        ascending=False,
    )


def write_findings(out_dir: Path, single: pd.DataFrame, ens: pd.DataFrame, pair: pd.DataFrame):
    best_single = single.iloc[0]
    best_ens = ens.iloc[0] if len(ens) else None

    lines = []
    lines.append("# v0.8.1 Unified Orchestration Evaluator 初版结果\n")
    lines.append("## 1. 当前范围\n")
    lines.append("本轮只实现基础统一评测器：单模型性能、静态 ensemble、pairwise complementarity。")
    lines.append("暂不包含 sparse routing curve、risk enrichment 和 cost-performance frontier。\n")

    lines.append("## 2. 单模型最佳结果\n")
    lines.append(
        f"- 当前最佳单模型为 `{best_single['name']}`，"
        f"accuracy={best_single['accuracy']:.4f}，"
        f"macro_f1={best_single['macro_f1']:.4f}，"
        f"QWK={best_single['qwk']:.4f}。"
    )

    if best_ens is not None:
        lines.append("\n## 3. 静态 ensemble 最佳结果\n")
        lines.append(
            f"- 当前最佳静态 ensemble 为 `{best_ens['name']}`，"
            f"accuracy={best_ens['accuracy']:.4f}，"
            f"macro_f1={best_ens['macro_f1']:.4f}，"
            f"QWK={best_ens['qwk']:.4f}。"
        )

    lines.append("\n## 4. 互补性观察\n")
    if len(pair):
        top = pair.iloc[0]
        lines.append(
            f"- `{top['model_a']}` 与 `{top['model_b']}` 的互补性最高；"
            f"`{top['model_a']}` 错而 `{top['model_b']}` 对的样本数为 {int(top['a_wrong_b_correct'])}，"
            f"`{top['model_b']}` 错而 `{top['model_a']}` 对的样本数为 {int(top['b_wrong_a_correct'])}。"
        )
    else:
        lines.append("- 当前模型数量不足以计算 pairwise complementarity。")

    lines.append("\n## 5. 下一步\n")
    lines.append("- 在该基础 evaluator 上继续补 sparse routing curve、random baseline、oracle upper bound 与 cost-performance frontier。")

    (out_dir / "key_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        default="experiments/v0_8_1_unified_orchestration/configs/model_registry.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/v0_8_1_unified_orchestration/outputs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = read_csv(args.registry)
    preds = load_enabled_predictions(registry)

    single = single_model_summary(preds)
    ens = ensemble_summary(preds)
    pair = pairwise_complementarity(preds)

    single.to_csv(out_dir / "single_model_summary.csv", index=False)
    ens.to_csv(out_dir / "static_ensemble_summary.csv", index=False)
    pair.to_csv(out_dir / "pairwise_complementarity.csv", index=False)

    write_findings(out_dir, single, ens, pair)

    print("[DONE] v0.8.1 basic evaluator")
    print("outputs:", out_dir)
    print("\nSingle model summary:")
    print(single.to_string(index=False))
    print("\nStatic ensemble summary:")
    print(ens.to_string(index=False))
    print("\nPairwise complementarity:")
    print(pair.to_string(index=False))


if __name__ == "__main__":
    main()
