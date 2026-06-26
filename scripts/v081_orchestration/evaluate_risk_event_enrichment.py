#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


OUT_DIR = Path("experiments/v0_8_1_unified_orchestration/outputs")
EVENT_CONFIG = Path("experiments/v0_8_1_unified_orchestration/configs/event_definitions.yaml")
REGISTRY = Path("experiments/v0_8_1_unified_orchestration/configs/model_registry.csv")
SPARSE_CSV = OUT_DIR / "sparse_routing_curve.csv"

PROB_COLS = [f"prob_{i}" for i in range(5)]


def load_events() -> dict:
    with open(EVENT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["events"]




def load_registry_predictions() -> pd.DataFrame:
    registry = pd.read_csv(REGISTRY)
    enabled = registry[registry["enabled"].astype(int) == 1].copy()

    frames = []
    cache = {}

    for _, r in enabled.iterrows():
        model_name = r["model_name"]
        pred_path = r["prediction_csv"]

        if pred_path not in cache:
            cache[pred_path] = pd.read_csv(pred_path)

        sub = cache[pred_path][cache[pred_path]["model_name"] == model_name].copy()
        if sub.empty:
            raise ValueError(f"no prediction rows for model: {model_name}")
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def add_margin_entropy(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    probs = out[PROB_COLS].to_numpy(dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    out["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    p = np.clip(probs, 1e-12, 1.0)
    out["entropy"] = -(p * np.log(p)).sum(axis=1)
    return out


def select_indices(df: pd.DataFrame, policy: str, k: int):
    if policy == "low_confidence":
        order = df.sort_values(["confidence", "image_key"], ascending=[True, True]).index
    elif policy == "low_margin":
        order = df.sort_values(["margin", "image_key"], ascending=[True, True]).index
    elif policy == "high_entropy":
        order = df.sort_values(["entropy", "image_key"], ascending=[False, True]).index
    else:
        raise ValueError(policy)
    return set(order[:k])


def main():
    events = load_events()
    pred = add_margin_entropy(load_registry_predictions())
    sparse = pd.read_csv(SPARSE_CSV)

    rows = []

    for _, r in sparse.iterrows():
        scout = r["scout"]
        budget = float(r["budget"])
        policy = r["policy"]
        selected_n = int(r["selected_n"])

        scout_df = (
            pred[pred["model_name"] == scout]
            .sort_values("image_key")
            .reset_index(drop=True)
        )

        selected = select_indices(scout_df, policy, selected_n)

        for event_name, cfg in events.items():
            event_mask = (
                (scout_df["true_label"].astype(int) >= int(cfg["true_min"]))
                & (scout_df["pred_label"].astype(int) <= int(cfg["pred_max"]))
            )

            event_idx = set(scout_df.index[event_mask].tolist())
            event_total = len(event_idx)
            selected_event_n = len(event_idx & selected)

            rows.append({
                "setting": r["setting"],
                "scout": scout,
                "experts": r["experts"],
                "budget": budget,
                "policy": policy,
                "selected_n": selected_n,
                "event_name": event_name,
                "event_total": event_total,
                "selected_event_n": selected_event_n,
                "event_recall": selected_event_n / event_total if event_total else 0.0,
                "event_precision": selected_event_n / selected_n if selected_n else 0.0,
                "event_lift_vs_budget": (selected_event_n / event_total) / budget if event_total and budget else 0.0,
                "event_definition": cfg["description"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "risk_event_enrichment.csv", index=False)

    key = (
        out.sort_values(["event_name", "event_recall", "event_lift_vs_budget"], ascending=[True, False, False])
        .groupby("event_name")
        .head(5)
        .reset_index(drop=True)
    )
    key.to_csv(OUT_DIR / "risk_event_enrichment_top.csv", index=False)

    lines = []
    lines.append("# v0.8.1 Risk Event Enrichment 初版结果\n")
    lines.append("## 1. 当前范围\n")
    lines.append("本轮将 DR-specific 风险事件接入统一 evaluator。事件定义基于 scout 的预测错误，用于衡量不同 sparse routing policy 是否能优先选中高风险低估样本。\n")
    lines.append("## 2. 事件定义\n")
    for name, cfg in events.items():
        lines.append(f"- `{name}`: {cfg['description']} true_label >= {cfg['true_min']}, pred_label <= {cfg['pred_max']}。")
    lines.append("\n## 3. 每类事件 Top 结果\n")
    for event_name, sub in key.groupby("event_name"):
        best = sub.iloc[0]
        lines.append(
            f"- `{event_name}`: best setting=`{best['setting']}`, policy=`{best['policy']}`, "
            f"budget={best['budget']:.1f}, recall={best['event_recall']:.4f}, "
            f"selected={int(best['selected_event_n'])}/{int(best['event_total'])}。"
        )
    lines.append("\n## 4. 当前边界\n")
    lines.append("- 当前事件是 DR 五分类任务上的 DR-specific 定义，不能直接泛化到其他眼病。")
    lines.append("- 事件基于 scout prediction 定义，因此不同 scout 的 event_total 可能不同。")
    lines.append("- 当前只评估 selected set 是否捕获事件，尚未区分 expert 是否最终修正该事件。")

    (OUT_DIR / "risk_event_enrichment_key_findings.md").write_text("\n".join(lines), encoding="utf-8")

    print("[DONE] risk event enrichment")
    print("\nTop event rows:")
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
