#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd


CFG = Path("experiments/v0_8_2b_controlled_protocols/configs/controlled_protocols.yaml")
OUT_DIR = Path("experiments/v0_8_2b_controlled_protocols/outputs")
PROB_COLS = [f"prob_{i}" for i in range(5)]


RISK_EVENTS = {
    "large_undergrading_dr_specific": {
        "description": "True label is PDR, but base/scout prediction is Moderate or below.",
        "true_min": 4,
        "pred_max": 2,
    },
    "severe_pdr_miss_dr_specific": {
        "description": "True label is Severe/PDR, but base/scout prediction is non-severe.",
        "true_min": 3,
        "pred_max": 2,
    },
    "referable_dr_miss_dr_specific": {
        "description": "True label is referable DR or above, but base/scout prediction is No/Mild DR.",
        "true_min": 2,
        "pred_max": 1,
    },
}


def load_registry_predictions(registry_path: str) -> pd.DataFrame:
    reg = pd.read_csv(registry_path)
    reg = reg[reg["enabled"].astype(int) == 1].copy()

    frames = []
    for _, r in reg.iterrows():
        model = r["model_name"]
        p = r["prediction_csv"]
        df = pd.read_csv(p)
        sub = df[df["model_name"] == model].copy()
        if sub.empty:
            raise ValueError(f"no prediction rows for model: {model}")
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    return out


def model_df(pred: pd.DataFrame, model: str) -> pd.DataFrame:
    d = pred[pred["model_name"] == model].sort_values("image_key").reset_index(drop=True)
    if d.empty:
        raise ValueError(f"missing model: {model}")
    return d


def uncertainty_from_probs(probs: np.ndarray):
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    p = np.clip(probs, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(probs.shape[1])
    return confidence, margin, entropy


def average_prediction(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    dfs = [model_df(pred, m) for m in models]
    base = dfs[0][["image_key", "true_label"]].copy()
    probs = sum(d[PROB_COLS].to_numpy(dtype=float) for d in dfs) / len(dfs)

    confidence, margin, entropy = uncertainty_from_probs(probs)

    out = base.copy()
    out[PROB_COLS] = probs
    out["pred_label"] = probs.argmax(axis=1)
    out["confidence"] = confidence
    out["margin"] = margin
    out["entropy"] = entropy
    return out


def single_scout_base_and_score(pred: pd.DataFrame, scout: str, policy: str) -> pd.DataFrame:
    base = model_df(pred, scout).copy()
    probs = base[PROB_COLS].to_numpy(dtype=float)
    confidence, margin, entropy = uncertainty_from_probs(probs)

    base["confidence"] = confidence
    base["margin"] = margin
    base["entropy"] = entropy

    if policy == "low_confidence":
        base["routing_score"] = 1.0 - base["confidence"]
    elif policy == "high_entropy":
        base["routing_score"] = base["entropy"]
    elif policy == "low_margin":
        base["routing_score"] = -base["margin"]
    else:
        raise ValueError(policy)

    return base


def multi_scout_base_and_score(pred: pd.DataFrame, scouts: list[str], signal: str) -> pd.DataFrame:
    scout_dfs = [model_df(pred, m) for m in scouts]
    base = average_prediction(pred, scouts)

    probs_list = [d[PROB_COLS].to_numpy(dtype=float) for d in scout_dfs]
    pred_list = [d["pred_label"].to_numpy(dtype=int) for d in scout_dfs]

    entropies = []
    margins = []
    confidences = []

    for probs in probs_list:
        conf, margin, entropy = uncertainty_from_probs(probs)
        confidences.append(conf)
        margins.append(margin)
        entropies.append(entropy)

    entropies = np.stack(entropies, axis=1)
    margins = np.stack(margins, axis=1)
    confidences = np.stack(confidences, axis=1)
    pred_stack = np.stack(pred_list, axis=1)

    disagreement = np.array([len(set(row.tolist())) > 1 for row in pred_stack], dtype=float)

    if signal == "mean_uncertainty":
        score = entropies.mean(axis=1)
    elif signal == "max_uncertainty":
        score = entropies.max(axis=1)
    elif signal == "disagreement_then_uncertainty":
        score = disagreement * 10.0 + entropies.mean(axis=1)
    else:
        raise ValueError(signal)

    base["routing_score"] = score
    base["multi_scout_disagreement"] = disagreement.astype(bool)
    base["mean_entropy"] = entropies.mean(axis=1)
    base["max_entropy"] = entropies.max(axis=1)
    base["mean_margin"] = margins.mean(axis=1)
    base["min_confidence"] = confidences.min(axis=1)
    return base


def select_top_budget(base: pd.DataFrame, budget: float) -> set[int]:
    k = int(round(len(base) * float(budget)))
    order = (
        base.reset_index()
        .sort_values(["routing_score", "image_key"], ascending=[False, True])["index"]
        .to_numpy()
    )
    return set(order[:k].tolist())


def event_mask(base: pd.DataFrame, true_min: int, pred_max: int) -> np.ndarray:
    y = base["true_label"].to_numpy(dtype=int)
    p = base["pred_label"].to_numpy(dtype=int)
    return (y >= true_min) & (p <= pred_max)


def risk_rows_for_protocol(protocol_family, protocol_name, role, scouts, experts, budget, policy, base):
    selected = select_top_budget(base, budget)
    selected_mask = np.zeros(len(base), dtype=bool)
    selected_mask[list(selected)] = True

    rows = []
    for event_name, spec in RISK_EVENTS.items():
        mask = event_mask(base, spec["true_min"], spec["pred_max"])
        event_total = int(mask.sum())
        selected_event_n = int((mask & selected_mask).sum())
        residual_event_n = int(event_total - selected_event_n)
        selected_n = int(selected_mask.sum())

        rows.append({
            "protocol_family": protocol_family,
            "protocol_name": protocol_name,
            "role": role,
            "scouts": "+".join(scouts) if isinstance(scouts, list) else scouts,
            "experts": "+".join(experts) if isinstance(experts, list) else experts,
            "budget": float(budget),
            "policy": policy,
            "selected_n": selected_n,
            "event_name": event_name,
            "event_description": spec["description"],
            "event_total": event_total,
            "selected_event_n": selected_event_n,
            "residual_event_n": residual_event_n,
            "event_recall": selected_event_n / event_total if event_total else np.nan,
            "event_precision": selected_event_n / selected_n if selected_n else np.nan,
            "event_lift_vs_budget": (selected_event_n / event_total) / float(budget) if event_total and budget else np.nan,
        })

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    pred = load_registry_predictions(cfg["source"]["prediction_registry"])

    rows = []

    for item in cfg["single_scout_protocols"]:
        scout = item["scout"]
        experts = item["experts"]
        role = item.get("role", "main")

        for budget in item["budgets"]:
            for policy in item["policies"]:
                base = single_scout_base_and_score(pred, scout, policy)
                rows.extend(risk_rows_for_protocol(
                    protocol_family="single_scout",
                    protocol_name=item["name"],
                    role=role,
                    scouts=scout,
                    experts=experts,
                    budget=budget,
                    policy=policy,
                    base=base,
                ))

    for item in cfg["multi_scout_protocols"]:
        scouts = item["scouts"]
        expert = item["expert"]
        role = item.get("role", "main")

        for budget in item["budgets"]:
            for signal in item["routing_signals"]:
                base = multi_scout_base_and_score(pred, scouts, signal)
                rows.extend(risk_rows_for_protocol(
                    protocol_family="multi_scout",
                    protocol_name=item["name"],
                    role=role,
                    scouts=scouts,
                    experts=expert,
                    budget=budget,
                    policy=signal,
                    base=base,
                ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "controlled_risk_event_coverage.csv", index=False)

    main_df = df[df["role"] == "main"].copy()
    main_df.to_csv(OUT_DIR / "controlled_risk_event_main_coverage.csv", index=False)

    best = (
        main_df.sort_values(
            ["event_name", "event_recall", "budget", "selected_n"],
            ascending=[True, False, True, True],
        )
        .groupby("event_name", as_index=False)
        .head(5)
    )
    best.to_csv(OUT_DIR / "controlled_risk_event_best_by_event.csv", index=False)

    print("[DONE] controlled risk event coverage")
    print("\nBest main rows by event:")
    cols = [
        "event_name", "protocol_family", "protocol_name", "scouts", "experts",
        "budget", "policy", "selected_n", "event_total",
        "selected_event_n", "residual_event_n", "event_recall",
        "event_precision", "event_lift_vs_budget",
    ]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
