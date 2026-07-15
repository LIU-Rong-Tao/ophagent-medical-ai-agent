#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd


CFG = Path("experiments/v0_8_2c_residual_risk_cases/configs/operating_points.yaml")
OUT_DIR = Path("experiments/v0_8_2c_residual_risk_cases/outputs")
PROB_COLS = [f"prob_{i}" for i in range(5)]


def uncertainty_from_probs(probs: np.ndarray):
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    p = np.clip(probs, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(probs.shape[1])
    return confidence, margin, entropy


def load_predictions(registry_path: str) -> pd.DataFrame:
    reg = pd.read_csv(registry_path)
    reg = reg[reg["enabled"].astype(int) == 1].copy()

    frames = []
    for _, r in reg.iterrows():
        model = r["model_name"]
        df = pd.read_csv(r["prediction_csv"])
        sub = df[df["model_name"] == model].copy()
        if sub.empty:
            raise ValueError(f"Missing prediction rows for model: {model}")
        frames.append(sub)

    return pd.concat(frames, ignore_index=True)


def model_df(pred: pd.DataFrame, model: str) -> pd.DataFrame:
    d = pred[pred["model_name"] == model].sort_values("image_key").reset_index(drop=True)
    if d.empty:
        raise ValueError(f"Missing model: {model}")
    return d


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


def single_scout_base(pred: pd.DataFrame, scout: str, policy: str) -> pd.DataFrame:
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

    base["scout_disagreement"] = False
    return base.sort_values("image_key").reset_index(drop=True)


def multi_scout_base(pred: pd.DataFrame, scouts: list[str], policy: str) -> pd.DataFrame:
    scout_dfs = [model_df(pred, m) for m in scouts]
    base = average_prediction(pred, scouts)

    probs_list = [d[PROB_COLS].to_numpy(dtype=float) for d in scout_dfs]
    pred_list = [d["pred_label"].to_numpy(dtype=int) for d in scout_dfs]

    entropies = []
    for probs in probs_list:
        _, _, entropy = uncertainty_from_probs(probs)
        entropies.append(entropy)

    entropies = np.stack(entropies, axis=1)
    pred_stack = np.stack(pred_list, axis=1)
    disagreement = np.array([len(set(row.tolist())) > 1 for row in pred_stack], dtype=float)

    if policy == "mean_uncertainty":
        score = entropies.mean(axis=1)
    elif policy == "max_uncertainty":
        score = entropies.max(axis=1)
    elif policy == "disagreement_then_uncertainty":
        score = disagreement * 10.0 + entropies.mean(axis=1)
    else:
        raise ValueError(policy)

    base["routing_score"] = score
    base["scout_disagreement"] = disagreement.astype(bool)
    base["mean_scout_entropy"] = entropies.mean(axis=1)
    base["max_scout_entropy"] = entropies.max(axis=1)

    for i, d in enumerate(scout_dfs):
        model = d["model_name"].iloc[0]
        base[f"{model}_pred"] = d["pred_label"].to_numpy(dtype=int)
        base[f"{model}_confidence"] = d["confidence"].to_numpy(dtype=float) if "confidence" in d.columns else d[PROB_COLS].max(axis=1).to_numpy(dtype=float)

    return base.sort_values("image_key").reset_index(drop=True)


def select_top_budget(base: pd.DataFrame, budget: float) -> np.ndarray:
    k = int(round(len(base) * float(budget)))
    order = (
        base.reset_index()
        .sort_values(["routing_score", "image_key"], ascending=[False, True])["index"]
        .to_numpy()
    )
    selected = np.zeros(len(base), dtype=bool)
    selected[order[:k]] = True
    return selected


def event_flags(true_label: np.ndarray, pred_label: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "large_undergrading_event": (true_label >= 4) & (pred_label <= 2),
        "referable_miss_event": (true_label >= 2) & (pred_label <= 1),
        "severe_pdr_miss_event": (true_label >= 3) & (pred_label <= 2),
    }



def build_fixed_union_risk_pools(pred: pd.DataFrame) -> pd.DataFrame:
    """Build the same fixed union risk pools used by v0.8.2b.

    The union pool is defined from main base/scout candidates:
    ConvNeXt, Swin, and ConvNeXt+Swin average-base.
    """
    base_defs = {
        "convnext_base": ["convnext_tiny"],
        "swin_base": ["swin_tiny"],
        "convnext_swin_average_base": ["convnext_tiny", "swin_tiny"],
    }

    bases = {
        name: average_prediction(pred, models).sort_values("image_key").reset_index(drop=True)
        for name, models in base_defs.items()
    }

    ref = bases["convnext_base"][["image_key", "true_label"]].copy()
    y = ref["true_label"].to_numpy(dtype=int)

    event_specs = {
        "fixed_large_undergrading_union_pool": (4, 2),
        "fixed_referable_miss_union_pool": (2, 1),
        "fixed_severe_pdr_miss_union_pool": (3, 2),
    }

    for event_name, (true_min, pred_max) in event_specs.items():
        union_mask = np.zeros(len(ref), dtype=bool)
        for base in bases.values():
            pred_label = base["pred_label"].to_numpy(dtype=int)
            union_mask |= (y >= true_min) & (pred_label <= pred_max)
        ref[event_name] = union_mask

    ref["fixed_any_union_risk_pool"] = (
        ref["fixed_large_undergrading_union_pool"]
        | ref["fixed_referable_miss_union_pool"]
        | ref["fixed_severe_pdr_miss_union_pool"]
    )

    return ref.sort_values("image_key").reset_index(drop=True)



def build_case_table_for_op(pred: pd.DataFrame, op: dict, fixed_pool: pd.DataFrame) -> pd.DataFrame:
    expert = op["expert"]
    expert_df = model_df(pred, expert)

    if op["family"] == "single_scout":
        scout = op["scout"]
        base = single_scout_base(pred, scout, op["policy"])
        scouts_label = scout
    elif op["family"] == "multi_scout":
        scouts = op["scouts"]
        base = multi_scout_base(pred, scouts, op["policy"])
        scouts_label = "+".join(scouts)
    else:
        raise ValueError(op["family"])

    assert np.array_equal(base["image_key"].to_numpy(), expert_df["image_key"].to_numpy())

    selected = select_top_budget(base, op["budget"])

    y = base["true_label"].to_numpy(dtype=int)
    base_pred = base["pred_label"].to_numpy(dtype=int)
    expert_pred = expert_df["pred_label"].to_numpy(dtype=int)

    final_pred = base_pred.copy()
    final_pred[selected] = expert_pred[selected]

    base_correct = base_pred == y
    expert_correct = expert_pred == y
    final_correct = final_pred == y

    expert_corrected = selected & (~base_correct) & expert_correct
    expert_induced_error = selected & base_correct & (~expert_correct)

    base_events = event_flags(y, base_pred)
    final_events = event_flags(y, final_pred)

    any_base_risk = np.zeros(len(base), dtype=bool)
    any_final_risk = np.zeros(len(base), dtype=bool)
    for k in base_events:
        any_base_risk |= base_events[k]
        any_final_risk |= final_events[k]

    residual_risk_after_routing = (~selected) & (any_base_risk | (~base_correct))
    residual_danger_after_routing = any_final_risk | (~final_correct)

    out = pd.DataFrame({
        "protocol_name": op["protocol_name"],
        "operating_point": op["name"],
        "role": op["role"],
        "family": op["family"],
        "scouts": scouts_label,
        "expert": expert,
        "budget": float(op["budget"]),
        "policy": op["policy"],
        "image_key": base["image_key"],
        "case_id": base["image_key"],
        "true_label": y,
        "scout_pred": base_pred,
        "expert_pred": expert_pred,
        "final_pred": final_pred,
        "scout_confidence": base["confidence"].to_numpy(dtype=float),
        "scout_margin": base["margin"].to_numpy(dtype=float),
        "scout_entropy": base["entropy"].to_numpy(dtype=float),
        "routing_score": base["routing_score"].to_numpy(dtype=float),
        "scout_disagreement": base["scout_disagreement"].to_numpy(dtype=bool),
        "selected_for_expert": selected,
        "base_correct": base_correct,
        "expert_correct": expert_correct,
        "final_correct": final_correct,
        "expert_corrected": expert_corrected,
        "expert_induced_error": expert_induced_error,
        "base_error": ~base_correct,
        "final_error": ~final_correct,
        "residual_risk_after_routing": residual_risk_after_routing,
        "residual_danger_after_routing": residual_danger_after_routing,
    })

    for k, v in base_events.items():
        out[k] = v
        out[f"residual_{k}"] = (~selected) & v
        out[f"final_{k}"] = final_events[k]

    extra_cols = [c for c in base.columns if c.endswith("_pred") or c.endswith("_confidence")]
    for c in extra_cols:
        out[c] = base[c].to_numpy()

    fixed_pool = fixed_pool.sort_values("image_key").reset_index(drop=True)
    assert np.array_equal(out["image_key"].to_numpy(), fixed_pool["image_key"].to_numpy())

    for c in [
        "fixed_large_undergrading_union_pool",
        "fixed_referable_miss_union_pool",
        "fixed_severe_pdr_miss_union_pool",
        "fixed_any_union_risk_pool",
    ]:
        out[c] = fixed_pool[c].to_numpy(dtype=bool)
        out[f"selected_{c}"] = out["selected_for_expert"].to_numpy(dtype=bool) & out[c].to_numpy(dtype=bool)
        out[f"residual_{c}"] = (~out["selected_for_expert"].to_numpy(dtype=bool)) & out[c].to_numpy(dtype=bool)

    return out


def summarize_cases(all_cases: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for op, df in all_cases.groupby("operating_point"):
        row = {
            "operating_point": op,
            "protocol_name": df["protocol_name"].iloc[0],
            "role": df["role"].iloc[0],
            "family": df["family"].iloc[0],
            "budget": df["budget"].iloc[0],
            "policy": df["policy"].iloc[0],
            "n_cases": len(df),
            "selected_n": int(df["selected_for_expert"].sum()),
            "n_base_error": int(df["base_error"].sum()),
            "n_final_error": int(df["final_error"].sum()),
            "n_corrected_by_expert": int(df["expert_corrected"].sum()),
            "n_expert_induced_error": int(df["expert_induced_error"].sum()),
            "n_residual_risk_after_routing": int(df["residual_risk_after_routing"].sum()),
            "n_residual_danger_after_routing": int(df["residual_danger_after_routing"].sum()),
        }

        for event in ["large_undergrading_event", "referable_miss_event", "severe_pdr_miss_event"]:
            row[f"n_{event}"] = int(df[event].sum())
            row[f"n_selected_{event}"] = int((df[event] & df["selected_for_expert"]).sum())
            row[f"n_residual_{event}"] = int(df[f"residual_{event}"].sum())
            row[f"recall_{event}"] = (
                row[f"n_selected_{event}"] / row[f"n_{event}"]
                if row[f"n_{event}"] else np.nan
            )

        for event in [
            "fixed_large_undergrading_union_pool",
            "fixed_referable_miss_union_pool",
            "fixed_severe_pdr_miss_union_pool",
        ]:
            row[f"n_{event}"] = int(df[event].sum())
            row[f"n_selected_{event}"] = int(df[f"selected_{event}"].sum())
            row[f"n_residual_{event}"] = int(df[f"residual_{event}"].sum())
            row[f"recall_{event}"] = (
                row[f"n_selected_{event}"] / row[f"n_{event}"]
                if row[f"n_{event}"] else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_overlap(all_cases: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["image_key", "true_label"]

    flags = [
        "selected_for_expert",
        "expert_corrected",
        "expert_induced_error",
        "residual_risk_after_routing",
        "residual_danger_after_routing",
        "large_undergrading_event",
        "referable_miss_event",
        "severe_pdr_miss_event",
        "fixed_large_undergrading_union_pool",
        "fixed_referable_miss_union_pool",
        "fixed_severe_pdr_miss_union_pool",
        "fixed_any_union_risk_pool",
    ]

    base = all_cases[key_cols].drop_duplicates().sort_values("image_key").reset_index(drop=True)

    for op, df in all_cases.groupby("operating_point"):
        sub = df[key_cols + flags].copy()
        rename = {f: f"{op}__{f}" for f in flags}
        sub = sub.rename(columns=rename)
        base = base.merge(sub, on=key_cols, how="left")

    op_names = sorted(all_cases["operating_point"].unique().tolist())
    selected_cols = [f"{op}__selected_for_expert" for op in op_names]
    residual_cols = [f"{op}__residual_risk_after_routing" for op in op_names]

    base["selected_by_n_protocols"] = base[selected_cols].sum(axis=1).astype(int)
    base["residual_by_n_protocols"] = base[residual_cols].sum(axis=1).astype(int)

    return base


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    pred = load_predictions(cfg["source"]["prediction_registry"])
    fixed_pool = build_fixed_union_risk_pools(pred)

    all_tables = []
    for op in cfg["operating_points"]:
        all_tables.append(build_case_table_for_op(pred, op, fixed_pool))

    all_cases = pd.concat(all_tables, ignore_index=True)
    all_cases.to_csv(OUT_DIR / cfg["outputs"]["all_cases"], index=False)

    selected_risk = all_cases[
        all_cases["selected_for_expert"]
        & (
            all_cases["large_undergrading_event"]
            | all_cases["referable_miss_event"]
            | all_cases["severe_pdr_miss_event"]
            | all_cases["base_error"]
        )
    ].copy()
    selected_risk.to_csv(OUT_DIR / cfg["outputs"]["selected_risk_cases"], index=False)

    residual_risk = all_cases[
        all_cases["residual_risk_after_routing"]
        | all_cases["residual_danger_after_routing"]
    ].copy()
    residual_risk.to_csv(OUT_DIR / cfg["outputs"]["residual_risk_cases"], index=False)

    overlap = build_overlap(all_cases)
    overlap.to_csv(OUT_DIR / cfg["outputs"]["protocol_overlap_cases"], index=False)

    summary = summarize_cases(all_cases)
    summary.to_csv(OUT_DIR / cfg["outputs"]["summary"], index=False)

    print("[DONE] v0.8.2c residual risk case export")
    print("\nCase export summary:")
    print(summary.to_string(index=False))
    print("\nWrote outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
