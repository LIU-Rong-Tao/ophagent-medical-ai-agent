#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


CFG = Path("experiments/v0_8_2b_controlled_protocols/configs/controlled_protocols.yaml")
OUT_DIR = Path("experiments/v0_8_2b_controlled_protocols/outputs")
PROB_COLS = [f"prob_{i}" for i in range(5)]
N_RANDOM = 2000
SEED = 2026


def metrics(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        "n_error": int((np.asarray(y) != np.asarray(pred)).sum()),
    }


def load_registry_predictions(registry_path):
    reg = pd.read_csv(registry_path)
    reg = reg[reg["enabled"].astype(int) == 1].copy()

    frames = []
    cache = {}

    for _, r in reg.iterrows():
        model = r["model_name"]
        p = r["prediction_csv"]
        if p not in cache:
            cache[p] = pd.read_csv(p)
        sub = cache[p][cache[p]["model_name"] == model].copy()
        if sub.empty:
            raise ValueError(f"no prediction rows for model: {model}")
        frames.append(sub)

    pred = pd.concat(frames, ignore_index=True)
    return pred, reg


def cost_map(reg):
    rows = []
    for _, r in reg.iterrows():
        model = r["model_name"]
        c = pd.read_csv(r["cost_csv"])
        sub = c[c["model_name"] == model].copy()
        if sub.empty:
            raise ValueError(f"no cost row for model: {model}")
        row = sub.iloc[0].to_dict()
        row["model_name"] = model
        rows.append(row)
    df = pd.DataFrame(rows)
    return dict(zip(df["model_name"], df["mean_ms_per_image"]))


def model_df(pred, model):
    d = pred[pred["model_name"] == model].sort_values("image_key").reset_index(drop=True)
    if d.empty:
        raise ValueError(f"missing model: {model}")
    return d


def average_prediction(pred, models):
    dfs = [model_df(pred, m) for m in models]
    base = dfs[0][["image_key", "true_label"]].copy()
    probs = sum(d[PROB_COLS].to_numpy(dtype=float) for d in dfs) / len(dfs)
    out = base.copy()
    out[PROB_COLS] = probs
    out["pred_label"] = probs.argmax(axis=1)
    out["confidence"] = probs.max(axis=1)
    return out


def uncertainty_from_probs(probs):
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    p = np.clip(probs, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(probs.shape[1])
    return confidence, margin, entropy


def multi_scout_scores(pred, scouts, signal):
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


def apply_expert(base, expert_df, selected_idx):
    pred = base["pred_label"].to_numpy(dtype=int).copy()
    expert_pred = expert_df["pred_label"].to_numpy(dtype=int)
    pred[list(selected_idx)] = expert_pred[list(selected_idx)]
    return pred


def random_and_oracle(y, base_pred, expert_pred, k):
    rng = np.random.default_rng(SEED)
    n = len(y)

    random_acc = []
    random_err = []
    all_idx = np.arange(n)

    for _ in range(N_RANDOM):
        sel = set(rng.choice(all_idx, size=k, replace=False).tolist())
        p = base_pred.copy()
        idx = list(sel)
        p[idx] = expert_pred[idx]
        m = metrics(y, p)
        random_acc.append(m["accuracy"])
        random_err.append(m["n_error"])

    base_wrong = base_pred != y
    expert_correct = expert_pred == y
    fixable = np.where(base_wrong & expert_correct)[0].tolist()
    nonfix = [i for i in range(n) if i not in fixable]

    oracle_sel = fixable[:k]
    if len(oracle_sel) < k:
        oracle_sel += nonfix[: k - len(oracle_sel)]

    p = base_pred.copy()
    p[oracle_sel] = expert_pred[oracle_sel]
    om = metrics(y, p)

    return {
        "random_accuracy_mean": float(np.mean(random_acc)),
        "random_accuracy_p025": float(np.quantile(random_acc, 0.025)),
        "random_accuracy_p975": float(np.quantile(random_acc, 0.975)),
        "random_n_error_mean": float(np.mean(random_err)),
        "random_n_error_p025": float(np.quantile(random_err, 0.025)),
        "random_n_error_p975": float(np.quantile(random_err, 0.975)),
        "oracle_accuracy": om["accuracy"],
        "oracle_n_error": om["n_error"],
        "oracle_fixable_total": int(len(fixable)),
    }


def dense_rows(cfg, pred, costs):
    rows = []
    for item in cfg["dense_baselines"]:
        models = item["models"]
        d = average_prediction(pred, models)
        y = d["true_label"].to_numpy(dtype=int)
        p = d["pred_label"].to_numpy(dtype=int)
        m = metrics(y, p)
        rows.append({
            "protocol_family": "dense_baseline",
            "protocol_name": item["name"],
            "role": item.get("role", "main"),
            "scouts": "",
            "experts": "+".join(models),
            "budget": 1.0,
            "policy": "dense",
            "routing_signal": "",
            "selected_n": len(d),
            "ms_per_image": sum(costs[x] for x in models),
            **m,
        })
    return rows


def single_scout_rows(cfg):
    sparse = pd.read_csv(cfg["source"]["full_sparse_curve"])
    rows = []

    for item in cfg["single_scout_protocols"]:
        scout = item["scout"]
        experts = "+".join(item["experts"])
        for budget in item["budgets"]:
            for policy in item["policies"]:
                sub = sparse[
                    (sparse["scout"] == scout)
                    & (sparse["experts"] == experts)
                    & (sparse["budget"].astype(float) == float(budget))
                    & (sparse["policy"] == policy)
                ].copy()

                if sub.empty:
                    continue

                r = sub.iloc[0].to_dict()
                rows.append({
                    "protocol_family": "single_scout",
                    "protocol_name": item["name"],
                    "role": item.get("role", "main"),
                    "scouts": scout,
                    "experts": experts,
                    "budget": float(budget),
                    "policy": policy,
                    "routing_signal": policy,
                    "selected_n": int(r["selected_n"]),
                    "ms_per_image": float(r["online_no_cache_ms_per_image"]),
                    "accuracy": float(r["accuracy"]),
                    "macro_f1": float(r["macro_f1"]),
                    "qwk": float(r["qwk"]),
                    "n_error": int(r["n_error"]),
                    "above_random_p975": bool(r["above_random_p975"]),
                    "random_accuracy_p975": float(r["random_accuracy_p975"]),
                    "oracle_accuracy": float(r["oracle_accuracy"]),
                    "gap_to_oracle_accuracy": float(r["gap_to_oracle_accuracy"]),
                })
    return rows


def multi_scout_rows(cfg, pred, costs):
    rows = []

    for item in cfg["multi_scout_protocols"]:
        scouts = item["scouts"]
        expert = item["expert"]
        expert_df = model_df(pred, expert)
        y = expert_df["true_label"].to_numpy(dtype=int)
        expert_pred = expert_df["pred_label"].to_numpy(dtype=int)

        for signal in item["routing_signals"]:
            base = multi_scout_scores(pred, scouts, signal)
            base_pred = base["pred_label"].to_numpy(dtype=int)

            order = base.sort_values(["routing_score", "image_key"], ascending=[False, True]).index.to_numpy()

            for budget in item["budgets"]:
                k = int(round(len(base) * float(budget)))
                selected = set(order[:k].tolist())
                routed_pred = apply_expert(base, expert_df, selected)
                m = metrics(y, routed_pred)

                rb = random_and_oracle(y, base_pred, expert_pred, k)

                rows.append({
                    "protocol_family": "multi_scout",
                    "protocol_name": item["name"],
                    "role": item.get("role", "main"),
                    "scouts": "+".join(scouts),
                    "experts": expert,
                    "budget": float(budget),
                    "policy": signal,
                    "routing_signal": signal,
                    "selected_n": k,
                    "ms_per_image": sum(costs[s] for s in scouts) + float(budget) * costs[expert],
                    **m,
                    **rb,
                    "above_random_p975": m["accuracy"] > rb["random_accuracy_p975"],
                    "gap_to_oracle_accuracy": rb["oracle_accuracy"] - m["accuracy"],
                })

    return rows


def write_report(df):
    main = df[df["role"].isin(["main", "dense_expert_reference"])].copy()
    best = main.sort_values(["accuracy", "ms_per_image"], ascending=[False, True]).iloc[0]
    frontier = main.sort_values("ms_per_image").copy()

    keep = []
    best_acc = -1
    for _, r in frontier.iterrows():
        if r["accuracy"] > best_acc:
            keep.append(r)
            best_acc = r["accuracy"]
    frontier = pd.DataFrame(keep)

    lines = []
    lines.append("# v0.8.2b 受控协议评测结果\n")
    lines.append("## 1. 当前定位\n")
    lines.append("本报告只汇总 controlled_protocols.yaml 中预定义的协议。全组合搜索只作为 exploratory screening，不作为主结论。\n")
    lines.append("## 2. 最佳主协议\n")
    lines.append(f"- protocol: `{best['protocol_name']}`")
    lines.append(f"- family: `{best['protocol_family']}`")
    lines.append(f"- scouts: `{best['scouts']}`")
    lines.append(f"- experts: `{best['experts']}`")
    lines.append(f"- budget: {best['budget']}")
    lines.append(f"- policy/signal: `{best['policy']}`")
    lines.append(f"- accuracy: {best['accuracy']:.6f}")
    lines.append(f"- macro_f1: {best['macro_f1']:.6f}")
    lines.append(f"- qwk: {best['qwk']:.6f}")
    lines.append(f"- n_error: {int(best['n_error'])}")
    lines.append(f"- ms_per_image: {best['ms_per_image']:.6f}\n")

    lines.append("## 3. 主协议 cost-performance frontier\n")
    for _, r in frontier.iterrows():
        lines.append(
            f"- `{r['protocol_name']}` / {r['policy']} / budget={r['budget']}: "
            f"Acc={r['accuracy']:.6f}, ms/image={r['ms_per_image']:.6f}, n_error={int(r['n_error'])}"
        )

    lines.append("\n## 4. 解释边界\n")
    lines.append("- 本结果不是全局最优组合搜索。")
    lines.append("- candidate ranking score 不作为最终证据。")
    lines.append("- ViT-B 和 GreenScout 当前主要作为 screening/ablation 对照。")
    lines.append("- Multi-scout routing 用于验证 scout 不必只有一个。")

    (OUT_DIR / "controlled_protocol_key_findings.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))

    pred, reg = load_registry_predictions(cfg["source"]["prediction_registry"])
    costs = cost_map(reg)

    rows = []
    rows.extend(dense_rows(cfg, pred, costs))
    rows.extend(single_scout_rows(cfg))
    rows.extend(multi_scout_rows(cfg, pred, costs))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "controlled_protocol_results.csv", index=False)

    main_df = df[df["role"].isin(["main", "dense_expert_reference"])].copy()
    main_df.to_csv(OUT_DIR / "controlled_protocol_main_results.csv", index=False)

    best_per_protocol = (
        main_df.sort_values(["protocol_name", "accuracy", "ms_per_image"], ascending=[True, False, True])
        .groupby("protocol_name", as_index=False)
        .head(1)
        .sort_values(["accuracy", "ms_per_image"], ascending=[False, True])
    )
    best_per_protocol.to_csv(OUT_DIR / "controlled_protocol_best_per_protocol.csv", index=False)

    frontier_rows = []
    best_acc = -1.0
    for _, r in main_df.sort_values("ms_per_image").iterrows():
        if float(r["accuracy"]) > best_acc:
            frontier_rows.append(r)
            best_acc = float(r["accuracy"])
    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(OUT_DIR / "controlled_protocol_cost_frontier.csv", index=False)

    write_report(df)

    cols = [
        "protocol_family", "protocol_name", "role", "scouts", "experts",
        "budget", "policy", "selected_n", "ms_per_image",
        "accuracy", "macro_f1", "qwk", "n_error",
        "above_random_p975", "random_accuracy_p975", "oracle_accuracy", "gap_to_oracle_accuracy",
    ]

    print("[DONE] controlled protocol evaluation")
    print("\nBest per protocol:")
    print(best_per_protocol[cols].to_string(index=False))

    print("\nControlled cost-performance frontier:")
    print(frontier_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
