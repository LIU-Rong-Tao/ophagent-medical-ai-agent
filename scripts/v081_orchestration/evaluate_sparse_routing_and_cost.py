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
BUDGETS = [0.1, 0.2, 0.3, 0.4, 0.5]
POLICIES = ["low_confidence", "low_margin", "high_entropy"]
N_RANDOM = 2000
SEED = 2026


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    probs = out[PROB_COLS].to_numpy(dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    out["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["entropy"] = entropy(probs)
    out["correct"] = out["pred_label"].astype(int) == out["true_label"].astype(int)
    return out


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "setting": name,
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "n_error": int((y_true != y_pred).sum()),
    }


def load_predictions(registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    enabled = registry[registry["enabled"].astype(int) == 1].copy()
    cache = {}
    preds = {}

    for _, r in enabled.iterrows():
        name = r["model_name"]
        path = r["prediction_csv"]
        if path not in cache:
            cache[path] = read_csv(path)
        sub = cache[path][cache[path]["model_name"] == name].copy()
        if sub.empty:
            raise ValueError(f"no prediction rows for {name}")
        preds[name] = add_derived(sub.sort_values("image_key").reset_index(drop=True))

    return preds


def load_costs(registry: pd.DataFrame) -> dict[str, dict]:
    enabled = registry[registry["enabled"].astype(int) == 1].copy()
    cache = {}
    costs = {}

    for _, r in enabled.iterrows():
        name = r["model_name"]
        path = r["cost_csv"]
        if path not in cache:
            cache[path] = read_csv(path)
        sub = cache[path][cache[path]["model_name"] == name].copy()
        if sub.empty:
            raise ValueError(f"no cost row for {name}")
        row = sub.iloc[0].to_dict()
        costs[name] = row

    return costs


def expert_group_name(group: tuple[str, ...]) -> str:
    return "+".join(group)


def average_probs(preds: dict[str, pd.DataFrame], group: tuple[str, ...]) -> np.ndarray:
    return np.mean([preds[m][PROB_COLS].to_numpy(dtype=float) for m in group], axis=0)


def select_indices(scout_df: pd.DataFrame, policy: str, k: int) -> np.ndarray:
    if policy == "low_confidence":
        order = scout_df.sort_values(["confidence", "image_key"], ascending=[True, True]).index
    elif policy == "low_margin":
        order = scout_df.sort_values(["margin", "image_key"], ascending=[True, True]).index
    elif policy == "high_entropy":
        order = scout_df.sort_values(["entropy", "image_key"], ascending=[False, True]).index
    else:
        raise ValueError(policy)
    return np.asarray(order[:k], dtype=int)


def role_allowed_models(registry: pd.DataFrame, role_keyword: str) -> list[str]:
    enabled = registry[registry["enabled"].astype(int) == 1].copy()
    return [
        r["model_name"]
        for _, r in enabled.iterrows()
        if role_keyword in str(r["role_hint"]).lower()
    ]


def sparse_routing_curve(
    preds: dict[str, pd.DataFrame],
    costs: dict[str, dict],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    names = list(preds.keys())
    scouts = role_allowed_models(registry, "scout")
    experts_allowed = role_allowed_models(registry, "expert")

    y_true = preds[names[0]]["true_label"].to_numpy(dtype=int)
    n = len(y_true)
    rows = []

    for scout in scouts:
        expert_candidates = [m for m in experts_allowed if m != scout]

        expert_groups: list[tuple[str, ...]] = []
        expert_groups.extend((m,) for m in expert_candidates)

        if len(expert_candidates) >= 2:
            expert_groups.append(tuple(expert_candidates))

        scout_df = preds[scout]
        scout_probs = scout_df[PROB_COLS].to_numpy(dtype=float)

        for experts in expert_groups:
            expert_probs = average_probs(preds, experts)

            for budget in BUDGETS:
                k = int(round(n * budget))

                for policy in POLICIES:
                    selected = select_indices(scout_df, policy, k)
                    final_probs = scout_probs.copy()
                    final_probs[selected] = expert_probs[selected]
                    y_pred = final_probs.argmax(axis=1)

                    setting = f"{scout} -> {expert_group_name(experts)}"
                    r = metric_row(setting, y_true, y_pred)
                    r.update({
                        "scout": scout,
                        "experts": expert_group_name(experts),
                        "budget": budget,
                        "selected_n": int(k),
                        "policy": policy,
                        "expert_models_n": int(len(experts)),
                    })

                    scout_ms = float(costs[scout]["total_forward_ms"]) / n
                    expert_ms = sum(float(costs[e]["total_forward_ms"]) / n for e in experts)

                    r["online_no_cache_ms_per_image"] = float(scout_ms + budget * expert_ms)
                    r["cached_scout_ms_per_image"] = float(budget * expert_ms)
                    r["scout_ms_per_image"] = float(scout_ms)
                    r["expert_ms_per_image_if_dense"] = float(expert_ms)
                    rows.append(r)

    return pd.DataFrame(rows).sort_values(
        ["accuracy", "macro_f1", "online_no_cache_ms_per_image"],
        ascending=[False, False, True],
    )


def dense_cost_rows(preds: dict[str, pd.DataFrame], costs: dict[str, dict]) -> pd.DataFrame:
    names = list(preds.keys())
    y_true = preds[names[0]]["true_label"].to_numpy(dtype=int)
    n = len(y_true)
    rows = []

    for k in range(1, len(names) + 1):
        for group in combinations(names, k):
            probs = average_probs(preds, group)
            y_pred = probs.argmax(axis=1)

            setting = "dense:" + expert_group_name(group)
            r = metric_row(setting, y_true, y_pred)
            r.update({
                "scenario": "dense",
                "models": expert_group_name(group),
                "models_n": k,
                "ms_per_image": float(sum(float(costs[m]["total_forward_ms"]) / n for m in group)),
            })
            rows.append(r)

    return pd.DataFrame(rows)


def cost_performance_frontier(dense: pd.DataFrame, sparse: pd.DataFrame) -> pd.DataFrame:
    sparse_rows = sparse.copy()
    sparse_rows["scenario"] = "sparse_online_no_cache"
    sparse_rows["models"] = sparse_rows["setting"]
    sparse_rows["models_n"] = 1 + sparse_rows["expert_models_n"]
    sparse_rows["ms_per_image"] = sparse_rows["online_no_cache_ms_per_image"]

    cols = ["scenario", "setting", "models", "models_n", "accuracy", "macro_f1", "qwk", "n_error", "ms_per_image"]
    all_rows = pd.concat([dense[cols], sparse_rows[cols]], ignore_index=True)

    all_rows = all_rows.sort_values(["ms_per_image", "accuracy"], ascending=[True, False]).reset_index(drop=True)

    frontier = []
    best_acc = -1.0
    for _, r in all_rows.iterrows():
        if float(r["accuracy"]) > best_acc + 1e-12:
            frontier.append(r.to_dict())
            best_acc = float(r["accuracy"])

    return pd.DataFrame(frontier)




def random_and_oracle_baselines(
    preds: dict[str, pd.DataFrame],
    costs: dict[str, dict],
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    names = list(preds.keys())
    scouts = role_allowed_models(registry, "scout")
    experts_allowed = role_allowed_models(registry, "expert")

    y_true = preds[names[0]]["true_label"].to_numpy(dtype=int)
    n = len(y_true)

    random_rows = []
    oracle_rows = []

    for scout in scouts:
        expert_candidates = [m for m in experts_allowed if m != scout]

        expert_groups: list[tuple[str, ...]] = []
        expert_groups.extend((m,) for m in expert_candidates)
        if len(expert_candidates) >= 2:
            expert_groups.append(tuple(expert_candidates))

        scout_probs = preds[scout][PROB_COLS].to_numpy(dtype=float)
        scout_pred = scout_probs.argmax(axis=1)

        for experts in expert_groups:
            expert_probs = average_probs(preds, experts)
            expert_pred = expert_probs.argmax(axis=1)

            expert_improves = (scout_pred != y_true) & (expert_pred == y_true)

            for budget in BUDGETS:
                k = int(round(n * budget))

                random_acc = []
                random_err = []

                for _ in range(N_RANDOM):
                    selected = rng.choice(n, size=k, replace=False)
                    final_pred = scout_pred.copy()
                    final_pred[selected] = expert_pred[selected]
                    random_acc.append(float(accuracy_score(y_true, final_pred)))
                    random_err.append(int((y_true != final_pred).sum()))

                random_rows.append({
                    "scout": scout,
                    "experts": expert_group_name(experts),
                    "budget": budget,
                    "selected_n": k,
                    "n_random": N_RANDOM,
                    "random_accuracy_mean": float(np.mean(random_acc)),
                    "random_accuracy_p025": float(np.quantile(random_acc, 0.025)),
                    "random_accuracy_p975": float(np.quantile(random_acc, 0.975)),
                    "random_n_error_mean": float(np.mean(random_err)),
                    "random_n_error_p025": float(np.quantile(random_err, 0.025)),
                    "random_n_error_p975": float(np.quantile(random_err, 0.975)),
                })

                # Oracle: first select images where expert fixes scout error.
                oracle_selected = np.where(expert_improves)[0][:k]
                if len(oracle_selected) < k:
                    remaining = np.setdiff1d(np.arange(n), oracle_selected, assume_unique=False)
                    oracle_selected = np.concatenate([oracle_selected, remaining[: k - len(oracle_selected)]])

                final_pred = scout_pred.copy()
                final_pred[oracle_selected] = expert_pred[oracle_selected]

                r = metric_row(f"{scout} -> {expert_group_name(experts)}", y_true, final_pred)
                r.update({
                    "scout": scout,
                    "experts": expert_group_name(experts),
                    "budget": budget,
                    "selected_n": k,
                    "oracle_fixable_total": int(expert_improves.sum()),
                })
                oracle_rows.append(r)

    return pd.DataFrame(random_rows), pd.DataFrame(oracle_rows)




def attach_random_oracle(
    sparse: pd.DataFrame,
    random_base: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    out = sparse.copy()

    rand_cols = [
        "scout", "experts", "budget", "selected_n",
        "random_accuracy_mean",
        "random_accuracy_p025",
        "random_accuracy_p975",
        "random_n_error_mean",
        "random_n_error_p025",
        "random_n_error_p975",
    ]
    oracle_cols = [
        "scout", "experts", "budget", "selected_n",
        "accuracy",
        "n_error",
        "oracle_fixable_total",
    ]

    rand = random_base[rand_cols].copy()
    ora = oracle[oracle_cols].copy().rename(columns={
        "accuracy": "oracle_accuracy",
        "n_error": "oracle_n_error",
    })

    out = out.merge(
        rand,
        on=["scout", "experts", "budget", "selected_n"],
        how="left",
    )
    out = out.merge(
        ora,
        on=["scout", "experts", "budget", "selected_n"],
        how="left",
    )

    out["above_random_p975"] = out["accuracy"] > out["random_accuracy_p975"]
    out["accuracy_gain_vs_random_mean"] = out["accuracy"] - out["random_accuracy_mean"]
    out["gap_to_oracle_accuracy"] = out["oracle_accuracy"] - out["accuracy"]
    out["error_reduction_vs_random_mean"] = out["random_n_error_mean"] - out["n_error"]
    out["gap_to_oracle_n_error"] = out["n_error"] - out["oracle_n_error"]

    return out


def write_findings(out_dir: Path, sparse: pd.DataFrame, frontier: pd.DataFrame):
    best_acc = sparse.iloc[0]
    best_tradeoff = sparse.sort_values(
        ["online_no_cache_ms_per_image", "accuracy"],
        ascending=[True, False],
    ).iloc[0]

    lines = []
    lines.append("# v0.8.1 Sparse Routing and Cost Frontier 初版结果\n")
    lines.append("## 1. 当前范围\n")
    lines.append("本轮在统一 evaluator 中加入 sparse routing curve 与 cost-performance frontier。")
    lines.append("当前只基于已有三模型 prediction CSV 与 v0.8.0e forward-cost 表进行估算。\n")

    lines.append("## 2. Sparse routing 最佳 accuracy\n")
    lines.append(
        f"- 最佳 sparse 设置为 `{best_acc['setting']}`，policy=`{best_acc['policy']}`，"
        f"budget={best_acc['budget']:.1f}，accuracy={best_acc['accuracy']:.4f}，"
        f"online no-cache cost={best_acc['online_no_cache_ms_per_image']:.3f} ms/image。"
    )

    lines.append("\n## 3. 最低 online no-cache 成本的 sparse 设置\n")
    lines.append(
        f"- 最低成本 sparse 设置为 `{best_tradeoff['setting']}`，policy=`{best_tradeoff['policy']}`，"
        f"budget={best_tradeoff['budget']:.1f}，accuracy={best_tradeoff['accuracy']:.4f}，"
        f"online no-cache cost={best_tradeoff['online_no_cache_ms_per_image']:.3f} ms/image。"
    )

    lines.append("\n## 4. Cost-performance frontier\n")
    lines.append("- `cost_performance_frontier.csv` 给出当前非支配 accuracy-cost 前沿。")
    lines.append("- 后续新增模型后，该文件应作为筛选 scout / expert 组合的主要入口之一。\n")

    lines.append("## 5. 当前边界\n")
    lines.append("- 当前 sparse cost 是基于 measured forward-only cost 与 selected_n 的估算。")
    lines.append("- 当前 sparse routing 遵循 `model_registry.csv` 中的 `role_hint`：只有 scout 候选可作为 scout，只有 expert 候选可作为 expert。")
    lines.append("- 当前还未加入 random baseline、oracle upper bound、DR-specific risk event enrichment。")

    (out_dir / "sparse_cost_key_findings.md").write_text("\n".join(lines), encoding="utf-8")


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
    preds = load_predictions(registry)
    costs = load_costs(registry)

    sparse = sparse_routing_curve(preds, costs, registry)
    dense = dense_cost_rows(preds, costs)
    frontier = cost_performance_frontier(dense, sparse)
    random_base, oracle = random_and_oracle_baselines(preds, costs, registry)
    sparse_with_baselines = attach_random_oracle(sparse, random_base, oracle)

    sparse_with_baselines.to_csv(out_dir / "sparse_routing_curve.csv", index=False)
    dense.to_csv(out_dir / "dense_cost_performance.csv", index=False)
    frontier.to_csv(out_dir / "cost_performance_frontier.csv", index=False)
    random_base.to_csv(out_dir / "sparse_random_baseline.csv", index=False)
    oracle.to_csv(out_dir / "sparse_oracle_upper_bound.csv", index=False)
    write_findings(out_dir, sparse_with_baselines, frontier)

    print("[DONE] sparse routing + cost frontier")
    print("\nTop sparse rows:")
    show_cols = [
        "setting", "accuracy", "macro_f1", "qwk", "n_error",
        "budget", "policy", "online_no_cache_ms_per_image",
        "above_random_p975", "random_accuracy_p975",
        "oracle_accuracy", "gap_to_oracle_accuracy",
    ]
    print(sparse_with_baselines[show_cols].head(20).to_string(index=False))
    print("\nCost-performance frontier:")
    print(frontier.to_string(index=False))


if __name__ == "__main__":
    main()
