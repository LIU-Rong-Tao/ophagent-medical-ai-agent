#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

IN_CSV = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/predictions/greenscout_three_model_standardized.csv")
OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/scout_ablation")

GREEN = "retfound_green_linear_probe"
CONV = "convnext_tiny"
RETF = "retfound_mae_cfp_official_protocol"
PROB_COLS = [f"prob_{i}" for i in range(5)]

BUDGETS = [0.30, 0.40, 0.50]
POLICIES = ["low_confidence", "low_margin", "high_entropy"]
N_RANDOM = 2000
SEED = 2026


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "n_error": int((y_true != y_pred).sum()),
    }


def entropy_norm(prob):
    prob = np.clip(prob, 1e-12, 1.0)
    return -(prob * np.log(prob)).sum(axis=1) / np.log(prob.shape[1])


def topk_mask(score, k):
    order = np.argsort(-score)
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:k]] = True
    return mask


def apply_sparse(base_pred, expert_pred, selected):
    pred = base_pred.copy()
    pred[selected] = expert_pred[selected]
    return pred


def transition_stats(y_true, scout_pred, expert_pred, selected):
    scout_error = scout_pred != y_true
    scout_correct = scout_pred == y_true
    expert_correct = expert_pred == y_true
    expert_wrong = expert_pred != y_true

    expert_correctable = scout_error & expert_correct
    rescued = selected & scout_error & expert_correct
    induced = selected & scout_correct & expert_wrong

    final_pred = apply_sparse(scout_pred, expert_pred, selected)

    return {
        "scout_error_total": int(scout_error.sum()),
        "selected_scout_errors": int((selected & scout_error).sum()),
        "scout_error_capture_recall": float((selected & scout_error).sum() / max(scout_error.sum(), 1)),
        "expert_correctable_total": int(expert_correctable.sum()),
        "selected_expert_correctable": int((selected & expert_correctable).sum()),
        "expert_correctable_capture_recall": float((selected & expert_correctable).sum() / max(expert_correctable.sum(), 1)),
        "rescued_scout_errors": int(rescued.sum()),
        "expert_induced_errors": int(induced.sum()),
        "net_error_reduction": int(rescued.sum() - induced.sum()),
        "final_error_reduction": int((scout_pred != y_true).sum() - (final_pred != y_true).sum()),
    }


def event_capture(y_true, scout_pred, expert_pred, selected):
    events = {
        "scout_error": scout_pred != y_true,
        "expert_correctable_scout_error": (scout_pred != y_true) & (expert_pred == y_true),

        # DR five-grade task specific events.
        "severe_pdr_miss_dr_specific": (y_true >= 3) & (scout_pred <= 2),
        "large_undergrading_dr_specific": (y_true == 4) & (scout_pred <= 2),
        "referable_dr_miss_dr_specific": (y_true >= 2) & (scout_pred <= 1),
    }

    rows = []
    for name, mask in events.items():
        total = int(mask.sum())
        selected_n = int((selected & mask).sum())
        rows.append({
            "event": name,
            "event_total": total,
            "selected_event_n": selected_n,
            "event_recall": float(selected_n / max(total, 1)),
            "event_precision_in_selected": float(selected_n / max(selected.sum(), 1)),
        })
    return rows


def policy_score(prob, policy):
    confidence = prob.max(axis=1)
    sorted_prob = np.sort(prob, axis=1)
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]
    entropy = entropy_norm(prob)

    if policy == "low_confidence":
        return -confidence
    if policy == "low_margin":
        return -margin
    if policy == "high_entropy":
        return entropy
    raise ValueError(policy)


def oracle_mask(y_true, scout_pred, expert_pred, k):
    scout_correct = scout_pred == y_true
    expert_correct = expert_pred == y_true

    delta = np.zeros(len(y_true), dtype=int)
    delta[(~scout_correct) & expert_correct] = 1
    delta[scout_correct & (~expert_correct)] = -1

    order = np.argsort(-delta, kind="stable")
    selected = np.zeros(len(y_true), dtype=bool)
    selected[order[:k]] = True
    return selected


def md_table(df):
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV)
    required = {"image_key", "true_label", "pred_label", "model_name", *PROB_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    models = {GREEN, CONV, RETF}
    present = set(df["model_name"].unique())
    if models - present:
        raise ValueError(f"Missing models: {sorted(models - present)}")

    if df.duplicated(["image_key", "model_name"]).any():
        raise ValueError("Duplicate (image_key, model_name) rows found.")

    keys = sorted(df["image_key"].unique())
    if len(keys) != 1100:
        print(f"[WARN] expected 1100 images, got {len(keys)}")

    y_true = (
        df.groupby("image_key")["true_label"]
        .first()
        .loc[keys]
        .to_numpy(dtype=int)
    )

    prob = {}
    pred = {}
    for m in [GREEN, CONV, RETF]:
        sub = df[df["model_name"] == m].set_index("image_key").loc[keys]
        prob[m] = sub[PROB_COLS].to_numpy(dtype=float)
        pred[m] = sub["pred_label"].to_numpy(dtype=int)

    prob["convnext_retfound_avg"] = (prob[CONV] + prob[RETF]) / 2.0
    pred["convnext_retfound_avg"] = prob["convnext_retfound_avg"].argmax(axis=1)

    prob["all_three_avg"] = (prob[GREEN] + prob[CONV] + prob[RETF]) / 3.0
    pred["all_three_avg"] = prob["all_three_avg"].argmax(axis=1)

    n = len(y_true)
    rng = np.random.default_rng(SEED)

    dense_rows = []
    for name, p in [
        ("retfound_green_linear_probe", pred[GREEN]),
        ("convnext_tiny", pred[CONV]),
        ("retfound_mae_cfp_official_protocol", pred[RETF]),
        ("experts_only_average", pred["convnext_retfound_avg"]),
        ("all_three_average", pred["all_three_avg"]),
    ]:
        row = {"method": name}
        row.update(metrics(y_true, p))
        dense_rows.append(row)

    settings = [
        {
            "setting": "A_green_scout_to_convnext_retfound_avg",
            "scout": GREEN,
            "expert": "convnext+retfound_avg",
            "scout_prob": prob[GREEN],
            "scout_pred": pred[GREEN],
            "expert_pred": pred["convnext_retfound_avg"],
            "expert_calls_per_selected": 2,
        },
        {
            "setting": "B_green_scout_to_convnext_only",
            "scout": GREEN,
            "expert": CONV,
            "scout_prob": prob[GREEN],
            "scout_pred": pred[GREEN],
            "expert_pred": pred[CONV],
            "expert_calls_per_selected": 1,
        },
        {
            "setting": "C_green_scout_to_retfound_only",
            "scout": GREEN,
            "expert": RETF,
            "scout_prob": prob[GREEN],
            "scout_pred": pred[GREEN],
            "expert_pred": pred[RETF],
            "expert_calls_per_selected": 1,
        },
        {
            "setting": "D_convnext_scout_to_retfound_only",
            "scout": CONV,
            "expert": RETF,
            "scout_prob": prob[CONV],
            "scout_pred": pred[CONV],
            "expert_pred": pred[RETF],
            "expert_calls_per_selected": 1,
        },
    ]

    curve_rows = []
    transition_rows = []
    event_rows = []
    random_rows = []
    oracle_rows = []

    for s in settings:
        for budget in BUDGETS:
            k = int(round(n * budget))

            # random baseline
            random_acc = []
            random_f1 = []
            random_qwk = []
            for _ in range(N_RANDOM):
                selected = np.zeros(n, dtype=bool)
                selected[rng.choice(n, size=k, replace=False)] = True
                final_pred = apply_sparse(s["scout_pred"], s["expert_pred"], selected)
                mm = metrics(y_true, final_pred)
                random_acc.append(mm["accuracy"])
                random_f1.append(mm["macro_f1"])
                random_qwk.append(mm["qwk"])

            random_rows.append({
                "setting": s["setting"],
                "budget": budget,
                "random_n": N_RANDOM,
                "random_acc_mean": float(np.mean(random_acc)),
                "random_acc_p025": float(np.quantile(random_acc, 0.025)),
                "random_acc_p975": float(np.quantile(random_acc, 0.975)),
                "random_macro_f1_mean": float(np.mean(random_f1)),
                "random_qwk_mean": float(np.mean(random_qwk)),
                "random_qwk_p975": float(np.quantile(random_qwk, 0.975)),
            })

            # oracle same-budget upper bound
            selected = oracle_mask(y_true, s["scout_pred"], s["expert_pred"], k)
            final_pred = apply_sparse(s["scout_pred"], s["expert_pred"], selected)
            row = {
                "setting": s["setting"],
                "scout": s["scout"],
                "expert": s["expert"],
                "budget": budget,
                "selected_n": int(selected.sum()),
                "policy": "oracle_same_budget_posthoc",
                "expert_forward_calls": int(selected.sum() * s["expert_calls_per_selected"]),
            }
            row.update(metrics(y_true, final_pred))
            oracle_rows.append(row)

            # uncertainty policies
            for policy in POLICIES:
                score = policy_score(s["scout_prob"], policy)
                selected = topk_mask(score, k)
                final_pred = apply_sparse(s["scout_pred"], s["expert_pred"], selected)

                row = {
                    "setting": s["setting"],
                    "scout": s["scout"],
                    "expert": s["expert"],
                    "budget": budget,
                    "selected_n": int(selected.sum()),
                    "policy": policy,
                    "expert_forward_calls": int(selected.sum() * s["expert_calls_per_selected"]),
                }
                row.update(metrics(y_true, final_pred))
                curve_rows.append(row)

                tr = {
                    "setting": s["setting"],
                    "scout": s["scout"],
                    "expert": s["expert"],
                    "budget": budget,
                    "policy": policy,
                }
                tr.update(transition_stats(y_true, s["scout_pred"], s["expert_pred"], selected))
                transition_rows.append(tr)

                for ev in event_capture(y_true, s["scout_pred"], s["expert_pred"], selected):
                    ev_row = {
                        "setting": s["setting"],
                        "scout": s["scout"],
                        "expert": s["expert"],
                        "budget": budget,
                        "policy": policy,
                    }
                    ev_row.update(ev)
                    event_rows.append(ev_row)

    dense = pd.DataFrame(dense_rows)
    curve = pd.DataFrame(curve_rows)
    transition = pd.DataFrame(transition_rows)
    events = pd.DataFrame(event_rows)
    random_df = pd.DataFrame(random_rows)
    oracle = pd.DataFrame(oracle_rows)

    dense.to_csv(OUT_DIR / "scout_ablation_dense_baselines.csv", index=False)
    curve.to_csv(OUT_DIR / "scout_ablation_curve.csv", index=False)
    transition.to_csv(OUT_DIR / "scout_ablation_transition_accounting.csv", index=False)
    events.to_csv(OUT_DIR / "scout_ablation_event_capture.csv", index=False)
    random_df.to_csv(OUT_DIR / "scout_ablation_random_summary.csv", index=False)
    oracle.to_csv(OUT_DIR / "scout_ablation_oracle_same_budget.csv", index=False)

    best = (
        curve.sort_values(["setting", "budget", "accuracy", "macro_f1", "qwk"], ascending=[True, True, False, False, False])
        .groupby(["setting", "budget"], as_index=False)
        .head(1)
    )

    best = best.merge(
        random_df[["setting", "budget", "random_acc_mean", "random_acc_p975", "random_qwk_mean", "random_qwk_p975"]],
        on=["setting", "budget"],
        how="left",
    )

    best = best.merge(
        oracle[["setting", "budget", "accuracy", "macro_f1", "qwk", "n_error"]].rename(
            columns={
                "accuracy": "oracle_accuracy",
                "macro_f1": "oracle_macro_f1",
                "qwk": "oracle_qwk",
                "n_error": "oracle_n_error",
            }
        ),
        on=["setting", "budget"],
        how="left",
    )

    best = best.merge(
        transition,
        on=["setting", "scout", "expert", "budget", "policy"],
        how="left",
    )

    best["above_random_p975_acc"] = best["accuracy"] > best["random_acc_p975"]
    best["gap_to_oracle_acc"] = best["oracle_accuracy"] - best["accuracy"]
    best = best.sort_values(["budget", "accuracy"], ascending=[True, False])

    best.to_csv(OUT_DIR / "scout_ablation_key_summary.csv", index=False)

    with open(OUT_DIR / "scout_ablation_key_summary.md", "w", encoding="utf-8") as f:
        f.write("# v0.8.0d Scout Ablation 关键结果\n\n")
        f.write("## 1. 实验设置\n\n")
        f.write("本轮实验比较 RETFound-Green 与 ConvNeXt-Tiny 作为 scout 时的稀疏专家调用效果。\n\n")
        f.write("模型池包括：\n\n")
        f.write("- Scout 候选：RETFound-Green linear probe、ConvNeXt-Tiny\n")
        f.write("- Expert 候选：ConvNeXt-Tiny、RETFound-MAE official-protocol\n\n")
        f.write("budget 表示进入专家模型通道的样本比例，即 expert-call equivalent budget；不等同于真实端到端推理成本下降比例。\n\n")

        f.write("## 2. Dense / Single Baselines\n\n")
        f.write(md_table(dense[["method", "accuracy", "macro_f1", "qwk", "n_error"]]))
        f.write("\n\n")

        f.write("## 3. Best Sparse Policy at 30/40/50% Budget\n\n")
        cols = [
            "setting", "budget", "policy", "accuracy", "macro_f1", "qwk", "n_error",
            "random_acc_p975", "above_random_p975_acc", "oracle_accuracy", "gap_to_oracle_acc",
            "expert_correctable_total", "selected_expert_correctable",
            "expert_correctable_capture_recall", "rescued_scout_errors",
            "expert_induced_errors", "net_error_reduction"
        ]
        f.write(md_table(best[cols]))
        f.write("\n\n")

        f.write("## 4. 当前边界\n\n")
        f.write("- 本结果仍是 prediction-level sparse routing simulation。\n")
        f.write("- expert-call equivalent budget 不等于真实 wall-clock、吞吐量或显存收益。\n")
        f.write("- DR-specific risk events 只适用于当前 APTOS DR 五级分级任务，不应直接泛化到所有眼科疾病。\n")
        f.write("- Oracle same-budget 为事后上界，不可部署。\n")

    print("[DONE] v0.8.0d scout ablation")
    print("output:", OUT_DIR)
    print("\nDense baselines:")
    print(dense[["method", "accuracy", "macro_f1", "qwk", "n_error"]].to_string(index=False))
    print("\nKey summary:")
    print(best[[
        "setting", "budget", "policy", "accuracy", "macro_f1", "qwk", "n_error",
        "random_acc_p975", "above_random_p975_acc",
        "oracle_accuracy", "expert_correctable_capture_recall",
        "rescued_scout_errors", "expert_induced_errors", "net_error_reduction"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
