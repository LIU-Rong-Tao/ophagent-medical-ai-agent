#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

COST_PATH = Path("experiments/v0_8_0_greenscout_feasibility/cost/inference_cost_table.csv")
GREEN_PATH = Path("experiments/v0_8_0_greenscout_feasibility/green_smoke/retfound_green_smoke_test.csv")
EXPERT_PATH = Path("experiments/v0_8_0_greenscout_feasibility/predictions/demo_existing_expert_predictions.csv")
OUT_PATH = Path("experiments/v0_8_0_greenscout_feasibility/cost/demo_cost_robust_summary.csv")
MD_PATH = Path("experiments/v0_8_0_greenscout_feasibility/reports/demo_cost_robust_summary.md")

def summarize_latency(df, model_name):
    x = df[df["model_name"] == model_name].copy()
    x = x[x["inference_ms"] > 0].reset_index(drop=True)

    if len(x) == 0:
        return None

    # 去掉每个模型第一张图，降低残余冷启动影响
    x_wo_first = x.iloc[1:].copy() if len(x) > 1 else x.copy()

    return {
        "model_name": model_name,
        "n_images": int(len(x)),
        "n_images_excluding_first": int(len(x_wo_first)),
        "mean_ms_raw": float(x["inference_ms"].mean()),
        "median_ms_raw": float(x["inference_ms"].median()),
        "max_ms_raw": float(x["inference_ms"].max()),
        "mean_ms_excluding_first": float(x_wo_first["inference_ms"].mean()),
        "median_ms_excluding_first": float(x_wo_first["inference_ms"].median()),
        "max_ms_excluding_first": float(x_wo_first["inference_ms"].max()),
        "peak_mem_mb": float(x["peak_mem_mb"].max()),
    }

def main():
    green = pd.read_csv(GREEN_PATH)
    expert = pd.read_csv(EXPERT_PATH)

    rows = []
    rows.append(summarize_latency(green, "retfound_green"))

    for m in sorted(expert["model_name"].unique()):
        rows.append(summarize_latency(expert, m))

    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)

    cost = pd.read_csv(COST_PATH)
    meta_cols = ["model_name", "role", "stage", "checkpoint_mb", "img_size", "note"]
    out = out.merge(cost[meta_cols], on="model_name", how="left")

    out = out[
        [
            "model_name", "role", "stage", "checkpoint_mb", "img_size",
            "n_images", "n_images_excluding_first",
            "mean_ms_raw", "median_ms_raw", "max_ms_raw",
            "mean_ms_excluding_first", "median_ms_excluding_first", "max_ms_excluding_first",
            "peak_mem_mb", "note",
        ]
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    md = "# v0.8.0 demo_samples 稳健成本摘要\n\n"
    md += "## 说明\n\n"
    md += "本表基于 demo_samples 15 张图像。由于单次测试中第一张图可能包含 CUDA kernel / 缓存残余冷启动开销，因此同时报告 raw latency 和 excluding-first latency。当前结果只用于 smoke/cost feasibility，不作为最终推理基准。\n\n"
    md += "## 成本摘要\n\n"
    md += "| model | role | checkpoint MB | img size | median ms raw | median ms excl. first | peak mem MB |\n"
    md += "|---|---|---:|---:|---:|---:|---:|\n"

    for _, r in out.iterrows():
        md += (
            f"| {r['model_name']} | {r['role']} | "
            f"{r['checkpoint_mb']:.2f} | {int(r['img_size'])} | "
            f"{r['median_ms_raw']:.2f} | {r['median_ms_excluding_first']:.2f} | "
            f"{r['peak_mem_mb']:.2f} |\n"
        )

    md += "\n## 初步判断\n\n"
    md += "- RETFound-Green 已通过当前环境加载与真实图像 embedding smoke test。\n"
    md += "- 相比 RETFound-MAE official-like，RETFound-Green 在 checkpoint 大小和显存上有明显优势。\n"
    md += "- 相比 ConvNeXt-Tiny，RETFound-Green 的延迟同量级，显存和 checkpoint 略低；但 Green 当前输出为 embedding，不是五分类 logits。\n"
    md += "- 当前不能把 demo_samples 15 张图的 latency 当最终部署成本，应补充重复测量或全量测试。\n"

    MD_PATH.write_text(md, encoding="utf-8")

    print("[DONE]")
    print("csv:", OUT_PATH)
    print("md:", MD_PATH)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
