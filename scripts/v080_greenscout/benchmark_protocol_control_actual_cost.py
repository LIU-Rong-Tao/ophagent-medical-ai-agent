#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


DATA_ROOT = Path("/data/LRT/RETFound/Data_split/APTOS2019/test")
OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/protocol_control/actual_cost")

SCOUT_ABLATION_SUMMARY = Path(
    "experiments/v0_8_0_greenscout_feasibility/protocol_control/scout_ablation/scout_ablation_key_summary.csv"
)

GREEN_CKPT = Path("checkpoints/retfound_green/retfoundgreen_statedict.pth")
GREEN_PROBE = Path("experiments/v0_8_0_greenscout_feasibility/green_probe/retfound_green_linear_probe.joblib")

CONV_CKPT = Path("experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth")
RETF_CKPT = Path(
    "experiments/aptos_retfound_mae_cfp_official_protocol/"
    "official_protocol_bs24_epoch50_seed0/checkpoints/retfound_mae_cfp_official_protocol_best.pth"
)

BATCH_SIZE = 32
WARMUP_BATCHES = 3
PROB_COLS = [f"prob_{i}" for i in range(5)]


def collect_test_images() -> pd.DataFrame:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    rows = []
    for p in sorted(DATA_ROOT.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rows.append({"image_key": p.stem, "image_path": str(p)})
    if not rows:
        raise RuntimeError(f"No images found under {DATA_ROOT}")
    return pd.DataFrame(rows).sort_values("image_key").reset_index(drop=True)


def build_transform(size: int, norm: str):
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "half":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f"unknown norm: {norm}")

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_batch(paths, tfm, device):
    xs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append(tfm(img))
    return torch.stack(xs, dim=0).to(device)


def load_green(device):
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(392, 392),
        num_classes=0,
        checkpoint_path=str(GREEN_CKPT),
    )
    model.global_pool = "avg"
    model.eval().to(device)
    probe = joblib.load(GREEN_PROBE)
    return model, probe


def load_convnext(device):
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=5)
    state = torch.load(CONV_CKPT, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def load_retfound(device):
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=5,
        global_pool="avg",
    )
    state = torch.load(RETF_CKPT, map_location="cpu")["model"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def benchmark_green(samples: pd.DataFrame, device: str):
    model, probe = load_green(device)
    tfm = build_transform(392, norm="half")

    return benchmark_model(
        model_name="retfound_green_linear_probe",
        role="scout",
        chain_note="Green encoder forward + sklearn linear-probe predict_proba",
        model=model,
        tfm=tfm,
        samples=samples,
        device=device,
        checkpoint_path=GREEN_CKPT,
        extra_predict_fn=lambda emb: probe.predict_proba(emb.detach().float().cpu().numpy()),
    )


def benchmark_classifier(model_name, role, model, tfm, samples, device, checkpoint_path, note):
    return benchmark_model(
        model_name=model_name,
        role=role,
        chain_note=note,
        model=model,
        tfm=tfm,
        samples=samples,
        device=device,
        checkpoint_path=checkpoint_path,
        extra_predict_fn=None,
    )


def benchmark_model(
    model_name,
    role,
    chain_note,
    model,
    tfm,
    samples,
    device,
    checkpoint_path,
    extra_predict_fn=None,
):
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    paths = samples["image_path"].tolist()

    # warmup
    warm_paths = paths[: min(BATCH_SIZE, len(paths))]
    x = load_batch(warm_paths, tfm, device)
    with torch.no_grad():
        for _ in range(WARMUP_BATCHES):
            out = model(x)
            if extra_predict_fn is not None:
                _ = extra_predict_fn(out)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    batch_rows = []
    pred_rows = []

    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc=model_name):
        part = samples.iloc[start:start + BATCH_SIZE]
        batch_paths = part["image_path"].tolist()
        x = load_batch(batch_paths, tfm, device)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
            if extra_predict_fn is None:
                prob = torch.softmax(out, dim=1).detach().float().cpu().numpy()
            else:
                prob = extra_predict_fn(out)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_ms = (t1 - t0) * 1000.0
        per_img_ms = batch_ms / len(part)

        batch_rows.append({
            "model_name": model_name,
            "role": role,
            "batch_start": int(start),
            "batch_size": int(len(part)),
            "batch_ms": float(batch_ms),
            "per_image_ms": float(per_img_ms),
        })

        for i, (_, r) in enumerate(part.iterrows()):
            pred = int(np.argmax(prob[i]))
            row = {
                "model_name": model_name,
                "image_key": r["image_key"],
                "pred_label": pred,
                "confidence": float(prob[i, pred]),
            }
            for c in range(5):
                row[f"prob_{c}"] = float(prob[i, c])
            pred_rows.append(row)

    batch_df = pd.DataFrame(batch_rows)
    pred_df = pd.DataFrame(pred_rows)

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0

    summary = {
        "model_name": model_name,
        "role": role,
        "chain_note": chain_note,
        "n_images": int(len(samples)),
        "batch_size": BATCH_SIZE,
        "mean_ms_per_image": float(batch_df["per_image_ms"].mean()),
        "median_ms_per_image": float(batch_df["per_image_ms"].median()),
        "p95_ms_per_image": float(batch_df["per_image_ms"].quantile(0.95)),
        "total_forward_ms": float(batch_df["batch_ms"].sum()),
        "images_per_second": float(len(samples) / (batch_df["batch_ms"].sum() / 1000.0)),
        "pytorch_peak_allocated_mem_mb": float(peak_mem_mb),
        "checkpoint_mb": float(checkpoint_path.stat().st_size / 1024 / 1024),
        "device": device,
    }

    return summary, batch_df, pred_df


def build_system_cost(model_cost: pd.DataFrame):
    ab = pd.read_csv(SCOUT_ABLATION_SUMMARY)
    required_cols = {"setting", "budget", "policy", "selected_n"}
    missing_cols = required_cols - set(ab.columns)
    if missing_cols:
        raise ValueError(f"missing columns in scout ablation summary: {sorted(missing_cols)}")

    cost = model_cost.set_index("model_name").to_dict(orient="index")
    n = int(model_cost["n_images"].iloc[0])

    def per_img(model):
        return float(cost[model]["total_forward_ms"]) / n

    def total(model):
        return float(cost[model]["total_forward_ms"])

    rows = []

    # Dense / single online baselines
    dense_systems = [
        ("green_only", ["retfound_green_linear_probe"]),
        ("convnext_only", ["convnext_tiny"]),
        ("retfound_only", ["retfound_mae_cfp_official_protocol"]),
        ("experts_only_dense", ["convnext_tiny", "retfound_mae_cfp_official_protocol"]),
        ("all_three_dense", ["retfound_green_linear_probe", "convnext_tiny", "retfound_mae_cfp_official_protocol"]),
    ]

    for name, models in dense_systems:
        rows.append({
            "scenario": "online_no_cache",
            "setting": name,
            "budget": 1.0 if "dense" in name else 0.0,
            "policy": "none",
            "n_images": n,
            "selected_n": n if "dense" in name else 0,
            "estimated_total_ms": float(sum(total(m) for m in models)),
            "estimated_ms_per_image": float(sum(total(m) for m in models) / n),
            "models_called": "+".join(models),
            "cost_note": "serial sum of measured forward-only model cost across required models",
        })

    # Sparse systems from scout ablation
    for _, r in ab.iterrows():
        setting = r["setting"]
        budget = float(r["budget"])
        policy = r["policy"]
        selected_n = int(r["selected_n"])

        if setting.startswith("A_green_scout_to_convnext_retfound_avg"):
            scout = "retfound_green_linear_probe"
            experts = ["convnext_tiny", "retfound_mae_cfp_official_protocol"]
        elif setting.startswith("B_green_scout_to_convnext_only"):
            scout = "retfound_green_linear_probe"
            experts = ["convnext_tiny"]
        elif setting.startswith("C_green_scout_to_retfound_only"):
            scout = "retfound_green_linear_probe"
            experts = ["retfound_mae_cfp_official_protocol"]
        elif setting.startswith("D_convnext_scout_to_retfound_only"):
            scout = "convnext_tiny"
            experts = ["retfound_mae_cfp_official_protocol"]
        else:
            continue

        expert_ms = sum(selected_n * per_img(e) for e in experts)
        scout_ms = total(scout)

        rows.append({
            "scenario": "online_no_cache",
            "setting": setting,
            "budget": budget,
            "policy": policy,
            "n_images": n,
            "selected_n": selected_n,
            "estimated_total_ms": float(scout_ms + expert_ms),
            "estimated_ms_per_image": float((scout_ms + expert_ms) / n),
            "models_called": scout + " -> " + "+".join(experts),
            "cost_note": "serial forward-equivalent estimate: measured scout forward-only cost + selected_n times measured expert forward-only per-image cost",
        })

        rows.append({
            "scenario": "cached_scout",
            "setting": setting,
            "budget": budget,
            "policy": policy,
            "n_images": n,
            "selected_n": selected_n,
            "estimated_total_ms": float(expert_ms),
            "estimated_ms_per_image": float(expert_ms / n),
            "models_called": "+".join(experts),
            "cost_note": "cached-scout estimate: assumes scout outputs already exist; only selected expert forward-only calls counted",
        })

    out = pd.DataFrame(rows)
    out["estimated_images_per_second"] = out["n_images"] / (out["estimated_total_ms"] / 1000.0)
    return out


def md_table(df):
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

    assert DATA_ROOT.exists(), DATA_ROOT
    assert GREEN_CKPT.exists(), GREEN_CKPT
    assert GREEN_PROBE.exists(), GREEN_PROBE
    assert CONV_CKPT.exists(), CONV_CKPT
    assert RETF_CKPT.exists(), RETF_CKPT
    assert SCOUT_ABLATION_SUMMARY.exists(), SCOUT_ABLATION_SUMMARY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = collect_test_images()
    print(f"[INFO] device={device}")
    print(f"[INFO] n_images={len(samples)}")
    print(f"[INFO] batch_size={BATCH_SIZE}")

    all_summaries = []
    all_batches = []
    all_preds = []

    print("\n[RUN] Green online chain")
    summary, batch_df, pred_df = benchmark_green(samples, device)
    all_summaries.append(summary)
    all_batches.append(batch_df)
    all_preds.append(pred_df)
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[RUN] ConvNeXt classifier")
    conv = load_convnext(device)
    summary, batch_df, pred_df = benchmark_classifier(
        "convnext_tiny",
        "scout_or_expert",
        conv,
        build_transform(224, norm="imagenet"),
        samples,
        device,
        CONV_CKPT,
        "ConvNeXt-Tiny classifier forward",
    )
    all_summaries.append(summary)
    all_batches.append(batch_df)
    all_preds.append(pred_df)
    del conv
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[RUN] RETFound official-protocol classifier")
    retf = load_retfound(device)
    summary, batch_df, pred_df = benchmark_classifier(
        "retfound_mae_cfp_official_protocol",
        "expert",
        retf,
        build_transform(224, norm="imagenet"),
        samples,
        device,
        RETF_CKPT,
        "RETFound-MAE official-protocol classifier forward via timm global_pool=avg",
    )
    all_summaries.append(summary)
    all_batches.append(batch_df)
    all_preds.append(pred_df)
    del retf
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    model_cost = pd.DataFrame(all_summaries)
    batch_cost = pd.concat(all_batches, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    system_cost = build_system_cost(model_cost)

    model_cost.to_csv(OUT_DIR / "actual_online_model_cost.csv", index=False)
    batch_cost.to_csv(OUT_DIR / "actual_online_batch_cost.csv", index=False)
    preds.to_csv(OUT_DIR / "actual_online_predictions_smoke.csv", index=False)
    system_cost.to_csv(OUT_DIR / "actual_sparse_system_cost_estimates.csv", index=False)

    key_settings = system_cost[
        (
            (system_cost["setting"].isin(["green_only", "convnext_only", "retfound_only", "experts_only_dense", "all_three_dense"]))
            | (
                system_cost["setting"].isin([
                    "A_green_scout_to_convnext_retfound_avg",
                    "C_green_scout_to_retfound_only",
                    "D_convnext_scout_to_retfound_only",
                ])
                & (system_cost["budget"] == 0.5)
            )
        )
    ].copy()

    with open(OUT_DIR / "actual_cost_key_findings.md", "w", encoding="utf-8") as f:
        f.write("# v0.8.0e Actual Forward-Cost Benchmark and Sparse System Estimate 关键结果\n\n")
        f.write("## 1. 实验设置\n\n")
        f.write("本轮实验在 APTOS2019 test split 上重新测量三条 online inference 链路的 forward-only cost，并基于 v0.8.0d scout ablation 的 selected_n 估算 sparse system cost。\n\n")
        f.write("成本口径：\n\n")
        f.write("- Green online：RETFound-Green encoder forward + sklearn linear-probe predict_proba。\n")
        f.write("- ConvNeXt online：ConvNeXt-Tiny classifier forward。\n")
        f.write("- RETFound online：RETFound-MAE official-protocol classifier forward，使用 timm `vit_large_patch16_224` + `global_pool='avg'` 严格加载 checkpoint。\n")
        f.write("- online no-cache：包含 scout 全量前向 + 被选中样本的 expert 前向估算。\n")
        f.write("- cached scout：假设 scout 输出已经存在，只统计被选中样本的 expert 前向估算。\n\n")
        f.write("当前结果仍是单 GPU forward benchmark 与 system-level estimate，不等同于完整生产服务中的 I/O、队列、并发和模型加载成本。\n\n")

        f.write("## 2. 单模型 online forward cost\n\n")
        cols = [
            "model_name", "mean_ms_per_image", "median_ms_per_image",
            "images_per_second", "pytorch_peak_allocated_mem_mb", "checkpoint_mb"
        ]
        f.write(md_table(model_cost[cols]))
        f.write("\n\n")

        f.write("## 3. 关键系统成本估算\n\n")
        cols2 = [
            "scenario", "setting", "budget", "selected_n",
            "estimated_ms_per_image", "estimated_images_per_second", "models_called"
        ]
        f.write(md_table(key_settings[cols2]))
        f.write("\n\n")

        f.write("## 4. 当前边界\n\n")
        f.write("- sparse system cost 基于 measured per-model forward cost 与 selected_n 估算，不是实际部署服务压测。\n")
        f.write("- 当前计时从 tensor batch 送入模型前开始，不包含 PIL 图像解码、Resize、Normalize、DataLoader workers、磁盘 I/O、请求排队、模型动态加载和并发调度成本。\n")
        f.write("- Green 与 ConvNeXt 的真实部署成本差异需要结合 batch size、显存占用和服务常驻方式解释。\n")

    print("\n[DONE]")
    print("out_dir:", OUT_DIR)
    print("\nModel cost:")
    print(model_cost.to_string(index=False))
    print("\nKey system cost:")
    print(key_settings.to_string(index=False))


if __name__ == "__main__":
    main()
