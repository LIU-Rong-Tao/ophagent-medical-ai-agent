#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torchvision import transforms
from tqdm import tqdm


REGISTRY = Path("experiments/v0_8_2_model_onboarding/configs/model_onboarding_registry.csv")
OUT_DIR = Path("experiments/v0_8_2_model_onboarding/outputs")
PROB_COLS = [f"prob_{i}" for i in range(5)]


def clean_cell(x, default=""):
    if pd.isna(x):
        return default
    return str(x).strip()


def truthy(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def build_transform(img_size: int, norm: str):
    if norm == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm == "half":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f"unsupported norm: {norm}")

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def extract_state_dict(raw, checkpoint_key: str):
    if checkpoint_key:
        cur = raw
        for part in checkpoint_key.split("."):
            cur = cur[part]
        return cur

    if isinstance(raw, dict):
        for k in ["model", "state_dict", "model_state_dict", "net"]:
            if k in raw and isinstance(raw[k], dict):
                return raw[k]

    return raw


def strip_module_prefix(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def build_model(row):
    kwargs = {
        "pretrained": False,
        "num_classes": int(row["num_classes"]),
    }
    global_pool = clean_cell(row.get("global_pool", ""))
    if global_pool:
        kwargs["global_pool"] = global_pool

    return timm.create_model(clean_cell(row["arch"]), **kwargs)


def validate_prediction(pred_path: Path, model_name: str):
    required = ["image_key", "true_label", "pred_label", "confidence", "model_name", *PROB_COLS]
    df = pd.read_csv(pred_path)

    errors = []
    for c in required:
        if c not in df.columns:
            errors.append(f"missing column: {c}")

    if "model_name" in df.columns:
        models = sorted(df["model_name"].dropna().unique().tolist())
        if models != [model_name]:
            errors.append(f"model_name mismatch: {models} != {[model_name]}")

    if not errors:
        prob_sum = df[PROB_COLS].sum(axis=1)
        if not np.allclose(prob_sum, 1.0, atol=1e-4):
            errors.append(f"prob_sum invalid: min={prob_sum.min()}, max={prob_sum.max()}")

        argmax = df[PROB_COLS].to_numpy().argmax(axis=1)
        mismatch = int((argmax != df["pred_label"].astype(int).to_numpy()).sum())
        if mismatch:
            errors.append(f"pred_label != argmax prob: {mismatch}")

    if errors:
        return {
            "prediction_valid": False,
            "prediction_errors": "; ".join(errors),
            "n": len(df),
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "qwk": np.nan,
            "n_error": np.nan,
        }

    y = df["true_label"].astype(int)
    p = df["pred_label"].astype(int)

    return {
        "prediction_valid": True,
        "prediction_errors": "",
        "n": len(df),
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "qwk": cohen_kappa_score(y, p, weights="quadratic"),
        "n_error": int((y != p).sum()),
    }


def image_paths_from_prediction(pred_path: Path):
    df = pd.read_csv(pred_path)
    if "image_path" not in df.columns:
        raise ValueError(f"prediction csv has no image_path: {pred_path}")
    paths = [Path(x) for x in df["image_path"].tolist()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing image files, first 5: {missing[:5]}")
    return paths


def load_batch(paths, tfm, device):
    xs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append(tfm(img))
    return torch.stack(xs, dim=0).to(device)


def benchmark_model(model, paths, tfm, batch_size: int, n_runs: int, device: str, model_name: str):
    rows = []

    for run_id in range(n_runs):
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        warm = load_batch(paths[:min(batch_size, len(paths))], tfm, device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(warm)

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        batch_rows = []
        for start in tqdm(range(0, len(paths), batch_size), desc=f"{model_name} run {run_id+1}/{n_runs}"):
            part = paths[start:start + batch_size]
            x = load_batch(part, tfm, device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                _ = model(x)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_ms = (t1 - t0) * 1000.0
            batch_rows.append({
                "batch_start": start,
                "batch_size_actual": len(part),
                "batch_ms": batch_ms,
                "per_image_ms": batch_ms / len(part),
            })

        bdf = pd.DataFrame(batch_rows)
        total_ms = float(bdf["batch_ms"].sum())
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0

        rows.append({
            "model_name": model_name,
            "run_id": run_id,
            "n_images": len(paths),
            "mean_ms_per_image": total_ms / len(paths),
            "median_ms_per_image": float(bdf["per_image_ms"].median()),
            "p95_ms_per_image": float(bdf["per_image_ms"].quantile(0.95)),
            "total_forward_ms": total_ms,
            "images_per_second": len(paths) / (total_ms / 1000.0),
            "pytorch_peak_allocated_mem_mb": float(peak),
            "batch_size": batch_size,
            "device": device,
        })

    return pd.DataFrame(rows)


def mark_latency_outliers(runs: pd.DataFrame) -> pd.DataFrame:
    out = runs.copy()
    out["latency_relative_to_median"] = 0.0
    out["latency_robust_z"] = 0.0
    out["latency_outlier_rel10pct"] = False
    out["latency_outlier_mad"] = False

    for model_name, idx in out.groupby("model_name").groups.items():
        vals = out.loc[idx, "mean_ms_per_image"].astype(float)
        med = float(vals.median())
        abs_dev = (vals - med).abs()
        mad = float(abs_dev.median())

        out.loc[idx, "latency_relative_to_median"] = abs_dev / med if med else 0.0
        if mad > 0:
            out.loc[idx, "latency_robust_z"] = 0.6745 * abs_dev / mad
        else:
            out.loc[idx, "latency_robust_z"] = 0.0

        out.loc[idx, "latency_outlier_rel10pct"] = out.loc[idx, "latency_relative_to_median"] > 0.10
        out.loc[idx, "latency_outlier_mad"] = out.loc[idx, "latency_robust_z"] > 3.5

    # Formal outlier: only large relative deviation is treated as a real instability.
    # MAD is kept as a sensitive diagnostic flag because with n_runs=5 and very stable latency,
    # tiny absolute changes can produce a large robust-z score.
    out["latency_mad_sensitive_flag"] = out["latency_outlier_mad"]
    out["latency_outlier_any"] = out["latency_outlier_rel10pct"]
    return out


def summarize_runs(runs: pd.DataFrame, ckpt_mb: float, img_size: int, arch: str, norm: str):
    runs = mark_latency_outliers(runs)

    g = runs.groupby("model_name", as_index=False).agg(
        n_runs=("run_id", "count"),
        n_latency_outlier_any=("latency_outlier_any", "sum"),
        n_latency_outlier_rel10pct=("latency_outlier_rel10pct", "sum"),
        n_latency_outlier_mad=("latency_outlier_mad", "sum"),
        n_latency_mad_sensitive_flag=("latency_mad_sensitive_flag", "sum"),
        mean_ms_per_image=("mean_ms_per_image", "median"),
        mean_ms_per_image_mean=("mean_ms_per_image", "mean"),
        mean_ms_per_image_std=("mean_ms_per_image", "std"),
        median_ms_per_image=("median_ms_per_image", "median"),
        p95_ms_per_image=("p95_ms_per_image", "median"),
        total_forward_ms=("total_forward_ms", "median"),
        total_forward_ms_mean=("total_forward_ms", "mean"),
        images_per_second=("images_per_second", "median"),
        pytorch_peak_allocated_mem_mb=("pytorch_peak_allocated_mem_mb", "max"),
        batch_size=("batch_size", "first"),
        device=("device", "first"),
    )
    g["mean_ms_per_image_cv"] = g["mean_ms_per_image_std"] / g["mean_ms_per_image_mean"]
    g["checkpoint_mb"] = ckpt_mb
    g["img_size"] = img_size
    g["arch"] = arch
    g["norm"] = norm
    g["latency_cv_gt_10pct"] = g["mean_ms_per_image_cv"] > 0.10
    g["latency_has_run_outlier"] = g["n_latency_outlier_any"] > 0
    g["latency_has_mad_sensitive_flag"] = g["n_latency_mad_sensitive_flag"] > 0
    g["cost_note"] = "multi-run single-GPU forward-only benchmark; excludes PIL decode/transform timing"
    return g


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reg = pd.read_csv(REGISTRY)
    reg = reg[(reg["enabled"].astype(int) == 1) & (reg["model_type"] == "timm_classifier")].copy()

    all_runs = []
    summaries = []
    reports = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for _, row in reg.iterrows():
        model_name = clean_cell(row["model_name"])
        ckpt = Path(clean_cell(row["checkpoint_path"]))
        pred_path = Path(clean_cell(row["prediction_csv"]))
        img_size = int(row["img_size"])
        batch_size = int(row["batch_size"])
        n_runs = int(row["n_runs"])
        norm = clean_cell(row["norm"], "imagenet")
        arch = clean_cell(row["arch"])

        load_ok = False
        load_error = ""

        pred_result = validate_prediction(pred_path, model_name)

        try:
            model = build_model(row)
            raw = torch.load(ckpt, map_location="cpu")
            sd = extract_state_dict(raw, clean_cell(row.get("checkpoint_key", "")))
            if truthy(row.get("strip_module_prefix", 0)):
                sd = strip_module_prefix(sd)
            model.load_state_dict(sd, strict=True)
            model.eval().to(device)
            load_ok = True
        except Exception as e:
            model = None
            load_error = repr(e)

        reports.append({
            "model_name": model_name,
            "arch": arch,
            "checkpoint_path": str(ckpt),
            "prediction_csv": str(pred_path),
            "strict_load_ok": load_ok,
            "strict_load_error": load_error,
            **pred_result,
        })

        if not load_ok:
            continue

        paths = image_paths_from_prediction(pred_path)
        tfm = build_transform(img_size, norm)

        runs = benchmark_model(model, paths, tfm, batch_size, n_runs, device, model_name)
        runs = mark_latency_outliers(runs)
        runs["checkpoint_mb"] = ckpt.stat().st_size / 1024 / 1024
        runs["img_size"] = img_size
        runs["arch"] = arch
        runs["norm"] = norm
        all_runs.append(runs)

        summary = summarize_runs(
            runs,
            ckpt_mb=ckpt.stat().st_size / 1024 / 1024,
            img_size=img_size,
            arch=arch,
            norm=norm,
        )
        summaries.append(summary)

        del model
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    report_df = pd.DataFrame(reports)
    report_df.to_csv(OUT_DIR / "model_onboarding_validation.csv", index=False)

    if all_runs:
        runs_df = pd.concat(all_runs, ignore_index=True)
        runs_df.to_csv(OUT_DIR / "model_forward_cost_runs.csv", index=False)
    else:
        runs_df = pd.DataFrame()

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        summary_df.to_csv(OUT_DIR / "model_forward_cost_summary.csv", index=False)
    else:
        summary_df = pd.DataFrame()

    lines = []
    lines.append("# v0.8.2 Model Onboarding Report\n")
    lines.append("## 1. Scope\n")
    lines.append("This report validates registry-driven onboarding for timm classifier models.\n")
    lines.append("## 2. Validation Summary\n")

    for _, r in report_df.iterrows():
        lines.append(f"### {r['model_name']}\n")
        lines.append(f"- arch: `{r['arch']}`")
        lines.append(f"- strict_load_ok: `{r['strict_load_ok']}`")
        lines.append(f"- prediction_valid: `{r['prediction_valid']}`")
        lines.append(f"- n: {r['n']}")
        lines.append(f"- accuracy: {r['accuracy']:.6f}")
        lines.append(f"- macro_f1: {r['macro_f1']:.6f}")
        lines.append(f"- qwk: {r['qwk']:.6f}")
        lines.append(f"- n_error: {r['n_error']}")
        if r["strict_load_error"]:
            lines.append(f"- strict_load_error: `{r['strict_load_error']}`")
        if r["prediction_errors"]:
            lines.append(f"- prediction_errors: `{r['prediction_errors']}`")
        lines.append("")

    if not summary_df.empty:
        lines.append("## 3. Forward-Cost Summary\n")
        for _, r in summary_df.iterrows():
            lines.append(f"### {r['model_name']}\n")
            lines.append(f"- n_runs: {int(r['n_runs'])}")
            lines.append(f"- mean_ms_per_image_median: {r['mean_ms_per_image']:.6f}")
            lines.append(f"- mean_ms_per_image_mean: {r['mean_ms_per_image_mean']:.6f}")
            lines.append(f"- mean_ms_per_image_std: {r['mean_ms_per_image_std']:.6f}")
            lines.append(f"- mean_ms_per_image_cv: {r['mean_ms_per_image_cv']:.6f}")
            lines.append(f"- total_forward_ms_median: {r['total_forward_ms']:.3f}")
            lines.append(f"- images_per_second_median: {r['images_per_second']:.3f}")
            lines.append(f"- peak_mem_max_mb: {r['pytorch_peak_allocated_mem_mb']:.3f}")
            lines.append(f"- checkpoint_mb: {r['checkpoint_mb']:.3f}")
            lines.append(f"- latency_cv_gt_10pct: `{r['latency_cv_gt_10pct']}`")
            lines.append(f"- latency_has_run_outlier: `{r['latency_has_run_outlier']}`")
            lines.append(f"- n_latency_outlier_any: {int(r['n_latency_outlier_any'])}")
            lines.append(f"- latency_has_mad_sensitive_flag: `{r['latency_has_mad_sensitive_flag']}`")
            lines.append(f"- n_latency_mad_sensitive_flag: {int(r['n_latency_mad_sensitive_flag'])}")
            lines.append("")

    lines.append("## 4. Boundary\n")
    lines.append("- Current script only supports registry-declared timm classifier models.")
    lines.append("- Cost is forward-only latency and excludes image decode / transform / dataloader / service overhead.")
    lines.append("- Non-timm foundation or embedding models should use a separate onboarding script.")

    (OUT_DIR / "model_onboarding_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("[DONE] v0.8.2 onboarding")
    print("\nValidation:")
    print(report_df.to_string(index=False))
    if not summary_df.empty:
        print("\nCost summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
