#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


DATA_ROOT = Path("/data/LRT/RETFound/Data_split/APTOS2019")
CKPT = Path("checkpoints/retfound_green/retfoundgreen_statedict.pth")

OUT_DIR = Path("experiments/v0_8_0_greenscout_feasibility/green_embeddings")
SUMMARY_OUT = Path("experiments/v0_8_0_greenscout_feasibility/cost/green_full_embedding_cost.csv")

MODEL_NAME = "retfound_green"
ARCH = "vit_small_patch14_reg4_dinov2"
IMG_SIZE = 392
BATCH_SIZE = 32

LABEL_MAP = {
    "anodr": 0,
    "bmilddr": 1,
    "cmoderatedr": 2,
    "dseveredr": 3,
    "eproliferativedr": 4,
    "eprodr": 4,
}

LABEL_NAME = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def collect_split(split: str) -> pd.DataFrame:
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        raise FileNotFoundError(f"missing split dir: {split_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    rows = []

    for p in sorted(split_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        label_dir = p.parent.name.lower()
        if label_dir not in LABEL_MAP:
            raise ValueError(f"unknown label dir: {p}")

        y = LABEL_MAP[label_dir]
        rows.append({
            "split": split,
            "image_key": p.stem,
            "image_path": str(p),
            "true_label": y,
            "true_label_name": LABEL_NAME[y],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no images found for split={split}")

    return df.sort_values(["true_label", "image_key"]).reset_index(drop=True)


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_model(device: str):
    model = timm.create_model(
        ARCH,
        img_size=(IMG_SIZE, IMG_SIZE),
        num_classes=0,
        checkpoint_path=str(CKPT),
    )
    model.global_pool = "avg"
    model.eval().to(device)
    return model


def load_batch(paths, tfm, device):
    xs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append(tfm(img))
    return torch.stack(xs, dim=0).to(device)


def export_split(split: str, model, tfm, device: str):
    df = collect_split(split)

    rows = []
    latencies = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # warmup
    warm_paths = df["image_path"].head(min(BATCH_SIZE, len(df))).tolist()
    x = load_batch(warm_paths, tfm, device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc=f"export {split}"):
        part = df.iloc[start:start + BATCH_SIZE].copy()
        paths = part["image_path"].tolist()

        x = load_batch(paths, tfm, device)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            emb = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_ms = (t1 - t0) * 1000.0
        per_image_ms = batch_ms / len(part)
        latencies.extend([per_image_ms] * len(part))

        emb = emb.detach().float().cpu().numpy()

        for i, (_, r) in enumerate(part.iterrows()):
            row = {
                "model_name": MODEL_NAME,
                "split": r["split"],
                "image_key": r["image_key"],
                "image_path": r["image_path"],
                "true_label": int(r["true_label"]),
                "true_label_name": r["true_label_name"],
                "embedding_dim": int(emb.shape[1]),
                "batch_size": len(part),
                "inference_ms_per_image": float(per_image_ms),
            }
            for j in range(emb.shape[1]):
                row[f"emb_{j}"] = float(emb[i, j])
            rows.append(row)

    out = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"retfound_green_{split}_embeddings.csv"
    out.to_csv(out_path, index=False)

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0

    summary = {
        "model_name": MODEL_NAME,
        "split": split,
        "n_images": int(len(df)),
        "embedding_dim": 384,
        "batch_size": BATCH_SIZE,
        "mean_ms_per_image": float(pd.Series(latencies).mean()),
        "median_ms_per_image": float(pd.Series(latencies).median()),
        "max_ms_per_image": float(pd.Series(latencies).max()),
        "peak_mem_mb": float(peak_mem_mb),
        "checkpoint_mb": CKPT.stat().st_size / 1024 / 1024,
        "img_size": IMG_SIZE,
        "device": device,
        "output_csv": str(out_path),
    }

    return summary


def main():
    assert DATA_ROOT.exists(), f"missing data root: {DATA_ROOT}"
    assert CKPT.exists(), f"missing checkpoint: {CKPT}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] data_root={DATA_ROOT}")
    print(f"[INFO] checkpoint={CKPT}")

    model = load_model(device)
    tfm = build_transform()

    summaries = []
    for split in ["train", "val", "test"]:
        summaries.append(export_split(split, model, tfm, device))

    summary_df = pd.DataFrame(summaries)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("\n[DONE]")
    print("summary:", SUMMARY_OUT)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
