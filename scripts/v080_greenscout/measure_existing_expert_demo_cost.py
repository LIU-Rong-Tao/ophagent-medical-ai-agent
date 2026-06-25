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


DEMO_DIR = Path("demo_samples")
COST_OUT = Path("experiments/v0_8_0_greenscout_feasibility/cost/inference_cost_table.csv")
PRED_OUT = Path("experiments/v0_8_0_greenscout_feasibility/predictions/demo_existing_expert_predictions.csv")

LABEL_MAP = {
    "anodr": 0,
    "bmilddr": 1,
    "cmoderatedr": 2,
    "dseveredr": 3,
    "eproliferativedr": 4,
}

LABEL_NAME = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

MODELS = [
    {
        "model_name": "convnext_tiny",
        "role": "existing_expert",
        "arch": "convnext_tiny",
        "checkpoint": "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth",
        "img_size": 224,
        "num_classes": 5,
    },
    {
        "model_name": "retfound_mae_cfp_official_like",
        "role": "existing_expert",
        "arch": "vit_large_patch16_224",
        "checkpoint": "experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/checkpoints/retfound_mae_cfp_best.pth",
        "img_size": 224,
        "num_classes": 5,
    },
]


def collect_demo_images() -> pd.DataFrame:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted([p for p in DEMO_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    rows = []
    for p in paths:
        dirname = p.parent.name.lower()
        if dirname not in LABEL_MAP:
            raise ValueError(f"unknown label dir: {p}")
        y = LABEL_MAP[dirname]
        rows.append({
            "image_key": p.stem,
            "image_path": str(p),
            "true_label": y,
            "true_label_name": LABEL_NAME[y],
        })

    return pd.DataFrame(rows).sort_values(["true_label", "image_key"]).reset_index(drop=True)


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_model(spec: dict, device: str):
    model = timm.create_model(
        spec["arch"],
        pretrained=False,
        num_classes=spec["num_classes"],
    )
    state = torch.load(spec["checkpoint"], map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def main():
    assert DEMO_DIR.exists(), f"missing demo dir: {DEMO_DIR}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = collect_demo_images()

    all_pred_rows = []
    cost_rows = []

    for spec in MODELS:
        print(f"\n[INFO] loading {spec['model_name']}")
        ckpt = Path(spec["checkpoint"])
        assert ckpt.exists(), f"missing checkpoint: {ckpt}"

        model = load_model(spec, device)
        tfm = build_transform(spec["img_size"])

        # warmup
        first = Image.open(samples.iloc[0]["image_path"]).convert("RGB")
        x = tfm(first).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()

        rows = []

        for _, r in tqdm(samples.iterrows(), total=len(samples), desc=spec["model_name"]):
            img = Image.open(r["image_path"]).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            probs_cpu = probs.detach().float().cpu()[0]
            pred = int(torch.argmax(probs_cpu).item())
            conf = float(probs_cpu[pred].item())
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0

            row = {
                "model_name": spec["model_name"],
                "role": spec["role"],
                "image_key": r["image_key"],
                "image_path": r["image_path"],
                "true_label": int(r["true_label"]),
                "true_label_name": r["true_label_name"],
                "pred_label": pred,
                "pred_label_name": LABEL_NAME[pred],
                "confidence": conf,
                "correct": bool(pred == int(r["true_label"])),
                "inference_ms": float((t1 - t0) * 1000.0),
                "peak_mem_mb": float(peak_mem_mb),
                "device": device,
                "img_size": int(spec["img_size"]),
            }

            for i in range(5):
                row[f"prob_{i}"] = float(probs_cpu[i].item())

            rows.append(row)

        pred_df = pd.DataFrame(rows)
        all_pred_rows.append(pred_df)

        cost_rows.append({
            "model_name": spec["model_name"],
            "role": spec["role"],
            "stage": "demo_samples_classifier_inference",
            "checkpoint_mb": ckpt.stat().st_size / 1024 / 1024,
            "n_images": int(len(pred_df)),
            "n_ok": int(len(pred_df)),
            "ok_rate": 1.0,
            "mean_inference_ms": float(pred_df["inference_ms"].mean()),
            "median_inference_ms": float(pred_df["inference_ms"].median()),
            "max_inference_ms": float(pred_df["inference_ms"].max()),
            "max_peak_mem_mb": float(pred_df["peak_mem_mb"].max()),
            "embedding_dim": "",
            "device": device,
            "img_size": int(spec["img_size"]),
            "note": "Existing expert classifier inference on demo_samples.",
        })

    pred_all = pd.concat(all_pred_rows, ignore_index=True)
    cost_new = pd.DataFrame(cost_rows)

    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    COST_OUT.parent.mkdir(parents=True, exist_ok=True)

    pred_all.to_csv(PRED_OUT, index=False)

    if COST_OUT.exists():
        old = pd.read_csv(COST_OUT)
        # 避免重复追加同名专家
        old = old[~old["model_name"].isin(cost_new["model_name"])]
        cost_all = pd.concat([old, cost_new], ignore_index=True)
    else:
        cost_all = cost_new

    cost_all.to_csv(COST_OUT, index=False)

    print("\n[DONE]")
    print("pred:", PRED_OUT)
    print("cost:", COST_OUT)

    print("\n=== cost summary ===")
    print(cost_all.to_string(index=False))

    print("\n=== expert predictions ===")
    print(pred_all[[
        "model_name", "image_key", "true_label", "pred_label",
        "confidence", "correct", "inference_ms", "peak_mem_mb"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
