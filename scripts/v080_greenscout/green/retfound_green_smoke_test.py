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


CKPT = Path("checkpoints/retfound_green/retfoundgreen_statedict.pth")
DEMO_DIR = Path("demo_samples")

SMOKE_OUT = Path("experiments/v0_8_0_greenscout_feasibility/green_smoke/retfound_green_smoke_test.csv")
COST_OUT = Path("experiments/v0_8_0_greenscout_feasibility/cost/inference_cost_table.csv")

MODEL_NAME = "retfound_green"
ARCH = "vit_small_patch14_reg4_dinov2"
IMG_SIZE = 392

LABEL_MAP = {
    "anodr": 0,
    "no_dr": 0,
    "no dr": 0,
    "0": 0,

    "bmilddr": 1,
    "mild": 1,
    "milddr": 1,
    "mild_dr": 1,
    "1": 1,

    "cmoderatedr": 2,
    "moderate": 2,
    "moderatedr": 2,
    "moderate_dr": 2,
    "2": 2,

    "dseveredr": 3,
    "severe": 3,
    "severedr": 3,
    "severe_dr": 3,
    "3": 3,

    "eprodr": 4,
    "eproliferativedr": 4,
    "e_proliferative_dr": 4,
    "proliferative": 4,
    "proliferativedr": 4,
    "proliferative_dr": 4,
    "4": 4,
}

LABEL_NAME = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def normalize_dir_name(s: str) -> str:
    return (
        s.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def infer_label_from_path(path: Path) -> int:
    # 优先用父目录名判断
    candidates = [path.parent.name, path.parent.parent.name]
    for name in candidates:
        key = normalize_dir_name(name)
        key_compact = key.replace("_", "")
        if key in LABEL_MAP:
            return LABEL_MAP[key]
        if key_compact in LABEL_MAP:
            return LABEL_MAP[key_compact]

    raise ValueError(f"cannot infer label from path: {path}")


def collect_demo_images() -> pd.DataFrame:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted([p for p in DEMO_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    if not paths:
        raise FileNotFoundError(f"no image files found under {DEMO_DIR}")

    rows = []
    for p in paths:
        y = infer_label_from_path(p)
        rows.append({
            "image_key": p.stem,
            "image_path": str(p),
            "true_label": y,
            "true_label_name": LABEL_NAME[y],
        })

    df = pd.DataFrame(rows).sort_values(["true_label", "image_key"]).reset_index(drop=True)
    return df


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def main():
    assert CKPT.exists(), f"checkpoint missing: {CKPT}"
    assert DEMO_DIR.exists(), f"demo dir missing: {DEMO_DIR}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] loading RETFound-Green")
    print(f"[INFO] checkpoint={CKPT}")
    print(f"[INFO] demo_dir={DEMO_DIR}")
    print(f"[INFO] device={device}")

    model = timm.create_model(
        ARCH,
        img_size=(IMG_SIZE, IMG_SIZE),
        num_classes=0,
        checkpoint_path=str(CKPT),
    )
    model.global_pool = "avg"
    model.eval().to(device)

    tfm = build_transform()
    samples = collect_demo_images()

    print("\n[INFO] demo samples:")
    print(samples.to_string(index=False))

    # warmup
    first_path = samples.iloc[0]["image_path"]
    img = Image.open(first_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    rows = []

    for _, r in tqdm(samples.iterrows(), total=len(samples)):
        image_key = r["image_key"]
        image_path = r["image_path"]

        try:
            img = Image.open(image_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            with torch.no_grad():
                y = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            y_cpu = y.detach().float().cpu()
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0

            rows.append({
                "model_name": MODEL_NAME,
                "image_key": image_key,
                "image_path": image_path,
                "true_label": int(r["true_label"]),
                "true_label_name": r["true_label_name"],
                "ok": True,
                "error": "",
                "embedding_shape": "x".join(map(str, y_cpu.shape)),
                "embedding_dim": int(y_cpu.shape[-1]),
                "embedding_norm": float(torch.linalg.norm(y_cpu, dim=1).mean().item()),
                "embedding_mean": float(y_cpu.mean().item()),
                "embedding_std": float(y_cpu.std().item()),
                "inference_ms": float((t1 - t0) * 1000.0),
                "peak_mem_mb": float(peak_mem_mb),
                "device": device,
                "img_size": IMG_SIZE,
            })

        except Exception as e:
            rows.append({
                "model_name": MODEL_NAME,
                "image_key": image_key,
                "image_path": image_path,
                "true_label": int(r["true_label"]),
                "true_label_name": r["true_label_name"],
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "embedding_shape": "",
                "embedding_dim": -1,
                "embedding_norm": -1,
                "embedding_mean": -1,
                "embedding_std": -1,
                "inference_ms": -1,
                "peak_mem_mb": -1,
                "device": device,
                "img_size": IMG_SIZE,
            })

    smoke = pd.DataFrame(rows)
    SMOKE_OUT.parent.mkdir(parents=True, exist_ok=True)
    COST_OUT.parent.mkdir(parents=True, exist_ok=True)
    smoke.to_csv(SMOKE_OUT, index=False)

    ok = smoke[smoke["ok"] == True].copy()

    cost = pd.DataFrame([{
        "model_name": MODEL_NAME,
        "role": "scout",
        "stage": "demo_samples_embedding_smoke_test",
        "checkpoint_mb": CKPT.stat().st_size / 1024 / 1024,
        "n_images": int(len(smoke)),
        "n_ok": int(len(ok)),
        "ok_rate": float(len(ok) / len(smoke)) if len(smoke) else 0.0,
        "mean_inference_ms": float(ok["inference_ms"].mean()) if len(ok) else -1,
        "median_inference_ms": float(ok["inference_ms"].median()) if len(ok) else -1,
        "max_inference_ms": float(ok["inference_ms"].max()) if len(ok) else -1,
        "max_peak_mem_mb": float(ok["peak_mem_mb"].max()) if len(ok) else -1,
        "embedding_dim": int(ok["embedding_dim"].mode().iloc[0]) if len(ok) else -1,
        "device": device,
        "img_size": IMG_SIZE,
        "note": "RETFound-Green demo_samples smoke test only; embedding output; no classification head.",
    }])

    cost.to_csv(COST_OUT, index=False)

    print("\n[DONE]")
    print("smoke:", SMOKE_OUT)
    print("cost:", COST_OUT)

    print("\n=== label counts ===")
    print(smoke.groupby(["true_label", "true_label_name"]).size().to_string())

    print("\n=== smoke summary ===")
    print(smoke[[
        "image_key", "true_label", "true_label_name", "ok",
        "embedding_shape", "inference_ms", "peak_mem_mb", "error"
    ]].to_string(index=False))

    print("\n=== cost summary ===")
    print(cost.to_string(index=False))


if __name__ == "__main__":
    main()
