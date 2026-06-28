#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retfound-root",
        default="/data/LRT/RETFound",
        help="RETFound repo root containing models_vit.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="/data/LRT/RETFound/output_dir/retfound_dinov2_Glaucoma_fundus_finetune/checkpoint-best.pth",
    )
    parser.add_argument(
        "--data-root",
        default="/data/LRT/RETFound/Data_split/Glaucoma_fundus",
    )
    parser.add_argument(
        "--out-csv",
        default=(
            "experiments/v0_8_3_glaucoma_scout_routing/"
            "retfound_dinov2_glaucoma_expert/evaluation/test/test_predictions.csv"
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", default="retfound_dinov2_glaucoma_expert")
    return parser.parse_args()


def build_eval_transform(input_size: int, norm: str):
    if str(norm).upper() == "IMAGENET":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported norm={norm}. 当前只按 checkpoint args.norm=IMAGENET 处理。")

    return transforms.Compose([
        transforms.Resize(int(input_size / 0.875), interpolation=Image.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_checkpoint_and_model(retfound_root: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(retfound_root))

    import models_vit  # noqa: E402

    # v0.8.3 export only needs the architecture.
    # The glaucoma expert checkpoint is a full fine-tuned checkpoint,
    # so do not let timm download vit_large_patch14_dinov2.lvd142m here.
    import timm  # noqa: E402

    _orig_create_model = timm.create_model

    def _create_model_no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return _orig_create_model(*args, **kwargs)

    timm.create_model = _create_model_no_pretrained
    if hasattr(models_vit, "timm"):
        models_vit.timm.create_model = _create_model_no_pretrained

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt or "args" not in ckpt:
        raise RuntimeError(f"Unexpected checkpoint format: keys={ckpt.keys() if isinstance(ckpt, dict) else type(ckpt)}")

    ckpt_args = copy.deepcopy(ckpt["args"])

    model_name = ckpt_args.model
    if not hasattr(models_vit, model_name):
        raise RuntimeError(f"models_vit has no constructor named {model_name}")

    constructor = getattr(models_vit, model_name)

    # 兼容 RETFound_dinov2(args, **kwargs) / RETFound_dinov2(args=args, **kwargs) 两种写法。
    try:
        model = constructor(
            args=ckpt_args,
            num_classes=int(ckpt_args.nb_classes),
            drop_path_rate=float(ckpt_args.drop_path),
            global_pool="token",
        )
    except TypeError:
        model = constructor(
            ckpt_args,
            num_classes=int(ckpt_args.nb_classes),
            drop_path_rate=float(ckpt_args.drop_path),
            global_pool="token",
        )

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)

    # 这里 head 是已 fine-tune 的三分类 head，不能缺。
    bad_missing = [k for k in missing if k.startswith("head.")]
    bad_unexpected = [k for k in unexpected if k.startswith("head.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Head loading mismatch. missing={bad_missing}, unexpected={bad_unexpected}"
        )

    model.to(device)
    model.eval()

    return model, ckpt_args, missing, unexpected


def make_image_key(image_path: str, split_root: Path) -> str:
    p = Path(image_path)
    try:
        return p.relative_to(split_root).as_posix()
    except ValueError:
        return p.name


@torch.no_grad()
def main() -> None:
    args = parse_args()

    retfound_root = Path(args.retfound_root)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    split_root = data_root / args.split
    out_csv = Path(args.out_csv)
    device = torch.device(args.device)

    if not retfound_root.exists():
        raise FileNotFoundError(f"RETFound root not found: {retfound_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not split_root.exists():
        raise FileNotFoundError(f"split root not found: {split_root}")

    model, ckpt_args, missing, unexpected = load_checkpoint_and_model(
        retfound_root=retfound_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    eval_transform = build_eval_transform(
        input_size=int(ckpt_args.input_size),
        norm=str(getattr(ckpt_args, "norm", "IMAGENET")),
    )

    dataset = datasets.ImageFolder(root=str(split_root), transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    expected_class_to_idx = {
        "anormal_control": 0,
        "bearly_glaucoma": 1,
        "cadvanced_glaucoma": 2,
    }
    if dataset.class_to_idx != expected_class_to_idx:
        raise RuntimeError(
            "class_to_idx mismatch:\n"
            f"dataset={dataset.class_to_idx}\n"
            f"expected={expected_class_to_idx}"
        )

    rows = []
    offset = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        top2_conf, top2_idx = torch.topk(probs, k=2, dim=1)
        pred = top2_idx[:, 0]
        confidence = top2_conf[:, 0]
        margin = top2_conf[:, 0] - top2_conf[:, 1]
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

        probs_cpu = probs.cpu()
        labels_cpu = labels.cpu()
        pred_cpu = pred.cpu()
        confidence_cpu = confidence.cpu()
        entropy_cpu = entropy.cpu()
        margin_cpu = margin.cpu()

        batch_n = images.size(0)
        batch_samples = dataset.samples[offset : offset + batch_n]

        for i, (image_path, _) in enumerate(batch_samples):
            rows.append({
                "image_path": str(image_path),
                "image_key": make_image_key(str(image_path), split_root),
                "true_label": int(labels_cpu[i].item()),
                "pred_label": int(pred_cpu[i].item()),
                "prob_0": float(probs_cpu[i, 0].item()),
                "prob_1": float(probs_cpu[i, 1].item()),
                "prob_2": float(probs_cpu[i, 2].item()),
                "confidence": float(confidence_cpu[i].item()),
                "entropy": float(entropy_cpu[i].item()),
                "margin": float(margin_cpu[i].item()),
                "model_name": args.model_name,
                "split": args.split,
            })

        offset += batch_n

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()

    metrics = {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "class_to_idx": dataset.class_to_idx,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(ckpt_args.epochs) if False else None,
        "ckpt_model": str(ckpt_args.model),
        "ckpt_finetune": str(ckpt_args.finetune),
        "input_size": int(ckpt_args.input_size),
        "norm": str(ckpt_args.norm),
        "missing_keys_n": len(missing),
        "unexpected_keys_n": len(unexpected),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["normal_control", "early_glaucoma", "advanced_glaucoma"],
            digits=4,
            output_dict=True,
        ),
    }

    metrics_path = out_csv.parent / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("========== v0.8.3 RETFound-DINOv2 glaucoma expert prediction cache exported ==========")
    print(f"rows: {len(df)}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"macro_f1: {metrics['macro_f1']:.6f}")
    print(f"kappa: {metrics['kappa']:.6f}")
    print(f"out_csv: {out_csv}")
    print(f"metrics_json: {metrics_path}")
    print("class distribution:")
    print(df["true_label"].value_counts().sort_index().to_string())
    print("prediction distribution:")
    print(df["pred_label"].value_counts().sort_index().to_string())
    print("missing_keys_n:", len(missing))
    print("unexpected_keys_n:", len(unexpected))
    print("missing_keys:", missing)
    print("unexpected_keys:", unexpected)


if __name__ == "__main__":
    main()
