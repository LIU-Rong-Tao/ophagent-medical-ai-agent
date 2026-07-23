#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import timm
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct script execution is the controlled runner's established stage contract.
from models.classifiers.builder import build_model  # noqa: E402
from models.datasets.aptos_dataset import build_aptos_dataloader  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_to_idx(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def build_experiment_dir(config: dict) -> Path:
    return Path(config["experiment_root"]) / config["experiment_name"] / config["run_name"]


def structured_config_value(config: dict, section: str, key: str, legacy_key: str):
    section_value = config.get(section)
    if isinstance(section_value, dict) and key in section_value:
        return section_value[key]
    return config[legacy_key]


def build_inference_model(
    config: dict,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        return build_model(
            config=config,
            checkpoint_path=str(checkpoint),
            device=device,
            training=False,
        )

    architecture = str(model_config["architecture"])
    num_classes = int(structured_config_value(config, "data", "num_classes", "num_classes"))
    model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def make_image_key(image_path: str, split_root: Path) -> str:
    p = Path(image_path)
    try:
        return p.relative_to(split_root).as_posix()
    except ValueError:
        return p.name


@torch.no_grad()
def export_predictions(
    *,
    config: dict,
    checkpoint: Path,
    class_to_idx_path: Path,
    out_csv: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model_name: str,
    max_samples: int | None = None,
) -> pd.DataFrame:
    data_root = Path(structured_config_value(config, "data", "root", "data_root"))
    image_size = int(structured_config_value(config, "training", "image_size", "image_size"))

    saved_class_to_idx = load_class_to_idx(class_to_idx_path)

    dataset, loader = build_aptos_dataloader(
        data_root=str(data_root),
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    dataset_class_to_idx = {str(k): int(v) for k, v in dataset.class_to_idx.items()}
    if dataset_class_to_idx != saved_class_to_idx:
        raise RuntimeError(
            "class_to_idx mismatch:\n"
            f"dataset={dataset_class_to_idx}\n"
            f"saved={saved_class_to_idx}\n"
            "停止导出，避免类别顺序错位。"
        )

    model = build_inference_model(config, checkpoint, device)

    split_root = data_root / split
    samples = dataset.samples

    rows = []
    offset = 0

    for images, labels in loader:
        if max_samples is not None:
            remaining = max_samples - offset
            if remaining <= 0:
                break
            images = images[:remaining]
            labels = labels[:remaining]
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
        batch_samples = samples[offset : offset + batch_n]

        for i, (image_path, _) in enumerate(batch_samples):
            row = {
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
                "model_name": model_name,
                "split": split,
            }
            rows.append(row)

        offset += batch_n

    pred_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)

    return pred_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/v0_8_3_glaucoma_scout_routing/configs/convnext_tiny_glaucoma_scout.yaml",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="覆盖历史配置中的数据根目录，便于在迁移后的服务器重放。",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--class-to-idx", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config = dict(config)
        if isinstance(config.get("data"), dict):
            config["data"] = dict(config["data"])
            config["data"]["root"] = args.data_root
        else:
            config["data_root"] = args.data_root

    needs_experiment_dir = not all((args.checkpoint, args.class_to_idx, args.out_csv))
    experiment_dir = build_experiment_dir(config) if needs_experiment_dir else None

    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        experiment_dir / "checkpoints" / f"{config['backbone']}_best.pth"  # type: ignore[operator]
    )
    class_to_idx_path = Path(args.class_to_idx) if args.class_to_idx else (
        experiment_dir / "configs" / "class_to_idx.json"  # type: ignore[operator]
    )
    out_csv = Path(args.out_csv) if args.out_csv else (
        experiment_dir / "evaluation" / args.split / f"{args.split}_predictions.csv"  # type: ignore[operator]
    )

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not class_to_idx_path.exists():
        raise FileNotFoundError(f"class_to_idx not found: {class_to_idx_path}")

    device = torch.device(args.device)
    fallback_name = config.get("experiment_name")
    if not fallback_name:
        fallback_name = structured_config_value(config, "model", "architecture", "backbone")
    model_name = args.model_name or str(fallback_name)

    pred_df = export_predictions(
        config=config,
        checkpoint=checkpoint,
        class_to_idx_path=class_to_idx_path,
        out_csv=out_csv,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        model_name=model_name,
        max_samples=args.max_samples,
    )

    acc = (pred_df["true_label"] == pred_df["pred_label"]).mean()

    print("========== v0.8.3 glaucoma scout prediction cache exported ==========")
    print(f"rows: {len(pred_df)}")
    print(f"accuracy: {acc:.6f}")
    print(f"out_csv: {out_csv}")
    print("class distribution:")
    print(pred_df["true_label"].value_counts().sort_index().to_string())
    print("prediction distribution:")
    print(pred_df["pred_label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
