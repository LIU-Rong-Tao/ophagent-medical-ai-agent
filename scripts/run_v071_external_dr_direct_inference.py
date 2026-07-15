#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v0.7.1 外部 DR 数据集直接推理脚本。

使用 APTOS-trained frozen checkpoints，直接在 IDRiD_data / MESSIDOR2 test split 上推理。
本脚本不使用外部 train / val 训练或调参，只输出分类迁移结果和后续复核排序所需信号。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, recall_score


BACKBONES = {
    "convnext_tiny": {
        "model_name": "convnext_tiny",
        "checkpoint": "experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth",
    },
    "swin_tiny": {
        "model_name": "swin_tiny_patch4_window7_224.ms_in1k",
        "checkpoint": "experiments/aptos_swin_tiny/lr1e-4_bs32_seed42/checkpoints/swin_tiny_patch4_window7_224.ms_in1k_best.pth",
    },
    "vit_b_imagenet": {
        "model_name": "vit_base_patch16_224",
        "checkpoint": "experiments/aptos_vit_base_patch16_imagenet/lr1e-4_bs32_seed42/checkpoints/vit_base_patch16_best.pth",
    },
    "vit_b_official_like": {
        "model_name": "vit_base_patch16_224",
        "checkpoint": "experiments/aptos_vit_base_patch16_official_like/official_like_bs32_epoch50_seed42/checkpoints/vit_base_patch16_best.pth",
    },
    "vit_l_official_like": {
        "model_name": "vit_large_patch16_224",
        "checkpoint": "experiments/aptos_vit_large_patch16_official_like/official_like_bs32_epoch50_seed42/checkpoints/vit_large_patch16_best.pth",
    },
    "retfound_mae_cfp_official_like": {
        "model_name": "vit_large_patch16_224",
        "checkpoint": "experiments/aptos_retfound_mae_cfp_official_like/official_like_bs32_epoch50_seed42/checkpoints/retfound_mae_cfp_best.pth",
    },
}


def build_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ExternalDRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, image_transform):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = str(row["relative_image_path"])
        image_path = self.data_root / rel_path

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        return {
            "image": image_tensor,
            "dataset": str(row["dataset"]),
            "split": str(row["split"]),
            "relative_image_path": rel_path,
            "image_name": str(row["image_name"]),
            "image_key": f'{row["dataset"]}::{rel_path}',
            "true_grade": int(row["grade"]),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    out = {k: [item[k] for item in batch] for k in batch[0] if k != "image"}
    out["image"] = images
    return out


def load_model(backbone: str, repo_root: Path, device: torch.device) -> torch.nn.Module:
    cfg = BACKBONES[backbone]
    ckpt_path = repo_root / cfg["checkpoint"]

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")

    model = timm.create_model(
        cfg["model_name"],
        pretrained=False,
        num_classes=5,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_one_backbone(
    *,
    backbone: str,
    df: pd.DataFrame,
    data_root: Path,
    repo_root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    dataset = ExternalDRDataset(df, data_root, build_transform(224))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    model = load_model(backbone, repo_root, device)
    grade_values = torch.arange(5, dtype=torch.float32, device=device)

    rows = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        top2_conf, top2_idx = torch.topk(probs, k=2, dim=1)

        pred_grade = top2_idx[:, 0]
        confidence = top2_conf[:, 0]
        top2_grade = top2_idx[:, 1]
        top2_confidence = top2_conf[:, 1]
        margin = confidence - top2_confidence

        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        expected_grade = (probs * grade_values[None, :]).sum(dim=1)
        severe_prob_mass = probs[:, 3] + probs[:, 4]
        expected_gap = expected_grade - pred_grade.float()
        gated_severe_prob_mass = torch.where(
            pred_grade <= 2,
            severe_prob_mass,
            torch.zeros_like(severe_prob_mass),
        )

        probs_np = probs.cpu().numpy()

        for i in range(len(batch["image_key"])):
            row = {
                "dataset": batch["dataset"][i],
                "split": batch["split"][i],
                "image_key": batch["image_key"][i],
                "relative_image_path": batch["relative_image_path"][i],
                "image_name": batch["image_name"][i],
                "true_grade": int(batch["true_grade"][i]),
                "backbone": backbone,
                "model_name": BACKBONES[backbone]["model_name"],
                "pred_grade": int(pred_grade[i].item()),
                "confidence": float(confidence[i].item()),
                "top2_grade": int(top2_grade[i].item()),
                "top2_confidence": float(top2_confidence[i].item()),
                "margin": float(margin[i].item()),
                "entropy": float(entropy[i].item()),
                "expected_grade": float(expected_grade[i].item()),
                "severe_prob_mass": float(severe_prob_mass[i].item()),
                "expected_gap": float(expected_gap[i].item()),
                "gated_severe_prob_mass": float(gated_severe_prob_mass[i].item()),
            }

            for c in range(5):
                row[f"prob_{c}"] = float(probs_np[i, c])

            rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [0, 1, 2, 3, 4]
    metric_rows = []
    cm_rows = []

    for (dataset, backbone), g in pred.groupby(["dataset", "backbone"], sort=True):
        y_true = g["true_grade"].astype(int).to_numpy()
        y_pred = g["pred_grade"].astype(int).to_numpy()

        metric_rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "metric": "accuracy",
            "class_grade": "",
            "value": float(accuracy_score(y_true, y_pred)),
            "n": int(len(g)),
        })
        metric_rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "metric": "macro_f1",
            "class_grade": "",
            "value": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "n": int(len(g)),
        })
        metric_rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "metric": "weighted_f1",
            "class_grade": "",
            "value": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "n": int(len(g)),
        })
        metric_rows.append({
            "dataset": dataset,
            "backbone": backbone,
            "metric": "qwk",
            "class_grade": "",
            "value": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
            "n": int(len(g)),
        })

        recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        for grade, rec in zip(labels, recalls):
            metric_rows.append({
                "dataset": dataset,
                "backbone": backbone,
                "metric": "recall",
                "class_grade": grade,
                "value": float(rec),
                "n": int((y_true == grade).sum()),
            })

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        for true_grade in labels:
            for pred_grade in labels:
                cm_rows.append({
                    "dataset": dataset,
                    "backbone": backbone,
                    "true_grade": true_grade,
                    "pred_grade": pred_grade,
                    "count": int(cm[true_grade, pred_grade]),
                })

    return pd.DataFrame(metric_rows), pd.DataFrame(cm_rows)


def write_summary(out_dir: Path, pred: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = []
    lines.append("# v0.7.1 External DR Direct Inference Summary")
    lines.append("")
    lines.append("## 版本定位")
    lines.append("")
    lines.append("本结果使用 v0.7.0 冻结的 APTOS-trained checkpoints，直接在 IDRiD_data / MESSIDOR2 test split 上推理。")
    lines.append("")
    lines.append("本阶段不使用外部 train / val 训练，不根据外部结果重新选择排序信号或复核预算。")
    lines.append("")
    lines.append("## 样本规模")
    lines.append("")

    for (dataset, backbone), g in pred.groupby(["dataset", "backbone"], sort=True):
        lines.append(f"- {dataset} / {backbone}: {len(g)} records")

    lines.append("")
    lines.append("## 分类迁移指标")
    lines.append("")

    show = metrics[metrics["metric"].isin(["accuracy", "macro_f1", "weighted_f1", "qwk"])]
    for _, row in show.sort_values(["dataset", "backbone", "metric"]).iterrows():
        lines.append(
            f"- {row['dataset']} / {row['backbone']} / {row['metric']}: "
            f"{row['value']:.4f} (n={int(row['n'])})"
        )

    lines.append("")
    lines.append("## 解释边界")
    lines.append("")
    lines.append("- 这是 direct external inference，不是外部数据重训。")
    lines.append("- 外部分类性能无论好坏都应报告。")
    lines.append("- 若分类迁移表现严重不足，后续 review prioritization 只能作为 failure analysis，不能强称 protocol 泛化成功。")
    lines.append("- IDRiD_data 内部 train/test 重复不影响本阶段，因为本阶段不使用 IDRiD train。")
    lines.append("")

    (out_dir / "external_dr_direct_inference_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/LRT/RETFound/Data_split")
    parser.add_argument("--inventory", default="experiments/summary/v0_7_0/external_dr_dataset_inventory.csv")
    parser.add_argument("--out-dir", default="experiments/summary/v0_7_1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    data_root = Path(args.data_root)
    inventory_path = Path(args.inventory)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(inventory_path)

    df = inventory[
        (inventory["dataset"].isin(["IDRiD_data", "MESSIDOR2"]))
        & (inventory["split"] == "test")
        & inventory["image_readable"].eq(True)
    ].copy()

    if df.empty:
        raise RuntimeError("没有找到可推理的外部 test 图像。")

    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"external test images: {len(df)}")

    all_predictions = []

    for backbone in BACKBONES:
        print(f"\n正在推理 backbone: {backbone}")
        one = infer_one_backbone(
            backbone=backbone,
            df=df,
            data_root=data_root,
            repo_root=repo_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        print(f"{backbone}: {len(one)} rows")
        all_predictions.append(one)

    pred = pd.concat(all_predictions, ignore_index=True)
    metrics, cm = compute_metrics(pred)

    pred.to_csv(out_dir / "external_dr_direct_inference_predictions.csv", index=False)
    metrics.to_csv(out_dir / "external_dr_classification_metrics.csv", index=False)
    cm.to_csv(out_dir / "external_dr_confusion_matrix.csv", index=False)
    write_summary(out_dir, pred, metrics)

    print("\n已保存：")
    print(out_dir / "external_dr_direct_inference_predictions.csv")
    print(out_dir / "external_dr_classification_metrics.csv")
    print(out_dir / "external_dr_confusion_matrix.csv")
    print(out_dir / "external_dr_direct_inference_summary.md")


if __name__ == "__main__":
    main()
