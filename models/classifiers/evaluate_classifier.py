from pathlib import Path

import argparse
import json
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import timm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from models.datasets.aptos_dataset import build_aptos_dataloader


# =====================================================
# 读取 YAML 配置文件
# =====================================================

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# =====================================================
# 读取类别映射
# =====================================================

def load_class_mapping(class_to_idx_path: str):
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {
        int(v): k
        for k, v in class_to_idx.items()
    }

    return class_to_idx, idx_to_class


# =====================================================
# 类别显示名称
# =====================================================

CLASS_DISPLAY_NAMES = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}


# =====================================================
# 绘制混淆矩阵
# =====================================================

def plot_confusion_matrix(cm, class_names, save_path):

    # =================================================
    # 转成百分比
    # 按行归一化
    # =================================================

    cm_percent = (
        cm.astype("float")
        / cm.sum(axis=1)[:, np.newaxis]
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        cm_percent,
        interpolation="nearest",
    )

    plt.title(
        "Confusion Matrix (%)"
    )

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        class_names,
    )

    thresh = cm_percent.max() / 2.0

    # =================================================
    # 显示百分比
    # =================================================

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            percentage = (
                cm_percent[i, j] * 100
            )

            plt.text(
                j,
                i,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm_percent[i, j] > thresh
                    else "black"
                ),
            )

    plt.ylabel("True Label")

    plt.xlabel("Predicted Label")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()


# =====================================================
# 主函数
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )

    args = parser.parse_args()

    # =================================================
    # 加载配置
    # =================================================

    config = load_config(args.config)

    data_root = config["data_root"]
    backbone = config["backbone"]
    num_classes = config["num_classes"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    experiment_root = config["experiment_root"]
    experiment_name = config["experiment_name"]
    run_name = config["run_name"]

    experiment_dir = (
        f"{experiment_root}/"
        f"{experiment_name}/"
        f"{run_name}"
    )

    config_dir = f"{experiment_dir}/configs"

    evaluation_dir = f"{experiment_dir}/evaluation/{args.split}"

    Path(evaluation_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    class_to_idx_path = (
        f"{config_dir}/class_to_idx.json"
    )

    class_to_idx, idx_to_class = load_class_mapping(
        class_to_idx_path
    )

    class_names = [
        CLASS_DISPLAY_NAMES.get(
            idx_to_class[i],
            idx_to_class[i],
        )
        for i in range(num_classes)
    ]

    # =================================================
    # 设备
    # =================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("========== Evaluation Config ==========")
    print(f"Backbone: {backbone}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Output Dir: {evaluation_dir}")
    print("=======================================")

    # =================================================
    # 构建 DataLoader
    # =================================================

    dataset, loader = build_aptos_dataloader(
        data_root=data_root,
        split=args.split,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # =================================================
    # 构建模型
    # =================================================

    model = timm.create_model(
        backbone,
        pretrained=False,
        num_classes=num_classes,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    # =================================================
    # 批量推理
    # =================================================

    all_true = []

    all_pred = []

    all_confidence = []

    all_probabilities = []

    all_image_paths = [
        path
        for path, _ in dataset.samples
    ]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            labels = labels.to(device)

            logits = model(images)

            probs = F.softmax(logits, dim=1)

            confidence, preds = torch.max(
                probs,
                dim=1,
            )

            all_true.extend(
                labels.cpu().numpy().tolist()
            )

            all_pred.extend(
                preds.cpu().numpy().tolist()
            )

            all_confidence.extend(
                confidence.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probs.cpu().numpy().tolist()
            )

    # =================================================
    # 保存逐图预测结果
    # =================================================

    records = []

    for idx, image_path in enumerate(all_image_paths):
        true_idx = all_true[idx]
        pred_idx = all_pred[idx]

        true_raw = idx_to_class[true_idx]
        pred_raw = idx_to_class[pred_idx]

        record = {
            "image_path": image_path,
            "true_idx": true_idx,
            "true_label": CLASS_DISPLAY_NAMES.get(
                true_raw,
                true_raw,
            ),
            "pred_idx": pred_idx,
            "pred_label": CLASS_DISPLAY_NAMES.get(
                pred_raw,
                pred_raw,
            ),
            "confidence": all_confidence[idx],
            "correct": true_idx == pred_idx,
        }

        for class_idx in range(num_classes):
            raw_name = idx_to_class[class_idx]

            display_name = CLASS_DISPLAY_NAMES.get(
                raw_name,
                raw_name,
            )

            record[f"prob_{display_name}"] = (
                all_probabilities[idx][class_idx]
            )

        records.append(record)

    pred_df = pd.DataFrame(records)

    pred_df.to_csv(
        f"{evaluation_dir}/test_predictions.csv",
        index=False,
    )

    # =================================================
    # 计算整体指标
    # =================================================

    accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            all_true,
            all_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            all_true,
            all_pred,
            average="weighted",
            zero_division=0,
        )
    )

    report_text = classification_report(
        all_true,
        all_pred,
        target_names=class_names,
        zero_division=0,
    )

    with open(
        f"{evaluation_dir}/classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_text)

    metrics = {
        "split": args.split,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "num_samples": len(all_true),
        "backbone": backbone,
        "checkpoint": args.checkpoint,
    }

    with open(
        f"{evaluation_dir}/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # =================================================
    # 混淆矩阵
    # =================================================

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(num_classes)),
    )

    np.save(
        f"{evaluation_dir}/confusion_matrix.npy",
        cm,
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        save_path=f"{evaluation_dir}/confusion_matrix.png",
    )

    # =================================================
    # 控制台输出
    # =================================================

    print("========== Test Evaluation Result ==========")
    print(f"Split: {args.split}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print("===========================================")

    print(report_text)

    print(f"评估结果已保存到: {evaluation_dir}")


# =====================================================
# 程序入口
# =====================================================

if __name__ == "__main__":
    main()