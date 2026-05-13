from pathlib import Path

import argparse
import json
import yaml
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW

import timm
from tqdm import tqdm

from models.datasets.aptos_dataset import build_aptos_dataloader


# =====================================================
# 固定随机种子
# =====================================================

def set_seed(seed: int):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =====================================================
# 读取 YAML 配置文件
# =====================================================

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# =====================================================
# 验证函数
# =====================================================

def validate(model, loader, criterion, device):
    model.eval()

    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()

    accuracy = correct / total

    avg_loss = val_loss / len(loader)

    return avg_loss, accuracy


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
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    pretrained = config["pretrained"]
    seed = config["seed"]

    experiment_root = config["experiment_root"]
    experiment_name = config["experiment_name"]
    run_name = config["run_name"]

    experiment_dir = (
        f"{experiment_root}/"
        f"{experiment_name}/"
        f"{run_name}"
    )

    checkpoint_dir = f"{experiment_dir}/checkpoints"
    log_dir = f"{experiment_dir}/logs"
    figure_dir = f"{experiment_dir}/figures"
    config_dir = f"{experiment_dir}/configs"

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    Path(config_dir).mkdir(parents=True, exist_ok=True)

    # =================================================
    # 固定随机种子
    # =================================================

    set_seed(seed)

    # =================================================
    # 设备信息
    # =================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total_gb = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        )
    else:
        gpu_name = "cpu"
        gpu_memory_total_gb = 0

    # =================================================
    # 保存实验配置
    # =================================================

    with open(
        f"{config_dir}/config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # =================================================
    # 保存环境与硬件信息
    # =================================================

    env_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "gpu_name": gpu_name,
        "gpu_memory_total_gb": gpu_memory_total_gb,
        "seed": seed,
    }

    with open(
        f"{config_dir}/env_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            env_info,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("========== 实验配置 ==========")
    print(f"Backbone: {backbone}")
    print(f"Dataset: {data_root}")
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory_total_gb} GB")
    print(f"Seed: {seed}")
    print("==============================")

    # =================================================
    # 构建 DataLoader
    # =================================================

    train_dataset, train_loader = build_aptos_dataloader(
        data_root=data_root,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
    )

    val_dataset, val_loader = build_aptos_dataloader(
        data_root=data_root,
        split="val",
        image_size=image_size,
        batch_size=batch_size,
    )

    # =================================================
    # 保存类别映射
    # 防止推理阶段类别顺序错位
    # =================================================

    with open(
        f"{config_dir}/class_to_idx.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            train_dataset.class_to_idx,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # =================================================
    # 构建模型
    # =================================================

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    model = model.to(device)

    # =================================================
    # 损失函数与优化器
    # =================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # =================================================
    # 训练主循环
    # =================================================

    best_acc = 0.0
    best_epoch = 0
    history = []

    train_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(train_loader)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}"
            )

            progress_bar.set_postfix(
                loss=loss.item()
            )

        train_loss = running_loss / len(train_loader)

        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time_sec = time.time() - epoch_start_time

        print(
            f"\nEpoch [{epoch + 1}/{num_epochs}]"
            f"\nTrain Loss: {train_loss:.4f}"
            f"\nVal Loss: {val_loss:.4f}"
            f"\nVal Acc: {val_acc:.4f}"
            f"\nEpoch Time: {epoch_time_sec:.2f} sec\n"
        )

        # =============================================
        # 保存最佳模型
        # =============================================

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1

            save_path = (
                f"{checkpoint_dir}/"
                f"{backbone}_best.pth"
            )

            torch.save(
                model.state_dict(),
                save_path,
            )

            print(f"最佳模型已保存: {save_path}")

        # =============================================
        # 保存每轮训练日志
        # =============================================

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "epoch_time_sec": epoch_time_sec,
            }
        )

        df = pd.DataFrame(history)

        df.to_csv(
            f"{log_dir}/train_log.csv",
            index=False,
        )

    # =================================================
    # 总训练耗时
    # =================================================

    total_train_time_sec = time.time() - train_start_time

    summary = {
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_train_time_sec": total_train_time_sec,
        "total_train_time_min": total_train_time_sec / 60,
        "avg_epoch_time_sec": total_train_time_sec / num_epochs,
        "backbone": backbone,
        "dataset": "APTOS2019",
        "run_name": run_name,
        "seed": seed,
        "device": device,
        "gpu_name": gpu_name,
        "gpu_memory_total_gb": gpu_memory_total_gb,
    }

    with open(
        f"{log_dir}/summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # =================================================
    # 保存 Loss 曲线
    # =================================================

    df = pd.DataFrame(history)

    plt.figure()

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Val Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.savefig(
        f"{figure_dir}/loss_curve.png",
        dpi=300,
    )

    plt.close()

    # =================================================
    # 保存 Validation Accuracy 曲线
    # =================================================

    plt.figure()

    plt.plot(
        df["epoch"],
        df["val_acc"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()

    plt.savefig(
        f"{figure_dir}/val_acc_curve.png",
        dpi=300,
    )

    plt.close()

    print("========== 训练结束 ==========")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Total Train Time: {total_train_time_sec / 60:.2f} min")
    print(f"Average Epoch Time: {total_train_time_sec / num_epochs:.2f} sec")
    print(f"实验目录: {experiment_dir}")


# =====================================================
# 程序入口
# =====================================================

if __name__ == "__main__":
    main()