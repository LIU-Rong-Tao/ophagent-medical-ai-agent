from pathlib import Path

import argparse
import json
import yaml
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.optim import AdamW

import timm

from tqdm import tqdm

from models.datasets.aptos_dataset import (
    build_aptos_dataloader
)


# =====================================================
# 固定随机种子
# 为了保证实验可复现
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

def load_config(config_path):

    with open(config_path, "r") as f:

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

            correct += (
                preds == labels
            ).sum().item()

    accuracy = correct / total

    avg_loss = val_loss / len(loader)

    return avg_loss, accuracy


# =====================================================
# 主函数
# =====================================================

def main():

    # =================================================
    # argparse
    # 支持命令行读取配置文件
    # =================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # =================================================
    # 加载配置文件
    # =================================================

    config = load_config(args.config)

    DATA_ROOT = config["data_root"]

    BACKBONE = config["backbone"]

    NUM_CLASSES = config["num_classes"]

    IMAGE_SIZE = config["image_size"]

    BATCH_SIZE = config["batch_size"]

    NUM_EPOCHS = config["num_epochs"]

    LEARNING_RATE = config["learning_rate"]

    PRETRAINED = config["pretrained"]

    SEED = config["seed"]

    # =================================================
    # 实验管理
    # experiment_name:
    #     aptos_convnext_tiny
    #
    # run_name:
    #     lr1e-4_bs32_seed42
    # =================================================

    EXPERIMENT_ROOT = config["experiment_root"]

    EXPERIMENT_NAME = config["experiment_name"]

    RUN_NAME = config["run_name"]

    EXPERIMENT_DIR = (
        f"{EXPERIMENT_ROOT}/"
        f"{EXPERIMENT_NAME}/"
        f"{RUN_NAME}"
    )

    # =================================================
    # checkpoint 保存目录
    # 每个实验独立保存
    # =================================================

    CHECKPOINT_DIR = (
        f"{EXPERIMENT_DIR}/checkpoints"
    )

    LOG_DIR = (
        f"{EXPERIMENT_DIR}/logs"
    )

    FIGURE_DIR = (
        f"{EXPERIMENT_DIR}/figures"
    )

    CONFIG_DIR = (
        f"{EXPERIMENT_DIR}/configs"
    )

    # =================================================
    # 创建目录
    # =================================================

    Path(CHECKPOINT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(LOG_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(FIGURE_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(CONFIG_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    # =================================================
    # 固定随机种子
    # =================================================

    set_seed(SEED)

    # =================================================
    # 设备
    # =================================================

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =================================================
    # 保存 config.json
    # =================================================

    with open(
        f"{CONFIG_DIR}/config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
            ensure_ascii=False
        )

    # =================================================
    # 保存环境信息
    # =================================================

    env_info = {

        "torch_version":
            torch.__version__,

        "cuda_available":
            torch.cuda.is_available(),

        "device":
            DEVICE,

        "gpu_name":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            )
    }

    with open(
        f"{CONFIG_DIR}/env_info.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            env_info,
            f,
            indent=4,
            ensure_ascii=False
        )

    # =================================================
    # 构建 DataLoader
    # =================================================

    train_dataset, train_loader = (
        build_aptos_dataloader(
            data_root=DATA_ROOT,
            split="train",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
    )

    val_dataset, val_loader = (
        build_aptos_dataloader(
            data_root=DATA_ROOT,
            split="val",
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
        )
    )

    # =================================================
    # 保存类别映射
    # 防止推理阶段类别错位
    # =================================================

    with open(
        f"{CONFIG_DIR}/class_to_idx.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            train_dataset.class_to_idx,
            f,
            indent=4,
            ensure_ascii=False
        )

    # =================================================
    # 构建模型
    # =================================================

    model = timm.create_model(
        BACKBONE,
        pretrained=PRETRAINED,
        num_classes=NUM_CLASSES,
    )

    model = model.to(DEVICE)

    # =================================================
    # Loss 与 Optimizer
    # =================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # =================================================
    # Training Loop
    # =================================================

    best_acc = 0.0

    history = []

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(train_loader)

        for images, labels in progress_bar:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch+1}/{NUM_EPOCHS}"
            )

            progress_bar.set_postfix(
                loss=loss.item()
            )

        train_loss = (
            running_loss / len(train_loader)
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        print(
            f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]"
            f"\nTrain Loss: {train_loss:.4f}"
            f"\nVal Loss: {val_loss:.4f}"
            f"\nVal Acc: {val_acc:.4f}\n"
        )

        # =============================================
        # 保存训练日志
        # =============================================

        history.append({

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "val_acc":
                val_acc,

            "best_acc":
                best_acc,
        })

        df = pd.DataFrame(history)

        df.to_csv(
            f"{LOG_DIR}/train_log.csv",
            index=False
        )

        # =============================================
        # 保存最佳模型
        # =============================================

        if val_acc > best_acc:

            best_acc = val_acc

            save_path = (
                f"{CHECKPOINT_DIR}/"
                f"{BACKBONE}_best.pth"
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"最佳模型已保存: "
                f"{save_path}"
            )

    # =================================================
    # 保存 Loss 曲线
    # =================================================

    plt.figure()

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Val Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.legend()

    plt.savefig(
        f"{FIGURE_DIR}/loss_curve.png",
        dpi=300
    )

    plt.close()

    # =================================================
    # 保存 Accuracy 曲线
    # =================================================

    plt.figure()

    plt.plot(
        df["epoch"],
        df["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "Validation Accuracy Curve"
    )

    plt.legend()

    plt.savefig(
        f"{FIGURE_DIR}/val_acc_curve.png",
        dpi=300
    )

    plt.close()

    print("训练结束")


# =====================================================
# 入口
# =====================================================

if __name__ == "__main__":

    main()