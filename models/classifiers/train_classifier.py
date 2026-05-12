from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

# 补训练结果记录
import json
import pandas as pd
import matplotlib.pyplot as plt

# timm:
# ------------------------------------------------
# 一个非常常用的视觉模型库
#
# 支持：
# ResNet
# ConvNeXt
# ViT
# Swin Transformer
# 等大量 backbone
#
# 当前我们使用：
# ConvNeXt-Tiny
#
import timm

# tqdm:
# ------------------------------------------------
# 用于显示训练进度条
#
from tqdm import tqdm

# 导入我们之前写好的 DataLoader
from models.datasets.aptos_dataset import build_aptos_dataloader


# =====================================================
# 配置部分（Config）
# =====================================================

# 数据集路径
DATA_ROOT = "/data/LRT/RETFound/Data_split/APTOS2019"

# DR 五分类
NUM_CLASSES = 5

# 输入图像尺寸
IMAGE_SIZE = 224

# batch size
#
# 表示：
# 一次送多少张图进GPU
#
BATCH_SIZE = 32

# 训练轮数
NUM_EPOCHS = 10

# 学习率
LEARNING_RATE = 1e-4

# 自动检测 GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint 保存目录
CHECKPOINT_DIR = "models/checkpoints"

# 自动创建目录
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# =====================================================
# 实验记录目录
# =====================================================

EXPERIMENT_DIR = "experiments/aptos_convnext_tiny"

LOG_DIR = f"{EXPERIMENT_DIR}/logs"

FIGURE_DIR = f"{EXPERIMENT_DIR}/figures"

CONFIG_DIR = f"{EXPERIMENT_DIR}/configs"

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)

Path(CONFIG_DIR).mkdir(parents=True, exist_ok=True)


# =====================================================
# 保存实验配置
# =====================================================

config = {
    "dataset": "APTOS2019",
    "backbone": "convnext_tiny",
    "num_classes": NUM_CLASSES,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": "AdamW",
    "loss_function": "CrossEntropyLoss",
    "pretrained": True,
}

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


# =====================================================
# 用于保存训练历史
# =====================================================

history = []

# =====================================================
# 构建 DataLoader
# =====================================================

# train dataloader
train_dataset, train_loader = build_aptos_dataloader(
    data_root=DATA_ROOT,
    split="train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# validation dataloader
val_dataset, val_loader = build_aptos_dataloader(
    data_root=DATA_ROOT,
    split="val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)


# =====================================================
# 构建模型（Model）
# =====================================================
#
# 当前 backbone:
# ConvNeXt-Tiny
#
# 为什么选它：
# ------------------------------------------------
# 1. 比 ResNet 更强
# 2. 比 ViT 更稳定
# 3. 显存占用适中
# 4. 医学图像效果很好
# 5. 非常适合作为 baseline
#
# pretrained=True:
# ------------------------------------------------
# 使用 ImageNet 预训练权重
#
# 本质：
# Transfer Learning（迁移学习）
#
# num_classes=5:
# ------------------------------------------------
# 修改最后分类头
#
# 对应 DR 五分类
#
model = timm.create_model(
    "convnext_tiny",

    pretrained=True,

    num_classes=NUM_CLASSES,
)

# 模型送入 GPU
model = model.to(DEVICE)


# =====================================================
# Loss Function（损失函数）
# =====================================================
#
# CrossEntropyLoss:
# ------------------------------------------------
# 分类任务最常用损失函数
#
# 当前：
# 多分类任务
#
criterion = nn.CrossEntropyLoss()


# =====================================================
# Optimizer（优化器）
# =====================================================
#
# AdamW:
# ------------------------------------------------
# Transformer / ConvNeXt 常用优化器
#
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# =====================================================
# 验证函数
# =====================================================
#
# 用于：
# 每轮训练后评估模型效果
#
def validate(model, loader):

    # eval模式
    #
    # 会关闭：
    # dropout
    # batchnorm更新
    #
    model.eval()

    total = 0
    correct = 0

    val_loss = 0.0

    # 不计算梯度
    #
    # 节省显存
    #
    with torch.no_grad():

        for images, labels in loader:

            # 数据送入 GPU
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # 前向传播
            outputs = model(images)

            # 计算 loss
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # 取最大概率类别
            preds = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()

    # 分类准确率
    accuracy = correct / total

    avg_loss = val_loss / len(loader)

    return avg_loss, accuracy


# =====================================================
# Training Loop（训练主循环）
# =====================================================

best_acc = 0.0

for epoch in range(NUM_EPOCHS):

    # train模式
    #
    # 开启：
    # dropout
    # batchnorm更新
    #
    model.train()

    running_loss = 0.0

    # tqdm进度条
    progress_bar = tqdm(train_loader)

    for images, labels in progress_bar:

        # 数据送入 GPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # 前向传播
        outputs = model(images)

        # 计算 loss
        loss = criterion(outputs, labels)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        running_loss += loss.item()

        # 更新进度条显示
        progress_bar.set_description(
            f"Epoch {epoch+1}/{NUM_EPOCHS}"
        )

        progress_bar.set_postfix(
            loss=loss.item()
        )

    train_loss = running_loss / len(train_loader)

    # 验证集评估
    val_loss, val_acc = validate(model, val_loader)

    print(
        f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]"
        f"\nTrain Loss: {train_loss:.4f}"
        f"\nVal Loss: {val_loss:.4f}"
        f"\nVal Acc: {val_acc:.4f}\n"
    )

# =================================================
# 保存每轮训练结果
# =================================================

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "best_acc": best_acc,
    })

    df = pd.DataFrame(history)

    df.to_csv(
        f"{LOG_DIR}/train_log.csv",
        index=False
    )
    # =================================================
    # 保存最佳模型
    # =================================================
    #
    # 如果当前准确率更高
    # 就保存 checkpoint
    #
    if val_acc > best_acc:

        best_acc = val_acc

        save_path = (
            f"{CHECKPOINT_DIR}/"
            f"convnext_tiny_best.pth"
        )

        torch.save(model.state_dict(), save_path)

        print(f"最佳模型已保存: {save_path}")

# =====================================================
# 保存训练曲线
# =====================================================

df = pd.DataFrame(history)


# =====================================================
# Loss Curve
# =====================================================

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


# =====================================================
# Validation Accuracy Curve
# =====================================================

plt.figure()

plt.plot(
    df["epoch"],
    df["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Validation Accuracy Curve")

plt.legend()

plt.savefig(
    f"{FIGURE_DIR}/val_acc_curve.png",
    dpi=300
)

plt.close()


print(
    f"实验日志已保存到: "
    f"{LOG_DIR}/train_log.csv"
)

print(
    f"训练曲线已保存到: "
    f"{FIGURE_DIR}"
)

print("训练结束")