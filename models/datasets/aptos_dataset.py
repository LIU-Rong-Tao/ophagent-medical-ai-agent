"""
APTOS2019 Dataset + DataLoader
=================================

当前阶段架构定位：
---------------------------------
这是 OphAgent 中的 Vision Tool 基础模块。

作用：
Fundus Image
→ Dataset
→ DataLoader
→ Tensor
→ 后续送入 ConvNeXt / ResNet / ViT 等视觉模型

当前采用：
---------------------------------
Dataset Framework:
    PyTorch Dataset + DataLoader

Data Organization:
    torchvision.datasets.ImageFolder

Backbone Planning:
    ConvNeXt-Tiny（第一版 baseline）

Training Paradigm:
    ImageNet Pretrained Transfer Learning

Reason:
    先建立稳定 Vision Pipeline，
    而不是一开始追求复杂大模型。
"""

from pathlib import Path
from typing import Tuple

# torchvision:
# ---------------------------------
# PyTorch 官方视觉工具库
#
# datasets:
#     用于读取图像数据集
#
# transforms:
#     图像预处理与数据增强
#
# 当前属于：
#     Vision Pipeline 基础层
#
from torchvision import datasets, transforms

# DataLoader:
# ---------------------------------
# PyTorch 数据流核心组件
#
# 负责：
#   batch加载
#   shuffle
#   多线程读取
#   GPU训练数据供给
#
from torch.utils.data import DataLoader


# =========================================
# APTOS2019 类别定义
# =========================================
#
# 当前任务：
#     Diabetic Retinopathy 5-class grading
#
# 对应：
#     0 -> No DR
#     1 -> Mild DR
#     2 -> Moderate DR
#     3 -> Severe DR
#     4 -> Proliferative DR
#
APTOS_CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]


# =========================================
# 图像预处理模块
# =========================================
#
# 当前属于：
#     Vision Preprocessing Pipeline
#
# 作用：
#     将原始眼底图转换成模型可训练输入
#
# 当前对应架构：
# ---------------------------------
# ImageNet Transfer Learning Pipeline
#
# 原因：
#     后续 ConvNeXt / ResNet / ViT
#     都采用 ImageNet pretrained weights
#
# 因此：
#     输入分布需要与 ImageNet 一致
#
def build_transforms(image_size: int = 224, train: bool = True):

    # =====================================
    # Train Transform
    # =====================================
    #
    # 用于训练阶段
    #
    # 包含：
    #     数据增强
    #
    # 目的：
    #     提高泛化能力
    #
    if train:

        return transforms.Compose([

            # Resize:
            # ---------------------------------
            # Vision Backbone 输入统一尺寸
            #
            # ConvNeXt / ResNet 默认:
            #     224x224
            #
            transforms.Resize((image_size, image_size)),

            # RandomHorizontalFlip:
            # ---------------------------------
            # 数据增强
            #
            # 防止过拟合
            #
            transforms.RandomHorizontalFlip(),

            # RandomRotation:
            # ---------------------------------
            # 医学图像轻度旋转增强
            #
            # 提高模型鲁棒性
            #
            transforms.RandomRotation(10),

            # ToTensor:
            # ---------------------------------
            # PIL Image
            # → Tensor
            #
            # shape:
            #     [3,224,224]
            #
            transforms.ToTensor(),

            # Normalize:
            # ---------------------------------
            # ImageNet标准化参数
            #
            # mean/std 来源：
            #     ImageNet RGB统计值
            #
            # 当前属于：
            #     Transfer Learning 标准流程
            #
            # 数学公式：
            #
            # x_norm = (x - mean) / std
            #
            # 作用：
            #     让输入分布稳定
            #     加速模型收敛
            #
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # =====================================
    # Validation / Test Transform
    # =====================================
    #
    # 验证阶段：
    #     不做随机增强
    #
    # 原因：
    #     保证评测稳定性
    #
    return transforms.Compose([

        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =========================================
# DataLoader 构建模块
# =========================================
#
# 当前属于：
#     Vision Data Pipeline
#
# 输入：
#     APTOS2019 数据路径
#
# 输出：
#     dataset
#     dataloader
#
# 后续：
#     train loop
#         ↓
#     ConvNeXt forward
#         ↓
#     loss
#         ↓
#     backward
#
def build_aptos_dataloader(
    data_root: str,
    split: str = "train",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Tuple[datasets.ImageFolder, DataLoader]:

    split_dir = Path(data_root) / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # =====================================
    # ImageFolder
    # =====================================
    #
    # 当前数据组织：
    #
    # train/
    #   anodr/
    #   bmilddr/
    #   ...
    #
    # ImageFolder 自动：
    #     读取图片
    #     分配label
    #
    dataset = datasets.ImageFolder(
        root=str(split_dir),

        transform=build_transforms(
            image_size=image_size,
            train=(split == "train"),
        ),
    )

    # =====================================
    # DataLoader
    # =====================================
    #
    # Vision Training Pipeline 核心组件
    #
    # 功能：
    #     batch读取
    #     shuffle
    #     多线程加速
    #
    dataloader = DataLoader(
        dataset,

        batch_size=batch_size,

        shuffle=shuffle if split == "train" else False,

        num_workers=num_workers,

        # pin_memory:
        # ---------------------------------
        # GPU训练加速
        #
        pin_memory=True,
    )

    return dataset, dataloader


# =========================================
# Debug / Pipeline Test
# =========================================
#
# 用于：
#     验证整个 Data Pipeline 是否正常
#
if __name__ == "__main__":

    data_root = "/data/LRT/RETFound/Data_split/APTOS2019"

    dataset, loader = build_aptos_dataloader(
        data_root=data_root,
        split="train",
        batch_size=8,
    )

    print("Class to index:", dataset.class_to_idx)

    print("Number of images:", len(dataset))

    # =====================================
    # 获取一个 batch
    # =====================================
    #
    # 输出：
    #     images:
    #         [B,3,224,224]
    #
    #     labels:
    #         [B]
    #
    images, labels = next(iter(loader))

    print("Image batch shape:", images.shape)

    print("Label batch shape:", labels.shape)

    print("Labels:", labels)