from pathlib import Path

import argparse
import json
import yaml

from PIL import Image

import torch
import torch.nn.functional as F

import timm

from torchvision import transforms


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
#
# 训练时 ImageFolder 会自动生成：
# {
#   "anodr": 0,
#   "bmilddr": 1,
#   ...
# }
#
# 推理时必须读取同一个 class_to_idx.json
# 防止类别顺序错位。
#

def load_class_mapping(class_to_idx_path: str):
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # 反转成：
    # 0 -> anodr
    # 1 -> bmilddr
    idx_to_class = {
        int(v): k
        for k, v in class_to_idx.items()
    }

    return idx_to_class


# =====================================================
# 类别名称美化
# =====================================================

CLASS_DISPLAY_NAMES = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}


# =====================================================
# 构建推理阶段 transform
# =====================================================
#
# 必须和验证集 transform 保持一致：
# Resize
# ToTensor
# Normalize
#

def build_infer_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =====================================================
# 主函数
# =====================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input fundus image.",
    )

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
        "--class_to_idx",
        type=str,
        default=None,
        help="Path to class_to_idx.json. If not provided, use experiment config path.",
    )

    args = parser.parse_args()

    # =================================================
    # 加载配置
    # =================================================

    config = load_config(args.config)

    backbone = config["backbone"]
    num_classes = config["num_classes"]
    image_size = config["image_size"]

    experiment_root = config["experiment_root"]
    experiment_name = config["experiment_name"]
    run_name = config["run_name"]

    experiment_dir = (
        f"{experiment_root}/"
        f"{experiment_name}/"
        f"{run_name}"
    )

    # =================================================
    # 类别映射路径
    # =================================================

    if args.class_to_idx is None:
        class_to_idx_path = (
            f"{experiment_dir}/"
            f"configs/class_to_idx.json"
        )
    else:
        class_to_idx_path = args.class_to_idx

    idx_to_class = load_class_mapping(class_to_idx_path)

    # =================================================
    # 设备
    # =================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =================================================
    # 构建模型
    # =================================================
    #
    # 推理阶段不需要再次加载 ImageNet 预训练权重。
    # 因为我们会加载自己训练好的 checkpoint。
    #

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
    # 读取图像
    # =================================================

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    transform = build_infer_transform(image_size)

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # =================================================
    # 推理
    # =================================================

    with torch.no_grad():
        logits = model(image_tensor)

        probs = F.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(
            probs,
            dim=1,
        )

    pred_idx = pred_idx.item()

    confidence = confidence.item()

    raw_class_name = idx_to_class[pred_idx]

    display_name = CLASS_DISPLAY_NAMES.get(
        raw_class_name,
        raw_class_name,
    )

    # =================================================
    # 输出结果
    # =================================================

    print("========== OphAgent 单图推理结果 ==========")

    print(f"Image: {image_path}")

    print(f"Backbone: {backbone}")

    print(f"Prediction: {display_name}")

    print(f"Raw Class: {raw_class_name}")

    print(f"Confidence: {confidence:.4f}")

    print("==========================================")


# =====================================================
# 程序入口
# =====================================================

if __name__ == "__main__":
    main()