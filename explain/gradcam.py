"""
OphAgent Grad-CAM Explainability

功能：
1. 加载 OphAgent classifier checkpoint
2. 对单张眼底图像进行预测
3. 通过 CAM adapter 支持 ConvNeXt / Swin / ViT / RETFound
4. 使用 Grad-CAM / HiResCAM / EigenCAM / LayerCAM 生成模型关注区域
5. 保存 original / heatmap / overlay 三张图

注意：
CAM 仅用于模型可解释性展示，不等同于医学病灶定位。
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from explain.cam_adapter import get_cam_target_layers, get_cam_reshape_transform
from models.classifiers.builder import build_model


CLASS_DISPLAY_NAMES = {
    "anodr": "No DR",
    "bmilddr": "Mild DR",
    "cmoderatedr": "Moderate DR",
    "dseveredr": "Severe DR",
    "eproliferativedr": "Proliferative DR",
}


def parse_args():
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Generate CAM visualization for OphAgent classifier."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入眼底图像路径。",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="训练配置文件路径，例如 configs/vision_baseline.yaml。",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型权重路径，例如 convnext_tiny_best.pth。",
    )

    parser.add_argument(
        "--class-to-idx",
        type=str,
        required=True,
        help="训练时保存的 class_to_idx.json 路径。",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="CAM 输出目录。",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=["gradcam", "hirescam", "eigencam", "layercam"],
        help="CAM 方法，可选：gradcam, hirescam, eigencam, layercam。",
    )

    parser.add_argument(
        "--target-layer",
        type=str,
        default="auto",
        help=(
            "CAM target layer. ConvNeXt: auto/stage4/stage3/stage2; "
            "Transformer: auto/early/middle/late/block<N>; "
            "Swin: auto/early/middle/late."
        ),
    )

    parser.add_argument(
        "--aug-smooth",
        action="store_true",
        help="使用 test-time augmentation smoothing，通常更平滑但速度更慢。",
    )

    parser.add_argument(
        "--eigen-smooth",
        action="store_true",
        help="使用 eigen smoothing，通常可减少噪声。",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """读取 YAML 配置文件。"""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_mapping(class_to_idx_path: str):
    """读取类别映射。"""

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    return class_to_idx, idx_to_class


def build_transform(image_size: int):
    """
    构建与训练 / 推理阶段一致的图像预处理。

    注意：
    CAM 推理输入必须和模型训练时保持一致。
    """

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


def load_image_for_model(image_path: str, image_size: int):
    """读取图像，并分别返回 PIL 图像、模型输入 Tensor、可视化用 RGB 图像。"""

    image_pil = Image.open(image_path).convert("RGB")

    transform = build_transform(image_size)
    input_tensor = transform(image_pil).unsqueeze(0)

    image_resized = image_pil.resize((image_size, image_size))
    image_rgb = np.array(image_resized).astype(np.float32) / 255.0

    return image_pil, input_tensor, image_rgb


def build_cam(method: str, model, target_layers, reshape_transform=None):
    """根据命令行参数构建不同 CAM 方法。"""

    method = method.lower()

    if method == "gradcam":
        return GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    if method == "hirescam":
        return HiResCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    if method == "eigencam":
        return EigenCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    if method == "layercam":
        return LayerCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

    raise ValueError(
        f"Unknown CAM method: {method}. "
        "Choose from: gradcam, hirescam, eigencam, layercam."
    )


def predict(model, input_tensor: torch.Tensor, device: torch.device):
    """执行分类推理，返回预测类别索引、置信度和概率分布。"""

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_idx].item()

    return pred_idx, confidence, probs


def save_gradcam_outputs(
    image_rgb: np.ndarray,
    grayscale_cam: np.ndarray,
    overlay_rgb: np.ndarray,
    output_dir: str,
):
    """保存 original、heatmap、overlay 三张图。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_uint8 = (image_rgb * 255).astype(np.uint8)
    heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    cv2.imwrite(
        str(output_dir / "original.png"),
        cv2.cvtColor(original_uint8, cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        str(output_dir / "heatmap.png"),
        heatmap_color,
    )

    cv2.imwrite(
        str(output_dir / "overlay.png"),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    _, idx_to_class = load_class_mapping(args.class_to_idx)

    backbone = config["backbone"]

    model = build_model(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    _, input_tensor, image_rgb = load_image_for_model(
        image_path=args.image,
        image_size=config["image_size"],
    )

    pred_idx, confidence, _ = predict(
        model=model,
        input_tensor=input_tensor,
        device=device,
    )

    pred_raw_class = idx_to_class[pred_idx]
    pred_display_class = CLASS_DISPLAY_NAMES.get(
        pred_raw_class,
        pred_raw_class,
    )

    print(f"Device: {device}")
    print(f"Backbone: {backbone}")
    print(f"CAM Method: {args.method}")
    print(f"Target Layer: {args.target_layer}")
    print(f"Aug Smooth: {args.aug_smooth}")
    print(f"Eigen Smooth: {args.eigen_smooth}")
    print(f"Prediction: {pred_display_class}")
    print(f"Raw Class: {pred_raw_class}")
    print(f"Confidence: {confidence:.4f}")

    target_layers = get_cam_target_layers(
        model=model,
        backbone=backbone,
        target_layer=args.target_layer,
    )

    reshape_transform = get_cam_reshape_transform(
        model=model,
        backbone=backbone,
    )

    cam = build_cam(
        method=args.method,
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    targets = [ClassifierOutputTarget(pred_idx)]

    grayscale_cam = cam(
        input_tensor=input_tensor.to(device),
        targets=targets,
        aug_smooth=args.aug_smooth,
        eigen_smooth=args.eigen_smooth,
    )[0]

    overlay_rgb = show_cam_on_image(
        image_rgb,
        grayscale_cam,
        use_rgb=True,
    )

    save_gradcam_outputs(
        image_rgb=image_rgb,
        grayscale_cam=grayscale_cam,
        overlay_rgb=overlay_rgb,
        output_dir=args.output,
    )

    print(f"CAM results saved to: {args.output}")


if __name__ == "__main__":
    main()