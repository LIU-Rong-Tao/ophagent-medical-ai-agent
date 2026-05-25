"""
批量生成不同 CAM 参数组合的可视化结果。

用途：
- 对比不同 CAM 方法：gradcam / hirescam / eigencam / layercam
- ConvNeXt 对比不同目标层：stage2 / stage3 / stage4
- Transformer backbone 对比不同 target depth：early / middle / late
- 对比是否使用 smoothing
- 为 v0.5.3 CAM adapter foundation 与 v0.6 explainability consistency benchmark 做准备

示例运行：

python scripts/run_gradcam_grid.py \
  --image demo_samples/cmoderatedr/d9bbdc33db83.png \
  --config configs/vit_large_patch16_official_like_clean.yaml \
  --checkpoint experiments/aptos_vit_large_patch16_official_like/official_like_bs32_epoch50_seed42/checkpoints/vit_large_patch16_best.pth \
  --class-to-idx experiments/aptos_vit_large_patch16_official_like/official_like_bs32_epoch50_seed42/configs/class_to_idx.json \
  --output-root experiments/summary/v0_5_3/cam_grid_compare/d9bbdc33db83/vit_l
"""

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args():
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Run CAM grid comparison for one fundus image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图像路径。",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径。",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 路径。",
    )

    parser.add_argument(
        "--class-to-idx",
        type=str,
        required=True,
        help="class_to_idx.json 路径。",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="所有对比结果的输出根目录。",
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=["gradcam", "hirescam", "eigencam", "layercam"],
        help="CAM 方法列表。",
    )

    parser.add_argument(
        "--target-layers",
        nargs="*",
        default=None,
        help=(
            "手动指定 target layers。"
            "ConvNeXt 可用 stage2/stage3/stage4；"
            "Transformer 可用 early/middle/late/block<N>。"
            "不指定时根据 backbone 自动选择。"
        ),
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """读取 YAML 配置。"""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_target_layers(backbone: str):
    """根据 backbone 自动选择 CAM target layers。"""

    backbone = backbone.lower()

    if backbone == "convnext_tiny":
        return ["stage2", "stage3", "stage4"]

    if backbone in [
        "swin_tiny",
        "swin_tiny_patch4_window7_224",
        "swin_tiny_patch4_window7_224.ms_in1k",
        "vit_base_patch16",
        "vit_large_patch16",
        "retfound_mae_cfp",
    ]:
        return ["early", "middle", "late"]

    raise ValueError(f"Unsupported backbone for CAM grid: {backbone}")


def run_command(command):
    """执行单条命令，并实时打印。"""

    print("\n" + "=" * 80)
    print("运行命令：")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(command, check=True)


def main():
    args = parse_args()

    config = load_config(args.config)
    backbone = config["backbone"]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    methods = args.methods

    if args.target_layers is not None and len(args.target_layers) > 0:
        target_layers = args.target_layers
    else:
        target_layers = infer_target_layers(backbone)

    smooth_settings = [
        {
            "name": "none",
            "extra_args": [],
        },
        {
            "name": "eigen",
            "extra_args": ["--eigen-smooth"],
        },
        {
            "name": "aug_eigen",
            "extra_args": ["--aug-smooth", "--eigen-smooth"],
        },
    ]

    total = len(methods) * len(target_layers) * len(smooth_settings)
    current = 0

    print(f"Backbone: {backbone}")
    print(f"Methods: {methods}")
    print(f"Target layers: {target_layers}")
    print(f"Output root: {output_root}")

    for method in methods:
        for target_layer in target_layers:
            for smooth in smooth_settings:
                current += 1

                output_dir = (
                    output_root
                    / f"{method}_{target_layer}_{smooth['name']}"
                )

                print(
                    f"\n[{current}/{total}] "
                    f"method={method}, "
                    f"target_layer={target_layer}, "
                    f"smooth={smooth['name']}"
                )

                command = [
                    "python",
                    "-m",
                    "explain.gradcam",
                    "--image",
                    args.image,
                    "--config",
                    args.config,
                    "--checkpoint",
                    args.checkpoint,
                    "--class-to-idx",
                    args.class_to_idx,
                    "--method",
                    method,
                    "--target-layer",
                    target_layer,
                    "--output",
                    str(output_dir),
                ]

                command.extend(smooth["extra_args"])

                run_command(command)

    print("\n全部 CAM 参数组合已生成完成。")
    print(f"结果目录：{output_root}")
    print("\n建议重点查看每个子目录中的 overlay.png。")


if __name__ == "__main__":
    main()