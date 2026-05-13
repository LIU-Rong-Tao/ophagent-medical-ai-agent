"""
批量生成不同 Grad-CAM 参数组合的可视化结果。

用途：
- 对比不同 CAM 方法：gradcam / hirescam / eigencam / layercam
- 对比不同目标层：stage2 / stage3 / stage4
- 对比是否使用 smoothing
- 为 v0.2 explainability 选择默认配置提供依据

示例运行：

python scripts/run_gradcam_grid.py \
  --image demo_samples/cmoderatedr/b9127e38d9b9.png \
  --config configs/vision_baseline.yaml \
  --checkpoint experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/checkpoints/convnext_tiny_best.pth \
  --class-to-idx experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/configs/class_to_idx.json \
  --output-root experiments/aptos_convnext_tiny/lr1e-4_bs32_seed42/explain/grid_compare/b9127e38d9b9
"""

import argparse
import subprocess
from pathlib import Path


def parse_args():
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Run Grad-CAM grid comparison for one fundus image."
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

    return parser.parse_args()


def run_command(command):
    """执行单条命令，并实时打印。"""

    print("\n" + "=" * 80)
    print("运行命令：")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(command, check=True)


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # CAM 方法候选
    methods = [
        "gradcam",
        "hirescam",
        "eigencam",
        "layercam",
    ]

    # ConvNeXt 目标层候选
    target_layers = [
        "stage2",
        "stage3",
        "stage4",
    ]

    # smoothing 组合
    # none：不使用 smoothing，更忠实
    # eigen：只使用 eigen smoothing
    # aug_eigen：同时使用 aug smoothing 和 eigen smoothing，更平滑但更慢
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