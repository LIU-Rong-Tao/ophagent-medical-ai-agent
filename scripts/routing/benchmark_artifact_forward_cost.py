#!/usr/bin/env python3
"""对 registry artifact 执行可复现的单 GPU forward-only 成本测量。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repository-root bootstrap above must run before these imports.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SUPPORTED_ADAPTERS = {
    "glaucoma_convnext_tiny",
    "glaucoma_retfound_dinov2",
}
CONFIG_FIELDS = {
    "benchmark_id",
    "timing_scope",
    "device",
    "precision",
    "warmup_runs",
    "timed_runs",
    "artifacts",
}
ARTIFACT_FIELDS = {
    "artifact_id",
    "cost_profile_id",
    "adapter",
    "task_id",
    "dataset_id",
    "model_family",
    "batch_size",
    "checkpoint_path",
    "data_root",
    "split",
}
RUN_COLUMNS = [
    "artifact_id",
    "cost_profile_id",
    "task_id",
    "dataset_id",
    "model_family",
    "repeat_index",
    "n_images",
    "batch_size",
    "device",
    "precision",
    "warmup_runs",
    "timed_runs",
    "total_forward_ms",
    "ms_per_image",
    "peak_allocated_memory_mb",
    "checkpoint_mb",
    "timing_scope",
    "timing_source",
]


class BenchmarkError(RuntimeError):
    """成本测量配置或运行结果不满足约束。"""


@dataclass
class PreparedArtifact:
    model: Any
    loader: Any
    n_images: int
    checkpoint_path: Path
    adapter_note: str


def load_benchmark_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BenchmarkError(f"成本测量配置不存在：{path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml

        except ImportError as exc:
            raise BenchmarkError("读取 YAML 配置需要安装 PyYAML") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise BenchmarkError("成本测量配置根节点必须是对象")
    missing = sorted(CONFIG_FIELDS - set(payload))
    if missing:
        raise BenchmarkError(f"成本测量配置缺少字段：{', '.join(missing)}")
    if payload["timing_scope"] != "forward_only":
        raise BenchmarkError("v0.8.4b 只接受 timing_scope=forward_only")
    if payload["precision"] != "fp32":
        raise BenchmarkError("v0.8.4b 仅测量 fp32，其他精度需单独建立 cost profile")
    if int(payload["warmup_runs"]) < 1:
        raise BenchmarkError("warmup_runs 必须大于 0")
    if int(payload["timed_runs"]) < 2:
        raise BenchmarkError("timed_runs 至少为 2，才能检查测量波动")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise BenchmarkError("artifacts 必须是非空列表")
    seen: set[tuple[str, str]] = set()
    profile_batch_sizes: dict[str, int] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise BenchmarkError("每个 artifact 配置必须是对象")
        artifact_missing = sorted(ARTIFACT_FIELDS - set(artifact))
        if artifact_missing:
            raise BenchmarkError(
                f"artifact 缺少字段：{', '.join(artifact_missing)}"
            )
        if artifact["adapter"] not in SUPPORTED_ADAPTERS:
            raise BenchmarkError(f"不支持的 adapter：{artifact['adapter']}")
        adapter_fields = {
            "glaucoma_convnext_tiny": {"model_config", "class_to_idx_path"},
            "glaucoma_retfound_dinov2": {"retfound_root"},
        }[artifact["adapter"]]
        adapter_missing = sorted(adapter_fields - set(artifact))
        if adapter_missing:
            raise BenchmarkError(
                f"{artifact['adapter']} 缺少字段：{', '.join(adapter_missing)}"
            )
        if int(artifact["batch_size"]) < 1:
            raise BenchmarkError("batch_size 必须大于 0")
        profile_id = str(artifact["cost_profile_id"])
        batch_size = int(artifact["batch_size"])
        previous_batch_size = profile_batch_sizes.setdefault(profile_id, batch_size)
        if previous_batch_size != batch_size:
            raise BenchmarkError(
                "同一 cost_profile_id 不能混用不同 batch_size："
                f"{profile_id} -> {previous_batch_size} / {batch_size}"
            )
        key = (str(artifact["artifact_id"]), str(artifact["cost_profile_id"]))
        if key in seen:
            raise BenchmarkError(
                "artifact_id + cost_profile_id 必须唯一：" f"{key[0]} / {key[1]}"
            )
        seen.add(key)
    return payload


def _single_value(group: pd.DataFrame, column: str) -> Any:
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise BenchmarkError(f"同一成本 profile 的 {column} 不一致")
    return values.iloc[0]


def aggregate_cost_runs(runs: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RUN_COLUMNS) - set(runs.columns))
    if missing:
        raise BenchmarkError(f"原始成本记录缺少字段：{', '.join(missing)}")
    if runs.empty:
        raise BenchmarkError("原始成本记录为空")

    rows: list[dict[str, Any]] = []
    for (artifact_id, cost_profile_id), group in runs.groupby(
        ["artifact_id", "cost_profile_id"], sort=True
    ):
        declared_runs = int(_single_value(group, "timed_runs"))
        observed_indices = sorted(group["repeat_index"].astype(int).tolist())
        if len(group) != declared_runs or observed_indices != list(
            range(1, declared_runs + 1)
        ):
            raise BenchmarkError(
                f"{artifact_id} / {cost_profile_id} 的重复测量数量不完整："
                f"expected={declared_runs}, observed={observed_indices}"
            )
        values = group["ms_per_image"].astype(float)
        mean = float(values.mean())
        median = float(values.median())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(
            {
                "artifact_id": artifact_id,
                "cost_profile_id": cost_profile_id,
                "task_id": _single_value(group, "task_id"),
                "dataset_id": _single_value(group, "dataset_id"),
                "model_family": _single_value(group, "model_family"),
                "n_images": int(_single_value(group, "n_images")),
                "batch_size": int(_single_value(group, "batch_size")),
                "device": _single_value(group, "device"),
                "precision": _single_value(group, "precision"),
                "warmup_runs": int(_single_value(group, "warmup_runs")),
                "timed_runs": declared_runs,
                "n_repeats": int(len(group)),
                "estimated_forward_ms_per_image": median,
                "mean_ms_per_image": mean,
                "median_ms_per_image": median,
                "std_ms_per_image": std,
                "cv_ms_per_image": std / mean if mean > 0 else np.nan,
                "images_per_second": 1000.0 / median if median > 0 else np.nan,
                "peak_allocated_memory_mb": float(
                    group["peak_allocated_memory_mb"].astype(float).max()
                ),
                "checkpoint_mb": float(group["checkpoint_mb"].astype(float).max()),
                "timing_scope": _single_value(group, "timing_scope"),
                "cost_status": "measured",
                "timing_source": _single_value(group, "timing_source"),
            }
        )
    return pd.DataFrame(rows)


def validate_runtime_device(config: dict[str, Any], actual_device_name: str) -> str:
    expected = str(config.get("expected_device_name_contains", "")).strip()
    if expected and expected.lower() not in actual_device_name.lower():
        raise BenchmarkError(
            f"实际 GPU={actual_device_name!r} 与成本 profile 声明的 {expected!r} 不一致"
        )
    return actual_device_name


def _load_json_mapping(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in data.items()}


def _prepare_convnext(spec: dict[str, Any], device: Any) -> PreparedArtifact:
    import yaml

    from models.classifiers.builder import build_model
    from models.datasets.aptos_dataset import build_aptos_dataloader

    config_path = Path(spec["model_config"])
    checkpoint_path = Path(spec["checkpoint_path"])
    class_mapping_path = Path(spec["class_to_idx_path"])
    for path in (config_path, checkpoint_path, class_mapping_path):
        if not path.exists():
            raise BenchmarkError(f"ConvNeXt 运行文件不存在：{path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data_root"] = str(spec["data_root"])
    dataset, loader = build_aptos_dataloader(
        data_root=config["data_root"],
        split=str(spec["split"]),
        image_size=int(config["image_size"]),
        batch_size=int(spec["batch_size"]),
        num_workers=int(spec.get("num_workers", 4)),
        shuffle=False,
    )
    expected_mapping = _load_json_mapping(class_mapping_path)
    observed_mapping = {str(key): int(value) for key, value in dataset.class_to_idx.items()}
    if observed_mapping != expected_mapping:
        raise BenchmarkError(
            "ConvNeXt class_to_idx 不一致，停止成本测量："
            f"dataset={observed_mapping}, checkpoint={expected_mapping}"
        )
    model = build_model(
        config=config,
        checkpoint_path=str(checkpoint_path),
        device=device,
        training=False,
    )
    return PreparedArtifact(
        model=model,
        loader=loader,
        n_images=len(dataset),
        checkpoint_path=checkpoint_path,
        adapter_note="复用 v0.8.3 ConvNeXt 青光眼分类器 loader",
    )


def _prepare_retfound(spec: dict[str, Any], device: Any) -> PreparedArtifact:
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from scripts.v083_glaucoma.export_retfound_dinov2_glaucoma_expert_predictions import (
        build_eval_transform,
        load_checkpoint_and_model,
    )

    retfound_root = Path(spec["retfound_root"])
    checkpoint_path = Path(spec["checkpoint_path"])
    split_root = Path(spec["data_root"]) / str(spec["split"])
    for path in (retfound_root, checkpoint_path, split_root):
        if not path.exists():
            raise BenchmarkError(f"RETFound 运行文件不存在：{path}")
    model, checkpoint_args, _, _ = load_checkpoint_and_model(
        retfound_root=retfound_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    transform = build_eval_transform(
        input_size=int(checkpoint_args.input_size),
        norm=str(getattr(checkpoint_args, "norm", "IMAGENET")),
    )
    dataset = datasets.ImageFolder(root=str(split_root), transform=transform)
    expected_mapping = {
        str(key): int(value)
        for key, value in spec.get("expected_class_to_idx", {}).items()
    }
    if expected_mapping and dataset.class_to_idx != expected_mapping:
        raise BenchmarkError(
            "RETFound class_to_idx 不一致，停止成本测量："
            f"dataset={dataset.class_to_idx}, expected={expected_mapping}"
        )
    loader = DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=False,
        num_workers=int(spec.get("num_workers", 8)),
        pin_memory=True,
    )
    return PreparedArtifact(
        model=model,
        loader=loader,
        n_images=len(dataset),
        checkpoint_path=checkpoint_path,
        adapter_note="复用 v0.8.3 RETFound-DINOv2 青光眼专家 loader",
    )


def prepare_artifact(spec: dict[str, Any], device: Any) -> PreparedArtifact:
    if spec["adapter"] == "glaucoma_convnext_tiny":
        return _prepare_convnext(spec, device)
    if spec["adapter"] == "glaucoma_retfound_dinov2":
        return _prepare_retfound(spec, device)
    raise BenchmarkError(f"不支持的 adapter：{spec['adapter']}")


def benchmark_prepared_artifact(
    spec: dict[str, Any],
    prepared: PreparedArtifact,
    config: dict[str, Any],
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    if str(config["precision"]) != "fp32":
        raise BenchmarkError("当前正式成本协议只支持 fp32")
    model = prepared.model
    model.eval()
    try:
        first_batch = next(iter(prepared.loader))
    except StopIteration as exc:
        raise BenchmarkError(f"{spec['artifact_id']} 数据集为空") from exc
    images = first_batch[0].to(device, non_blocking=True)
    with torch.inference_mode():
        for _ in range(int(config["warmup_runs"])):
            model(images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    del images, first_batch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    for repeat_index in range(1, int(config["timed_runs"]) + 1):
        total_forward_ms = 0.0
        seen = 0
        for batch in prepared.loader:
            batch_images = batch[0].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                model(batch_images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_forward_ms += (time.perf_counter() - started) * 1000.0
            seen += int(batch_images.shape[0])
        if seen != prepared.n_images:
            raise BenchmarkError(
                f"{spec['artifact_id']} 本轮计时图像数异常：{seen} != {prepared.n_images}"
            )
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if device.type == "cuda"
            else 0.0
        )
        rows.append(
            {
                "artifact_id": spec["artifact_id"],
                "cost_profile_id": spec["cost_profile_id"],
                "task_id": spec["task_id"],
                "dataset_id": spec["dataset_id"],
                "model_family": spec["model_family"],
                "repeat_index": repeat_index,
                "n_images": seen,
                "batch_size": int(spec["batch_size"]),
                "device": str(config["runtime_device_name"]),
                "precision": config["precision"],
                "warmup_runs": int(config["warmup_runs"]),
                "timed_runs": int(config["timed_runs"]),
                "total_forward_ms": total_forward_ms,
                "ms_per_image": total_forward_ms / seen,
                "peak_allocated_memory_mb": peak_memory,
                "checkpoint_mb": prepared.checkpoint_path.stat().st_size / 1024 / 1024,
                "timing_scope": config["timing_scope"],
                "timing_source": (
                    f"{config['benchmark_id']} | {prepared.adapter_note} | "
                    "不含解码、预处理、I/O、CPU-GPU 传输和服务开销"
                ),
            }
        )
    return rows


def run_benchmark(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import torch
    except ImportError as exc:
        raise BenchmarkError("正式成本测量需要安装项目 PyTorch 环境") from exc
    device = torch.device(str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise BenchmarkError("正式 v0.8.4b 成本测量要求可用 CUDA GPU")
    runtime_config = dict(config)
    runtime_config["runtime_device_name"] = validate_runtime_device(
        config, torch.cuda.get_device_name(device)
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_rows: list[dict[str, Any]] = []
    for spec in config["artifacts"]:
        prepared = prepare_artifact(spec, device)
        try:
            all_rows.extend(
                benchmark_prepared_artifact(spec, prepared, runtime_config, device)
            )
        finally:
            del prepared
            gc.collect()
            torch.cuda.empty_cache()
    runs = pd.DataFrame(all_rows, columns=RUN_COLUMNS)
    return runs, aggregate_cost_runs(runs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_benchmark_config(args.config)
    if args.dry_run:
        print(f"[DRY-RUN] benchmark_id={config['benchmark_id']}")
        for artifact in config["artifacts"]:
            print(
                "[PLANNED] "
                f"{artifact['artifact_id']} | adapter={artifact['adapter']} | "
                f"profile={artifact['cost_profile_id']}"
            )
        print("[DRY-RUN COMPLETE] 未加载模型，未写入文件")
        return 0

    runs, summary = run_benchmark(config)
    args.runs_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.runs_output, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
    print(f"[完成] 原始重复测量：{args.runs_output}")
    print(f"[完成] 正式成本汇总：{args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
