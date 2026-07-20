#!/usr/bin/env python3
"""Run the repaired official-repository EyeCLIP APTOS fine-tuning recipe."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eyeclip_task_adapter import (  # noqa: E402
    EYECLIP_CHECKPOINT_SHA256,
    EYECLIP_SOURCE_COMMIT,
    EyeClipAptosTaskAdapter,
    EyeClipClassifier,
    load_eyeclip_foundation,
    sha256_file,
)
from scripts.training.aptos_downstream_common import (  # noqa: E402
    APTOS_LABELS,
    APTOS_MANIFEST_SHA256,
    classification_metrics,
    dataset_manifest,
    load_config,
    prediction_frame,
    set_seed,
    utc_now,
)

ARTIFACT_ID = "aptos2019-eyeclip-vitb32-official-recipe-repaired-v1"


def strict_preflight(config):
    source = Path(config["foundation"]["source_root"])
    checkpoint = Path(config["foundation"]["checkpoint_path"])
    actual_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != EYECLIP_SOURCE_COMMIT:
        raise ValueError(f"EyeCLIP 实际源码 commit 不匹配：{actual_commit}")
    if sha256_file(checkpoint) != EYECLIP_CHECKPOINT_SHA256:
        raise ValueError("EyeCLIP checkpoint SHA256 不匹配")
    manifest, digest = dataset_manifest(config["data"]["root"])
    if digest != APTOS_MANIFEST_SHA256:
        raise ValueError("APTOS 数据清单与冻结划分不一致")
    expected = {"epochs": 50, "batch_size": 64, "image_size": 224, "seeds": [0, 1, 2, 3, 4]}
    for key, value in expected.items():
        if config["training"][key] != value:
            raise ValueError(f"EyeCLIP 官方仓库 recipe 要求 training.{key}={value}")
    if config["evaluation"]["selection_split"] != "val":
        raise ValueError("只允许使用 validation 选择 checkpoint")
    if config["evaluation"]["save_best_by"] != "macro_auc_ovr":
        raise ValueError("EyeCLIP 官方仓库按 validation AUC 保存 checkpoint")
    return {
        "strict_preflight": True,
        "source_commit": actual_commit,
        "checkpoint_sha256": EYECLIP_CHECKPOINT_SHA256,
        "dataset_manifest_sha256": digest,
        "split_sizes": {key: value["samples"] for key, value in manifest["splits"].items()},
        "test_used_for_selection": False,
        "official_repository_recipe_executable_without_repairs": False,
    }


def build_loaders(config, seed):
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
    from torchvision.transforms import v2 as transforms

    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=None,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    root = Path(config["data"]["root"])
    datasets = {
        "train": ImageFolder(root / "train", transform=train_transform),
        "val": ImageFolder(root / "val", transform=eval_transform),
        "test": ImageFolder(root / "test", transform=eval_transform),
    }
    common = {
        "batch_size": 64,
        "num_workers": int(config["runtime"]["num_workers"]),
        "pin_memory": True,
    }
    generator = torch.Generator().manual_seed(seed)
    return datasets, {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, generator=generator, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def _layer_id(name: str, block_count: int) -> int:
    if name.startswith(("visual.conv1", "visual.class_embedding", "visual.positional_embedding", "visual.ln_pre")):
        return 0
    prefix = "visual.transformer.resblocks."
    if name.startswith(prefix):
        return int(name[len(prefix) :].split(".", 1)[0]) + 1
    return block_count + 1


def parameter_groups(model, weight_decay: float, layer_decay: float):
    block_count = len(model.visual.transformer.resblocks)
    scales = [layer_decay ** (block_count + 1 - index) for index in range(block_count + 2)]
    groups = {}
    for name, parameter in model.named_parameters():
        layer = _layer_id(name, block_count)
        no_decay = parameter.ndim == 1 or name.endswith(("class_embedding", "positional_embedding"))
        key = (layer, no_decay)
        groups.setdefault(
            key,
            {
                "params": [],
                "lr_scale": scales[layer],
                "weight_decay": 0.0 if no_decay else weight_decay,
            },
        )["params"].append(parameter)
    return list(groups.values())


def _set_lr(optimizer, base_lr, progress, config):
    warmup = config["scheduler"]["warmup_epochs"]
    epochs = config["training"]["epochs"]
    minimum = config["scheduler"]["minimum_learning_rate"]
    if progress < warmup:
        lr = base_lr * progress / warmup
    else:
        lr = minimum + (base_lr - minimum) * 0.5 * (
            1 + math.cos(math.pi * (progress - warmup) / (epochs - warmup))
        )
    for group in optimizer.param_groups:
        group["lr"] = lr * group["lr_scale"]
    return lr


def _train_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    losses = []
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for index, (images, labels) in enumerate(loader):
        _set_lr(optimizer, 1e-4, epoch + index / len(loader), config)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def _evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(logits.float(), 1).cpu().numpy())
        labels.append(targets.numpy())
    probs = np.concatenate(probabilities)
    truth = np.concatenate(labels)
    return classification_metrics(truth, probs), probs


def train_seed(config, seed, seed_dir):
    set_seed(seed)
    datasets, loaders = build_loaders(config, seed)
    device = config["runtime"]["device"]
    visual = load_eyeclip_foundation(
        config["foundation"]["source_root"], config["foundation"]["checkpoint_path"], device
    )
    model = EyeClipClassifier(visual, len(APTOS_LABELS)).to(device)
    torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)
    torch.nn.init.zeros_(model.head.bias)
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01, 0.75), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    best = None
    history = []
    for epoch in range(50):
        started = time.perf_counter()
        loss = _train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            epoch,
            config,
        )
        metrics, _ = _evaluate(model, loaders["val"], device)
        history.append({"seed": seed, "epoch": epoch + 1, "train_loss": loss, **{f"val_{k}": v for k, v in metrics.items()}, "elapsed_seconds": time.perf_counter() - started})
        score = (metrics["macro_auc_ovr"], metrics["quadratic_kappa"], metrics["macro_f1"])
        if best is None or score > best[0]:
            best = (score, epoch + 1, copy.deepcopy(metrics), {k: v.detach().cpu().clone() for k, v in model.visual.state_dict().items()}, {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()})
        print(f"seed={seed} epoch={epoch + 1:02d}/50 loss={loss:.4f} val_auc={metrics['macro_auc_ovr']:.4f} val_qwk={metrics['quadratic_kappa']:.4f}", flush=True)
    model.visual.load_state_dict(best[3], strict=True)
    model.head.load_state_dict(best[4], strict=True)
    test_metrics, test_probs = _evaluate(model, loaders["test"], device)
    seed_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = seed_dir / "eyeclip_aptos_task_checkpoint.pth"
    torch.save({"schema_version": 1, "artifact_id": ARTIFACT_ID, "seed": seed, "labels": APTOS_LABELS, "source_commit": EYECLIP_SOURCE_COMMIT, "encoder_checkpoint_sha256": EYECLIP_CHECKPOINT_SHA256, "dataset_manifest_sha256": APTOS_MANIFEST_SHA256, "selected_epoch": best[1], "visual_state_dict": best[3], "classifier_state_dict": best[4]}, checkpoint)
    prediction_frame(datasets["test"], test_probs, config["data"]["root"]).to_csv(seed_dir / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(seed_dir / "training_history.csv", index=False)
    metrics = {"seed": seed, "selection_split": "val", "selection_metric": "macro_auc_ovr", "selected_epoch": best[1], "validation": best[2], "test": test_metrics, "test_used_for_selection": False}
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, checkpoint


def run_training(config_path: Path, overrides=None):
    config = load_config(config_path)
    for dotted, value in dict(overrides or {}).items():
        if value is not None:
            section, key = dotted.split(".", 1)
            config[section][key] = value
    preflight = strict_preflight(config)
    output = Path(config["output"]["run_dir"])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "base_protocol.yaml")
    (output / "effective_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest, _ = dataset_manifest(config["data"]["root"])
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "validation_report.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    primary_checkpoint = None
    for seed in config["training"]["seeds"]:
        metrics, checkpoint = train_seed(config, int(seed), output / "seed_runs" / f"seed_{seed}")
        results.append(metrics)
        if int(seed) == 0:
            primary_checkpoint = checkpoint
    summary = pd.DataFrame([{ "seed": item["seed"], **{f"val_{k}": v for k, v in item["validation"].items()}, **{f"test_{k}": v for k, v in item["test"].items()} } for item in results])
    summary.to_csv(output / "seed_metrics.csv", index=False)
    aggregate = {column: {"mean": float(summary[column].mean()), "std": float(summary[column].std(ddof=1))} for column in summary if column.startswith("test_") and column != "test_n"}
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    primary = output / "seed_runs/seed_0"
    record = {"model_id": f"aptos_dr_5class::{ARTIFACT_ID}", "task_id": "aptos_dr_5class", "dataset_id": "APTOS2019", "dataset_display_name": "APTOS 2019", "dataset_source": "public", "artifact_id": ARTIFACT_ID, "model_family": "eyeclip", "architecture": "EyeCLIP ViT-B/32", "label_space": "dr_icdr_0_4", "n_classes": 5, "prediction_source": "adapter", "prediction_path": str(primary / "test_predictions.csv"), "adapter_status": "completed", "compatibility_status": "offline_evaluation_ready", "role_candidates": "scout|expert", "checkpoint_path": str(primary_checkpoint), "base_model_provider": "ophbench", "base_model_id": "eyeclip", "base_checkpoint_id": "eyeclip-default", "encoder_checkpoint_sha256": EYECLIP_CHECKPOINT_SHA256, "task_checkpoint": True, "task_adapted": True, "task_inference_ready": True, "offline_evaluation_eligible": True, "unified_evaluation_completed": True, "inference_cost_measured": False, "route_eligible": False, "output_dir": str(output), "evaluation_role": "official_repository_recipe_repaired", "lifecycle_status": "candidate", "research_claim_status": "official_repository_recipe_with_documented_repairs", "cost_status": "unmeasured", "selection_split": "val", "selection_metric": "macro_auc_ovr", "test_used_for_selection": False, "trainer_adapter": "eyeclip_aptos_official_recipe_repaired_v1", "task_checkpoint_sha256": sha256_file(primary_checkpoint)}
    pd.DataFrame([record]).to_csv(output / "registration_record.csv", index=False)
    run_manifest = {"created_at_utc": utc_now(), "artifact_id": ARTIFACT_ID, "source_commit": EYECLIP_SOURCE_COMMIT, "official_command_parameters_preserved": True, "five_seed_protocol": [0, 1, 2, 3, 4], "primary_operational_seed": 0, "test_used_for_selection": False, "route_eligible": False, "documented_repairs": ["README filename main_finetune_ophthal.py corrected to actual main_finetune_opthal.py", "layer decay mapped from nonexistent visual.blocks to visual.transformer.resblocks", "removed per-epoch test evaluation; test runs once after validation checkpoint selection"]}
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    adapter = EyeClipAptosTaskAdapter.load(source_root=config["foundation"]["source_root"], encoder_checkpoint=config["foundation"]["checkpoint_path"], task_checkpoint=primary_checkpoint, device=config["runtime"]["device"])
    images = [ImageFolder(Path(config["data"]["root"]) / "test").loader(path) for path, _ in ImageFolder(Path(config["data"]["root"]) / "test").samples[:2]]
    if adapter.predict_proba(images).shape != (2, 5):
        raise ValueError("EyeCLIP 任务 Adapter 最终 Smoke 失败")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    output = run_training(args.config, overrides={"data.root": str(args.data_root) if args.data_root else None, "foundation.checkpoint_path": str(args.checkpoint) if args.checkpoint else None, "foundation.source_root": str(args.source_root) if args.source_root else None, "output.run_dir": str(args.output_dir) if args.output_dir else None, "runtime.device": args.device})
    print(f"EyeCLIP APTOS adaptation completed: {output}")


if __name__ == "__main__":
    main()
