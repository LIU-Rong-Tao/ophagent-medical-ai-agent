#!/usr/bin/env python3
"""Shared frozen-encoder APTOS linear-probe adaptation for registered CFP assets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eyeclip_task_adapter import load_eyeclip_foundation, sha256_file as eyeclip_sha256  # noqa: E402
from app.keepfit_task_adapter import load_keepfit_vision, preprocess_keepfit_image  # noqa: E402
from scripts.training.aptos_downstream_common import (  # noqa: E402
    APTOS_MANIFEST_SHA256,
    classification_metrics,
    dataset_manifest,
    set_seed,
)


ASSET_ROOT = Path("/training_data/lizekun/model_cache/ophbench")
SOURCE_ROOT = ASSET_ROOT / "upstream_sources"
DATA_ROOT = Path("/training_data/lizekun/data/RETFound/Data_split/APTOS2019")
EXPERIMENT_ROOT = ROOT / "experiments/opening_risk_routing_closure/replays"
CLASS_ORDER = ("anodr", "bmilddr", "cmoderatedr", "dseveredr", "eproliferativedr")

MODEL_SPECS = {
    "eyeclip_cfp": {
        "model_id": "eyeclip_cfp",
        "checkpoint_id": "eyeclip-default",
        "checkpoint": ASSET_ROOT / "eyeclip/eyeclip-default/eyeclip.pt",
        "source": SOURCE_ROOT / "eyeclip/2fcf6034552e6006c94bd84cbdc6f4a5897b29c0",
        "preprocessing_id": "eyeclip_clip_transform_224",
        "adapter_type": "eyeclip_frozen_encoder_linear_probe",
        "c_candidates": [0.01, 0.1, 1.0, 10.0],
    },
    "keepfit_cfp": {
        "model_id": "keepfit_cfp",
        "checkpoint_id": "keepfit-flair-mmretinal-cfp",
        "checkpoint": ASSET_ROOT / "keepfit/keepfit-flair-mmretinal-cfp/KeepFIT (flair+MM).pth",
        "source": SOURCE_ROOT / "keepfit/dbbb1f05b9d27278b01e15e5f837b44b22d32cee",
        "preprocessing_id": "keepfit_preserve_aspect_pad_512_scale_01",
        "adapter_type": "keepfit_frozen_encoder_linear_probe",
        "c_candidates": [0.316],
    },
    "ret_clip": {
        "model_id": "ret_clip",
        "checkpoint_id": "ret-clip-default",
        "checkpoint": ASSET_ROOT / "ret-clip/ret-clip-default/ret-clip.pt",
        "source": SOURCE_ROOT / "ret-clip/1ddb9a1d331eba9e0a2675f3273e7cbcea0914bd",
        "preprocessing_id": "ret_clip_image_transform_224",
        "adapter_type": "ret_clip_frozen_encoder_linear_probe",
        "c_candidates": [0.01, 0.1, 1.0, 10.0],
    },
    "retizero": {
        "model_id": "retizero",
        "checkpoint_id": "retizero-default",
        "checkpoint": ASSET_ROOT / "retizero/retizero-default/RetiZero.pth",
        "source": SOURCE_ROOT / "retizero/d72aadc692fbe33b182c79711bccb397edffb419",
        "preprocessing_id": "retizero_resize_224_imagenet",
        "adapter_type": "retizero_frozen_encoder_linear_probe",
        "c_candidates": [0.01, 0.1, 1.0, 10.0],
    },
}


class ManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: Callable):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image = Image.open(DATA_ROOT / row.relative_path).convert("RGB")
        return self.transform(image), int(row.label), str(row.image_key)


def _source_commit(path: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _load_eyeclip(spec: dict, device: str):
    sys.path.insert(0, str(spec["source"]))
    from eyeclip.clip import _transform

    visual = load_eyeclip_foundation(spec["source"], spec["checkpoint"], device).eval()
    return visual, _transform(224), lambda batch: visual(batch.to(device))


def _load_keepfit(spec: dict, device: str):
    encoder = load_keepfit_vision(spec["checkpoint"], spec["checkpoint_id"], device).eval()
    return encoder, preprocess_keepfit_image, lambda batch: encoder(batch.to(device))


def _load_ret_clip(spec: dict, device: str):
    import torch

    sys.path.insert(0, str(spec["source"]))
    from RET_CLIP.clip.model import CLIP, convert_models_to_fp32, convert_weights
    from RET_CLIP.clip.utils import image_transform

    config_root = spec["source"] / "RET_CLIP/clip/model_configs"
    info = {}
    for name in ("ViT-B-16.json", "RoBERTa-wwm-ext-base-chinese.json"):
        info.update(json.loads((config_root / name).read_text(encoding="utf-8")))
    model = CLIP(**info)
    convert_weights(model)
    convert_models_to_fp32(model)
    raw = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in raw.items() if "bert.pooler" not in key}
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    return model, image_transform(224), lambda batch: model(batch.to(device), None, None)


def _load_retizero(spec: dict, device: str):
    import torch

    dependency_root = ASSET_ROOT / "runtime_deps/urfound"
    for name in tuple(sys.modules):
        if name == "transformers" or name.startswith("transformers.") or name == "zeroshot" or name.startswith("zeroshot."):
            del sys.modules[name]
    sys.path[:0] = [str(dependency_root), str(spec["source"])]
    from transformers import AutoConfig, AutoModel
    import zeroshot.modeling.model as retizero_model

    config = AutoConfig.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", local_files_only=True)
    config.output_hidden_states = True
    original_tokenizer = retizero_model.AutoTokenizer.from_pretrained
    retizero_model.AutoTokenizer.from_pretrained = lambda name, *args, **kwargs: original_tokenizer(name, *args, local_files_only=True, **kwargs)
    retizero_model.AutoModel.from_pretrained = lambda *args, **kwargs: AutoModel.from_config(config)
    model = retizero_model.CLIPRModel(vision_type="lora", vision_pretrained=False, from_checkpoint=False, R=8)
    model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=False), strict=True)
    model = model.eval().to(device)
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    transform = Compose([Resize((224, 224)), ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return model, transform, lambda batch: model.vision_model(batch.to(device))


LOADERS = {
    "eyeclip_cfp": _load_eyeclip,
    "keepfit_cfp": _load_keepfit,
    "ret_clip": _load_ret_clip,
    "retizero": _load_retizero,
}


def _manifests() -> dict[str, pd.DataFrame]:
    root = EXPERIMENT_ROOT / "preti/seed42_20260722/preflight"
    result = {split: pd.read_csv(root / f"{split}_manifest.csv") for split in ("train", "val", "test")}
    expected = {"train": 2048, "val": 514, "test": 1100}
    if {split: len(frame) for split, frame in result.items()} != expected:
        raise ValueError("冻结 APTOS manifest 样本数不一致")
    _, digest = dataset_manifest(DATA_ROOT)
    if digest != APTOS_MANIFEST_SHA256:
        raise ValueError("APTOS 目录与冻结 manifest 不一致")
    return result


def _features(forward: Callable, dataset: Dataset, batch_size: int, device: str):
    import torch

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    values, labels, keys = [], [], []
    with torch.inference_mode():
        for images, target, sample_keys in loader:
            output = forward(images)
            if not torch.isfinite(output).all():
                raise ValueError("冻结编码器产生 NaN/Inf")
            values.append(output.float().cpu().numpy())
            labels.append(target.numpy())
            keys.extend(sample_keys)
    return np.concatenate(values), np.concatenate(labels), keys


def _prediction_frame(manifest: pd.DataFrame, probabilities: np.ndarray, spec: dict, split: str, run_id: str, sha256: str) -> pd.DataFrame:
    ordered = manifest.reset_index(drop=True).copy()
    output = pd.DataFrame({
        "case_id": ordered.image_key,
        "patient_id": "",
        "split": split,
        "y_true": ordered.label.astype(int),
        "y_pred": probabilities.argmax(axis=1),
        "image_key": ordered.image_key,
        "image_path": ordered.relative_path,
    })
    for index in range(5):
        output[f"prob_{index}"] = probabilities[:, index]
    output["model_id"] = spec["model_id"]
    output["checkpoint_sha256"] = sha256
    output["preprocessing_id"] = spec["preprocessing_id"]
    output["split_id"] = APTOS_MANIFEST_SHA256
    output["inference_run_id"] = run_id
    output["inference_dtype"] = "float32"
    return output


def _cost(forward: Callable, dataset: Dataset, device: str) -> dict:
    import torch

    results = {}
    for batch_size in (1, 16):
        images = torch.stack([dataset[index][0] for index in range(batch_size)]).to(device)
        for _ in range(10):
            forward(images)
        torch.cuda.synchronize()
        elapsed = []
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(30):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            forward(images)
            end.record()
            torch.cuda.synchronize()
            elapsed.append(start.elapsed_time(end))
        results[f"batch_{batch_size}"] = {
            "median_forward_ms": float(np.median(elapsed)),
            "median_ms_per_image": float(np.median(elapsed) / batch_size),
            "throughput_images_per_second": float(batch_size * 1000 / np.median(elapsed)),
            "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        }
    return {"device": device, "dtype": "float32", "warmup": 10, "repeats": 30, "forward_only": True, "results": results}


def run(model_name: str, device: str, smoke_only: bool = False) -> Path:
    spec = MODEL_SPECS[model_name]
    set_seed(42)
    manifests = _manifests()
    checkpoint_sha256 = eyeclip_sha256(spec["checkpoint"])
    if not spec["checkpoint"].is_file() or not spec["source"].is_dir():
        raise FileNotFoundError("本地 checkpoint 或官方源码快照缺失")
    model, transform, forward = LOADERS[model_name](spec, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("基础编码器必须完全冻结")
    smoke_dataset = ManifestDataset(manifests["val"].iloc[:16], transform)
    smoke_x, _, _ = _features(forward, smoke_dataset, 16, device)
    repeat_x, _, _ = _features(forward, smoke_dataset, 16, device)
    if smoke_x.ndim != 2 or not np.allclose(smoke_x, repeat_x, atol=1e-6, rtol=1e-5):
        raise RuntimeError("16 张真实 APTOS 图像重复前向不一致")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = EXPERIMENT_ROOT / model_name / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "smoke.json").write_text(json.dumps({"status": "passed", "samples": 16, "feature_dim": int(smoke_x.shape[1]), "checkpoint_sha256": checkpoint_sha256}, indent=2), encoding="utf-8")
    if smoke_only:
        return output
    datasets = {split: ManifestDataset(frame, transform) for split, frame in manifests.items()}
    train_x, train_y, _ = _features(forward, datasets["train"], 32, device)
    val_x, val_y, _ = _features(forward, datasets["val"], 32, device)
    test_x, test_y, _ = _features(forward, datasets["test"], 32, device)
    choices = []
    for c_value in spec["c_candidates"]:
        probe = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=42).fit(train_x, train_y)
        probabilities = probe.predict_proba(val_x)
        choices.append((classification_metrics(val_y, probabilities)["macro_f1"], c_value, probe))
    _, selected_c, probe = max(choices, key=lambda item: item[0])
    val_probabilities, test_probabilities = probe.predict_proba(val_x), probe.predict_proba(test_x)
    (output / "predictions").mkdir()
    _prediction_frame(manifests["val"], val_probabilities, spec, "validation", run_id, checkpoint_sha256).to_csv(output / "predictions/validation_predictions.csv", index=False)
    _prediction_frame(manifests["test"], test_probabilities, spec, "test", run_id, checkpoint_sha256).to_csv(output / "predictions/test_predictions.csv", index=False)
    (output / "metrics").mkdir()
    metrics = {"selection_split": "validation", "selected_c": selected_c, "test_used_for_selection": False, "validation": classification_metrics(val_y, val_probabilities), "test": classification_metrics(test_y, test_probabilities)}
    (output / "metrics/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "costs").mkdir()
    (output / "costs/forward_cost.json").write_text(json.dumps(_cost(forward, datasets["val"], device), indent=2), encoding="utf-8")
    config = {"model": model_name, "adapter_type": spec["adapter_type"], "checkpoint": str(spec["checkpoint"]), "checkpoint_sha256": checkpoint_sha256, "source_commit": _source_commit(spec["source"]), "preprocessing_id": spec["preprocessing_id"], "manifest_sha256": APTOS_MANIFEST_SHA256, "selection": {"split": "validation", "c_candidates": spec["c_candidates"], "selected_c": selected_c}, "route_eligible": False}
    (output / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "run_summary.json").write_text(json.dumps({"status": "completed", "task_adapted": True, "task_inference_ready": False, "offline_evaluation_eligible": True, "validation_selection_eligible": True, "route_eligible": False, "feature_dim": int(train_x.shape[1]), "metrics": metrics}, indent=2), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    print(run(args.model, args.device, args.smoke_only), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
