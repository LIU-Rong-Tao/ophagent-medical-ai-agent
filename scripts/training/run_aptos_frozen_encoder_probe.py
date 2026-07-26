#!/usr/bin/env python3
"""Shared frozen-encoder APTOS linear-probe adaptation for registered CFP assets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eyeclip_task_adapter import load_eyeclip_foundation, sha256_file as eyeclip_sha256  # noqa: E402
from app.keepfit_task_adapter import load_keepfit_vision, preprocess_keepfit_image  # noqa: E402
from app.aptos_replay_adapters import (  # noqa: E402
    GreenAptosTaskAdapter,
    ReplayAdapterSpec,
    TimmAptosTaskAdapter,
    load_registered_aptos_adapter,
)
from app.flair_task_adapter import FlairAptosTaskAdapter  # noqa: E402
from app.preti_task_adapter import PretiAptosTaskAdapter  # noqa: E402
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
    "convnext_tiny": {
        "model_id": "convnext_tiny",
        "registered_manifest": Path(
            "/training_data/lizekun/ophagent_assets/experiments/model_hub/runs/"
            "external_transfer/deepdrid_v1.1/frozen_official_validation/"
            "convnext_tiny/inference_manifest.json"
        ),
    },
    "swin_tiny": {
        "model_id": "swin_tiny",
        "registered_manifest": Path(
            "/training_data/lizekun/ophagent_assets/experiments/model_hub/runs/"
            "external_transfer/deepdrid_v1.1/frozen_official_validation/"
            "swin_tiny/inference_manifest.json"
        ),
    },
    "retfound_cfp": {
        "model_id": "retfound_cfp",
        "registered_manifest": Path(
            "/training_data/lizekun/ophagent_assets/experiments/model_hub/runs/"
            "external_transfer/deepdrid_v1.1/frozen_official_validation/"
            "retfound_cfp/inference_manifest.json"
        ),
    },
    "flair": {
        "model_id": "flair",
        "registered_manifest": Path(
            "/training_data/lizekun/ophagent_assets/experiments/model_hub/runs/"
            "external_transfer/deepdrid_v1.1/frozen_official_validation/"
            "flair/inference_manifest.json"
        ),
    },
    "retfound_green": {
        "model_id": "retfound_green",
        "registered_manifest": Path(
            "/training_data/lizekun/ophagent_assets/experiments/model_hub/runs/"
            "external_transfer/deepdrid_v1.1/frozen_official_validation/"
            "retfound_green/inference_manifest.json"
        ),
    },
    "preti": {
        "model_id": "preti",
        "registered_manifest": Path(
            "/training_data/lizekun/ophagent_assets/experiments/model_hub/runs/"
            "external_transfer/deepdrid_v1.1/frozen_official_validation/"
            "preti/inference_manifest.json"
        ),
    },
}


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: Callable,
        data_root: Path = DATA_ROOT,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        image = Image.open(self.data_root / row.relative_path).convert("RGB")
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


def _load_registered(spec: dict, device: str):
    payload = json.loads(spec["registered_manifest"].read_text(encoding="utf-8"))
    adapter_spec = ReplayAdapterSpec(**payload["adapter_spec"])
    adapter = load_registered_aptos_adapter(adapter_spec, device=device)
    if isinstance(adapter, TimmAptosTaskAdapter):
        def forward(batch):
            return adapter.model.forward_head(
                adapter.model.forward_features(batch.to(device)),
                pre_logits=True,
            )

        model = adapter.model
    elif isinstance(adapter, FlairAptosTaskAdapter):
        from app.flair_task_adapter import preprocess_flair_image

        model = adapter.encoder

        def forward(batch):
            return model(batch.to(device))

        transform = preprocess_flair_image
    elif isinstance(adapter, GreenAptosTaskAdapter):
        model = adapter.encoder

        def forward(batch):
            return model(batch.to(device))

        transform = adapter.preprocess
    elif isinstance(adapter, PretiAptosTaskAdapter):
        model = adapter.model.encoder

        def forward(batch):
            return model.forward_encoder_no_masking(batch.to(device))[0][:, 0]

        transform = adapter.preprocess
    else:
        raise TypeError(f"不支持的登记任务适配器：{type(adapter).__name__}")
    if isinstance(adapter, TimmAptosTaskAdapter):
        transform = adapter.preprocess
    spec["checkpoint"] = Path(adapter_spec.checkpoint_path)
    spec["preprocessing_id"] = adapter_spec.preprocessing_id
    spec["adapter_type"] = f"{adapter_spec.adapter_type}_frozen_features"
    spec["source_commit"] = "registered_task_runtime"
    return model, transform, forward


LOADERS = {
    "eyeclip_cfp": _load_eyeclip,
    "keepfit_cfp": _load_keepfit,
    "ret_clip": _load_ret_clip,
    "retizero": _load_retizero,
    "convnext_tiny": _load_registered,
    "swin_tiny": _load_registered,
    "retfound_cfp": _load_registered,
    "flair": _load_registered,
    "retfound_green": _load_registered,
    "preti": _load_registered,
}


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_task_config(path: Path | None) -> dict:
    if path is None:
        return {
            "task_id": "aptos_dr_5class",
            "dataset_id": "APTOS2019",
            "data_root": str(DATA_ROOT),
            "output_root": str(EXPERIMENT_ROOT),
            "class_order": list(range(5)),
            "c_candidates": None,
        }
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("任务适配配置必须为 mapping")
    return payload


def prepare_task_manifests(task_config: dict) -> tuple[dict[str, pd.DataFrame], str]:
    source_path = Path(task_config["source_manifest"])
    source = pd.read_csv(source_path)
    required = {"case_id", "patient_id", "source_split", "y_true", "relative_image_path"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"源 manifest 缺少字段：{missing}")
    train_source = source[
        source["source_split"].eq(task_config["source_train_split"])
        & source["y_true"].notna()
    ].copy()
    test = source[
        source["source_split"].eq(task_config["frozen_test_split"])
        & source["y_true"].notna()
    ].copy()
    patient_labels = train_source.groupby("patient_id")["y_true"].max()
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(task_config["internal_validation_fraction"]),
        random_state=int(task_config["random_seed"]),
    )
    patient_ids = patient_labels.index.to_numpy()
    train_index, validation_index = next(
        splitter.split(patient_ids, patient_labels.to_numpy())
    )
    train_patients = set(patient_ids[train_index])
    validation_patients = set(patient_ids[validation_index])
    if train_patients & validation_patients:
        raise ValueError("患者级内部划分发生重叠")
    frames = {
        "train": train_source[train_source["patient_id"].isin(train_patients)].copy(),
        "val": train_source[
            train_source["patient_id"].isin(validation_patients)
        ].copy(),
        "test": test.copy(),
    }
    canonical_rows = []
    for split, frame in frames.items():
        frame["split"] = split
        frame["label"] = frame["y_true"].astype(int)
        frame["image_key"] = frame["case_id"].astype(str)
        frame["relative_path"] = frame["relative_image_path"].astype(str)
        frames[split] = frame.sort_values("case_id").reset_index(drop=True)
        canonical_rows.extend(
            frames[split][
                ["case_id", "patient_id", "split", "label", "relative_image_path"]
            ].to_dict("records")
        )
    canonical = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output_root = Path(task_config["manifest_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        target = output_root / f"{split}_manifest.csv"
        if target.exists():
            existing = pd.read_csv(target)
            if existing["case_id"].astype(str).tolist() != frame[
                "case_id"
            ].astype(str).tolist():
                raise ValueError(f"已冻结 {split} manifest 与确定性划分不一致")
        else:
            frame.to_csv(target, index=False)
    summary = {
        "task_id": task_config["task_id"],
        "source_manifest_sha256": _sha256_text(source_path),
        "split_manifest_sha256": digest,
        "random_seed": int(task_config["random_seed"]),
        "internal_validation_fraction": float(
            task_config["internal_validation_fraction"]
        ),
        "patient_counts": {
            split: int(frame["patient_id"].nunique())
            for split, frame in frames.items()
        },
        "image_counts": {split: int(len(frame)) for split, frame in frames.items()},
        "patient_overlap": {
            "train_validation": 0,
            "train_test": int(
                len(set(frames["train"].patient_id) & set(frames["test"].patient_id))
            ),
            "validation_test": int(
                len(set(frames["val"].patient_id) & set(frames["test"].patient_id))
            ),
        },
        "test_used_for_selection": False,
    }
    summary_path = output_root / "split_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise ValueError("已冻结 split summary 与当前协议不一致")
    else:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frames, digest


def _manifests(task_config: dict) -> tuple[dict[str, pd.DataFrame], str]:
    if task_config["task_id"] != "aptos_dr_5class":
        return prepare_task_manifests(task_config)
    root = EXPERIMENT_ROOT / "preti/seed42_20260722/preflight"
    result = {split: pd.read_csv(root / f"{split}_manifest.csv") for split in ("train", "val", "test")}
    expected = {"train": 2048, "val": 514, "test": 1100}
    if {split: len(frame) for split, frame in result.items()} != expected:
        raise ValueError("冻结 APTOS manifest 样本数不一致")
    _, digest = dataset_manifest(DATA_ROOT)
    if digest != APTOS_MANIFEST_SHA256:
        raise ValueError("APTOS 目录与冻结 manifest 不一致")
    return result, digest


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


def _prediction_frame(manifest: pd.DataFrame, probabilities: np.ndarray, spec: dict, split: str, run_id: str, sha256: str, split_id: str) -> pd.DataFrame:
    ordered = manifest.reset_index(drop=True).copy()
    output = pd.DataFrame({
        "case_id": ordered.image_key,
        "patient_id": ordered["patient_id"].astype(str)
        if "patient_id" in ordered
        else "",
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
    output["split_id"] = split_id
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


def run(
    model_name: str,
    device: str,
    smoke_only: bool = False,
    task_config_path: Path | None = None,
) -> Path:
    spec = dict(MODEL_SPECS[model_name])
    task_config = _load_task_config(task_config_path)
    set_seed(42)
    manifests, split_id = _manifests(task_config)
    if "registered_manifest" in spec:
        model, transform, forward = _load_registered(spec, device)
    else:
        model, transform, forward = LOADERS[model_name](spec, device)
    checkpoint_sha256 = eyeclip_sha256(spec["checkpoint"])
    if not spec["checkpoint"].is_file() or (
        "source" in spec and not spec["source"].is_dir()
    ):
        raise FileNotFoundError("本地 checkpoint 或官方源码快照缺失")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("基础编码器必须完全冻结")
    data_root = Path(task_config["data_root"])
    smoke_dataset = ManifestDataset(
        manifests["val"].iloc[:16],
        transform,
        data_root,
    )
    smoke_x, _, _ = _features(forward, smoke_dataset, 16, device)
    repeat_x, _, _ = _features(forward, smoke_dataset, 16, device)
    if smoke_x.ndim != 2 or not np.allclose(smoke_x, repeat_x, atol=1e-6, rtol=1e-5):
        raise RuntimeError("16 张真实 APTOS 图像重复前向不一致")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(task_config["output_root"]) / model_name / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "smoke.json").write_text(json.dumps({"status": "passed", "samples": 16, "feature_dim": int(smoke_x.shape[1]), "checkpoint_sha256": checkpoint_sha256}, indent=2), encoding="utf-8")
    if smoke_only:
        return output
    datasets = {
        split: ManifestDataset(frame, transform, data_root)
        for split, frame in manifests.items()
    }
    train_x, train_y, _ = _features(forward, datasets["train"], 32, device)
    val_x, val_y, _ = _features(forward, datasets["val"], 32, device)
    test_x, test_y, _ = _features(forward, datasets["test"], 32, device)
    choices = []
    c_candidates = task_config.get("c_candidates") or spec["c_candidates"]
    for c_value in c_candidates:
        probe = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, random_state=42).fit(train_x, train_y)
        probabilities = probe.predict_proba(val_x)
        choices.append((classification_metrics(val_y, probabilities)["macro_f1"], c_value, probe))
    _, selected_c, probe = max(choices, key=lambda item: item[0])
    class_order = np.asarray(task_config["class_order"])
    if not np.array_equal(probe.classes_, class_order):
        raise ValueError(
            f"探针类别顺序 {probe.classes_.tolist()} 与任务协议 {class_order.tolist()} 不一致"
        )
    val_probabilities, test_probabilities = probe.predict_proba(val_x), probe.predict_proba(test_x)
    (output / "predictions").mkdir()
    _prediction_frame(manifests["val"], val_probabilities, spec, "validation", run_id, checkpoint_sha256, split_id).to_csv(output / "predictions/validation_predictions.csv", index=False)
    _prediction_frame(manifests["test"], test_probabilities, spec, "test", run_id, checkpoint_sha256, split_id).to_csv(output / "predictions/test_predictions.csv", index=False)
    (output / "metrics").mkdir()
    metrics = {
        "evaluation_design": task_config.get(
            "evaluation_design",
            "aptos_frozen_encoder_probe",
        ),
        "selection_split": "internal_validation",
        "frozen_evaluation_split": task_config.get(
            "frozen_evaluation_split",
            "test",
        ),
        "selected_c": selected_c,
        "test_used_for_selection": False,
        "validation": classification_metrics(val_y, val_probabilities),
        "test": classification_metrics(test_y, test_probabilities),
    }
    (output / "metrics/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "costs").mkdir()
    (output / "costs/forward_cost.json").write_text(json.dumps(_cost(forward, datasets["val"], device), indent=2), encoding="utf-8")
    source_commit = spec.get("source_commit")
    if source_commit is None and "source" in spec:
        source_commit = _source_commit(spec["source"])
    probe_path = output / "probe.joblib"
    joblib.dump(
        {
            "probe": probe,
            "class_order": list(task_config["class_order"]),
            "selected_c": selected_c,
            "split_id": split_id,
        },
        probe_path,
    )
    config = {
        "task_id": task_config["task_id"],
        "dataset_id": task_config["dataset_id"],
        "model": model_name,
        "adapter_type": spec["adapter_type"],
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": source_commit,
        "preprocessing_id": spec["preprocessing_id"],
        "manifest_sha256": split_id,
        "selection": {
            "split": "internal_validation",
            "c_candidates": c_candidates,
            "selected_c": selected_c,
        },
        "frozen_test_used_for_selection": False,
        "route_eligible": False,
    }
    (output / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "run_summary.json").write_text(json.dumps({"status": "completed", "task_adapted": True, "task_inference_ready": False, "offline_evaluation_eligible": True, "validation_selection_eligible": True, "route_eligible": False, "feature_dim": int(train_x.shape[1]), "metrics": metrics}, indent=2), encoding="utf-8")
    artifact_rows = []
    for relative in (
        "config_resolved.json",
        "probe.joblib",
        "run_summary.json",
        "metrics/metrics.json",
        "costs/forward_cost.json",
        "predictions/validation_predictions.csv",
        "predictions/test_predictions.csv",
    ):
        path = output / relative
        artifact_rows.append(
            {
                "relative_path": relative,
                "sha256": eyeclip_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(artifact_rows).to_csv(output / "artifact_manifest.csv", index=False)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--prepare-task-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_task_only:
        if args.task_config is None:
            parser.error("--prepare-task-only 需要 --task-config")
        task_config = _load_task_config(args.task_config)
        _, digest = prepare_task_manifests(task_config)
        print(digest, flush=True)
        return 0
    if args.model is None:
        parser.error("运行模型适配时必须提供 --model")
    print(
        run(args.model, args.device, args.smoke_only, args.task_config),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
