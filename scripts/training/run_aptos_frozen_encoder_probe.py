#!/usr/bin/env python3
"""Shared frozen-encoder task adaptation for registered CFP assets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
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
from app.generic_result_audit import run_observed_positive_audit  # noqa: E402
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
    "keepfit_half_cfp": {
        "model_id": "keepfit_half_cfp",
        "checkpoint_id": "keepfit-half-flair-mmretinal-cfp",
        "checkpoint": ASSET_ROOT
        / "keepfit/keepfit-half-flair-mmretinal-cfp/KeepFIT (50%flair+MM).pth",
        "source": SOURCE_ROOT / "keepfit/dbbb1f05b9d27278b01e15e5f837b44b22d32cee",
        "preprocessing_id": "keepfit_preserve_aspect_pad_512_scale_01",
        "adapter_type": "keepfit_frozen_encoder_linear_probe",
        "c_candidates": [0.01, 0.1, 0.316, 1.0, 10.0],
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

CHECKPOINT_TASK_MODELS = {
    "eyeclip-default": "eyeclip_cfp",
    "flair-default": "flair",
    "keepfit-flair-mmretinal-cfp": "keepfit_cfp",
    "keepfit-half-flair-mmretinal-cfp": "keepfit_half_cfp",
    "preti-default": "preti",
    "ret-clip-default": "ret_clip",
    "retfound-green-v0.1": "retfound_green",
    "retfound-cfp": "retfound_cfp",
    "retizero-default": "retizero",
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
    "keepfit_half_cfp": _load_keepfit,
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


def _verify_expected_sha256(path: Path, expected: str) -> None:
    actual = _sha256_text(path)
    if actual != expected:
        raise ValueError(f"冻结资产 SHA256 不一致：{path.name}")


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_cfp_compatibility_gate(task_config: dict) -> pd.DataFrame:
    from ophbench import load_registry

    readiness_path = Path(task_config["readiness_registry"])
    readiness = pd.read_csv(readiness_path).fillna("")
    readiness_lookup = {
        (str(row.model_id), str(row.checkpoint_id)): row
        for row in readiness.itertuples(index=False)
        if str(row.scope) == "ophbench_catalog"
    }
    rows = []
    for checkpoint in load_registry().checkpoints:
        checkpoint_id = str(checkpoint.checkpoint_id)
        model_id = str(checkpoint.model_id)
        modalities = tuple(str(item) for item in checkpoint.modalities)
        task_model = CHECKPOINT_TASK_MODELS.get(checkpoint_id, "")
        readiness_row = readiness_lookup.get((model_id, checkpoint_id))
        local_asset = bool(
            readiness_row is not None and _as_bool(readiness_row.h100_local_asset)
        )
        smoke_passed = bool(
            readiness_row is not None
            and str(readiness_row.runtime_smoke_status) == "runtime_smoke_passed"
        )
        cfp_compatible = "CFP" in modalities
        applicable_type = str(checkpoint.artifact_type) not in {
            "generative_model",
            "task_checkpoint",
        }
        adapter_available = bool(task_model and task_model in MODEL_SPECS)
        passed = bool(
            cfp_compatible
            and applicable_type
            and local_asset
            and smoke_passed
            and adapter_available
        )
        blockers = []
        if not cfp_compatible:
            blockers.append("non_cfp_modality")
        if not applicable_type:
            blockers.append("artifact_type_not_applicable")
        if not local_asset:
            blockers.append("h100_local_asset_missing")
        if not smoke_passed:
            blockers.append("runtime_smoke_not_passed")
        if not adapter_available:
            blockers.append("task_adapter_unavailable")
        rows.append(
            {
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "modalities": "|".join(modalities),
                "artifact_type": str(checkpoint.artifact_type),
                "h100_local_asset": local_asset,
                "runtime_smoke_passed": smoke_passed,
                "task_adapter": task_model,
                "cfp_task_compatible": passed,
                "qualification_limited": not bool(checkpoint.license),
                "blocker": ";".join(blockers),
            }
        )
    gate = pd.DataFrame(rows)
    if len(gate) != 27:
        raise ValueError(f"OphBench checkpoint 数量异常：{len(gate)} != 27")
    selected = tuple(gate.loc[gate["cfp_task_compatible"], "task_adapter"])
    configured = tuple(task_config["models"])
    if set(selected) != set(configured):
        raise ValueError(
            f"配置模型与 CFP 门禁结果不一致：gate={selected}, config={configured}"
        )
    output = Path(task_config["compatibility_gate_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = pd.read_csv(output).fillna("")
        if existing.to_dict("records") != gate.fillna("").to_dict("records"):
            raise ValueError("已冻结 CFP 兼容门禁与当前 Registry/readiness 不一致")
    else:
        temporary = output.with_suffix(".csv.tmp")
        gate.to_csv(temporary, index=False)
        temporary.replace(output)
    return gate


def prepare_observed_label_manifests(
    task_config: dict,
) -> tuple[dict[str, pd.DataFrame], str]:
    protocol_root = Path(task_config["canonical_protocol_root"])
    paths = {
        "samples": protocol_root / "canonical_samples.csv",
        "source_mapping": protocol_root / "canonical_source_mapping.csv",
        "split": protocol_root / "canonical_split_manifest_seed2026.csv",
        "runtime": protocol_root / "h100_canonical_runtime_index.csv",
        "runtime_summary": protocol_root / "h100_canonical_runtime_index_summary.json",
        "class_mapping": protocol_root / "class_mapping.csv",
        "matrix": protocol_root / "observed_multilabel_matrix.npy",
    }
    for key, expected in task_config["canonical_asset_sha256"].items():
        _verify_expected_sha256(paths[key], str(expected))
    runtime_summary = json.loads(paths["runtime_summary"].read_text(encoding="utf-8"))
    if not runtime_summary.get("ready") or int(
        runtime_summary.get("mapped_canonical_rows", 0)
    ) != int(task_config["expected_samples"]):
        raise ValueError("H100 原图路径解析器未完整映射 canonical cohort")

    samples = pd.read_csv(paths["samples"])
    split = pd.read_csv(paths["split"])
    runtime = pd.read_csv(paths["runtime"])
    class_mapping = pd.read_csv(paths["class_mapping"])
    matrix = np.load(paths["matrix"], allow_pickle=False)
    n_classes = int(task_config["num_classes"])
    expected_samples = int(task_config["expected_samples"])
    if (
        len(samples) != expected_samples
        or len(split) != expected_samples
        or len(runtime) != expected_samples
        or matrix.shape != (expected_samples, n_classes)
        or class_mapping["class_id"].astype(int).tolist() != list(range(n_classes))
    ):
        raise ValueError("canonical 样本、标签矩阵、类别顺序或原图映射数量不一致")
    merged = (
        samples.merge(
            split[
                [
                    "canonical_index",
                    "canonical_id",
                    "near_duplicate_group",
                    "split",
                    "cv_fold",
                ]
            ],
            on=["canonical_index", "canonical_id"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            runtime[["canonical_id", "h100_relative_path"]],
            on="canonical_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("canonical_index")
        .reset_index(drop=True)
    )
    if len(merged) != expected_samples or merged["canonical_id"].duplicated().any():
        raise ValueError("canonical manifest 合并不完整")
    matrix_ids = [
        tuple(np.flatnonzero(matrix[index]).astype(int).tolist())
        for index in range(expected_samples)
    ]
    manifest_ids = [
        tuple(int(item) for item in re.split(r"[;|]", str(value)))
        for value in merged["observed_label_ids"]
    ]
    if matrix_ids != manifest_ids:
        raise ValueError("59 维观测标签矩阵与 canonical manifest 不一致")
    data_root = Path(task_config["data_root"])
    missing_images = [
        path
        for path in merged["h100_relative_path"].map(data_root.__truediv__)
        if not path.is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"canonical cohort 缺少 {len(missing_images)} 张原图")

    merged["image_key"] = merged["canonical_id"].astype(str)
    merged["relative_path"] = merged["h100_relative_path"].astype(str)
    merged["patient_id"] = ""
    merged["label"] = -1
    single = merged["label_count"].astype(int).eq(1)
    merged.loc[single, "label"] = merged.loc[single, "observed_label_ids"].astype(int)
    validation_fold = int(task_config["internal_validation_fold"])
    development = merged[merged["split"].eq("development")].copy()
    frozen_test = merged[merged["split"].eq("test")].copy()
    frames = {
        "development_observed": development.reset_index(drop=True),
        "test_observed": frozen_test.reset_index(drop=True),
        "train": development[
            single.loc[development.index]
            & development["cv_fold"].astype(int).ne(validation_fold)
        ].reset_index(drop=True),
        "val": development[
            single.loc[development.index]
            & development["cv_fold"].astype(int).eq(validation_fold)
        ].reset_index(drop=True),
        "test": frozen_test[single.loc[frozen_test.index]].reset_index(drop=True),
    }
    expected = task_config["expected_split_counts"]
    observed_counts = {name: len(frame) for name, frame in frames.items()}
    if observed_counts != {key: int(value) for key, value in expected.items()}:
        raise ValueError(
            f"TRHD59 冻结划分数量不一致：{observed_counts} != {expected}"
        )
    if task_config.get("patient_level_isolation") != "unverified":
        raise ValueError("TRHD59 当前不得声称患者级隔离已验证")
    canonical = json.dumps(
        [
            {
                "canonical_id": str(row.canonical_id),
                "split": str(row.split),
                "cv_fold": None if pd.isna(row.cv_fold) else int(row.cv_fold),
                "observed_label_ids": str(row.observed_label_ids),
            }
            for row in merged.itertuples(index=False)
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    split_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    output_root = Path(task_config["manifest_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        target = output_root / f"{name}_manifest.csv"
        if target.exists():
            existing = pd.read_csv(target)
            if existing["canonical_id"].astype(str).tolist() != frame[
                "canonical_id"
            ].astype(str).tolist():
                raise ValueError(f"已冻结 {name} manifest 与 canonical 协议不一致")
        else:
            temporary = target.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(target)
    contract = {
        "task_id": task_config["task_id"],
        "task_semantics": "observed_positive_multiclass",
        "class_order": list(range(n_classes)),
        "split_id": split_id,
        "counts": observed_counts,
        "unobserved_classes_treated_as_negative": False,
        "single_observed_label_comparison_is_weak": True,
        "primary_metric": "macro_f1",
        "qwk_status": "not_applicable_nominal_task",
        "patient_level_isolation": "unverified",
        "test_used_for_selection": False,
        "test_metrics_before_route_freeze": "forbidden",
    }
    contract_path = output_root / "task_contract_resolved.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("已冻结 TRHD59 任务契约与当前 canonical 协议不一致")
    else:
        temporary = contract_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(contract_path)
    build_cfp_compatibility_gate(task_config)
    return frames, split_id


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
    if task_config.get("task_semantics") == "observed_positive_multiclass":
        return prepare_observed_label_manifests(task_config)
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
    for index in range(probabilities.shape[1]):
        output[f"prob_{index}"] = probabilities[:, index]
    output["model_id"] = spec["model_id"]
    output["checkpoint_sha256"] = sha256
    output["preprocessing_id"] = spec["preprocessing_id"]
    output["split_id"] = split_id
    output["inference_run_id"] = run_id
    output["inference_dtype"] = "float32"
    return output


def _observed_prediction_frame(
    manifest: pd.DataFrame,
    probabilities: np.ndarray,
    spec: dict,
    split: str,
    run_id: str,
    checkpoint_sha256: str,
    split_id: str,
) -> pd.DataFrame:
    ordered = manifest.reset_index(drop=True)
    output = pd.DataFrame(
        {
            "case_id": ordered["canonical_id"].astype(str),
            "split": split,
            "observed_label_ids": ordered["observed_label_ids"].astype(str),
            "observed_label_names": ordered["observed_label_names"].astype(str),
            "observed_label_count": ordered["label_count"].astype(int),
            "y_pred": probabilities.argmax(axis=1),
        }
    )
    for index in range(probabilities.shape[1]):
        output[f"prob_{index}"] = probabilities[:, index]
    output["model_id"] = spec["model_id"]
    output["checkpoint_sha256"] = checkpoint_sha256
    output["preprocessing_id"] = spec["preprocessing_id"]
    output["split_id"] = split_id
    output["inference_run_id"] = run_id
    output["inference_dtype"] = "float32"
    return output


def _select_features(
    all_features: np.ndarray,
    all_keys: list[str],
    manifest: pd.DataFrame,
) -> np.ndarray:
    key_to_index = {str(key): index for index, key in enumerate(all_keys)}
    indices = [key_to_index[str(key)] for key in manifest["image_key"]]
    return all_features[np.asarray(indices, dtype=int)]


def _nominal_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    metrics = classification_metrics(labels, probabilities)
    metrics.pop("quadratic_kappa", None)
    metrics["qwk_status"] = "not_applicable_nominal_task"
    return metrics


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


def _run_observed_positive_task(
    *,
    output: Path,
    task_config: dict,
    spec: dict,
    manifests: dict[str, pd.DataFrame],
    split_id: str,
    checkpoint_sha256: str,
    run_id: str,
    transform: Callable,
    forward: Callable,
    device: str,
) -> Path:
    data_root = Path(task_config["data_root"])
    development_dataset = ManifestDataset(
        manifests["development_observed"],
        transform,
        data_root,
    )
    test_dataset = ManifestDataset(
        manifests["test_observed"],
        transform,
        data_root,
    )
    development_x, _, development_keys = _features(
        forward,
        development_dataset,
        int(task_config.get("feature_batch_size", 32)),
        device,
    )
    test_x, _, test_keys = _features(
        forward,
        test_dataset,
        int(task_config.get("feature_batch_size", 32)),
        device,
    )
    train_x = _select_features(development_x, development_keys, manifests["train"])
    val_x = _select_features(development_x, development_keys, manifests["val"])
    weak_test_x = _select_features(test_x, test_keys, manifests["test"])
    train_y = manifests["train"]["label"].astype(int).to_numpy()
    val_y = manifests["val"]["label"].astype(int).to_numpy()

    choices = []
    c_candidates = task_config.get("c_candidates") or spec["c_candidates"]
    for c_value in c_candidates:
        probe = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=int(task_config.get("probe_max_iter", 2000)),
            random_state=int(task_config["random_seed"]),
        ).fit(train_x, train_y)
        probabilities = probe.predict_proba(val_x)
        choices.append(
            (
                _nominal_metrics(val_y, probabilities)["macro_f1"],
                float(c_value),
                probe,
            )
        )
    _, selected_c, probe = max(choices, key=lambda item: item[0])
    class_order = np.asarray(task_config["class_order"])
    if not np.array_equal(probe.classes_, class_order):
        raise ValueError(
            f"探针类别顺序 {probe.classes_.tolist()} 与任务协议 "
            f"{class_order.tolist()} 不一致"
        )
    validation_probabilities = probe.predict_proba(val_x)
    test_probabilities = probe.predict_proba(weak_test_x)
    development_probabilities = probe.predict_proba(development_x)
    full_test_probabilities = probe.predict_proba(test_x)

    prediction_root = output / "predictions"
    prediction_root.mkdir()
    standard_predictions = {
        "validation_predictions.csv": _prediction_frame(
            manifests["val"],
            validation_probabilities,
            spec,
            "validation",
            run_id,
            checkpoint_sha256,
            split_id,
        ),
        "test_predictions.csv": _prediction_frame(
            manifests["test"],
            test_probabilities,
            spec,
            "test",
            run_id,
            checkpoint_sha256,
            split_id,
        ),
    }
    observed_predictions = {
        "development_observed_predictions.csv": _observed_prediction_frame(
            manifests["development_observed"],
            development_probabilities,
            spec,
            "development",
            run_id,
            checkpoint_sha256,
            split_id,
        ),
        "test_observed_predictions.csv": _observed_prediction_frame(
            manifests["test_observed"],
            full_test_probabilities,
            spec,
            "test",
            run_id,
            checkpoint_sha256,
            split_id,
        ),
    }
    for filename, frame in {**standard_predictions, **observed_predictions}.items():
        frame.to_csv(prediction_root / filename, index=False)

    metrics_root = output / "metrics"
    metrics_root.mkdir()
    weak_metrics = {
        "evaluation_design": "private_observed_label_task_validation",
        "selection_split": "development_internal_validation",
        "frozen_evaluation_split": "test",
        "selected_c": selected_c,
        "test_used_for_selection": False,
        "single_observed_label_comparison_is_weak": True,
        "validation": _nominal_metrics(
            val_y,
            validation_probabilities,
        ),
        "test_metrics_status": "sealed_not_computed_before_route_freeze",
    }
    probability_columns = [
        f"prob_{index}" for index in range(int(task_config["num_classes"]))
    ]
    observed_audits = {}
    development_audit = run_observed_positive_audit(
        observed_predictions["development_observed_predictions.csv"],
        probability_columns=probability_columns,
        high_confidence_threshold=float(task_config["high_confidence_threshold"]),
    )
    observed_audits["development"] = development_audit.summary
    observed_audits["test"] = {
        "status": "sealed_not_computed_before_route_freeze",
        "n_cases": int(len(observed_predictions["test_observed_predictions.csv"])),
    }
    (metrics_root / "metrics.json").write_text(
        json.dumps(
            {
                **weak_metrics,
                "observed_positive_audit": observed_audits,
                "unobserved_classes_treated_as_negative": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    cost_root = output / "costs"
    cost_root.mkdir()
    (cost_root / "forward_cost.json").write_text(
        json.dumps(
            _cost(forward, development_dataset, device),
            indent=2,
        ),
        encoding="utf-8",
    )
    probe_path = output / "probe.joblib"
    joblib.dump(
        {
            "probe": probe,
            "class_order": list(task_config["class_order"]),
            "selected_c": selected_c,
            "split_id": split_id,
            "label_semantics": "weak_single_observed_label",
        },
        probe_path,
    )
    config = {
        "task_id": task_config["task_id"],
        "dataset_id": task_config["dataset_id"],
        "model": spec["model_id"],
        "adapter_type": spec["adapter_type"],
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_id": spec["preprocessing_id"],
        "manifest_sha256": split_id,
        "task_semantics": "observed_positive_multiclass",
        "selection": {
            "split": "development_internal_validation",
            "c_candidates": c_candidates,
            "selected_c": selected_c,
        },
        "unobserved_classes_treated_as_negative": False,
        "frozen_test_used_for_selection": False,
        "patient_level_isolation": "unverified",
        "route_eligible": False,
    }
    (output / "config_resolved.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "task_adapted": True,
                "task_inference_ready": False,
                "offline_evaluation_eligible": True,
                "validation_selection_eligible": True,
                "route_eligible": False,
                "feature_dim": int(development_x.shape[1]),
                "metrics": weak_metrics,
                "observed_positive_audit": observed_audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.csv":
            continue
        artifact_rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": eyeclip_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(artifact_rows).to_csv(
        output / "artifact_manifest.csv",
        index=False,
    )
    return output


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
        raise RuntimeError("16 张真实任务图像重复前向不一致")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(task_config["output_root"]) / model_name / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "smoke.json").write_text(json.dumps({"status": "passed", "samples": 16, "feature_dim": int(smoke_x.shape[1]), "checkpoint_sha256": checkpoint_sha256}, indent=2), encoding="utf-8")
    if smoke_only:
        return output
    if task_config.get("task_semantics") == "observed_positive_multiclass":
        return _run_observed_positive_task(
            output=output,
            task_config=task_config,
            spec=spec,
            manifests=manifests,
            split_id=split_id,
            checkpoint_sha256=checkpoint_sha256,
            run_id=run_id,
            transform=transform,
            forward=forward,
            device=device,
        )
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
        _, digest = _manifests(task_config)
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
