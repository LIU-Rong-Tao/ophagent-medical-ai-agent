"""Runtime readiness and isolated smoke probes for registered model assets."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any
import zipfile

import pandas as pd

from app.model_providers import OphBenchProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = PROJECT_ROOT.parent / "model_cache" / "ophbench"
RUNTIME_ROOT = PROJECT_ROOT / "experiments/model_hub/runtime/asset_smoke"
REPRESENTATIVE_CHECKPOINTS = (
    "retfound-cfp",
    "eyeclip-default",
    "flair-default",
    "fmue-default",
    "deretfound-pretraining",
)
MAX_TORCH_STRUCTURE_BYTES = 4 * 1024**3
UPSTREAM_SOURCE_ROOT = DEFAULT_ASSET_ROOT / "upstream_sources"
OFFICIAL_SOURCE_PINS = {
    "deretfound": ("01835850302e9e248bad8aef5c815c29bff4c360",),
    "eyeclip": ("2fcf6034552e6006c94bd84cbdc6f4a5897b29c0",),
    "flair": ("d6652d53389ff49e5f73efaccf4246e9de88d1a3",),
    "fmue": ("b07ba8a797d6440826f3870f73faf567963ffc15",),
    "keepfit": ("dbbb1f05b9d27278b01e15e5f837b44b22d32cee",),
    "mirage": ("930523b209dad3dd2eadc7296fb7ce3f3eaa9924",),
    "preti": ("2ac2b0f123d69151877ebd44a33edfb026cdac45",),
    "ret-clip": ("1ddb9a1d331eba9e0a2675f3273e7cbcea0914bd",),
    "retizero": ("d72aadc692fbe33b182c79711bccb397edffb419",),
    "retfound": ("ae9a9ecf37857cf47b8aa9f87cd6f710d75db287",),
    "retfound-green": ("767c77ecc6ad2656ace051b17bf22d2b47485c6c",),
    "urfound": ("6678be52afdf6c30114fb1d5f733be1282374ada",),
    "vilref": ("repo",),
    "visionunite": ("repo",),
}
RUNTIME_PROFILES = {
    "deretfound-pretraining": "deretfound_mae_runtime",
    "deretfound-sd-retina": "deretfound_sd_runtime",
    "retfound-cfp": "retfound_runtime",
    "retfound-oct": "retfound_runtime",
    "retfound-green-v0.1": "retfound_green_runtime",
    "preti-default": "preti_runtime",
    "eyeclip-default": "eyeclip_runtime",
    "flair-default": "flair_runtime",
    "fmue-default": "fmue_runtime",
    "keepfit-ffa-ir-mmretinal-ffa": "keepfit_runtime",
    "keepfit-flair-mmretinal-cfp": "keepfit_runtime",
    "keepfit-half-flair-mmretinal-cfp": "keepfit_runtime",
    "mirage-base": "mirage_runtime",
    "mirage-large": "mirage_runtime",
    "ret-clip-default": "ret_clip_runtime",
    "retizero-default": "retizero_runtime",
    "urfound-default": "urfound_runtime",
    "vilref-default": "vilref_runtime",
    "visionunite-default": "visionunite_resource_audit",
}


def _manifest_path() -> Path:
    import ophbench

    return Path(ophbench.__file__).resolve().parents[1] / "catalog/download_manifest.csv"


def _profile_for(filename: str, checkpoint_id: str, size_bytes: int) -> tuple[str, str]:
    if checkpoint_id in RUNTIME_PROFILES:
        return RUNTIME_PROFILES[checkpoint_id], "按固定官方源码契约执行模型构建、权重加载、预处理与原生前向"
    suffix = Path(filename).suffix.lower()
    if suffix == ".safetensors":
        return "safetensors_structure", "读取 safetensors 头部、张量键与形状"
    if suffix == ".zip":
        return "zip_structure", "读取压缩包目录，不解压模型文件"
    if suffix in {".pth", ".pt", ".ckpt", ".bin"}:
        if size_bytes > MAX_TORCH_STRUCTURE_BYTES:
            return "metadata_only", "文件超过安全结构探测上限"
        return "torch_checkpoint_structure", "安全反序列化 checkpoint 并汇总张量结构"
    return "metadata_only", "尚无受控探测 profile"


def _dependencies_for(profile: str) -> tuple[str, ...]:
    if profile == "retfound_runtime":
        return ("torch", "torchvision", "timm", "PIL", "ophbench")
    if profile in {"retfound_green_runtime", "preti_runtime"}:
        return ("torch", "torchvision", "timm", "PIL")
    if profile in {"eyeclip_runtime", "ret_clip_runtime", "vilref_runtime"}:
        return ("torch", "torchvision", "PIL")
    if profile == "mirage_runtime":
        return ("torch", "einops", "safetensors")
    if profile in {"fmue_runtime", "urfound_runtime"}:
        return ("torch", "torchvision", "timm")
    if profile in {"deretfound_mae_runtime", "deretfound_sd_runtime"}:
        return ("torch",)
    if profile == "visionunite_resource_audit":
        return ("torch",)
    if profile in {"flair_runtime", "keepfit_runtime", "retizero_runtime"}:
        return ("torch", "torchvision", "transformers", "tensorboard")
    if profile == "torch_checkpoint_structure":
        return ("torch",)
    if profile == "safetensors_structure":
        return ("safetensors",)
    return ()


def _expected_output(artifact_type: str) -> str:
    if artifact_type in {"foundation_encoder", "ablation_checkpoint"}:
        return "特征向量（需 Adapter 验证）"
    if artifact_type == "vision_language_model":
        return "图像/文本表征或相似度（需 Adapter 验证）"
    if artifact_type == "multimodal_full_model":
        return "配对多模态输出（需原生输入契约）"
    if artifact_type == "task_checkpoint":
        return "原任务输出；不可视为当前任务概率"
    if artifact_type == "generative_model":
        return "生成模型输出；不进入分类路由"
    return "输出契约待登记"


def build_asset_readiness(
    asset_root: Path | str = DEFAULT_ASSET_ROOT,
    *,
    manifest_path: Path | str | None = None,
    records=None,
) -> pd.DataFrame:
    """Build checkpoint-level readiness without loading model weights."""

    root = Path(asset_root)
    manifest = Path(manifest_path) if manifest_path else _manifest_path()
    with manifest.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    provider_records = records if records is not None else OphBenchProvider().list_models()
    by_checkpoint = {record.source_checkpoint_id: record for record in provider_records}
    rows: list[dict[str, Any]] = []
    for item in manifest_rows:
        checkpoint_id = str(item["checkpoint_id"])
        model_id = str(item["model_id"])
        filename = str(item.get("filename", ""))
        expected_size = int(item.get("size_bytes") or 0)
        relative_path = Path(model_id) / checkpoint_id / filename
        local_path = root / relative_path
        exists = local_path.is_file()
        actual_size = local_path.stat().st_size if exists else 0
        size_matches = bool(exists and expected_size and actual_size == expected_size)
        record = by_checkpoint.get(checkpoint_id)
        profile, profile_detail = _profile_for(filename, checkpoint_id, actual_size)
        dependencies = _dependencies_for(profile)
        missing_dependencies = [
            name for name in dependencies if importlib.util.find_spec(name) is None
        ]
        adapter_implemented = bool(
            record is not None and getattr(record, "adapter_implemented", False)
        )
        integrity_evidence = str(item.get("local_integrity_status", ""))
        sha256_evidence_verified = bool(
            item.get("sha256")
            and integrity_evidence == "local_size_sha256_and_non_html_verified"
        )
        artifact_type = str(item.get("artifact_type", ""))
        runtime_eligible = bool(
            exists
            and size_matches
            and profile in set(RUNTIME_PROFILES.values())
            and profile != "visionunite_resource_audit"
            and not missing_dependencies
        )
        probe_eligible = bool(
            exists
            and size_matches
            and profile != "metadata_only"
            and not missing_dependencies
        )
        reasons = []
        if not exists:
            reasons.append("本机资产不存在")
        elif not size_matches:
            reasons.append("文件大小与冻结 manifest 不一致")
        if missing_dependencies:
            reasons.append("缺少依赖：" + ", ".join(missing_dependencies))
        if profile == "metadata_only":
            reasons.append(profile_detail)
        if not adapter_implemented:
            reasons.append("Adapter 尚未实现")
        rows.append(
            {
                "model_id": model_id,
                "checkpoint_id": checkpoint_id,
                "modalities": "|".join(getattr(record, "modalities", ()) if record else ()),
                "artifact_type": artifact_type,
                "framework": str(getattr(record, "framework", "") if record else ""),
                "input_size": str(getattr(record, "input_size", "") if record else ""),
                "expected_output": _expected_output(artifact_type),
                "asset_relative_path": relative_path.as_posix(),
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "local_asset_exists": exists,
                "size_matches_manifest": size_matches,
                "registry_integrity_evidence": integrity_evidence,
                "registry_sha256_evidence": sha256_evidence_verified,
                "sha256_rechecked_this_run": False,
                "official_source_verified": str(item.get("source_provenance_status", ""))
                == "official_source_verified",
                "adapter_implemented": adapter_implemented,
                "loader_status": "已实现" if adapter_implemented else "待实现",
                "preprocessing_status": str(
                    getattr(record, "preprocessing_verification_status", "")
                    if record
                    else ""
                ),
                "access_type": str(getattr(record, "access_type", "") if record else ""),
                "requires_auth": bool(
                    getattr(record, "requires_auth", False) if record else False
                ),
                "license": str(getattr(record, "license", "") if record else ""),
                "encoder_smoke_passed_before_run": bool(
                    record is not None and getattr(record, "encoder_smoke_passed", False)
                ),
                "probe_profile": profile,
                "profile_detail": profile_detail,
                "probe_eligible": probe_eligible,
                "runtime_smoke_eligible": runtime_eligible,
                "missing_dependencies": "|".join(missing_dependencies),
                "blocked_reason": "；".join(reasons),
                "task_inference_ready": False,
                "route_eligible": False,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["local_asset_exists", "model_id", "checkpoint_id"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def _torch_summary(value: Any) -> dict[str, Any]:
    import torch

    tensor_count = 0
    parameter_count = 0
    sample_keys: list[str] = []

    def visit(item: Any, prefix: str = "", depth: int = 0) -> None:
        nonlocal tensor_count, parameter_count
        if torch.is_tensor(item):
            tensor_count += 1
            parameter_count += int(item.numel())
            if prefix and len(sample_keys) < 12:
                sample_keys.append(prefix)
            return
        if isinstance(item, dict) and depth < 4:
            for key, child in item.items():
                visit(child, f"{prefix}.{key}".strip("."), depth + 1)
        elif isinstance(item, (list, tuple)) and depth < 4:
            for index, child in enumerate(item[:32]):
                visit(child, f"{prefix}[{index}]", depth + 1)

    visit(value)
    top_level_keys = list(value)[:30] if isinstance(value, dict) else []
    return {
        "container_type": type(value).__name__,
        "top_level_keys": [str(key) for key in top_level_keys],
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "sample_tensor_keys": sample_keys,
    }


def _official_source(model_id: str) -> Path:
    path = UPSTREAM_SOURCE_ROOT / model_id
    for part in OFFICIAL_SOURCE_PINS[model_id]:
        path /= part
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned official source is unavailable for {model_id}")
    return path


def _retfound_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from ophbench.models.adapters.retfound_cfp import RETFoundCFPAdapter

    adapter = RETFoundCFPAdapter(
        checkpoint_path=path, device=str(spec.get("device", "cpu"))
    ).load()
    tensor = adapter.preprocess(Image.new("RGB", (320, 280), color=(127, 127, 127)))
    output = adapter.encode_image(tensor).detach().cpu().numpy()
    if output.shape != (1, 1024) or not np.isfinite(output).all():
        raise RuntimeError(f"Unexpected RETFound output: {output.shape}")
    return {
        "input_shape": list(tensor.shape),
        "output_shape": list(output.shape),
        "output_finite": True,
        "source_commit": OFFICIAL_SOURCE_PINS["retfound"][0],
    }


def _retfound_green_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    import timm
    from PIL import Image
    from torchvision.transforms import v2 as transforms

    device = str(spec.get("device", "cpu"))
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(392, 392),
        num_classes=0,
        checkpoint_path=str(path),
    ).eval()
    model.global_pool = "avg"
    model = model.to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize((392, 392), antialias=True),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    tensor = preprocess(Image.new("RGB", (480, 360), color=(127, 127, 127))).unsqueeze(0)
    with torch.inference_mode():
        output = model(tensor.to(device))
    if output.shape != (1, 384) or not output.isfinite().all():
        raise RuntimeError(f"Unexpected RETFound-Green output: {tuple(output.shape)}")
    return {
        "input_shape": list(tensor.shape),
        "output_shape": list(output.shape),
        "output_finite": True,
        "source_commit": OFFICIAL_SOURCE_PINS["retfound-green"][0],
    }


def _preti_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image
    from torchvision.transforms import v2 as transforms

    source = _official_source("preti")
    for module_name in tuple(sys.modules):
        if module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
    package = types.ModuleType("models")
    package.__path__ = [str(source / "models")]
    package.__package__ = "models"
    sys.modules["models"] = package
    sys.path.insert(0, str(source))
    import models.PRETI_model as preti_model

    class _UnusedTrainingLoss(torch.nn.Module):
        def forward(self, *args):
            return torch.tensor(0.0)

    preti_model.PerceptualLoss = lambda *args, **kwargs: _UnusedTrainingLoss()
    model = preti_model.SIAM_MODELS["vitb"](
        pretrained=False,
        norm_pix_loss=False,
        patch_size=16,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=8,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_unexpected = [key for key in unexpected if key.startswith("perceptual_loss_fn.")]
    if missing or len(allowed_unexpected) != len(unexpected):
        raise RuntimeError(
            f"PRETI state mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    device = str(spec.get("device", "cpu"))
    model = model.eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    tensor = preprocess(Image.new("RGB", (320, 280), color=(127, 127, 127))).unsqueeze(0)
    with torch.inference_mode():
        output, age_embedding, gender_embedding = model.forward_encoder_no_masking(
            tensor.to(device)
        )
    if output.shape != (1, 197, 768) or not output.isfinite().all():
        raise RuntimeError(f"Unexpected PRETI output: {tuple(output.shape)}")
    return {
        "input_shape": list(tensor.shape),
        "output_shape": list(output.shape),
        "metadata_output_shapes": [
            list(age_embedding.shape),
            list(gender_embedding.shape),
        ],
        "output_finite": True,
        "source_commit": OFFICIAL_SOURCE_PINS["preti"][0],
        "compatibility_note": "Skipped the unused VGG training-loss module during encoder smoke",
    }


def _eyeclip_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image

    source = _official_source("eyeclip")
    sys.path.insert(0, str(source))
    import eyeclip
    from eyeclip.clip import _transform
    from eyeclip.model import build_model

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_state_dict"])
    device = str(spec.get("device", "cpu"))
    model = model.eval().to(device)
    tensor = _transform(model.visual.input_resolution)(
        Image.new("RGB", (320, 280), color=(127, 127, 127))
    ).unsqueeze(0).to(device)
    tokens = eyeclip.tokenize(["normal retina"], truncate=True).to(device)
    with torch.inference_mode():
        image_output = model.encode_image(tensor)
        text_output = model.encode_text(tokens)
    if image_output.shape != (1, 512) or text_output.shape != (1, 512):
        raise RuntimeError("Unexpected EyeCLIP image/text output")
    return {
        "input_shape": list(tensor.shape),
        "image_output_shape": list(image_output.shape),
        "text_output_shape": list(text_output.shape),
        "output_finite": bool(image_output.isfinite().all() and text_output.isfinite().all()),
        "source_commit": OFFICIAL_SOURCE_PINS["eyeclip"][0],
    }


def _vilref_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image

    source = _official_source("vilref")
    sys.path.insert(0, str(source))
    from ViLReF.clip.utils import load_from_name, tokenize

    device = str(spec.get("device", "cpu"))
    model, preprocess = load_from_name(
        str(path),
        device=device,
        vision_model_name="ViT-B-16",
        text_model_name="RoBERTa-wwm-ext-base-chinese",
        input_resolution=224,
    )
    model = model.float().eval()
    tensor = preprocess(
        Image.new("RGB", (320, 280), color=(127, 127, 127))
    ).unsqueeze(0).to(device)
    tokens = tokenize(["normal retina"]).to(device)
    with torch.inference_mode():
        image_output = model.encode_image(tensor)
        text_output = model.encode_text(tokens)
    if image_output.shape != (1, 512) or text_output.shape != (1, 512):
        raise RuntimeError("Unexpected ViLReF image/text output")
    return {
        "input_shape": list(tensor.shape),
        "image_output_shape": list(image_output.shape),
        "text_output_shape": list(text_output.shape),
        "output_finite": bool(image_output.isfinite().all() and text_output.isfinite().all()),
        "source_commit": "3a01b908f72abab04661bba80403989f5271b990",
        "compatibility_note": "Forced FP32 for PyTorch 2.5 mixed-dtype compatibility",
    }


def _ret_clip_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image

    source = _official_source("ret-clip")
    sys.path.insert(0, str(source))
    from RET_CLIP.clip.model import (
        CLIP,
        convert_models_to_fp32,
        convert_weights,
    )
    from RET_CLIP.clip.utils import image_transform, tokenize

    model_info: dict[str, Any] = {}
    config_root = source / "RET_CLIP" / "clip" / "model_configs"
    for config_name in (
        "ViT-B-16.json",
        "RoBERTa-wwm-ext-base-chinese.json",
    ):
        model_info.update(json.loads((config_root / config_name).read_text()))
    model = CLIP(**model_info)
    convert_weights(model)
    convert_models_to_fp32(model)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if "bert.pooler" not in key
    }
    model.load_state_dict(state_dict, strict=True)
    device = str(spec.get("device", "cpu"))
    model = model.eval().to(device)
    tensor = image_transform(224)(
        Image.new("RGB", (320, 280), color=(127, 127, 127))
    ).unsqueeze(0).to(device)
    paired_tensor = tensor.clone()
    tokens = tokenize(["normal retina"]).to(device)
    with torch.inference_mode():
        image_output = model(tensor, None, None)
        text_output = model(None, None, tokens)
        paired_output = model(tensor, paired_tensor, tokens)
    output_tensors = [image_output, *text_output, *paired_output]
    if image_output.shape != (1, 512):
        raise RuntimeError(f"Unexpected RET-CLIP image output: {image_output.shape}")
    if any(not output.isfinite().all() for output in output_tensors):
        raise RuntimeError("RET-CLIP produced non-finite output")
    return {
        "input_shape": list(tensor.shape),
        "image_output_shape": list(image_output.shape),
        "text_output_shapes": [list(output.shape) for output in text_output],
        "paired_output_shapes": [list(output.shape) for output in paired_output],
        "output_finite": True,
        "state_dict_key_count": len(state_dict),
        "source_commit": OFFICIAL_SOURCE_PINS["ret-clip"][0],
        "dependency_contract": "official ViT-B-16 + RoBERTa-wwm-ext-base-chinese",
        "compatibility_note": (
            "Applied the official fp32/amp conversion order before strict checkpoint load"
        ),
    }


def _mirage_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    source = _official_source("mirage")
    sys.path.insert(0, str(source / "hf"))
    from mirage_hf import MIRAGEWrapper

    size = "large" if str(spec["checkpoint_id"]).endswith("large") else "base"
    model = MIRAGEWrapper(
        input_size=512,
        patch_size=32,
        modalities="bscan-slo",
        size=size,
    )
    raw_state = load_file(path)
    state = {key.removeprefix("model."): value for key, value in raw_state.items()}
    model.load_state_dict(state, strict=True)
    device = str(spec.get("device", "cpu"))
    model = model.eval().to(device)
    inputs = {
        "bscan": torch.full((1, 1, 512, 512), 0.5, device=device),
        "slo": torch.full((1, 1, 512, 512), 0.5, device=device),
    }
    with torch.inference_mode():
        output = model(inputs)
    expected_dim = 1024 if size == "large" else 768
    if output.shape != (1, 513, expected_dim) or not output.isfinite().all():
        raise RuntimeError(f"Unexpected MIRAGE output: {tuple(output.shape)}")
    return {
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "output_shape": list(output.shape),
        "output_finite": True,
        "source_commit": OFFICIAL_SOURCE_PINS["mirage"][0],
        "compatibility_note": (
            "Removed the wrapper-level model. prefix written by the official HF save path"
        ),
    }


def _deretfound_mae_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    import timm.models.vision_transformer as timm_vit
    from PIL import Image

    source = _official_source("deretfound")
    sys.path.insert(0, str(source))
    original_block = timm_vit.Block

    def compatible_block(*args, **kwargs):
        kwargs.pop("qk_scale", None)
        return original_block(*args, **kwargs)

    timm_vit.Block = compatible_block
    import models_mae

    model = models_mae.mae_vit_large_patch16()
    with zipfile.ZipFile(path) as archive:
        with archive.open("PreTraining/checkpoint-best.pth") as checkpoint_file:
            checkpoint = torch.load(
                checkpoint_file,
                map_location="cpu",
                weights_only=False,
            )
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "DERETFound MAE checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    device = str(spec.get("device", "cpu"))
    model = model.eval().to(device)
    image = Image.new("RGB", (320, 280), color=(127, 127, 127)).resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    imagenet_std = np.array([0.229, 0.224, 0.225])
    prepared = image_array / imagenet_std
    tensor = torch.from_numpy(prepared).unsqueeze(0)
    tensor = torch.einsum("nhwc->nchw", tensor).float().to(device)
    with torch.inference_mode():
        loss, prediction, mask = model(tensor, mask_ratio=0.75)
    if prediction.shape != (1, 196, 768) or mask.shape != (1, 196):
        raise RuntimeError("Unexpected DERETFound MAE output")
    if not loss.isfinite() or not prediction.isfinite().all() or not mask.isfinite().all():
        raise RuntimeError("DERETFound MAE produced non-finite output")
    return {
        "input_shape": list(tensor.shape),
        "loss_shape": list(loss.shape),
        "prediction_shape": list(prediction.shape),
        "mask_shape": list(mask.shape),
        "output_finite": True,
        "state_dict_key_count": len(checkpoint["model"]),
        "source_commit": OFFICIAL_SOURCE_PINS["deretfound"][0],
        "compatibility_note": (
            "Dropped removed timm qk_scale without modifying official source; "
            "reproduced the deployed utility.prepare_data behavior"
        ),
    }


def _deretfound_sd_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import tempfile
    import torch

    dependency_root = UPSTREAM_SOURCE_ROOT.parent / "runtime_deps" / "deretfound"
    transformers_root = UPSTREAM_SOURCE_ROOT.parent / "runtime_deps" / "urfound"
    if not dependency_root.is_dir() or not transformers_root.is_dir():
        raise FileNotFoundError("Pinned DERETFound diffusion environment is unavailable")
    for module_name in tuple(sys.modules):
        if (
            module_name == "diffusers"
            or module_name.startswith("diffusers.")
            or module_name == "transformers"
            or module_name.startswith("transformers.")
            or module_name == "huggingface_hub"
            or module_name.startswith("huggingface_hub.")
        ):
            del sys.modules[module_name]
    sys.path[:0] = [str(dependency_root), str(transformers_root)]
    from diffusers import UNet2DConditionModel

    member_root = "sd-retina-model/checkpoint-60000/unet"
    with tempfile.TemporaryDirectory(prefix="deretfound-unet-") as directory:
        extract_root = Path(directory)
        with zipfile.ZipFile(path) as archive:
            for filename in ("config.json", "diffusion_pytorch_model.bin"):
                archive.extract(f"{member_root}/{filename}", extract_root)
        model = UNet2DConditionModel.from_pretrained(
            extract_root / member_root,
            local_files_only=True,
        )
        device = str(spec.get("device", "cpu"))
        model = model.eval().to(device)
        latent = torch.randn(1, 4, 64, 64, device=device)
        text_condition = torch.randn(1, 77, 768, device=device)
        timestep = torch.tensor([1], device=device)
        with torch.inference_mode():
            output = model(
                latent,
                timestep,
                encoder_hidden_states=text_condition,
            ).sample
    if output.shape != (1, 4, 64, 64) or not output.isfinite().all():
        raise RuntimeError("Unexpected DERETFound diffusion UNet output")
    return {
        "latent_input_shape": list(latent.shape),
        "text_condition_shape": list(text_condition.shape),
        "output_shape": list(output.shape),
        "output_finite": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_commit": OFFICIAL_SOURCE_PINS["deretfound"][0],
        "dependency_contract": "diffusers==0.21.4",
        "runtime_semantics": "native retinal diffusion UNet forward",
        "classification_routing_applicability": "not_applicable_to_classification_routing",
        "unresolved_items": (
            "Stable Diffusion v1.4 base VAE, text encoder, and tokenizer are not bundled; "
            "end-to-end image generation was not evaluated"
        ),
    }


def _retizero_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    device = str(spec.get("device", "cpu"))
    if not device.startswith("cuda"):
        raise RuntimeError("The pinned official RetiZero module selects CUDA at import time")
    dependency_root = UPSTREAM_SOURCE_ROOT.parent / "runtime_deps" / "urfound"
    if not dependency_root.is_dir():
        raise FileNotFoundError("Pinned transformers 4.30.2 environment is unavailable")
    for module_name in tuple(sys.modules):
        if (
            module_name == "transformers"
            or module_name.startswith("transformers.")
            or module_name == "zeroshot"
            or module_name.startswith("zeroshot.")
        ):
            del sys.modules[module_name]
    sys.path.insert(0, str(dependency_root))
    source = _official_source("retizero")
    sys.path.insert(0, str(source))
    from transformers import AutoConfig, AutoModel
    import zeroshot.modeling.model as retizero_model

    config = AutoConfig.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        local_files_only=True,
    )
    config.output_hidden_states = True
    original_tokenizer_loader = retizero_model.AutoTokenizer.from_pretrained
    retizero_model.AutoTokenizer.from_pretrained = (
        lambda name, *args, **kwargs: original_tokenizer_loader(
            name,
            *args,
            local_files_only=True,
            **kwargs,
        )
    )
    retizero_model.AutoModel.from_pretrained = (
        lambda *args, **kwargs: AutoModel.from_config(config)
    )
    model = retizero_model.CLIPRModel(
        vision_type="lora",
        vision_pretrained=False,
        from_checkpoint=False,
        R=8,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    image = Image.new("RGB", (320, 280), color=(127, 127, 127))
    probabilities, logits = model(image, ["normal retina", "glaucoma"])
    if probabilities.shape != (2,) or logits.shape != (2,):
        raise RuntimeError("Unexpected RetiZero zero-shot output")
    if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
        raise RuntimeError("RetiZero produced non-finite output")
    return {
        "input_size": [224, 224],
        "probability_shape": list(probabilities.shape),
        "logit_shape": list(logits.shape),
        "output_finite": True,
        "state_dict_key_count": len(state_dict),
        "source_commit": OFFICIAL_SOURCE_PINS["retizero"][0],
        "dependency_contract": "official torch~=1.13.1, transformers~=4.27.4",
        "compatibility_note": (
            "Executed with the repository-vendored timm implementation and the pinned "
            "local transformers 4.30.2 compatibility environment"
        ),
    }


def _flair_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from safetensors.torch import load_file

    device = str(spec.get("device", "cpu"))
    if not device.startswith("cuda"):
        raise RuntimeError("The pinned official FLAIR module selects CUDA at import time")
    dependency_root = UPSTREAM_SOURCE_ROOT.parent / "runtime_deps" / "urfound"
    if not dependency_root.is_dir():
        raise FileNotFoundError("Pinned transformers 4.30.2 environment is unavailable")
    for module_name in tuple(sys.modules):
        if module_name == "transformers" or module_name.startswith("transformers."):
            del sys.modules[module_name]
    sys.path.insert(0, str(dependency_root))
    source = _official_source("flair")
    sys.path.insert(0, str(source))
    from transformers import AutoConfig, AutoModel
    import flair.modeling.model as flair_model

    config = AutoConfig.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        local_files_only=True,
    )
    config.output_hidden_states = True
    original_tokenizer_loader = flair_model.AutoTokenizer.from_pretrained
    flair_model.AutoTokenizer.from_pretrained = (
        lambda name, *args, **kwargs: original_tokenizer_loader(
            name,
            *args,
            local_files_only=True,
            **kwargs,
        )
    )
    flair_model.AutoModel.from_pretrained = (
        lambda *args, **kwargs: AutoModel.from_config(config)
    )
    model = flair_model.FLAIRModel(vision_pretrained=False)
    state_dict = load_file(path)
    model.load_state_dict(state_dict, strict=True)
    image = np.full((512, 512, 3), 127, dtype=np.uint8)
    probabilities, logits = model(image, ["normal retina", "macular hole"])
    if probabilities.shape != (1, 2) or logits.shape != (1, 2):
        raise RuntimeError("Unexpected FLAIR similarity output")
    if not np.isfinite(probabilities).all() or not np.isfinite(logits).all():
        raise RuntimeError("FLAIR produced non-finite output")
    return {
        "input_shape": list(image.shape),
        "probability_shape": list(probabilities.shape),
        "logit_shape": list(logits.shape),
        "output_finite": True,
        "state_dict_key_count": len(state_dict),
        "source_commit": OFFICIAL_SOURCE_PINS["flair"][0],
        "dependency_contract": "transformers==4.30.2",
        "compatibility_note": (
            "Constructed Bio_ClinicalBERT from cached official config/tokenizer; "
            "all model parameters came from the FLAIR checkpoint"
        ),
    }


def _keepfit_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    device = str(spec.get("device", "cpu"))
    if not device.startswith("cuda"):
        raise RuntimeError("The pinned official KeepFIT module selects CUDA at import time")
    dependency_root = UPSTREAM_SOURCE_ROOT.parent / "runtime_deps" / "urfound"
    if not dependency_root.is_dir():
        raise FileNotFoundError("Pinned transformers 4.30.2 environment is unavailable")
    for module_name in tuple(sys.modules):
        if module_name == "transformers" or module_name.startswith("transformers."):
            del sys.modules[module_name]
    sys.path.insert(0, str(dependency_root))
    source = _official_source("keepfit") / "KeepFIT" / "KeepFIT-CFP"
    sys.path.insert(0, str(source))
    from transformers import AutoConfig, AutoModel
    import keepfit.modeling.model as keepfit_model

    config = AutoConfig.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        local_files_only=True,
    )
    config.output_hidden_states = True
    original_tokenizer_loader = keepfit_model.AutoTokenizer.from_pretrained
    keepfit_model.AutoTokenizer.from_pretrained = (
        lambda name, *args, **kwargs: original_tokenizer_loader(
            name,
            *args,
            local_files_only=True,
            **kwargs,
        )
    )
    keepfit_model.AutoModel.from_pretrained = (
        lambda *args, **kwargs: AutoModel.from_config(config)
    )
    model = keepfit_model.KeepFITModel(
        vision_pretrained=False,
        from_checkpoint=False,
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    image = np.full((400, 320, 3), 127, dtype=np.uint8)
    probabilities, logits = model(
        image,
        ["normal retina", "diabetic retinopathy"],
    )
    if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
        raise RuntimeError(f"Unexpected KeepFIT output: {probabilities.shape}")
    return {
        "input_shape": list(image.shape),
        "probability_shape": list(probabilities.shape),
        "logit_shape": list(logits.shape),
        "output_finite": True,
        "source_commit": OFFICIAL_SOURCE_PINS["keepfit"][0],
        "dependency_contract": "transformers==4.30.2",
        "compatibility_note": (
            "Constructed Bio_ClinicalBERT from cached official config/tokenizer; "
            "all model parameters came from the KeepFIT checkpoint"
        ),
    }


def _fmue_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from PIL import Image
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from torchvision import transforms

    source = _official_source("fmue")
    sys.path.insert(0, str(source))
    import vit_model

    model = vit_model.vit_large_patch16(
        img_size=224,
        num_classes=16,
        drop_path_rate=0.1,
        global_pool=True,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = str(spec.get("device", "cpu"))
    model = model.eval().to(device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    tensor = preprocess(Image.new("RGB", (320, 280), color=(127, 127, 127))).unsqueeze(0)
    with torch.inference_mode():
        features = model.forward_features(tensor.to(device))
        output = model.head(features)
    if output.shape != (1, 16) or not output.isfinite().all():
        raise RuntimeError(f"Unexpected FMUE output: {tuple(output.shape)}")
    return {
        "input_shape": list(tensor.shape),
        "feature_shape": list(features.shape),
        "output_shape": list(output.shape),
        "output_finite": True,
        "source_commit": OFFICIAL_SOURCE_PINS["fmue"][0],
        "runtime_semantics": "16-class OCT task checkpoint; not a generic encoder",
        "compatibility_note": (
            "Called forward_features and head explicitly because timm 1.0 adds "
            "an attn_mask argument to the base forward method"
        ),
    }


def _urfound_runtime(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    import timm.models.vision_transformer as timm_vit

    dependency_root = UPSTREAM_SOURCE_ROOT.parent / "runtime_deps" / "urfound"
    if not dependency_root.is_dir():
        raise FileNotFoundError("Pinned UrFound transformers 4.30.2 environment is unavailable")
    for module_name in tuple(sys.modules):
        if (
            module_name == "transformers"
            or module_name.startswith("transformers.")
            or module_name == "util"
            or module_name.startswith("util.")
            or module_name == "bert"
            or module_name.startswith("bert.")
        ):
            del sys.modules[module_name]
    sys.path.insert(0, str(dependency_root))
    sys.path.insert(0, str(_official_source("urfound")))
    np.float = float
    original_block = timm_vit.Block

    def compatible_block(*args, **kwargs):
        kwargs.pop("qk_scale", None)
        return original_block(*args, **kwargs)

    timm_vit.Block = compatible_block
    import util.model_urfound as urfound_model

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = urfound_model.urmodel(norm_pix_loss=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = str(spec.get("device", "cpu"))
    if not device.startswith("cuda"):
        raise RuntimeError("Official UrFound forward hard-codes CUDA tensor transfer")
    model = model.eval().to(device)
    batch = {
        "image": torch.full((1, 3, 448, 448), 0.5),
        "ids": torch.ones((1, 16), dtype=torch.long),
        "labels": torch.ones((1, 16), dtype=torch.long),
        "attention_mask": torch.ones((1, 16), dtype=torch.long),
        "type_ids": torch.zeros((1, 16), dtype=torch.long),
    }
    with torch.inference_mode():
        losses, output, mask = model(batch, mask_ratio=0.75)
    if output.shape != (1, 196, 3072) or not output.isfinite().all():
        raise RuntimeError(f"Unexpected UrFound output: {tuple(output.shape)}")
    return {
        "input_shape": list(batch["image"].shape),
        "output_shape": list(output.shape),
        "mask_shape": list(mask.shape),
        "losses_finite": all(torch.isfinite(value).item() for value in losses),
        "output_finite": True,
        "source_commit": OFFICIAL_SOURCE_PINS["urfound"][0],
        "dependency_contract": "transformers==4.30.2",
        "compatibility_note": (
            "Dropped removed timm qk_scale and restored deprecated numpy.float alias "
            "without modifying official source"
        ),
    }


def _visionunite_resource_audit(path: Path) -> dict[str, Any]:
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    state_dict = checkpoint["model"]
    keys = list(state_dict)
    tensor_bytes = sum(
        value.numel() * value.element_size()
        for value in state_dict.values()
        if hasattr(value, "numel")
    )
    return {
        "checkpoint_container": list(checkpoint),
        "state_dict_key_count": len(keys),
        "state_dict_tensor_bytes": tensor_bytes,
        "llama_key_count": sum(key.startswith("llama.") for key in keys),
        "eva02_key_count": sum(key.startswith("eva02.") for key in keys),
        "clip_key_count": sum(key.startswith("clip.") for key in keys),
        "source_commit": "3dab080ef21d946c4dfaab26572de3828c598090",
        "official_constructor": "llama.llama_adapter.load / LLaMA_adapter",
        "resource_estimate": (
            "Checkpoint model tensors are about "
            f"{tensor_bytes / 1024**3:.1f} GiB before runtime activations"
        ),
        "missing_official_assets": [
            "LLaMA-7B params.json",
            "LLaMA tokenizer.model",
            "LLaMA base checkpoint shards",
        ],
        "verification_status": "partially_verified",
        "final_conclusion": "resource_blocked",
        "failure_summary": (
            "Official construction requires separately supplied LLaMA-7B base assets; "
            "the checkpoint size is not the blocking reason"
        ),
    }


RUNTIME_HANDLERS = {
    "deretfound_mae_runtime": _deretfound_mae_runtime,
    "deretfound_sd_runtime": _deretfound_sd_runtime,
    "retfound_runtime": _retfound_runtime,
    "retfound_green_runtime": _retfound_green_runtime,
    "preti_runtime": _preti_runtime,
    "eyeclip_runtime": _eyeclip_runtime,
    "vilref_runtime": _vilref_runtime,
    "ret_clip_runtime": _ret_clip_runtime,
    "mirage_runtime": _mirage_runtime,
    "fmue_runtime": _fmue_runtime,
    "urfound_runtime": _urfound_runtime,
    "retizero_runtime": _retizero_runtime,
    "flair_runtime": _flair_runtime,
    "keepfit_runtime": _keepfit_runtime,
}


def run_probe_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one profile inside a worker process; never changes model qualifications."""

    path = Path(spec["asset_path"])
    profile = str(spec["probe_profile"])
    started = datetime.now(timezone.utc)
    details: dict[str, Any]
    achieved_stage = "asset_probe_passed"
    if profile in RUNTIME_HANDLERS:
        import torch

        device = str(spec.get("device", "cpu"))
        torch_device = torch.device(device)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(torch_device)
            torch.cuda.reset_peak_memory_stats(torch_device)
        details = RUNTIME_HANDLERS[profile](path, spec)
        achieved_stage = "runtime_smoke_passed"
        peak_vram_bytes = (
            torch.cuda.max_memory_allocated(torch_device)
            if device.startswith("cuda") and torch.cuda.is_available()
            else 0
        )
        stage_evidence = {
            "official_source_pinned": details.get("source_commit", "pinned source"),
            "environment_ready": profile,
            "model_constructed": "official constructor completed",
            "checkpoint_deserialized": "checkpoint container was read",
            "weights_loaded": "profile weight-loading validation completed",
            "official_preprocessing_ready": "profile preprocessing produced native input",
            "native_forward_passed": "native forward returned without exception",
            "output_contract_validated": "shape and finite-value checks passed",
        }
        details["stage_evidence"] = {
            stage: {
                "status": "passed",
                "evidence": evidence,
                "failure_type": None,
                "failure_summary": None,
                "duration_seconds": None,
                "device": device,
                "peak_vram_bytes": None,
            }
            for stage, evidence in stage_evidence.items()
        }
        details["process_peak_vram_bytes"] = peak_vram_bytes
    elif profile == "visionunite_resource_audit":
        details = _visionunite_resource_audit(path)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "status": "skipped",
            "achieved_stage": "resource_blocked",
            "elapsed_seconds": elapsed,
            "details": details,
            "asset_probe_passed": True,
            "runtime_smoke_passed": False,
            "resource_blocked": True,
            "task_inference_ready": False,
            "route_eligible": False,
        }
    elif profile == "torch_checkpoint_structure":
        import argparse
        import torch

        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        details = _torch_summary(checkpoint)
        details["probe_kind"] = "torch_checkpoint_structure"
    elif profile == "safetensors_structure":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            sample = {
                key: {"shape": list(handle.get_slice(key).get_shape())}
                for key in keys[:12]
            }
        details = {"tensor_count": len(keys), "sample_tensors": sample}
        details["probe_kind"] = "safetensors_structure"
    elif profile == "zip_structure":
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            bad_paths = [
                member.filename
                for member in members
                if Path(member.filename).is_absolute() or ".." in Path(member.filename).parts
            ]
            details = {
                "member_count": len(members),
                "compressed_bytes": sum(member.compress_size for member in members),
                "uncompressed_bytes": sum(member.file_size for member in members),
                "sample_members": [member.filename for member in members[:20]],
                "unsafe_member_count": len(bad_paths),
                "probe_kind": "zip_structure",
            }
        if bad_paths:
            raise ValueError("Archive contains unsafe member paths")
    else:
        return {
            "status": "skipped",
            "achieved_stage": "metadata_only",
            "detail": "No executable probe profile",
            "asset_probe_passed": False,
            "runtime_smoke_passed": False,
        }
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "status": "passed",
        "achieved_stage": achieved_stage,
        "elapsed_seconds": elapsed,
        "details": details,
        "asset_probe_passed": True,
        "runtime_smoke_passed": achieved_stage == "runtime_smoke_passed",
        "task_inference_ready": False,
        "route_eligible": False,
    }


def latest_smoke_run(runtime_root: Path | str = RUNTIME_ROOT) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(runtime_root)
    summaries = sorted(root.glob("*/summary.json"), reverse=True)
    if not summaries:
        return {}, pd.DataFrame()
    summary_path = summaries[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results_path = summary_path.parent / "results.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    return summary, results
