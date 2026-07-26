"""Thin, model-specific loading contracts for the unified APTOS replay pool.

Shared manifest alignment, prediction export, metrics, cost timing and replay
audits deliberately do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ADAPTER_TYPES = {
    "timm_classifier",
    "retfound_dinov2_classifier",
    "flair_frozen_encoder_linear_head",
    "frozen_encoder_sklearn_probe",
    "retfound_classifier",
    "preti_classifier",
}


@dataclass(frozen=True)
class ReplayAdapterSpec:
    adapter_type: str
    architecture: str
    preprocessing_id: str
    checkpoint_path: str
    task_checkpoint_path: str | None = None
    probe_path: str | None = None
    source_root: str | None = None
    checkpoint_key: str | None = None
    global_pool: str | None = None
    allow_argparse_namespace: bool = False
    source_num_classes: int = 5
    output_probability_groups: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.adapter_type not in ADAPTER_TYPES:
            raise ValueError(f"unsupported replay adapter: {self.adapter_type}")


def load_timm_classifier(spec: ReplayAdapterSpec, *, device: str) -> Any:
    """Load a strict five-class timm checkpoint without changing its recipe."""
    from argparse import Namespace

    import timm
    import torch

    checkpoint = Path(spec.checkpoint_path)
    kwargs: dict[str, Any] = {
        "pretrained": False,
        "num_classes": int(spec.source_num_classes),
    }
    if spec.global_pool:
        kwargs["global_pool"] = spec.global_pool
    model = timm.create_model(spec.architecture, **kwargs)
    if spec.allow_argparse_namespace:
        with torch.serialization.safe_globals([Namespace]):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state: Any = payload
    if spec.checkpoint_key:
        for key in spec.checkpoint_key.split("."):
            state = state[key]
    elif isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    if not isinstance(state, dict):
        raise TypeError("timm task checkpoint must be a bare state_dict")
    state = dict(state)
    for prefix in ("module.", "model."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key).removeprefix(prefix): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict timm loading returned key differences")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


def load_retfound_dinov2_classifier(spec: ReplayAdapterSpec, *, device: str) -> Any:
    """Load the registered RETFound-DINOv2 task checkpoint strictly."""
    import copy
    import sys

    import timm
    import torch

    if not spec.source_root:
        raise ValueError("RETFound-DINOv2 adapter requires source_root")
    source_root = Path(spec.source_root)
    sys.path.insert(0, str(source_root))
    import models_vit

    original_create_model = timm.create_model

    def create_model_without_download(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_without_download
    if hasattr(models_vit, "timm"):
        models_vit.timm.create_model = create_model_without_download
    try:
        checkpoint = torch.load(
            spec.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict) or "model" not in checkpoint or "args" not in checkpoint:
            raise TypeError("RETFound-DINOv2 checkpoint must contain model and args")
        checkpoint_args = copy.deepcopy(checkpoint["args"])
        constructor = getattr(models_vit, str(checkpoint_args.model))
        kwargs = {
            "num_classes": int(spec.source_num_classes),
            "drop_path_rate": float(checkpoint_args.drop_path),
            "global_pool": spec.global_pool or "token",
        }
        try:
            model = constructor(args=checkpoint_args, **kwargs)
        except TypeError:
            model = constructor(checkpoint_args, **kwargs)
        result = model.load_state_dict(checkpoint["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("strict RETFound-DINOv2 loading returned key differences")
    finally:
        timm.create_model = original_create_model
        if hasattr(models_vit, "timm"):
            models_vit.timm.create_model = original_create_model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


def load_flair_adapter(spec: ReplayAdapterSpec, *, device: str) -> Any:
    from app.flair_task_adapter import FlairAptosTaskAdapter

    if not spec.task_checkpoint_path:
        raise ValueError("FLAIR adapter requires task_checkpoint_path")
    return FlairAptosTaskAdapter.load(
        encoder_checkpoint=spec.checkpoint_path,
        task_checkpoint=spec.task_checkpoint_path,
        device=device,
    )


def load_preti_adapter(spec: ReplayAdapterSpec, *, device: str, source_root: str) -> Any:
    from app.preti_task_adapter import PretiAptosTaskAdapter

    if not spec.task_checkpoint_path:
        raise ValueError("PRETI adapter requires task_checkpoint_path")
    return PretiAptosTaskAdapter.load(
        source_root=source_root,
        encoder_checkpoint=spec.checkpoint_path,
        task_checkpoint=spec.task_checkpoint_path,
        device=device,
    )


def load_green_probe_adapter(spec: ReplayAdapterSpec, *, device: str) -> tuple[Any, Any]:
    """Load the historical Green encoder and fitted sklearn probe strictly."""
    import joblib
    import timm
    import torch

    if not spec.probe_path:
        raise ValueError("Green adapter requires probe_path")
    encoder = timm.create_model(
        spec.architecture, img_size=(392, 392), num_classes=0
    )
    encoder.global_pool = "avg"
    state = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=True)
    result = encoder.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict RETFound-Green loading returned key differences")
    probe = joblib.load(spec.probe_path)
    if not hasattr(probe, "predict_proba"):
        raise TypeError("Green probe is not a fitted probability classifier")
    return encoder.eval().to(device), probe


class TimmAptosTaskAdapter:
    """Frozen classifier with its registered evaluation transform."""

    def __init__(
        self,
        model: Any,
        *,
        preprocessing_id: str,
        device: str,
        num_classes: int = 5,
    ):
        import torch
        from torchvision import transforms

        if preprocessing_id == "imagenet_bicubic_resize256_centercrop224_v1":
            resize = [
                transforms.Resize(
                    256,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(224),
            ]
        elif preprocessing_id in {
            "timm_imagefolder_v1_eval_resize224_bilinear_imagenet_rgb_fp32",
            "timm_imagefolder_v1_resize224_imagenet_rgb_fp32",
        }:
            resize = [transforms.Resize((224, 224))]
        elif preprocessing_id == (
            "retfound_dinov2_bicubic_resize256_centercrop224_imagenet_rgb_fp32"
        ):
            resize = [
                transforms.Resize(
                    256,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(224),
            ]
        else:
            raise ValueError(f"unsupported task preprocessing: {preprocessing_id}")
        self.model = model
        self.device = device
        self.labels = tuple(range(int(num_classes)))
        self.preprocess = transforms.Compose(
            [
                *resize,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._torch = torch

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        tensors = [self.preprocess(image.convert("RGB")) for image in images]
        if not tensors:
            return np.empty((0, len(self.labels)), dtype=float)
        with self._torch.inference_mode():
            logits = self.model(self._torch.stack(tensors).to(self.device))
            probabilities = self._torch.softmax(logits.float(), dim=1).cpu().numpy()
        _validate_probabilities(probabilities, len(self.labels))
        return probabilities


class GroupedProbabilityAdapter:
    """Collapse a registered source label space without retraining or calibration."""

    def __init__(
        self,
        adapter: Any,
        groups: Iterable[Iterable[int]],
    ):
        self.adapter = adapter
        self.groups = tuple(tuple(int(index) for index in group) for group in groups)
        source_classes = len(adapter.labels)
        flattened = [index for group in self.groups for index in group]
        if sorted(flattened) != list(range(source_classes)):
            raise ValueError(
                "output_probability_groups must partition every source class exactly once"
            )
        self.labels = tuple(range(len(self.groups)))

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        source = np.asarray(self.adapter.predict_proba(images), dtype=float)
        grouped = np.column_stack(
            [source[:, list(group)].sum(axis=1) for group in self.groups]
        )
        _validate_probabilities(grouped, len(self.labels))
        return grouped


class GreenAptosTaskAdapter:
    """Frozen RETFound-Green encoder plus its registered sklearn probe."""

    def __init__(self, encoder: Any, probe: Any, *, device: str):
        import torch
        from torchvision import transforms

        self.encoder = encoder
        self.probe = probe
        self.device = device
        self.labels = tuple(range(5))
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((392, 392)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self._torch = torch

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        tensors = [self.preprocess(image.convert("RGB")) for image in images]
        if not tensors:
            return np.empty((0, len(self.labels)), dtype=float)
        with self._torch.inference_mode():
            features = self.encoder(self._torch.stack(tensors).to(self.device))
        probabilities = np.asarray(
            self.probe.predict_proba(features.detach().cpu().numpy()),
            dtype=float,
        )
        _validate_probabilities(probabilities, len(self.labels))
        return probabilities


def _validate_probabilities(probabilities: np.ndarray, num_classes: int) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError("task adapter probability columns do not match the label space")
    if not np.isfinite(probabilities).all():
        raise ValueError("task adapter returned NaN or Inf")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("task adapter probabilities do not sum to one")


def load_registered_aptos_adapter(
    spec: ReplayAdapterSpec,
    *,
    device: str,
) -> Any:
    """Resolve an existing APTOS task adapter without training or recalibration."""
    if spec.adapter_type in {"timm_classifier", "retfound_classifier"}:
        adapter = TimmAptosTaskAdapter(
            load_timm_classifier(spec, device=device),
            preprocessing_id=spec.preprocessing_id,
            device=device,
            num_classes=spec.source_num_classes,
        )
    elif spec.adapter_type == "retfound_dinov2_classifier":
        adapter = TimmAptosTaskAdapter(
            load_retfound_dinov2_classifier(spec, device=device),
            preprocessing_id=spec.preprocessing_id,
            device=device,
            num_classes=spec.source_num_classes,
        )
    elif spec.adapter_type == "flair_frozen_encoder_linear_head":
        adapter = load_flair_adapter(spec, device=device)
    elif spec.adapter_type == "preti_classifier":
        if not spec.source_root:
            raise ValueError("PRETI adapter requires source_root")
        adapter = load_preti_adapter(
            spec,
            device=device,
            source_root=spec.source_root,
        )
    elif spec.adapter_type == "frozen_encoder_sklearn_probe":
        encoder, probe = load_green_probe_adapter(spec, device=device)
        adapter = GreenAptosTaskAdapter(encoder, probe, device=device)
    else:
        raise ValueError(f"unsupported registered task adapter: {spec.adapter_type}")
    if spec.output_probability_groups:
        return GroupedProbabilityAdapter(adapter, spec.output_probability_groups)
    return adapter
