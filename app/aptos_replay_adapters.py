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

    def __post_init__(self) -> None:
        if self.adapter_type not in ADAPTER_TYPES:
            raise ValueError(f"unsupported replay adapter: {self.adapter_type}")


def load_timm_classifier(spec: ReplayAdapterSpec, *, device: str) -> Any:
    """Load a strict five-class timm checkpoint without changing its recipe."""
    from argparse import Namespace

    import timm
    import torch

    checkpoint = Path(spec.checkpoint_path)
    kwargs: dict[str, Any] = {"pretrained": False, "num_classes": 5}
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
    """Frozen timm classifier with its registered APTOS evaluation transform."""

    def __init__(self, model: Any, *, preprocessing_id: str, device: str):
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
        elif preprocessing_id == "timm_imagefolder_v1_eval_resize224_bilinear_imagenet_rgb_fp32":
            resize = [transforms.Resize((224, 224))]
        else:
            raise ValueError(f"unsupported timm APTOS preprocessing: {preprocessing_id}")
        self.model = model
        self.device = device
        self.labels = tuple(range(5))
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
        return TimmAptosTaskAdapter(
            load_timm_classifier(spec, device=device),
            preprocessing_id=spec.preprocessing_id,
            device=device,
        )
    if spec.adapter_type == "flair_frozen_encoder_linear_head":
        return load_flair_adapter(spec, device=device)
    if spec.adapter_type == "preti_classifier":
        if not spec.source_root:
            raise ValueError("PRETI adapter requires source_root")
        return load_preti_adapter(
            spec,
            device=device,
            source_root=spec.source_root,
        )
    if spec.adapter_type == "frozen_encoder_sklearn_probe":
        encoder, probe = load_green_probe_adapter(spec, device=device)
        return GreenAptosTaskAdapter(encoder, probe, device=device)
    raise ValueError(f"unsupported registered APTOS adapter: {spec.adapter_type}")
