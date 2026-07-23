"""Thin, model-specific loading contracts for the unified APTOS replay pool.

Shared manifest alignment, prediction export, metrics, cost timing and replay
audits deliberately do not live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    def __post_init__(self) -> None:
        if self.adapter_type not in ADAPTER_TYPES:
            raise ValueError(f"unsupported replay adapter: {self.adapter_type}")


def load_timm_classifier(spec: ReplayAdapterSpec, *, device: str) -> Any:
    """Load a strict five-class timm checkpoint without changing its recipe."""
    import timm
    import torch

    checkpoint = Path(spec.checkpoint_path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("timm task checkpoint must be a bare state_dict")
    model = timm.create_model(spec.architecture, pretrained=False, num_classes=5)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict timm loading returned key differences")
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
