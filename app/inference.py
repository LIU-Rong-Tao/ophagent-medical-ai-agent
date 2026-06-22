"""OphAgent 单图真实 checkpoint 推理与结构化阶段错误。"""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.checkpoints import ModelArtifact


GRADE_LABELS = (
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
)
VALID_ERROR_STAGES = {
    "unsupported_loader",
    "build_model",
    "load_checkpoint",
    "preprocess",
    "inference",
    "postprocess",
}


@dataclass(frozen=True)
class InferenceResult:
    ok: bool
    stage: str
    error_type: str | None = None
    error_message: str | None = None
    pred_grade: int | None = None
    probabilities: list[float] | None = None
    confidence: float | None = None
    margin: float | None = None
    entropy_norm: float | None = None
    source: str | None = None
    backbone: str | None = None
    labels: list[str] | None = None

    def to_display_payload(self) -> dict[str, Any]:
        if not self.ok or self.probabilities is None or self.pred_grade is None:
            raise ValueError("失败的推理结果不能转换为展示概率。")
        return {
            "labels": list(self.labels or GRADE_LABELS),
            "probabilities": list(self.probabilities),
            "pred_grade": int(self.pred_grade),
            "confidence": self.confidence,
            "margin": self.margin,
            "entropy_norm": self.entropy_norm,
            "source": self.source,
            "backbone": self.backbone,
        }


class InferenceStageError(RuntimeError):
    def __init__(self, stage: str, cause: Exception):
        if stage not in VALID_ERROR_STAGES:
            raise ValueError(f"未知推理失败阶段：{stage}")
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def _failure(
    stage: str,
    error_type: str,
    error_message: str,
    *,
    backbone: str | None,
) -> InferenceResult:
    return InferenceResult(
        ok=False,
        stage=stage,
        error_type=error_type,
        error_message=str(error_message).strip()[:600],
        backbone=backbone,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"JSON 内容为空或不是对象：{path}")
    return value


def _load_runtime():
    import timm
    import torch

    from agent.runner import build_transform

    return torch, timm, build_transform


def _build_model(
    artifact: ModelArtifact,
    config: dict[str, Any],
    timm_module,
):
    """与 v0.7.1 direct inference 一致地创建五分类 timm 模型。"""

    return timm_module.create_model(
        artifact.loader_model_name,
        pretrained=False,
        num_classes=int(config.get("num_classes", 5)),
    )


def _load_checkpoint(model, artifact: ModelArtifact, torch_module) -> None:
    """按 v0.7.1 冻结推理协议严格加载最终分类 checkpoint。"""

    state_dict = torch_module.load(
        artifact.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)


def _cache_key(artifact: ModelArtifact) -> tuple[str, str, int | None]:
    return (
        artifact.model_key,
        str(artifact.checkpoint_path),
        artifact.checkpoint_mtime_ns,
    )


def _release_cached_model(cache: MutableMapping[str, Any], torch_module) -> None:
    active = cache.get("_ophagent_active_model")
    if not isinstance(active, dict):
        return
    model = active.get("model")
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    cache.pop("_ophagent_active_model", None)
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _load_model_bundle(
    artifact: ModelArtifact,
    *,
    cache: MutableMapping[str, Any] | None,
    runtime_loader: Callable[[], tuple[Any, Any, Callable]],
):
    try:
        torch_module, timm_module, build_transform = runtime_loader()
        config = _read_json(artifact.config_path)
        class_to_idx = _read_json(artifact.class_to_idx_path)
        key = _cache_key(artifact)
        active = cache.get("_ophagent_active_model") if cache is not None else None
        if isinstance(active, dict) and active.get("key") == key:
            return (
                active["model"],
                active["device"],
                active["idx_to_class"],
                active["image_size"],
                torch_module,
                build_transform,
            )
        model = _build_model(artifact, config, timm_module)
    except InferenceStageError:
        raise
    except Exception as exc:
        raise InferenceStageError("build_model", exc) from exc

    try:
        _load_checkpoint(model, artifact, torch_module)
    except Exception as exc:
        raise InferenceStageError("load_checkpoint", exc) from exc

    try:
        device = torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
        model.to(device).eval()
        idx_to_class = {
            int(index): str(name)
            for name, index in class_to_idx.items()
        }
        image_size = int(config.get("image_size", 224))
    except Exception as exc:
        raise InferenceStageError("build_model", exc) from exc

    if cache is not None:
        _release_cached_model(cache, torch_module)
        cache["_ophagent_active_model"] = {
            "key": key,
            "model": model,
            "device": device,
            "idx_to_class": idx_to_class,
            "image_size": image_size,
        }
    return (
        model,
        device,
        idx_to_class,
        image_size,
        torch_module,
        build_transform,
    )


def postprocess_probabilities(
    probabilities: Sequence[float],
    *,
    display_name: str,
    backbone: str,
) -> InferenceResult:
    try:
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (5,):
            raise ValueError("DR 在线推理必须返回 5 个类别概率。")
        if not np.isfinite(values).all():
            raise ValueError("概率包含 NaN 或无穷值。")
        if (values < 0).any():
            raise ValueError("概率不能为负数。")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("概率和必须大于 0。")
        values = values / total
        order = np.argsort(values)
        pred_grade = int(order[-1])
        confidence = float(values[order[-1]])
        margin = float(values[order[-1]] - values[order[-2]])
        safe = np.clip(values, np.finfo(float).tiny, 1.0)
        entropy_norm = float(
            -(values * np.log(safe)).sum() / math.log(len(values))
        )
    except Exception as exc:
        return _failure(
            "postprocess",
            "InvalidProbabilityOutput",
            str(exc),
            backbone=backbone,
        )
    return InferenceResult(
        ok=True,
        stage="complete",
        pred_grade=pred_grade,
        probabilities=values.tolist(),
        confidence=confidence,
        margin=margin,
        entropy_norm=entropy_norm,
        source=f"{display_name} 在线 checkpoint 推理",
        backbone=backbone,
        labels=list(GRADE_LABELS),
    )


def run_single_image_inference(
    image: Image.Image,
    artifact: ModelArtifact,
    *,
    cache: MutableMapping[str, Any] | None = None,
    runtime_loader: Callable[[], tuple[Any, Any, Callable]] = _load_runtime,
) -> InferenceResult:
    """运行一个冻结 checkpoint；所有失败都返回固定阶段，不静默返回 None。"""

    if (
        not artifact.can_attempt_load
        or not artifact.loader_model_name
        or artifact.checkpoint_path is None
        or artifact.config_path is None
        or artifact.class_to_idx_path is None
    ):
        return _failure(
            "unsupported_loader",
            "UnsupportedLoader",
            "当前模型缺少已知 loader、checkpoint、config 或类别映射。",
            backbone=artifact.model_key,
        )

    try:
        (
            model,
            device,
            _idx_to_class,
            image_size,
            torch_module,
            build_transform,
        ) = _load_model_bundle(
            artifact,
            cache=cache,
            runtime_loader=runtime_loader,
        )
    except InferenceStageError as exc:
        return _failure(
            exc.stage,
            type(exc.cause).__name__,
            str(exc.cause),
            backbone=artifact.model_key,
        )

    try:
        tensor = (
            build_transform(image_size)(image.convert("RGB"))
            .unsqueeze(0)
            .to(device)
        )
    except Exception as exc:
        return _failure(
            "preprocess",
            type(exc).__name__,
            str(exc),
            backbone=artifact.model_key,
        )

    try:
        with torch_module.no_grad():
            logits = model(tensor)
            probabilities = (
                torch_module.softmax(logits, dim=1)[0]
                .detach()
                .cpu()
                .tolist()
            )
    except Exception as exc:
        return _failure(
            "inference",
            type(exc).__name__,
            str(exc),
            backbone=artifact.model_key,
        )

    return postprocess_probabilities(
        probabilities,
        display_name=artifact.display_name,
        backbone=artifact.model_key,
    )
