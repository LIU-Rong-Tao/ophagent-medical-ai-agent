"""模型中转台共享的模型元数据规范化规则。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_UNSPECIFIED_PRETRAINING = {"", "unspecified", "unknown", "none", "nan"}

_TIMM_PRETRAINED_MODELS = {
    "convnext_tiny": (
        "convnext_tiny.in12k_ft_in1k",
        "imagenet12k_ft_imagenet1k",
        "convnext_tiny",
    ),
    "swin_tiny_patch4_window7_224": (
        "swin_tiny_patch4_window7_224.ms_in1k",
        "imagenet1k",
        "swin_tiny",
    ),
    "vit_base_patch16_224": (
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "imagenet21k_ft_imagenet1k",
        "vit_b",
    ),
    "vit_large_patch16_224": (
        "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "imagenet21k_ft_imagenet1k",
        "vit_l",
    ),
}


def _architecture_base(architecture: str) -> str:
    return str(architecture).strip().lower().split(".", 1)[0]


def resolve_timm_pretrained_architecture(architecture: str) -> str:
    """将无标签 timm 架构解析为明确的官方预训练权重标识。"""

    value = str(architecture).strip()
    resolved = _TIMM_PRETRAINED_MODELS.get(_architecture_base(value))
    return resolved[0] if resolved else value


def timm_pretraining_source(architecture: str) -> str:
    resolved = _TIMM_PRETRAINED_MODELS.get(_architecture_base(architecture))
    return resolved[1] if resolved else "timm_pretrained"


def canonical_backbone_id(architecture: str) -> str:
    base = _architecture_base(architecture)
    resolved = _TIMM_PRETRAINED_MODELS.get(base)
    return resolved[2] if resolved else base


def canonical_timm_artifact_id(architecture: str, task_id: str) -> str:
    """生成不携带历史任务链的 timm 任务模型标识。"""

    backbone = canonical_backbone_id(architecture)
    task = str(task_id).strip()
    if not backbone or not task:
        raise ValueError("architecture 与 task_id 不能为空")
    return f"{backbone}_imagenet_{task}_adapter"


def _clean_pretraining_source(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _UNSPECIFIED_PRETRAINING else text


def _config_pretraining_source(training_config: Mapping[str, Any] | None) -> str:
    if not training_config:
        return ""
    for key in ("pretraining_source", "initialization_source"):
        source = _clean_pretraining_source(training_config.get(key))
        if source:
            return source
    pretrained = training_config.get("pretrained")
    if pretrained is True or str(pretrained).strip().lower() in {"1", "true", "yes"}:
        return "timm_pretrained"
    if pretrained is False or str(pretrained).strip().lower() in {"0", "false", "no"}:
        return "random_initialization"
    return ""


def normalized_model_metadata(
    artifact_id: str,
    architecture: str,
    *,
    pretraining_source: str = "",
    training_config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    artifact = str(artifact_id).strip()
    arch = str(architecture).strip() or artifact
    value = f"{artifact} {arch}".lower()

    if "retfound" in value:
        family = "retfound"
    elif "convnext" in value:
        family = "convnext"
    elif "swin" in value:
        family = "swin"
    elif "vit" in value:
        family = "vit"
    elif "mock" in value:
        family = "mock"
    else:
        family = "other"

    explicit_pretraining = _clean_pretraining_source(pretraining_source)
    config_pretraining = _config_pretraining_source(training_config)
    if explicit_pretraining:
        pretraining = explicit_pretraining
    elif config_pretraining:
        pretraining = config_pretraining
    elif "official_protocol" in value:
        pretraining = "official_protocol"
    elif "official_like" in value:
        pretraining = "official_like"
    elif "imagenet" in value or "in1k" in value:
        pretraining = "imagenet"
    elif "green" in value:
        pretraining = "retfound_green"
    elif "dinov2" in value:
        pretraining = "dinov2"
    else:
        pretraining = "unspecified"

    return {
        "model_family": family,
        "architecture": arch,
        "pretraining_source": pretraining,
    }
