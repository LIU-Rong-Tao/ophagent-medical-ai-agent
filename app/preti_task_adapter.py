"""PRETI image encoder and APTOS five-class task inference adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types
from typing import Iterable

import numpy as np


LABELS = (
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
)
PRETI_SOURCE_COMMIT = "2ac2b0f123d69151877ebd44a33edfb026cdac45"
ENCODER_PREFIXES = (
    "age_embedding",
    "blocks.",
    "cls_token",
    "gender_embedding",
    "norm.",
    "patch_embed.",
    "pos_embed",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_preti_foundation(
    *,
    source_root: Path | str,
    checkpoint_path: Path | str,
    device: str = "cpu",
):
    """Load the pinned official PRETI ViT-B encoder without its unused VGG loss."""

    import torch

    source = Path(source_root)
    checkpoint = Path(checkpoint_path)
    if not (source / "models/PRETI_model.py").is_file():
        raise FileNotFoundError("PRETI 固定源码不完整")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PRETI checkpoint 不存在：{checkpoint}")

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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("PRETI checkpoint 缺少 model state_dict")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    allowed_unexpected = [
        key for key in unexpected if key.startswith("perceptual_loss_fn.")
    ]
    if missing or len(allowed_unexpected) != len(unexpected):
        raise ValueError(
            f"PRETI 权重契约不匹配：missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return model.to(device)


def encoder_state_dict(model) -> dict[str, object]:
    """Return only parameters used by forward_encoder_no_masking."""

    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(ENCODER_PREFIXES)
    }


def set_encoder_trainable(model) -> list[object]:
    parameters = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(ENCODER_PREFIXES)
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(parameter)
    if not parameters:
        raise ValueError("PRETI encoder 没有可训练参数")
    return parameters


class PretiClassifier:
    def __new__(cls, foundation, num_classes: int = 5):
        import torch

        class _Module(torch.nn.Module):
            def __init__(self, encoder, classes):
                super().__init__()
                self.encoder = encoder
                self.head = torch.nn.Linear(768, classes)

            def forward(self, images):
                tokens, _, _ = self.encoder.forward_encoder_no_masking(images)
                return self.head(tokens[:, 0])

        return _Module(foundation, num_classes)


class PretiAptosTaskAdapter:
    def __init__(self, model, *, device: str):
        import torch
        from torchvision.transforms import v2 as transforms

        self.model = model.eval()
        self.device = device
        self.labels = LABELS
        self.preprocess = transforms.Compose(
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

    @classmethod
    def load(
        cls,
        *,
        source_root: Path | str,
        encoder_checkpoint: Path | str,
        task_checkpoint: Path | str,
        device: str = "cpu",
    ):
        import torch

        payload = torch.load(task_checkpoint, map_location="cpu", weights_only=False)
        expected_sha = str(payload.get("encoder_checkpoint_sha256", ""))
        actual_sha = sha256_file(encoder_checkpoint)
        if expected_sha != actual_sha:
            raise ValueError("PRETI 基础 checkpoint SHA256 与任务头记录不一致")
        foundation = load_preti_foundation(
            source_root=source_root,
            checkpoint_path=encoder_checkpoint,
            device="cpu",
        )
        unexpected = foundation.load_state_dict(
            payload["encoder_state_dict"], strict=False
        ).unexpected_keys
        if unexpected:
            raise ValueError(f"PRETI 任务 encoder 含未知键：{unexpected[:5]}")
        model = PretiClassifier(foundation, num_classes=len(LABELS))
        model.head.load_state_dict(payload["classifier_state_dict"])
        return cls(model.to(device), device=device)

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        import torch

        tensors = [self.preprocess(image) for image in images]
        if not tensors:
            return np.empty((0, len(self.labels)), dtype=float)
        with torch.inference_mode():
            logits = self.model(torch.stack(tensors).to(self.device))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        if probabilities.shape[1] != len(self.labels):
            raise ValueError("PRETI 任务头概率列与标签空间不一致")
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-6
        ):
            raise ValueError("PRETI 任务头返回了非法概率")
        return probabilities
