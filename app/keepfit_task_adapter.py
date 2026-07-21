"""KeepFIT frozen-encoder APTOS linear-probe inference adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from scripts.training.aptos_downstream_common import APTOS_LABELS


KEEPFIT_SOURCE_COMMIT = "dbbb1f05b9d27278b01e15e5f837b44b22d32cee"
KEEPFIT_CHECKPOINTS = {
    "keepfit-flair-mmretinal-cfp": {
        "sha256": "500904a3cae65f813c74ad9b87c2305c7b375c899fd064fc548c6b1f0e7104c9",
        "artifact_id": "aptos2019-keepfit-flair-mmretinal-cfp-official-lp-project-v1",
        "display_name": "KeepFIT flair+MM CFP",
    },
    "keepfit-half-flair-mmretinal-cfp": {
        "sha256": "12e7fd11f9572f63332c8a3c6e00786d670df02258ee9d5c50a8a311bb70345c",
        "artifact_id": "aptos2019-keepfit-half-flair-mmretinal-cfp-official-lp-project-v1",
        "display_name": "KeepFIT 50% flair+MM CFP（消融）",
    },
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_keepfit_backbone():
    import torch
    import torchvision

    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Identity()
    return model


def load_keepfit_vision(
    checkpoint_path: Path | str,
    checkpoint_id: str,
    device: str = "cpu",
):
    import torch

    spec = KEEPFIT_CHECKPOINTS.get(checkpoint_id)
    if spec is None:
        raise ValueError(f"不支持的 KeepFIT CFP checkpoint：{checkpoint_id}")
    checkpoint = Path(checkpoint_path)
    if sha256_file(checkpoint) != spec["sha256"]:
        raise ValueError("KeepFIT checkpoint SHA256 不匹配")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "vision_model.model."
    backbone_state = {
        key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)
    }
    model = build_keepfit_backbone()
    model.load_state_dict(backbone_state, strict=True)
    return model.eval().to(device)


def preprocess_keepfit_image(image):
    import torch
    from torchvision.transforms import Resize

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))
    height, width = tensor.shape[-2:]
    if height == width:
        return Resize((512, 512), antialias=True)(tensor)
    scale = max(height, width) / 512
    resized = Resize(
        (max(1, int(height / scale)), max(1, int(width / scale))), antialias=True
    )(tensor)
    return torch.nn.functional.pad(
        resized,
        (0, 512 - resized.shape[-1], 0, 512 - resized.shape[-2]),
    )


class KeepFITAptosTaskAdapter:
    def __init__(self, encoder, classifier, device: str, checkpoint_id: str):
        self.encoder = encoder.eval()
        self.classifier = classifier.eval()
        self.device = device
        self.checkpoint_id = checkpoint_id
        self.labels = APTOS_LABELS

    @classmethod
    def load(cls, *, encoder_checkpoint, task_checkpoint, device="cpu"):
        import torch

        payload = torch.load(task_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_id = str(payload["base_checkpoint_id"])
        if payload.get("encoder_checkpoint_sha256") != sha256_file(encoder_checkpoint):
            raise ValueError("KeepFIT 基础 checkpoint 与任务 checkpoint 记录不一致")
        encoder = load_keepfit_vision(encoder_checkpoint, checkpoint_id, device)
        classifier = torch.nn.Linear(2048, len(APTOS_LABELS))
        classifier.load_state_dict(payload["classifier_state_dict"], strict=True)
        return cls(encoder, classifier.to(device), device, checkpoint_id)

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        import torch

        tensors = [preprocess_keepfit_image(image) for image in images]
        if not tensors:
            return np.empty((0, len(self.labels)), dtype=float)
        with torch.inference_mode():
            features = self.encoder(torch.stack(tensors).to(self.device))
            probabilities = torch.softmax(self.classifier(features).float(), dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(1), 1, atol=1e-6
        ):
            raise ValueError("KeepFIT 任务头返回非法概率")
        return probabilities
