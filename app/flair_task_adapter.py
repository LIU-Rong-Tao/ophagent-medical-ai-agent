"""FLAIR frozen-encoder APTOS linear-probe inference adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from scripts.training.aptos_downstream_common import APTOS_LABELS


FLAIR_SOURCE_COMMIT = "d6652d53389ff49e5f73efaccf4246e9de88d1a3"
FLAIR_CHECKPOINT_SHA256 = "050334cd934fa3126435c202c41f493ca222fc3fe3ea9c50cf21f67c792a1440"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FlairVisionEncoder:
    def __new__(cls):
        import torch
        import torchvision

        class _Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(2048, 512, bias=False)

            def forward(self, features):
                features = torch.nn.functional.normalize(features, dim=-1)
                return torch.nn.functional.normalize(self.projection(features), dim=-1)

        class _Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet50(weights=None)
                self.model.fc = torch.nn.Identity()
                self.projection_head_vision = _Projection()

            def forward(self, images):
                return self.projection_head_vision(self.model(images))

        return _Encoder()


def load_flair_vision(checkpoint_path: Path | str, device: str = "cpu"):
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint_path)
    if sha256_file(checkpoint) != FLAIR_CHECKPOINT_SHA256:
        raise ValueError("FLAIR checkpoint SHA256 不匹配")
    state = {
        key.removeprefix("vision_model."): value
        for key, value in load_file(checkpoint).items()
        if key.startswith("vision_model.")
    }
    model = FlairVisionEncoder()
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def preprocess_flair_image(image):
    import torch
    from skimage.filters import threshold_li
    from skimage.measure import label, regionprops
    from torchvision.transforms import v2 as transforms

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    channel_first = np.transpose(array, (2, 0, 1))
    binary = channel_first[0] > threshold_li(channel_first[0])
    labels = label(binary)
    counts = np.bincount(labels.flat, weights=binary.flat)
    largest = labels == np.argmax(counts)
    regions = regionprops(label(largest))
    if regions:
        top, left, bottom, right = regions[0].bbox
        channel_first = channel_first[:, top:bottom, left:right]
    tensor = torch.from_numpy(channel_first)
    height, width = tensor.shape[-2:]
    scale = max(height, width) / 512
    resized = transforms.Resize(
        (max(1, int(height / scale)), max(1, int(width / scale))), antialias=True
    )(tensor)
    return torch.nn.functional.pad(
        resized,
        (0, 512 - resized.shape[-1], 0, 512 - resized.shape[-2]),
    )


class FlairAptosTaskAdapter:
    def __init__(self, encoder, classifier, device: str):
        self.encoder = encoder.eval()
        self.classifier = classifier.eval()
        self.device = device
        self.labels = APTOS_LABELS

    @classmethod
    def load(cls, *, encoder_checkpoint, task_checkpoint, device="cpu"):
        import torch

        payload = torch.load(task_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("encoder_checkpoint_sha256") != sha256_file(encoder_checkpoint):
            raise ValueError("FLAIR 基础 checkpoint 与任务 checkpoint 记录不一致")
        encoder = load_flair_vision(encoder_checkpoint, device)
        classifier = torch.nn.Linear(512, len(APTOS_LABELS))
        classifier.load_state_dict(payload["classifier_state_dict"], strict=True)
        return cls(encoder, classifier.to(device), device)

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        import torch

        tensors = [preprocess_flair_image(image) for image in images]
        if not tensors:
            return np.empty((0, len(self.labels)), dtype=float)
        with torch.inference_mode():
            features = self.encoder(torch.stack(tensors).to(self.device))
            probabilities = torch.softmax(self.classifier(features).float(), dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
            raise ValueError("FLAIR 任务头返回非法概率")
        return probabilities
