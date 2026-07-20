"""EyeCLIP APTOS five-class task inference adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

from scripts.training.aptos_downstream_common import APTOS_LABELS


EYECLIP_SOURCE_COMMIT = "2fcf6034552e6006c94bd84cbdc6f4a5897b29c0"
EYECLIP_CHECKPOINT_SHA256 = "dfb6990aa31d55e6eee23f5b79d4b170d86af9d0f533fedaa805f0cdb5d686d4"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_eyeclip_foundation(source_root: Path | str, checkpoint_path: Path | str, device: str = "cpu"):
    import torch

    source = Path(source_root)
    checkpoint = Path(checkpoint_path)
    if not (source / "eyeclip/model.py").is_file():
        raise FileNotFoundError("EyeCLIP 固定源码不完整")
    if sha256_file(checkpoint) != EYECLIP_CHECKPOINT_SHA256:
        raise ValueError("EyeCLIP checkpoint SHA256 不匹配")
    for name in tuple(sys.modules):
        if name == "eyeclip" or name.startswith("eyeclip."):
            del sys.modules[name]
    sys.path.insert(0, str(source))
    from eyeclip.model import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError("EyeCLIP checkpoint 缺少 model_state_dict")
    model = build_model(dict(payload["model_state_dict"])).float()
    visual = model.visual
    if (
        visual.input_resolution != 224
        or visual.conv1.kernel_size != (32, 32)
        or visual.output_dim != 512
        or len(visual.transformer.resblocks) != 12
    ):
        raise ValueError("EyeCLIP checkpoint 不是已登记的 ViT-B/32 视觉编码器")
    return visual.to(device)


class EyeClipClassifier:
    def __new__(cls, visual, num_classes: int = 5):
        import torch

        class _Module(torch.nn.Module):
            def __init__(self, encoder, classes):
                super().__init__()
                self.visual = encoder
                self.head = torch.nn.Linear(encoder.output_dim, classes)

            def forward(self, images):
                return self.head(self.visual(images))

        return _Module(visual, num_classes)


class EyeClipAptosTaskAdapter:
    def __init__(self, model, device: str):
        from torchvision.transforms import v2 as transforms

        self.model = model.eval()
        self.device = device
        self.labels = APTOS_LABELS
        self.preprocess = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToDtype(__import__("torch").float32, scale=True),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @classmethod
    def load(cls, *, source_root, encoder_checkpoint, task_checkpoint, device="cpu"):
        import torch

        payload = torch.load(task_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("encoder_checkpoint_sha256") != sha256_file(encoder_checkpoint):
            raise ValueError("EyeCLIP 基础 checkpoint 与任务 checkpoint 记录不一致")
        visual = load_eyeclip_foundation(source_root, encoder_checkpoint, "cpu")
        visual.load_state_dict(payload["visual_state_dict"], strict=True)
        model = EyeClipClassifier(visual, len(APTOS_LABELS))
        model.head.load_state_dict(payload["classifier_state_dict"])
        return cls(model.to(device), device)

    def predict_proba(self, images: Iterable[object]) -> np.ndarray:
        import torch

        tensors = [self.preprocess(image) for image in images]
        if not tensors:
            return np.empty((0, len(self.labels)), dtype=float)
        with torch.inference_mode():
            logits = self.model(torch.stack(tensors).to(self.device))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
            raise ValueError("EyeCLIP 任务头返回非法概率")
        return probabilities
