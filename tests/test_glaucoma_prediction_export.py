from pathlib import Path

import torch

from scripts.v083_glaucoma import export_glaucoma_scout_predictions as exporter


def test_structured_training_config_builds_exact_timm_architecture(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pth"
    expected_model = torch.nn.Linear(2, 3)
    torch.save(expected_model.state_dict(), checkpoint)
    calls: list[tuple[str, bool, int]] = []

    def create_model(architecture: str, *, pretrained: bool, num_classes: int):
        calls.append((architecture, pretrained, num_classes))
        return torch.nn.Linear(2, num_classes)

    monkeypatch.setattr(exporter.timm, "create_model", create_model)
    config = {
        "model": {"architecture": "convnext_tiny"},
        "training": {"image_size": 224},
        "data": {"root": "/dataset", "num_classes": 3},
    }

    model = exporter.build_inference_model(config, checkpoint, torch.device("cpu"))

    assert calls == [("convnext_tiny", False, 3)]
    assert model.training is False
    assert exporter.structured_config_value(config, "data", "root", "data_root") == "/dataset"
    assert exporter.structured_config_value(config, "training", "image_size", "image_size") == 224
