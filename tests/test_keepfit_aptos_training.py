from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from app.keepfit_task_adapter import KEEPFIT_CHECKPOINTS, preprocess_keepfit_image


def test_keepfit_preprocessing_returns_official_canvas_shape() -> None:
    image = Image.fromarray(np.full((300, 500, 3), 127, dtype=np.uint8))
    tensor = preprocess_keepfit_image(image)
    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor[:, 308:] == 0)


def test_keepfit_protocol_preserves_official_linear_probe_defaults() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/model_hub/training_protocols"
        / "keepfit_aptos_official_lp_project_v1.yaml"
    )
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert protocol["foundation"]["project_features"] is False
    assert protocol["preprocessing"]["crop_foreground"] is False
    assert protocol["classifier"]["c"] == 0.316
    assert protocol["evaluation"]["selection_split"] == "none_fixed_official_c"
    assert protocol["claim"]["official_aptos_recipe_available"] is False


def test_only_cfp_keepfit_checkpoints_are_aptos_candidates() -> None:
    assert set(KEEPFIT_CHECKPOINTS) == {
        "keepfit-flair-mmretinal-cfp",
        "keepfit-half-flair-mmretinal-cfp",
    }
    assert KEEPFIT_CHECKPOINTS["keepfit-half-flair-mmretinal-cfp"]["display_name"].endswith(
        "（消融）"
    )
