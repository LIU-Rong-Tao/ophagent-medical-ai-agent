from pathlib import Path

import yaml

from scripts.training.train_eyeclip_aptos import _layer_id, parameter_groups


def test_eyeclip_layer_decay_uses_actual_clip_visual_hierarchy() -> None:
    assert _layer_id("visual.conv1.weight", 12) == 0
    assert _layer_id("visual.transformer.resblocks.0.attn.in_proj_weight", 12) == 1
    assert _layer_id("visual.transformer.resblocks.11.ln_2.weight", 12) == 12
    assert _layer_id("visual.ln_post.weight", 12) == 13
    assert _layer_id("head.weight", 12) == 13


def test_eyeclip_protocol_records_official_repairs_and_test_isolation() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/model_hub/training_protocols"
        / "eyeclip_aptos_official_recipe_repaired_v1.yaml"
    )
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert protocol["training"]["seeds"] == [0, 1, 2, 3, 4]
    assert protocol["optimizer"]["learning_rate"] == 1e-4
    assert protocol["evaluation"]["selection_split"] == "val"
    assert protocol["evaluation"]["save_best_by"] == "macro_auc_ovr"
    assert protocol["evaluation"]["test_once_after_selection"] is True
    assert protocol["repairs"]["readme_entrypoint_typo"] is True
    assert protocol["repairs"]["per_epoch_test_evaluation_removed"] is True


def test_parameter_groups_cover_every_parameter_once() -> None:
    import torch

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 4, 2)
            self.class_embedding = torch.nn.Parameter(torch.zeros(4))
            self.positional_embedding = torch.nn.Parameter(torch.zeros(5, 4))
            self.ln_pre = torch.nn.LayerNorm(4)
            self.transformer = torch.nn.Module()
            self.transformer.resblocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
            )
            self.ln_post = torch.nn.LayerNorm(4)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()
            self.head = torch.nn.Linear(4, 5)

    model = Model()
    groups = parameter_groups(model, 0.01, 0.75)
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert len(grouped) == len(list(model.parameters()))
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
