import torch
import timm

from models.classifiers.vit import build_vit_model
from models.classifiers.retfound import build_retfound_mae_cfp_model


def build_model(
    config: dict,
    checkpoint_path: str | None = None,
    device=None,
    training: bool = False,
):
    backbone = config["backbone"]
    num_classes = config["num_classes"]
    pretrained = config.get("pretrained", False)
    drop_path = config.get("drop_path", 0.0)

    if backbone in [
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
        "swin_tiny_patch4_window7_224.ms_in1k",
    ]:
        model = timm.create_model(
            backbone,
            pretrained=pretrained if training else False,
            num_classes=num_classes,
        )

    elif backbone == "vit_base_patch16":
        model = build_vit_model(
            num_classes=num_classes,
            pretrained=pretrained if training else False,
            drop_path_rate=drop_path,
        )

    elif backbone == "vit_large_patch16":
        model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=pretrained if training else False,
            num_classes=num_classes,
            drop_path_rate=drop_path,
        )

    elif backbone == "retfound_mae_cfp":
        model = build_retfound_mae_cfp_model(
            num_classes=num_classes,
            checkpoint_path=config["retfound_checkpoint"],
            drop_path_rate=drop_path,
        )

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint)

    if device is not None:
        model.to(device)

    if training:
        model.train()
    else:
        model.eval()

    return model