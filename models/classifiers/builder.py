import torch
import timm


def build_model(config: dict, checkpoint_path: str | None = None, device=None):
    backbone = config["backbone"]
    num_classes = config["num_classes"]

    if backbone in [
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
    ]:
        model = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=num_classes,
        )

    elif backbone == "retfound":
        raise NotImplementedError(
            "RETFound is not implemented yet. "
            "Add RETFound model loading in models/classifiers/retfound.py later."
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

    model.eval()
    return model