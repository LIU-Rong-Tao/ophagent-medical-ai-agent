from pathlib import Path

import torch
import timm


def build_retfound_mae_cfp_model(
    num_classes: int,
    checkpoint_path: str,
    drop_path_rate: float = 0.0,
):
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (
            k.startswith("decoder_")
            or k.startswith("mask_token")
            or k.startswith("decoder")
            or k.startswith("head.")
        )
    }

    msg = model.load_state_dict(state_dict, strict=False)

    print("Loaded RETFound-MAE-CFP checkpoint:")
    print(f"  checkpoint: {Path(checkpoint_path)}")
    print(f"  missing keys: {len(msg.missing_keys)}")
    print(f"  unexpected keys: {len(msg.unexpected_keys)}")

    return model
