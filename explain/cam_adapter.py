"""
CAM adapter for OphAgent backbones.

This module provides backbone-specific target layers and reshape transforms
for Grad-CAM style methods.

Supported backbones:
- ConvNeXt-Tiny
- Swin-Tiny
- ViT-B/16
- ViT-L/16
- RETFound-MAE-CFP
"""

from typing import Callable, List, Optional, Tuple

import torch


def get_vit_grid_size(model) -> Tuple[int, int]:
    """
    Get ViT patch grid size from timm patch_embed.

    For 224x224 image and patch size 16:
        grid_size = (14, 14)

    Falls back to (14, 14) for current OphAgent 224/16 setting.
    """

    patch_embed = getattr(model, "patch_embed", None)
    grid_size = getattr(patch_embed, "grid_size", None)

    if grid_size is None:
        return 14, 14

    if isinstance(grid_size, tuple):
        return int(grid_size[0]), int(grid_size[1])

    return int(grid_size), int(grid_size)


def make_vit_reshape_transform(height: int, width: int) -> Callable:
    """
    Build reshape transform for ViT / RETFound token outputs.

    Input tensor shape:
        [B, num_tokens, C]

    Output tensor shape:
        [B, C, H, W]
    """

    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected ViT token tensor with shape [B, N, C], "
                f"got {tuple(tensor.shape)}"
            )

        expected_tokens = height * width

        # Remove cls token if present.
        if tensor.shape[1] == expected_tokens + 1:
            tensor = tensor[:, 1:, :]

        if tensor.shape[1] != expected_tokens:
            raise ValueError(
                f"ViT token count mismatch: got {tensor.shape[1]}, "
                f"expected {expected_tokens} or {expected_tokens + 1}"
            )

        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2).contiguous()

        return result

    return reshape_transform


def make_swin_reshape_transform(height: int = 7, width: int = 7) -> Callable:
    """
    Build reshape transform for Swin token outputs.

    timm Swin may return:
    - [B, H, W, C]
    - [B, N, C]

    Output tensor shape:
        [B, C, H, W]
    """

    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            # [B, H, W, C] -> [B, C, H, W]
            return tensor.permute(0, 3, 1, 2).contiguous()

        if tensor.ndim == 3:
            expected_tokens = height * width

            if tensor.shape[1] != expected_tokens:
                raise ValueError(
                    f"Swin token count mismatch: got {tensor.shape[1]}, "
                    f"expected {expected_tokens}"
                )

            result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
            result = result.permute(0, 3, 1, 2).contiguous()

            return result

        raise ValueError(
            f"Unsupported Swin tensor shape: {tuple(tensor.shape)}"
        )

    return reshape_transform


def get_transformer_target_block(model, target_layer: str):
    """
    Select target transformer block for ViT / RETFound.

    Supported target_layer:
    - auto / late: second last block
    - middle: middle block
    - early: quarter-depth block
    - block<N>: explicit block index, e.g. block18
    """

    target_layer = target_layer.lower()
    blocks = model.blocks
    n_blocks = len(blocks)

    if target_layer in ["auto", "late"]:
        index = max(n_blocks - 2, 0)

    elif target_layer == "middle":
        index = n_blocks // 2

    elif target_layer == "early":
        index = n_blocks // 4

    elif target_layer.startswith("block"):
        index_text = target_layer.replace("block", "")

        if not index_text.isdigit():
            raise ValueError(
                f"Invalid transformer target_layer: {target_layer}. "
                "Use block<N>, for example block11 or block22."
            )

        index = int(index_text)

        if index < 0 or index >= n_blocks:
            raise ValueError(
                f"Invalid block index {index}; model has {n_blocks} blocks."
            )

    else:
        raise ValueError(
            f"Unsupported transformer target_layer: {target_layer}. "
            "Choose from: auto, early, middle, late, block<N>."
        )

    return blocks[index].norm1


def get_swin_target_layer(model, target_layer: str):
    """
    Select target layer for Swin.

    Supported target_layer:
    - auto / late: final Swin stage
    - middle: third Swin stage
    - early: second Swin stage
    """

    target_layer = target_layer.lower()
    layers = model.layers
    n_layers = len(layers)

    if target_layer in ["auto", "late"]:
        layer_index = n_layers - 1

    elif target_layer == "middle":
        layer_index = min(2, n_layers - 1)

    elif target_layer == "early":
        layer_index = min(1, n_layers - 1)

    else:
        raise ValueError(
            f"Unsupported Swin target_layer: {target_layer}. "
            "Choose from: auto, early, middle, late."
        )

    return layers[layer_index].blocks[-1].norm1


def get_cam_target_layers(
    model,
    backbone: str,
    target_layer: str = "auto",
) -> List[torch.nn.Module]:
    """
    Return CAM target layers according to backbone.

    ConvNeXt:
        supports auto / stage2 / stage3 / stage4

    Swin:
        supports auto / early / middle / late

    ViT / RETFound:
        supports auto / early / middle / late / block<N>
    """

    backbone = backbone.lower()
    target_layer = target_layer.lower()

    if backbone == "convnext_tiny":
        if target_layer in ["auto", "stage3"]:
            return [model.stages[2].blocks[-1].conv_dw]

        if target_layer == "stage4":
            return [model.stages[-1].blocks[-1].conv_dw]

        if target_layer == "stage2":
            return [model.stages[1].blocks[-1].conv_dw]

        raise ValueError(
            f"Unsupported ConvNeXt target_layer: {target_layer}. "
            "Choose from: auto, stage4, stage3, stage2."
        )

    if backbone in [
        "swin_tiny",
        "swin_tiny_patch4_window7_224",
        "swin_tiny_patch4_window7_224.ms_in1k",
    ]:
        return [get_swin_target_layer(model, target_layer)]

    if backbone in [
        "vit_base_patch16",
        "vit_large_patch16",
        "retfound_mae_cfp",
    ]:
        return [get_transformer_target_block(model, target_layer)]

    raise ValueError(f"Unsupported backbone for CAM target layer: {backbone}")


def get_cam_reshape_transform(
    model,
    backbone: str,
) -> Optional[Callable]:
    """
    Return reshape_transform for transformer backbones.

    ConvNeXt outputs feature maps directly, so no reshape is needed.
    """

    backbone = backbone.lower()

    if backbone == "convnext_tiny":
        return None

    if backbone in [
        "swin_tiny",
        "swin_tiny_patch4_window7_224",
        "swin_tiny_patch4_window7_224.ms_in1k",
    ]:
        return make_swin_reshape_transform(height=7, width=7)

    if backbone in [
        "vit_base_patch16",
        "vit_large_patch16",
        "retfound_mae_cfp",
    ]:
        height, width = get_vit_grid_size(model)
        return make_vit_reshape_transform(height=height, width=width)

    raise ValueError(f"Unsupported backbone for CAM reshape transform: {backbone}")