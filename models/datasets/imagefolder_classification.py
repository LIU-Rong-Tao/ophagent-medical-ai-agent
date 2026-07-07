"""通用 ImageFolder 分类数据集预检与 DataLoader。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetInspection:
    class_to_idx: dict[str, int]
    split_sizes: dict[str, int]


def _split_classes(split_dir: Path) -> tuple[list[str], int]:
    if not split_dir.is_dir():
        raise ValueError(f"数据集缺少 split 目录：{split_dir}")
    classes = []
    n_images = 0
    for directory in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        images = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        if images:
            classes.append(directory.name)
            n_images += len(images)
    if not classes:
        raise ValueError(f"split 中没有可用类别图像：{split_dir}")
    return classes, n_images


def inspect_imagefolder_dataset(
    root: Path | str,
    required_splits: tuple[str, ...] = ("train", "val", "test"),
) -> DatasetInspection:
    data_root = Path(root)
    classes_by_split: dict[str, list[str]] = {}
    split_sizes: dict[str, int] = {}
    for split in required_splits:
        classes, n_images = _split_classes(data_root / split)
        classes_by_split[split] = classes
        split_sizes[split] = n_images
    reference = classes_by_split[required_splits[0]]
    mismatched = {split: classes for split, classes in classes_by_split.items() if classes != reference}
    if mismatched:
        raise ValueError(f"train/val/test 类别目录不一致：{mismatched}")
    return DatasetInspection(
        class_to_idx={class_name: index for index, class_name in enumerate(reference)},
        split_sizes=split_sizes,
    )


def should_pin_memory(requested_device: str, *, cuda_available: bool) -> bool:
    if requested_device == "cpu":
        return False
    if requested_device == "auto":
        return bool(cuda_available)
    return requested_device.startswith("cuda") and bool(cuda_available)


def build_imagefolder_loaders(config: Any):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    image_size = int(config.image_size)
    train_steps = []
    if bool(getattr(config, "random_resized_crop", False)):
        train_steps.append(transforms.RandomResizedCrop(image_size))
    else:
        train_steps.append(transforms.Resize((image_size, image_size)))
    flip_probability = float(getattr(config, "horizontal_flip_probability", 0.5))
    if flip_probability > 0:
        train_steps.append(transforms.RandomHorizontalFlip(p=flip_probability))
    rotation_degrees = float(getattr(config, "rotation_degrees", 10.0))
    if rotation_degrees > 0:
        train_steps.append(transforms.RandomRotation(rotation_degrees))
    color_jitter = float(getattr(config, "color_jitter", 0.0))
    if color_jitter > 0:
        train_steps.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    datasets_by_split = {
        "train": datasets.ImageFolder(Path(config.data_root) / "train", transform=train_transform),
        "val": datasets.ImageFolder(Path(config.data_root) / "val", transform=eval_transform),
        "test": datasets.ImageFolder(Path(config.data_root) / "test", transform=eval_transform),
    }
    pin_memory = should_pin_memory(str(config.device), cuda_available=torch.cuda.is_available())
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            shuffle=split == "train",
            num_workers=int(config.num_workers),
            pin_memory=pin_memory,
        )
        for split, dataset in datasets_by_split.items()
    }
    return datasets_by_split, loaders
