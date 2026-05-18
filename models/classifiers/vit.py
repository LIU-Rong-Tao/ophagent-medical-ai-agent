import timm


def build_vit_model(
    num_classes: int,
    pretrained: bool = True,
    drop_path_rate: float = 0.0,
):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    return model