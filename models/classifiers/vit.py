import timm


def build_vit_model(
    num_classes: int,
    pretrained: bool = True,
):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model