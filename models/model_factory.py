from models.unetformer import UNetFormer


def build_model(config: dict):
    model_cfg = config["model"]
    num_classes = config["data"]["num_classes"]
    model_name = model_cfg.get("name", "unetformer").lower()

    if model_name == "unetformer":
        return UNetFormer(
            decode_channels=model_cfg.get("decode_channels", 64),
            dropout=model_cfg.get("dropout", 0.1),
            backbone_name=model_cfg.get("backbone", "swsl_resnet18"),
            pretrained=model_cfg.get("pretrained", True),
            window_size=model_cfg.get("window_size", 8),
            num_classes=num_classes,
        )

    raise NotImplementedError(
        f"Model '{model_name}' is not implemented yet. "
        "Currently supported: unetformer"
    )
