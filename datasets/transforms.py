import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(image_size: int, augment_cfg: dict | None = None, is_train: bool = True):
    augment_cfg = augment_cfg or {}
    transforms = []

    if is_train:
        if augment_cfg.get("hflip", True):
            transforms.append(A.HorizontalFlip(p=0.5))
        if augment_cfg.get("vflip", True):
            transforms.append(A.VerticalFlip(p=0.5))
        if augment_cfg.get("rotate90", True):
            transforms.append(A.RandomRotate90(p=0.5))

    transforms.extend(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms, is_check_shapes=False)
