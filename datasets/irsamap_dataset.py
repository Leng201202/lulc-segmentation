from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.label_maps import IGNORE_INDEX, IRSAMAP_RGB_TO_CLASS
from datasets.transforms import build_transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_image_files(directory: Path | None) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rgb_mask_to_class_indices(mask_rgb: np.ndarray) -> np.ndarray:
    class_map = np.full(mask_rgb.shape[:2], IGNORE_INDEX, dtype=np.int64)
    rgb = mask_rgb[..., :3].astype(np.int32)

    for color, class_id in IRSAMAP_RGB_TO_CLASS.items():
        match = np.all(rgb == np.array(color, dtype=np.int32), axis=-1)
        class_map[match] = class_id

    return class_map


class IRSAMapDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int = 512,
        augment_cfg: dict | None = None,
        is_train: bool = True,
    ):
        self.image_paths = list_image_files(image_dir)
        self.mask_paths = []
        for image_path in self.image_paths:
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            self.mask_paths.append(mask_path)

        if not self.image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

        self.transform = build_transforms(image_size, augment_cfg, is_train=is_train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_rgb = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_rgb is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask = rgb_mask_to_class_indices(mask_rgb)

        transformed = self.transform(image=image, mask=mask)
        return {
            "image": transformed["image"],
            "mask": torch.as_tensor(transformed["mask"], dtype=torch.long),
            "image_path": str(image_path),
            "dataset": "irsamap",
        }
