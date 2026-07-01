from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.irsamap_dataset import list_image_files
from datasets.label_maps import IGNORE_INDEX, LOVEDA_MAPPINGS
from datasets.transforms import build_transforms


def remap_loveda_mask(mask: np.ndarray, mapping_name: str = "targeted") -> np.ndarray:
    mapping = LOVEDA_MAPPINGS[mapping_name]
    remapped = np.full(mask.shape, IGNORE_INDEX, dtype=np.int64)
    for src_value, dst_value in mapping.items():
        remapped[mask == src_value] = dst_value
    return remapped


class LoveDADataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int = 512,
        augment_cfg: dict | None = None,
        is_train: bool = True,
        mapping_name: str = "targeted",
    ):
        self.image_paths = list_image_files(image_dir)
        self.mask_paths = []
        for image_path in self.image_paths:
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing LoveDA mask for {image_path.name}: {mask_path}")
            self.mask_paths.append(mask_path)

        if not self.image_paths:
            raise RuntimeError(f"No LoveDA images found in {image_dir}")

        self.mapping_name = mapping_name
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

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = remap_loveda_mask(mask.astype(np.int64), self.mapping_name)

        transformed = self.transform(image=image, mask=mask)
        return {
            "image": transformed["image"],
            "mask": torch.as_tensor(transformed["mask"], dtype=torch.long),
            "image_path": str(image_path),
            "dataset": "loveda",
        }
