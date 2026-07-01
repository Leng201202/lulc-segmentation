from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import build_transforms
from tools.palette import mask_to_class_indices


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_image_files(directory: Path) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    files = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return files


def split_train_val(
    image_dir: Path,
    mask_dir: Path,
    val_split: float,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    images = list_image_files(image_dir)
    masks = list_image_files(mask_dir)

    pairs = []
    mask_by_stem = {p.stem: p for p in masks}
    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}{image_path.suffix}"
        if not mask_path.exists():
            mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is not None and mask_path.exists():
            pairs.append((image_path, mask_path))

    if not pairs:
        return [], [], [], []

    val_count = max(1, int(len(pairs) * val_split)) if len(pairs) > 1 else 0
    train_pairs = pairs[:-val_count] if val_count else pairs
    val_pairs = pairs[-val_count:] if val_count else []

    train_images, train_masks = zip(*train_pairs) if train_pairs else ([], [])
    val_images, val_masks = zip(*val_pairs) if val_pairs else ([], [])
    return list(train_images), list(train_masks), list(val_images), list(val_masks)


class IRSAMapDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        image_size: int = 512,
        augment_cfg: dict | None = None,
        is_train: bool = True,
        label_encoding: str = "category_code",
    ):
        if len(image_paths) != len(mask_paths):
            raise ValueError("Image and mask lists must have the same length.")
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.label_encoding = label_encoding
        self.transform = build_transforms(image_size, augment_cfg, is_train=is_train)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = cv2.cvtColor(cv2.imread(str(self.image_paths[index])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_paths[index]), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {self.mask_paths[index]}")

        if mask.ndim == 2:
            mask = mask_to_class_indices(mask, encoding=self.label_encoding)
        else:
            mask = mask_to_class_indices(
                cv2.cvtColor(mask, cv2.COLOR_BGR2RGB),
                encoding="rgb",
            )

        transformed = self.transform(image=image, mask=mask.astype(np.int64))
        image_tensor = transformed["image"]
        mask_tensor = torch.as_tensor(transformed["mask"], dtype=torch.long)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(self.image_paths[index]),
        }


def build_dataloaders(config: dict, project_root: Path):
    from tools.config import get_data_paths

    data_cfg = config["data"]
    paths = get_data_paths(config, project_root)
    image_size = data_cfg.get("image_size", 512)
    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    augment_cfg = config.get("augment", {})

    train_images = list_image_files(paths["train_images"])
    train_masks_dir = paths["train_masks"]
    train_pairs = []
    for image_path in train_images:
        mask_path = train_masks_dir / f"{image_path.stem}{image_path.suffix}"
        if mask_path.exists():
            train_pairs.append((image_path, mask_path))

    val_images = list_image_files(paths["val_images"])
    val_masks_dir = paths["val_masks"]
    val_pairs = []
    for image_path in val_images:
        mask_path = val_masks_dir / f"{image_path.stem}{image_path.suffix}"
        if mask_path.exists():
            val_pairs.append((image_path, mask_path))

    if not val_pairs and train_pairs:
        split_images, split_masks, val_images, val_masks = split_train_val(
            paths["train_images"],
            paths["train_masks"],
            data_cfg.get("val_split", 0.2),
        )
        if val_images:
            train_pairs = list(zip(split_images, split_masks))
            val_pairs = list(zip(val_images, val_masks))

    if not train_pairs:
        raise RuntimeError(
            "No training samples found. Download IRSAMap from "
            "https://github.com/ucas-dlg/IRSAMap and run:\n"
            "  python tools/prepare_irsamap.py --source /path/to/download"
        )

    label_encoding = data_cfg.get("label_encoding", "category_code")

    train_dataset = IRSAMapDataset(
        [p[0] for p in train_pairs],
        [p[1] for p in train_pairs],
        image_size=image_size,
        augment_cfg=augment_cfg,
        is_train=True,
        label_encoding=label_encoding,
    )
    val_dataset = IRSAMapDataset(
        [p[0] for p in val_pairs],
        [p[1] for p in val_pairs],
        image_size=image_size,
        augment_cfg=augment_cfg,
        is_train=False,
        label_encoding=label_encoding,
    ) if val_pairs else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader
