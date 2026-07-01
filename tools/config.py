from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(root: Path, relative: str | None) -> Path | None:
    if not relative:
        return None
    path = Path(relative)
    if path.is_absolute():
        return path
    return root / path


def get_data_paths(config: dict, project_root: Path) -> dict:
    data_cfg = config["data"]
    data_root = resolve_path(project_root, data_cfg["root"])

    def pair(image_key: str, mask_key: str) -> tuple[Path | None, Path | None]:
        image_dir = resolve_path(data_root, data_cfg[image_key])
        mask_dir = resolve_path(data_root, data_cfg[mask_key])
        return image_dir, mask_dir

    train_images, train_masks = pair("train_images", "train_masks")
    val_images, val_masks = pair("val_image_dir", "val_mask_dir")
    test_images, test_masks = pair("test_image_dir", "test_mask_dir")

    return {
        "root": data_root,
        "train_images": train_images,
        "train_masks": train_masks,
        "val_images": val_images,
        "val_masks": val_masks,
        "test_images": test_images,
        "test_masks": test_masks,
    }
