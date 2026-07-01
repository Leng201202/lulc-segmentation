#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare raw IRSAMap dataset into plural 'images' and 'masks_rgb' folders."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the raw IRSAMap directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Path where the processed/restructured IRSAMap dataset will be saved.",
    )
    parser.add_argument(
        "--mask-folder",
        type=str,
        default="SegLabel_rvwsb",
        choices=["SegLabel_rvwsb", "SegLabel_vwsbr"],
        help="Name of the source mask subfolder to use (e.g. SegLabel_rvwsb or SegLabel_vwsbr).",
    )
    return parser.parse_args()


def prepare_split(src_split: Path, dst_split: Path, mask_folder_name: str):
    src_image_dir = src_split / "image"
    src_mask_dir = src_split / mask_folder_name

    if not src_image_dir.exists():
        print(f"Warning: Image directory {src_image_dir} does not exist. Skipping split {src_split.name}.")
        return

    if not src_mask_dir.exists():
        print(f"Warning: Mask directory {src_mask_dir} does not exist. Skipping split {src_split.name}.")
        return

    dst_image_dir = dst_split / "images"
    dst_mask_dir = dst_split / "masks_rgb"

    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing split '{src_split.name}': {src_image_dir} -> {dst_split}")

    # Process all image files
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_count = 0
    mask_count = 0

    for path in src_image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            # Copy image
            shutil.copy2(path, dst_image_dir / path.name)
            img_count += 1

            # Copy corresponding mask
            mask_path = src_mask_dir / path.name
            if mask_path.exists():
                shutil.copy2(mask_path, dst_mask_dir / path.name)
                mask_count += 1
            else:
                print(f"Warning: Missing mask for {path.name} in {src_mask_dir}")

    print(f"  Done. Copied {img_count} images and {mask_count} masks.")


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    for split in ["train", "val", "test"]:
        prepare_split(src_root / split, dst_root / split, args.mask_folder)


if __name__ == "__main__":
    main()
