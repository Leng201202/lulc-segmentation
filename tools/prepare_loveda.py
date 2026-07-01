#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare raw LoveDA dataset by merging Rural and Urban directories into flat train/val/test splits."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the raw LoveDA directory containing Train/Val/Test subfolders.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Path where the processed/restructured LoveDA dataset will be saved.",
    )
    parser.add_argument(
        "--mask-folder",
        type=str,
        default="masks_png",
        help="Name of the mask subfolder to copy (e.g. masks_png, masks_png_convert).",
    )
    return parser.parse_args()


def prepare_split(src_split_dir: Path, dst_split_dir: Path, mask_folder_name: str):
    if not src_split_dir.exists():
        print(f"Warning: Split folder {src_split_dir} does not exist. Skipping.")
        return

    dst_images_dir = dst_split_dir / "images"
    dst_masks_dir = dst_split_dir / "masks"

    dst_images_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    mask_count = 0

    # LoveDA has 'Rural' and 'Urban' subfolders in each split
    for area in ["Rural", "Urban"]:
        src_area_dir = src_split_dir / area
        if not src_area_dir.exists():
            # Check for lowercase variation just in case
            src_area_dir = src_split_dir / area.lower()
            if not src_area_dir.exists():
                continue

        src_images_dir = src_area_dir / "images_png"
        src_masks_dir = src_area_dir / mask_folder_name

        if not src_images_dir.exists():
            continue

        print(f"Processing {src_split_dir.name} -> {area}...")

        # Copy images
        for img_path in src_images_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() == ".png":
                shutil.copy2(img_path, dst_images_dir / img_path.name)
                img_count += 1

                # Copy masks if they exist (Note: Test split does not have public masks)
                if src_masks_dir.exists():
                    dst_masks_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = src_masks_dir / img_path.name
                    if mask_path.exists():
                        shutil.copy2(mask_path, dst_masks_dir / img_path.name)
                        mask_count += 1

    print(f"  Split '{dst_split_dir.name}': Copied {img_count} images and {mask_count} masks.")


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    # Note: Case variations in the directory names of the raw dataset
    # E.g. 'Train', 'Val', 'Test' in raw LoveDA, mapped to lowercase train, val, test.
    splits_mapping = {
        "Train": "train",
        "Val": "val",
        "Test": "test"
    }

    for src_name, dst_name in splits_mapping.items():
        src_split_dir = src_root / src_name
        if not src_split_dir.exists():
            # try lowercase
            src_split_dir = src_root / src_name.lower()
            if not src_split_dir.exists():
                continue
        prepare_split(src_split_dir, dst_root / dst_name, args.mask_folder)


if __name__ == "__main__":
    main()
