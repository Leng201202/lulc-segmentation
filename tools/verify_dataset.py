#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify prepared IRSAMap and LoveDA datasets for matching files and correct image sizes."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory containing IRSAMap and LoveDA folders.",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=1024,
        help="Expected width and height of dataset images.",
    )
    return parser.parse_args()


def verify_split(image_dir: Path, mask_dir: Path | None, split_name: str, expected_size: int):
    if not image_dir.exists():
        print(f"❌ Split '{split_name}': Image directory does not exist: {image_dir}")
        return False

    images = sorted(list(image_dir.glob("*")))
    images = [p for p in images if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]

    if not images:
        print(f"❌ Split '{split_name}': No images found in {image_dir}")
        return False

    print(f"📋 Split '{split_name}': Found {len(images)} images.")

    # Verify matching masks if applicable
    missing_masks = 0
    if mask_dir is not None:
        if not mask_dir.exists():
            print(f"❌ Split '{split_name}': Mask directory does not exist: {mask_dir}")
            return False

        for img_path in images:
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                missing_masks += 1

        if missing_masks > 0:
            print(f"❌ Split '{split_name}': Missing masks for {missing_masks}/{len(images)} images.")
        else:
            print(f"✅ Split '{split_name}': All masks successfully matched.")
    else:
        print(f"ℹ️ Split '{split_name}': Mask checking skipped (Test split).")

    # Sample and check image sizes
    sample_limit = min(5, len(images))
    incorrect_sizes = 0
    for i in range(sample_limit):
        img_path = images[i]
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w != expected_size or h != expected_size:
                    print(f"⚠️ Image {img_path.name} has size {w}x{h} (Expected {expected_size}x{expected_size})")
                    incorrect_sizes += 1
        except Exception as e:
            print(f"⚠️ Could not open sample image {img_path.name}: {e}")
            incorrect_sizes += 1

    if incorrect_sizes == 0:
        print(f"✅ Split '{split_name}': Sampled size check passed ({expected_size}x{expected_size}).")
    else:
        print(f"❌ Split '{split_name}': {incorrect_sizes} sample(s) had incorrect size or failed to open.")

    return missing_masks == 0 and incorrect_sizes == 0


def main():
    args = parse_args()
    data_root = Path(args.data_dir)

    success = True

    # Check IRSAMap
    irsamap_root = data_root / "IRSAMap"
    print("\n=== Verifying IRSAMap ===")
    if irsamap_root.exists():
        for split in ["train", "val", "test"]:
            split_success = verify_split(
                image_dir=irsamap_root / split / "images",
                mask_dir=irsamap_root / split / "masks_rgb",
                split_name=f"IRSAMap {split}",
                expected_size=args.expected_size
            )
            success = success and split_success
    else:
        print(f"❌ IRSAMap directory not found at {irsamap_root}")
        success = False

    # Check LoveDA
    loveda_root = data_root / "LoveDA"
    print("\n=== Verifying LoveDA ===")
    if loveda_root.exists():
        for split in ["train", "val"]:
            split_success = verify_split(
                image_dir=loveda_root / split / "images",
                mask_dir=loveda_root / split / "masks",
                split_name=f"LoveDA {split}",
                expected_size=args.expected_size
            )
            success = success and split_success
        
        # Test split doesn't have public ground truth masks
        split_success = verify_split(
            image_dir=loveda_root / "test" / "images",
            mask_dir=None,
            split_name="LoveDA test",
            expected_size=args.expected_size
        )
        success = success and split_success
    else:
        print(f"ℹ️ LoveDA directory not found at {loveda_root} (Optional support dataset)")

    if success:
        print("\n🎉 Overall Validation Success! Dataset is ready for training.")
        sys.exit(0)
    else:
        print("\n❌ Validation Failed! Please check warning/error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
