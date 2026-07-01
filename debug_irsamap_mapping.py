"""Debug script to verify IRSAMap mask remapping."""
import cv2
import numpy as np
from pathlib import Path

from datasets.irsamap_dataset import mask_to_class_indices, list_image_files
from datasets.label_maps import CLASS_NAMES, IGNORE_INDEX

import sys

def main():
    if len(sys.argv) > 1:
        mask_dir = Path(sys.argv[1])
    else:
        candidates = [
            Path("data/IRSAMap_prepared/train/masks_rgb"),
            Path("data/IRSAMap_prepared/val/masks_rgb"),
            Path("data/irsamap/masks_rgb"),
            Path("data/irsamap/masks"),
        ]
        mask_dir = next((p for p in candidates if p.exists()), candidates[0])
    print(f"Using mask directory: {mask_dir}")

    mask_paths = list_image_files(mask_dir)
    if not mask_paths:
        print(f"No mask files found in {mask_dir}")
        return

    counts = {name: 0 for name in CLASS_NAMES}
    counts["Ignore"] = 0

    for mask_path in mask_paths[:10]:  # Check first 10 masks
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        remapped = mask_to_class_indices(mask)
        unique, counts_np = np.unique(remapped, return_counts=True)

        for val, cnt in zip(unique, counts_np):
            if val == IGNORE_INDEX:
                counts["Ignore"] += int(cnt)
            elif 0 <= val < len(CLASS_NAMES):
                counts[CLASS_NAMES[val]] += int(cnt)

    print("\nIRSAMap Mask Remapping Debug:")
    print("-" * 40)
    total = 0
    for name in CLASS_NAMES + ["Ignore"]:
        print(f"{name}: {counts[name]}")
        total += counts[name]
    print(f"\nTotal pixels: {total}")
    print(f"Bareland pixels: {counts['Bareland']}")

    if counts["Bareland"] > 0:
        print("\n✓ SUCCESS: Bareland class (7) has positive pixel count.")
    else:
        print("\n✗ WARNING: Bareland class (7) has 0 pixels. Check the mapping.")

if __name__ == "__main__":
    main()