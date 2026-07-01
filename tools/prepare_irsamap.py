"""
Prepare the official IRSAMap download for training.

Official dataset: https://github.com/ucas-dlg/IRSAMap
Download link is in the repo README (ScienceDB).

Expected official layout (after extracting the archive):
  <source>/
    Image/                  or Images/
      train/*.png
      val/*.png
      test/*.png
    SegLabel_vwsbr/         recommended (category codes: 10,11,12,21,...)
      train/*.png
      val/*.png
      test/*.png

Alternative label folders:
  - SegLabel_rvwsb          category codes (original)
  - SegLabel_vwsbr_0-11     ref-codes 0-11 (known label errors — verify first)
  - SegLabel_rvwsbr_0-11    ref-codes 0-11 (known label errors — verify first)

Usage:
  python tools/prepare_irsamap.py \\
    --source /path/to/IRSAMap_download \\
    --output data/IRSAMap \\
    --image-dir Image \\
    --label-dir SegLabel_vwsbr
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_DIR_CANDIDATES = ("Image", "Images", "image", "images", "Img")
LABEL_DIR_CANDIDATES = (
    "SegLabel_vwsbr",
    "SegLabel_rvwsb",
    "SegLabel_vwsbr_0-11",
    "SegLabel_rvwsbr_0-11",
)


def find_child(parent: Path, name: str | None, candidates: tuple[str, ...]) -> Path:
    if name:
        path = parent / name
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
        return path

    for candidate in candidates:
        path = parent / candidate
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find directory under {parent}. Tried: {candidates}"
    )


def link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def prepare_split(
    image_dir: Path,
    label_dir: Path,
    split: str,
    output_root: Path,
    use_symlinks: bool,
) -> tuple[int, int]:
    src_images = image_dir / split
    src_labels = label_dir / split
    if not src_images.exists():
        return 0, 0

    dst_images = output_root / split / "images"
    dst_masks = output_root / split / "masks"
    image_files = sorted(
        p for p in src_images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    paired = 0
    missing = 0
    for image_path in image_files:
        label_path = src_labels / image_path.name
        if not label_path.exists():
            missing += 1
            continue
        link_or_copy(image_path, dst_images / image_path.name, use_symlinks)
        link_or_copy(label_path, dst_masks / image_path.name, use_symlinks)
        paired += 1

    return paired, missing


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare official IRSAMap data for training.")
    parser.add_argument("--source", required=True, help="Path to extracted IRSAMap download.")
    parser.add_argument("--output", default="data/IRSAMap", help="Output directory for this project.")
    parser.add_argument("--image-dir", default=None, help="Image folder name inside source.")
    parser.add_argument("--label-dir", default=None, help="Label folder name inside source.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    use_symlinks = not args.copy

    image_root = find_child(source, args.image_dir, IMAGE_DIR_CANDIDATES)
    label_root = find_child(source, args.label_dir, LABEL_DIR_CANDIDATES)

    print(f"Source images: {image_root}")
    print(f"Source labels: {label_root}")
    print(f"Output:        {output}")
    print(f"Mode:          {'symlink' if use_symlinks else 'copy'}")
    print()

    totals = {"paired": 0, "missing": 0}
    for split in SPLITS:
        paired, missing = prepare_split(image_root, label_root, split, output, use_symlinks)
        totals["paired"] += paired
        totals["missing"] += missing
        print(f"  {split:5s}: {paired:5d} pairs, {missing:4d} images without labels")

    print()
    print(f"Total paired samples: {totals['paired']}")
    if totals["missing"]:
        print(f"Warning: {totals['missing']} images had no matching label file.")
    if totals["paired"] == 0:
        raise RuntimeError("No paired samples were prepared. Check --source and folder names.")

    print("\nOfficial IRSAMap split sizes (paper): train=4617, val=340, test=912")


if __name__ == "__main__":
    main()
