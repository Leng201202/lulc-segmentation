import argparse
import multiprocessing as mp
import multiprocessing.pool as mpp
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.irsamap import IRSAMapDataset, list_image_files
from models.unetformer import build_model
from tools.config import get_data_paths, load_config
from tools.metrics import compute_metrics, mean_iou, mean_dice
from tools.palette import CLASS_NAMES, class_indices_to_color
from tools.utils import get_device


def label2rgb(mask):
    """Convert class-ID mask to RGB using the palette."""
    mask_rgb = class_indices_to_color(mask)
    return mask_rgb


def img_writer(inp):
    """Write a single prediction mask to disk."""
    mask, output_path, rgb = inp
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rgb:
        mask_rgb = label2rgb(mask)
        mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), mask_bgr)
    else:
        cv2.imwrite(str(output_path), mask.astype(np.uint8))


def get_args():
    parser = argparse.ArgumentParser(description="Test UNetFormer with TTA and save predictions.")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path("config/unetformer_resnet18.yml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "-ckpt", "--checkpoint",
        type=Path,
        default=Path("checkpoints/best.pt"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "-o", "--output_path",
        type=Path,
        required=True,
        help="Directory where to save resulting masks.",
    )
    parser.add_argument(
        "-s", "--split",
        choices=["val", "test"],
        default="test",
        help="Which split to evaluate on.",
    )
    parser.add_argument(
        "-t", "--tta",
        help="Test time augmentation.",
        default=None,
        choices=[None, "lr", "d4"],
    )
    parser.add_argument(
        "--rgb",
        help="Output RGB-colored images instead of class-ID masks.",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / args.config)
    data_cfg = config["data"]
    paths = get_data_paths(config, project_root)

    if args.checkpoint.is_absolute():
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = project_root / args.checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.split == "val":
        image_dir, mask_dir = paths["val_images"], paths["val_masks"]
    else:
        image_dir, mask_dir = paths["test_images"], paths["test_masks"]

    image_paths = list_image_files(image_dir)
    mask_paths = [mask_dir / f"{p.stem}{p.suffix}" for p in image_paths]
    pairs = [(i, m, p) for i, m, p in zip(image_paths, mask_paths, image_paths) if m.exists()]

    if not pairs:
        raise RuntimeError(f"No samples found for split={args.split}.")

    test_dataset = IRSAMapDataset(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        image_size=data_cfg.get("image_size", 1024),
        augment_cfg=config.get("augment"),
        is_train=False,
        label_encoding=data_cfg.get("label_encoding", "category_code"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_cfg = config["train"]
    device = get_device(
        train_cfg.get("device", "cuda"),
        train_cfg.get("gpu_id", 0),
    )
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
        print(f"TTA enabled: HorizontalFlip + VerticalFlip")
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
        print(f"TTA enabled: D4 (HFlip + VFlip + 4x Rotate90)")

    args.output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving predictions to: {args.output_path}")
    print(f"Mode: {'RGB' if args.rgb else 'class-ID'}")

    # Initialize accumulators
    tp_total = [0] * data_cfg["num_classes"]
    fp_total = [0] * data_cfg["num_classes"]
    fn_total = [0] * data_cfg["num_classes"]
    total_oa_sum = 0.0
    total_batches = 0
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", ncols=80):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            raw_predictions = model(images)
            raw_predictions = nn.functional.softmax(raw_predictions, dim=1)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                gt = masks[i].cpu().numpy()

                # Accumulate metrics
                batch_metrics = compute_metrics(
                    torch.from_numpy(mask),
                    torch.from_numpy(gt),
                    data_cfg["num_classes"],
                    data_cfg.get("ignore_index", 255),
                )
                for cls in range(data_cfg["num_classes"]):
                    tp_total[cls] += batch_metrics["tp_per_class"][cls]
                    fp_total[cls] += batch_metrics["fp_per_class"][cls]
                    fn_total[cls] += batch_metrics["fn_per_class"][cls]
                total_oa_sum += batch_metrics["oa"]
                total_batches += 1

                # Use image stem (e.g. "1252") as mask name
                img_path = Path(batch["image_path"][i])
                mask_name = img_path.stem
                results.append(
                    (mask, str(args.output_path / f"{mask_name}.png"), args.rgb)
                )

    # Compute final per-class metrics
    per_class_iou = []
    per_class_f1 = []
    for cls in range(data_cfg["num_classes"]):
        tp = tp_total[cls]
        fp = fp_total[cls]
        fn = fn_total[cls]
        union = tp + fp + fn
        iou = tp / union if union > 0 else float("nan")
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        per_class_iou.append(iou)
        per_class_f1.append(f1)

    overall_oa = total_oa_sum / total_batches if total_batches > 0 else 0.0
    overall_miou = mean_iou(per_class_iou)
    overall_mf1 = mean_dice(per_class_f1)

    # Print per-class metrics
    print()
    print("=" * 60)
    print("Per-class metrics:")
    for name, iou, f1 in zip(CLASS_NAMES, per_class_iou, per_class_f1):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
        print(f"  {name:12s} | IoU: {iou_str:>7s} | F1: {f1_str:>7s}")
    print("=" * 60)
    print(f"OA:   {overall_oa:.4f}")
    print(f"mIoU: {overall_miou:.4f}")
    print(f"mF1:  {overall_mf1:.4f}")
    print("=" * 60)

    # Save metrics to CSV
    metrics_csv = args.output_path / "metrics.csv"
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("class,iou,f1\n")
        for name, iou, f1 in zip(CLASS_NAMES, per_class_iou, per_class_f1):
            iou_str = f"{iou:.6f}" if not np.isnan(iou) else ""
            f1_str = f"{f1:.6f}" if not np.isnan(f1) else ""
            f.write(f"{name},{iou_str},{f1_str}\n")
        f.write(f"OA,{overall_oa:.6f},\n")
        f.write(f"mIoU,{overall_miou:.6f},\n")
        f.write(f"mF1,{overall_mf1:.6f},\n")
    print(f"Metrics saved to: {metrics_csv}")

    # Write images with multiprocessing
    t0 = time.time()
    num_workers = min(4, mp.cpu_count())
    with mpp.Pool(processes=num_workers) as pool:
        pool.map(img_writer, results)
    t1 = time.time()
    print(f"Image writing: {t1 - t0:.2f}s ({len(results)} images)")


if __name__ == "__main__":
    main()
