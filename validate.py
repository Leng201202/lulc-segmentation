import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets.irsamap import IRSAMapDataset, list_image_files
from losses.composite import SegmentationLoss
from models.unetformer import build_model
from tools.config import get_data_paths, load_config
from tools.metrics import compute_metrics, mean_iou, mean_dice
from tools.palette import CLASS_NAMES
from tools.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Validate UNetFormer on IRSAMap.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/unetformer_resnet18.yml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which split to evaluate.",
    )
    return parser.parse_args()


def main():
    import numpy as np

    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / args.config)
    data_cfg = config["data"]
    paths = get_data_paths(config, project_root)

    if args.split == "val":
        image_dir, mask_dir = paths["val_images"], paths["val_masks"]
    else:
        image_dir, mask_dir = paths["test_images"], paths["test_masks"]

    image_paths = list_image_files(image_dir)
    mask_paths = [mask_dir / f"{p.stem}{p.suffix}" for p in image_paths]
    pairs = [(i, m) for i, m in zip(image_paths, mask_paths) if m.exists()]

    if not pairs:
        raise RuntimeError(f"No samples found for split={args.split}.")

    dataset = IRSAMapDataset(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        image_size=data_cfg.get("image_size", 512),
        augment_cfg=config.get("augment"),
        is_train=False,
        label_encoding=data_cfg.get("label_encoding", "category_code"),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    train_cfg = config["train"]
    device = get_device(
        train_cfg.get("device", "cuda"),
        train_cfg.get("gpu_id", 0),
    )
    model = build_model(config).to(device)
    checkpoint = torch.load(project_root / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = SegmentationLoss(
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg.get("ignore_index", 255),
        aux_weight=config["train"].get("aux_loss_weight", 0.4),
    )

    # Initialize accumulators
    total_loss = 0.0
    tp_total = [0] * data_cfg["num_classes"]
    fp_total = [0] * data_cfg["num_classes"]
    fn_total = [0] * data_cfg["num_classes"]
    total_oa_sum = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", ncols=80):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = logits.argmax(dim=1)

            batch_metrics = compute_metrics(preds, masks, data_cfg["num_classes"], data_cfg.get("ignore_index", 255))
            for cls in range(data_cfg["num_classes"]):
                tp_total[cls] += batch_metrics["tp_per_class"][cls]
                fp_total[cls] += batch_metrics["fp_per_class"][cls]
                fn_total[cls] += batch_metrics["fn_per_class"][cls]

            total_oa_sum += batch_metrics["oa"]
            total_batches += 1

    # Compute final per-class metrics
    per_class_iou = []
    per_class_f1 = []
    for cls in range(data_cfg["num_classes"]):
        tp = tp_total[cls]
        fp = fp_total[cls]
        fn = fn_total[cls]
        union = tp + fp + fn
        if union == 0:
            iou = float("nan")
        else:
            iou = tp / union

        if (2 * tp + fp + fn) == 0:
            f1 = float("nan")
        else:
            f1 = (2 * tp) / (2 * tp + fp + fn)

        per_class_iou.append(iou)
        per_class_f1.append(f1)

    overall_oa = total_oa_sum / total_batches
    overall_miou = mean_iou(per_class_iou)
    overall_mf1 = mean_dice(per_class_f1)

    print(f"Split: {args.split}")
    print(f"Loss: {total_loss / len(loader):.4f}")
    print(f"OA: {overall_oa:.4f}")
    print(f"mIoU: {overall_miou:.4f}")
    print(f"mF1: {overall_mf1:.4f}")
    print("Per-class IoU:")
    for name, iou in zip(CLASS_NAMES, per_class_iou):
        if not np.isnan(iou):
            print(f"  {name:12s}: {iou:.4f}")
    print("Per-class F1:")
    for name, f1 in zip(CLASS_NAMES, per_class_f1):
        if not np.isnan(f1):
            print(f"  {name:12s}: {f1:.4f}")


if __name__ == "__main__":
    main()
