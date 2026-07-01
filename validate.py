import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from datasets.irsamap import IRSAMapDataset, list_image_files
from losses.composite import SegmentationLoss
from models.unetformer import build_model
from tools.config import get_data_paths, load_config
from tools.metrics import compute_iou, mean_iou, pixel_accuracy
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

    total_loss = 0.0
    class_ious = [[] for _ in range(data_cfg["num_classes"])]
    accuracies = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = logits.argmax(dim=1)
            accuracies.append(pixel_accuracy(preds, masks, data_cfg.get("ignore_index", 255)))
            batch_ious = compute_iou(preds, masks, data_cfg["num_classes"], data_cfg.get("ignore_index", 255))
            for idx, value in enumerate(batch_ious):
                if value == value:
                    class_ious[idx].append(value)

    avg_class_ious = [
        sum(values) / len(values) if values else float("nan")
        for values in class_ious
    ]

    print(f"Split: {args.split}")
    print(f"Loss: {total_loss / len(loader):.4f}")
    print(f"Pixel Accuracy: {sum(accuracies) / len(accuracies):.4f}")
    print(f"mIoU: {mean_iou(avg_class_ious):.4f}")
    print("Per-class IoU:")
    for name, iou in zip(CLASS_NAMES, avg_class_ious):
        if iou == iou:
            print(f"  {name:12s}: {iou:.4f}")


if __name__ == "__main__":
    main()
