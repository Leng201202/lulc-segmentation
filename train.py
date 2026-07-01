import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets.irsamap import build_dataloaders
from losses.composite import SegmentationLoss
from models.unetformer import build_model
from tools.config import load_config
from tools.metrics import compute_metrics, mean_iou, mean_dice
from tools.utils import get_device, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNetFormer on IRSAMap.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/unetformer_resnet18.yml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def evaluate(model, loader, criterion, device, num_classes, ignore_index, class_names):
    model.eval()
    total_loss = 0.0

    # Initialize accumulators for metrics
    tp_total = [0] * num_classes
    fp_total = [0] * num_classes
    fn_total = [0] * num_classes
    total_oa_sum = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = logits.argmax(dim=1)

            # Compute batch metrics
            batch_metrics = compute_metrics(preds, masks, num_classes, ignore_index)

            # Accumulate metrics
            for cls in range(num_classes):
                tp_total[cls] += batch_metrics["tp_per_class"][cls]
                fp_total[cls] += batch_metrics["fp_per_class"][cls]
                fn_total[cls] += batch_metrics["fn_per_class"][cls]

            total_oa_sum += batch_metrics["oa"]
            total_batches += 1

    # Compute final per-class metrics
    per_class_iou = []
    per_class_f1 = []
    for cls in range(num_classes):
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

    # Compute overall metrics
    overall_oa = total_oa_sum / total_batches if total_batches > 0 else 0.0
    overall_miou = mean_iou(per_class_iou)
    overall_mf1 = mean_dice(per_class_f1)

    return {
        "loss": total_loss / max(len(loader), 1),
        "overall_oa": overall_oa,
        "overall_miou": overall_miou,
        "overall_mf1": overall_mf1,
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, log_interval):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="Train", leave=False)

    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % log_interval == 0:
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

    return running_loss / max(len(loader), 1)


def init_csv_logger(log_path, num_classes, class_names):
    """Initialize CSV log file with headers."""
    headers = ["epoch", "train_loss"]

    # Add validation metrics
    headers.extend([
        "val_loss",
        "val_oa",
        "val_miou",
        "val_mf1"
    ])

    # Add per-class IoU and F1
    for name in class_names:
        headers.append(f"iou_{name}")
    for name in class_names:
        headers.append(f"f1_{name}")

    # Create file and write headers
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    return log_path


def log_to_csv(log_path, epoch, train_loss, val_metrics, class_names):
    """Append metrics to CSV file."""
    row = [epoch, train_loss]

    # Add validation metrics
    row.extend([
        val_metrics["loss"],
        val_metrics["overall_oa"],
        val_metrics["overall_miou"],
        val_metrics["overall_mf1"]
    ])

    # Add per-class IoU
    for iou in val_metrics["per_class_iou"]:
        row.append(iou if not np.isnan(iou) else "")

    # Add per-class F1
    for f1 in val_metrics["per_class_f1"]:
        row.append(f1 if not np.isnan(f1) else "")

    with open(log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / args.config)

    train_cfg = config["train"]
    data_cfg = config["data"]
    set_seed(train_cfg.get("seed", 42))

    # Get class names
    from tools.palette import CLASS_NAMES
    class_names = CLASS_NAMES
    num_classes = data_cfg["num_classes"]

    device = get_device(
        train_cfg.get("device", "cuda"),
        train_cfg.get("gpu_id", 0),
    )
    train_loader, val_loader = build_dataloaders(config, project_root)

    model = build_model(config).to(device)
    criterion = SegmentationLoss(
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg.get("ignore_index", 255),
        aux_weight=train_cfg.get("aux_loss_weight", 0.4),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    start_epoch = 1
    checkpoint_dir = project_root / train_cfg.get("checkpoint_dir", "checkpoints")
    log_path = checkpoint_dir / "training_log.csv"
    best_miou = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_miou = checkpoint.get("metrics", {}).get("overall_miou", 0.0)

    # Initialize CSV logger
    init_csv_logger(log_path, num_classes, class_names)

    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Logging to: {log_path}")
    print()

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        print(f"Epoch {epoch}/{train_cfg['epochs']}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_cfg.get("log_interval", 10),
        )
        scheduler.step()

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                data_cfg["num_classes"],
                data_cfg.get("ignore_index", 255),
                class_names
            )

            # Print metrics
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val OA: {val_metrics['overall_oa']:.4f}")
            print(f"  Val mIoU: {val_metrics['overall_miou']:.4f}")
            print(f"  Val mF1: {val_metrics['overall_mf1']:.4f}")
            print("  Per-class IoU:")
            for name, iou in zip(class_names, val_metrics["per_class_iou"]):
                if not np.isnan(iou):
                    print(f"    {name:12s}: {iou:.4f}")
            print("  Per-class F1:")
            for name, f1 in zip(class_names, val_metrics["per_class_f1"]):
                if not np.isnan(f1):
                    print(f"    {name:12s}: {f1:.4f}")

            # Log to CSV
            log_to_csv(log_path, epoch, train_loss, val_metrics, class_names)

            # Save best model
            if val_metrics["overall_miou"] >= best_miou:
                best_miou = val_metrics["overall_miou"]
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    config,
                )
                print(f"  Saved best model (mIoU: {best_miou:.4f})")
        else:
            print(f"  Train Loss: {train_loss:.4f}")

        # Save periodic checkpoint
        if epoch % train_cfg.get("save_interval", 10) == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                val_metrics if val_metrics else {"train_loss": train_loss},
                config,
            )
            print(f"  Saved checkpoint epoch_{epoch:03d}.pt")

        print()

    # Save final checkpoint
    save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizer,
        train_cfg["epochs"],
        val_metrics if val_metrics else {"train_loss": train_loss},
        config,
    )
    print(f"Training complete!")
    print(f"Checkpoints and log saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
