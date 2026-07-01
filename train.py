import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets.combined_dataset import build_train_loader, build_val_loader
from datasets.label_maps import CLASS_NAMES, IGNORE_INDEX
from losses.losses import SegmentationLoss
from metrics.segmentation_metrics import compute_metrics, format_metrics
from models.model_factory import build_model
from utils.config import get_project_root, load_config
from utils.seed import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train LULC semantic segmentation model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def resolve_checkpoint_path(path: str | Path, project_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def save_checkpoint(path: Path, model, optimizer, epoch, metrics, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index):
    model.eval()
    total_loss = 0.0
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    total_correct = 0
    total_pixels = 0

    for batch in tqdm(loader, desc="Validate", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        total_loss += criterion(outputs, masks).item()

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = logits.argmax(dim=1)

        valid_mask = masks != ignore_index
        total_correct += (preds[valid_mask] == masks[valid_mask]).sum().item()
        total_pixels += valid_mask.sum().item()

        for cls in range(num_classes):
            pred_mask = (preds == cls) & valid_mask
            target_mask = (masks == cls) & valid_mask
            tp[cls] += torch.logical_and(pred_mask, target_mask).sum().item()
            fp[cls] += torch.logical_and(pred_mask, ~target_mask).sum().item()
            fn[cls] += torch.logical_and(~pred_mask, target_mask).sum().item()

    per_class_iou = []
    per_class_f1 = []
    for cls in range(num_classes):
        union = tp[cls] + fp[cls] + fn[cls]
        per_class_iou.append(0.0 if union == 0 else tp[cls] / union)
        denom = 2 * tp[cls] + fp[cls] + fn[cls]
        per_class_f1.append(0.0 if denom == 0 else (2 * tp[cls]) / denom)

    oa = total_correct / total_pixels if total_pixels > 0 else 0.0

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "oa": oa,
        "miou": sum(per_class_iou) / len(per_class_iou),
        "mf1": sum(per_class_f1) / len(per_class_f1),
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
    }
    return metrics


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
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item()
        if step % log_interval == 0:
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

    return running_loss / max(len(loader), 1)


def init_csv_logger(log_path: Path):
    headers = ["epoch", "train_loss", "val_loss", "val_oa", "val_miou", "val_mf1"]
    headers.extend([f"iou_{name}" for name in CLASS_NAMES])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(headers)


def append_csv_log(log_path: Path, epoch, train_loss, val_metrics):
    row = [
        epoch,
        train_loss,
        val_metrics["loss"],
        val_metrics["oa"],
        val_metrics["miou"],
        val_metrics["mf1"],
    ]
    row.extend(val_metrics["per_class_iou"])
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def main():
    args = parse_args()
    project_root = get_project_root()
    config = load_config(resolve_checkpoint_path(args.config, project_root))

    train_cfg = config["train"]
    data_cfg = config["data"]
    set_seed(train_cfg.get("seed", 42))

    device = get_device(train_cfg.get("device", "cuda"), train_cfg.get("gpu_id", 0))
    train_loader = build_train_loader(config)
    val_loader = build_val_loader(config)

    model = build_model(config).to(device)
    criterion = SegmentationLoss(
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg.get("ignore_index", IGNORE_INDEX),
        aux_weight=train_cfg.get("aux_loss_weight", 0.4),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    checkpoint_dir = project_root / train_cfg.get("checkpoint_dir", "checkpoints")
    best_checkpoint = checkpoint_dir / train_cfg.get("checkpoint_name", "best.pth")
    log_path = checkpoint_dir / f"{config['experiment']['name']}_training_log.csv"

    start_epoch = 1
    best_miou = -1.0
    if train_cfg.get("resume"):
        resume_path = resolve_checkpoint_path(train_cfg["resume"], project_root)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not train_cfg.get("finetune_reset_optimizer", False):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_miou = checkpoint.get("metrics", {}).get("miou", -1.0)
        print(f"Resumed from {resume_path} (epoch {checkpoint.get('epoch', '?')})")

    if start_epoch == 1:
        init_csv_logger(log_path)

    print(f"Experiment: {config['experiment']['name']} ({config['experiment']['stage']})")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Checkpoint: {best_checkpoint}")
    print()

    val_metrics = {}
    epochs_no_improve = 0
    patience = train_cfg.get("early_stopping_patience", 10)

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_cfg.get("log_interval", 10),
        )
        scheduler.step()

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            data_cfg["num_classes"],
            data_cfg.get("ignore_index", IGNORE_INDEX),
        )

        print(f"Epoch {epoch}/{train_cfg['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  OA:   {val_metrics['oa']:.4f}")
        print(f"  mIoU: {val_metrics['miou']:.4f}  mF1: {val_metrics['mf1']:.4f}")
        # Per-class IoU for key tracked classes
        iou = val_metrics["per_class_iou"]
        print("  Per-class IoU:")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            v = iou[cls_idx]
            bar = f"{v:.4f}" if not math.isnan(v) else "  N/A "
            print(f"    {cls_name:<12}: {bar}")
        append_csv_log(log_path, epoch, train_loss, val_metrics)

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            epochs_no_improve = 0
            save_checkpoint(best_checkpoint, model, optimizer, epoch, val_metrics, config)
            print(f"  Saved best checkpoint: {best_checkpoint} (mIoU={best_miou:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement in mIoU for {epochs_no_improve} epoch(s).")

        if epoch % train_cfg.get("save_interval", 10) == 0:
            save_checkpoint(
                checkpoint_dir / f"{config['experiment']['name']}_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
            )

        if epochs_no_improve >= patience:
            print(f"  Early stopping triggered. Validation mIoU did not improve for {patience} epochs.")
            # Write a flag file so the pipeline runner can detect early stopping and send an alert
            flag_path = checkpoint_dir / f"{config['experiment']['name']}_early_stopped.txt"
            with open(flag_path, "w", encoding="utf-8") as f:
                f.write(
                    f"experiment: {config['experiment']['name']}\n"
                    f"stage: {config['experiment']['stage']}\n"
                    f"stopped_at_epoch: {epoch}\n"
                    f"best_miou: {best_miou:.4f}\n"
                    f"patience: {patience}\n"
                )
            break
        print()

    save_checkpoint(
        checkpoint_dir / f"{config['experiment']['name']}_last.pt",
        model,
        optimizer,
        epoch,  # Use current epoch in case of early stopping
        val_metrics,
        config,
    )
    print(f"Training complete. Log: {log_path}")


if __name__ == "__main__":
    main()
