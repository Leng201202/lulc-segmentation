import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets.irsamap import build_dataloaders
from losses.composite import SegmentationLoss
from models.unetformer import build_model
from tools.config import load_config
from tools.metrics import compute_iou, mean_iou, pixel_accuracy
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


def evaluate(model, loader, criterion, device, num_classes, ignore_index):
    model.eval()
    total_loss = 0.0
    all_ious = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = logits.argmax(dim=1)
            batch_ious = compute_iou(preds, masks, num_classes, ignore_index)
            for idx, value in enumerate(batch_ious):
                if not torch.isnan(torch.tensor(value)):
                    all_ious[idx].append(value)

    avg_ious = [sum(values) / len(values) if values else float("nan") for values in all_ious]
    return total_loss / max(len(loader), 1), mean_iou(avg_ious)


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


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / args.config)

    train_cfg = config["train"]
    data_cfg = config["data"]
    set_seed(train_cfg.get("seed", 42))

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
    best_miou = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_miou = checkpoint.get("metrics", {}).get("miou", 0.0)

    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")

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

        metrics = {"train_loss": train_loss}
        message = f"Epoch {epoch}/{train_cfg['epochs']} | train_loss={train_loss:.4f}"

        if val_loader is not None:
            val_loss, val_miou = evaluate(
                model,
                val_loader,
                criterion,
                device,
                data_cfg["num_classes"],
                data_cfg.get("ignore_index", 255),
            )
            metrics.update({"val_loss": val_loss, "miou": val_miou})
            message += f" | val_loss={val_loss:.4f} | mIoU={val_miou:.4f}"

            if val_miou >= best_miou:
                best_miou = val_miou
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    config,
                )

        print(message)

        if epoch % train_cfg.get("save_interval", 10) == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                metrics,
                config,
            )

    save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizer,
        train_cfg["epochs"],
        metrics,
        config,
    )
    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
