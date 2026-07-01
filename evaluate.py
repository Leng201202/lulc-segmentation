import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from datasets.combined_dataset import build_test_loader
from datasets.label_maps import CLASS_NAMES, IGNORE_INDEX
from losses.losses import SegmentationLoss
from metrics.segmentation_metrics import compute_metrics, format_metrics, save_metrics
from models.model_factory import build_model
from utils.config import get_project_root, load_config, resolve_path
from utils.seed import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LULC model on IRSAMap test set.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--output",
        default="outputs/evaluation_results.json",
        help="Path to save evaluation metrics JSON.",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    project_root = get_project_root()
    config_path = resolve_path(project_root, args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = load_config(config_path)
    data_cfg = config["data"]
    train_cfg = config["train"]

    device = get_device(train_cfg.get("device", "cuda"), train_cfg.get("gpu_id", 0))
    loader = build_test_loader(config)

    model = build_model(config).to(device)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = SegmentationLoss(
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg.get("ignore_index", IGNORE_INDEX),
        aux_weight=train_cfg.get("aux_loss_weight", 0.4),
    )

    total_loss = 0.0
    tp = torch.zeros(data_cfg["num_classes"], dtype=torch.int64)
    fp = torch.zeros(data_cfg["num_classes"], dtype=torch.int64)
    fn = torch.zeros(data_cfg["num_classes"], dtype=torch.int64)
    total_correct = 0
    total_pixels = 0

    for batch in tqdm(loader, desc="Evaluate"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        total_loss += criterion(outputs, masks).item()

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = logits.argmax(dim=1)

        valid_mask = masks != data_cfg.get("ignore_index", IGNORE_INDEX)
        total_correct += (preds[valid_mask] == masks[valid_mask]).sum().item()
        total_pixels += valid_mask.sum().item()

        for cls in range(data_cfg["num_classes"]):
            pred_mask = (preds == cls) & valid_mask
            target_mask = (masks == cls) & valid_mask
            tp[cls] += torch.logical_and(pred_mask, target_mask).sum().cpu()
            fp[cls] += torch.logical_and(pred_mask, ~target_mask).sum().cpu()
            fn[cls] += torch.logical_and(~pred_mask, target_mask).sum().cpu()

    per_class_iou = []
    per_class_f1 = []
    for cls in range(data_cfg["num_classes"]):
        union = tp[cls] + fp[cls] + fn[cls]
        per_class_iou.append(0.0 if union == 0 else (tp[cls] / union).item())
        denom = 2 * tp[cls] + fp[cls] + fn[cls]
        per_class_f1.append(0.0 if denom == 0 else ((2 * tp[cls]) / denom).item())

    oa = total_correct / total_pixels if total_pixels > 0 else 0.0

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "oa": oa,
        "miou": sum(per_class_iou) / len(per_class_iou),
        "mf1": sum(per_class_f1) / len(per_class_f1),
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
    }

    print("IRSAMap Test Evaluation")
    print(format_metrics(metrics))
    print(f"Loss: {metrics['loss']:.4f}")

    output_path = resolve_path(project_root, args.output)
    save_metrics(metrics, output_path)

    summary = {
        "experiment": config["experiment"]["name"],
        "checkpoint": str(resolve_path(project_root, args.checkpoint)),
        "metrics": metrics,
        "class_names": CLASS_NAMES,
    }
    with open(output_path.with_name(output_path.stem + "_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
