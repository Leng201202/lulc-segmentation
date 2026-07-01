import json
from pathlib import Path

import numpy as np
import torch

from datasets.label_maps import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    preds = preds.detach().view(-1).cpu().numpy()
    targets = targets.detach().view(-1).cpu().numpy()

    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        tp[cls] = np.logical_and(pred_mask, target_mask).sum()
        fp[cls] = np.logical_and(pred_mask, ~target_mask).sum()
        fn[cls] = np.logical_and(~pred_mask, target_mask).sum()

    per_class_iou = []
    per_class_f1 = []
    for cls in range(num_classes):
        union = tp[cls] + fp[cls] + fn[cls]
        if union == 0:
            per_class_iou.append(0.0)
        else:
            per_class_iou.append(tp[cls] / union)

        denom = 2 * tp[cls] + fp[cls] + fn[cls]
        if denom == 0:
            per_class_f1.append(0.0)
        else:
            per_class_f1.append((2 * tp[cls]) / denom)

    oa = float((preds == targets).mean()) if len(targets) else 0.0
    miou = mean_ignore_nan(per_class_iou)
    mf1 = mean_ignore_nan(per_class_f1)

    return {
        "oa": oa,
        "miou": miou,
        "mf1": mf1,
        "per_class_iou": per_class_iou,
        "per_class_f1": per_class_f1,
    }


def mean_ignore_nan(values: list[float]) -> float:
    valid = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def format_metrics(metrics: dict, class_names: list[str] | None = None) -> str:
    class_names = class_names or CLASS_NAMES
    lines = [
        f"OA:   {metrics['oa']:.4f}",
        f"mIoU: {metrics['miou']:.4f}",
        f"mF1:  {metrics['mf1']:.4f}",
        "Per-class IoU:",
    ]
    for name, iou in zip(class_names, metrics["per_class_iou"]):
        lines.append(f"  {name:10s}: {iou:.4f}")
    return "\n".join(lines)


def save_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "oa": metrics["oa"],
        "miou": metrics["miou"],
        "mf1": metrics["mf1"],
        "per_class_iou": {
            name: float(iou)
            for name, iou in zip(CLASS_NAMES, metrics["per_class_iou"])
        },
        "per_class_f1": {
            name: float(f1)
            for name, f1 in zip(CLASS_NAMES, metrics["per_class_f1"])
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
