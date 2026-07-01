import numpy as np
import torch


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255):
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious


def mean_iou(ious: list[float]) -> float:
    valid = [v for v in ious if not np.isnan(v)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> float:
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    valid = targets != ignore_index
    if valid.sum() == 0:
        return 0.0
    return float((preds[valid] == targets[valid]).mean())
