import numpy as np
import torch


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """Compute IoU, Dice (F1), and pixel accuracy for all classes."""
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    ious = []
    dices = []  # Dice coefficient is equivalent to F1 score for segmentation
    tp_per_class = []
    fp_per_class = []
    fn_per_class = []

    for cls in range(num_classes):
        pred_mask = preds == cls
        target_mask = targets == cls

        tp = np.logical_and(pred_mask, target_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(target_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        tp_per_class.append(tp)
        fp_per_class.append(fp)
        fn_per_class.append(fn)

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(tp / union)

        if (2 * tp + fp + fn) == 0:
            dices.append(float("nan"))
        else:
            dices.append((2 * tp) / (2 * tp + fp + fn))

    # Overall Accuracy (OA)
    total_correct = (preds == targets).sum()
    total_valid = len(preds)
    oa = total_correct / total_valid if total_valid > 0 else 0.0

    return {
        "ious": ious,
        "dices": dices,
        "tp_per_class": tp_per_class,
        "fp_per_class": fp_per_class,
        "fn_per_class": fn_per_class,
        "oa": oa,
        "total_valid": total_valid,
    }


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """Legacy function for backward compatibility."""
    return compute_metrics(preds, targets, num_classes, ignore_index)["ious"]


def mean_iou(ious: list[float]) -> float:
    valid = [v for v in ious if not np.isnan(v)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def mean_dice(dices: list[float]) -> float:
    valid = [v for v in dices if not np.isnan(v)]
    if not valid:
        return 0.0
    return float(np.mean(valid))


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> float:
    """Legacy function for backward compatibility."""
    return compute_metrics(preds, targets, 1, ignore_index)["oa"]
