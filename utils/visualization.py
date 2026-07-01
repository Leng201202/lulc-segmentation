from pathlib import Path

import cv2
import numpy as np

from datasets.label_maps import CLASS_COLORS, IGNORE_INDEX, NUM_CLASSES


def class_mask_to_color(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color[mask == class_id] = CLASS_COLORS[class_id]
    color[mask == IGNORE_INDEX] = (0, 0, 0)
    return color


def save_side_by_side(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    gt_color = class_mask_to_color(ground_truth)
    pred_color = class_mask_to_color(prediction)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    panel = np.concatenate([image, gt_color, pred_color], axis=1)
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel_bgr)
