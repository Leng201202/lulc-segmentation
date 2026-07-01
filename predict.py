import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.irsamap_dataset import rgb_mask_to_class_indices
from datasets.transforms import build_transforms
from models.model_factory import build_model
from utils.config import get_project_root, load_config, resolve_path
from utils.seed import get_device
from utils.visualization import class_mask_to_color, save_side_by_side


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and save visualizations.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="IRSAMap split to run predictions on.",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions",
        help="Directory for prediction outputs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples to predict.",
    )
    return parser.parse_args()


@torch.no_grad()
def predict_image(model, image_rgb, transform, device, original_size):
    transformed = transform(image=image_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)
    outputs = model(tensor)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    if pred.shape != original_size:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return pred


def main():
    args = parse_args()
    project_root = get_project_root()
    config = load_config(resolve_path(project_root, args.config))
    data_cfg = config["data"]
    train_cfg = config["train"]
    irsamap_cfg = data_cfg["irsamap"]

    root = resolve_path(project_root, irsamap_cfg["root"])
    if args.split == "test":
        image_dir = resolve_path(root, irsamap_cfg["test_images"])
        mask_dir = resolve_path(root, irsamap_cfg["test_masks"])
    else:
        image_dir = resolve_path(root, irsamap_cfg["val_images"])
        mask_dir = resolve_path(root, irsamap_cfg["val_masks"])

    output_dir = resolve_path(project_root, args.output)
    pred_dir = output_dir / "masks"
    color_dir = output_dir / "color"
    vis_dir = output_dir / "visualizations"
    for directory in (pred_dir, color_dir, vis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = get_device(train_cfg.get("device", "cuda"), train_cfg.get("gpu_id", 0))
    model = build_model(config).to(device)
    checkpoint = torch.load(resolve_path(project_root, args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(data_cfg.get("image_size", 512), is_train=False)
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    for image_path in tqdm(image_paths, desc="Predict"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]

        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            mask_rgb = cv2.cvtColor(cv2.imread(str(mask_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            ground_truth = rgb_mask_to_class_indices(mask_rgb)
        else:
            ground_truth = np.zeros(original_size, dtype=np.int64)

        prediction = predict_image(model, image_rgb, transform, device, original_size)
        color_pred = class_mask_to_color(prediction)

        stem = image_path.stem
        cv2.imwrite(str(pred_dir / f"{stem}.png"), prediction.astype(np.uint8))
        cv2.imwrite(str(color_dir / f"{stem}.png"), cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
        save_side_by_side(
            image_rgb,
            ground_truth,
            prediction,
            vis_dir / f"{stem}_compare.png",
        )

    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
