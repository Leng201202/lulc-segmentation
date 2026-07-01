import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.transforms import build_transforms
from models.unetformer import build_model
from tools.config import get_data_paths, load_config
from tools.palette import class_indices_to_color
from tools.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run UNetFormer inference.")
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
        "-i",
        "--input",
        default=None,
        help="Input image or directory. Defaults to test split from config.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/predictions",
        help="Output directory for prediction masks.",
    )
    return parser.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )


def predict_image(model, image_path: Path, transform, device):
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    transformed = transform(image=image)
    tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
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
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / args.config)
    data_cfg = config["data"]
    paths = get_data_paths(config, project_root)

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = project_root / input_path
    else:
        input_path = paths["test_images"] or paths["val_images"] or paths["train_images"]

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_path)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_path}")

    train_cfg = config["train"]
    device = get_device(
        train_cfg.get("device", "cuda"),
        train_cfg.get("gpu_id", 0),
    )
    model = build_model(config).to(device)
    checkpoint = torch.load(project_root / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(data_cfg.get("image_size", 512), is_train=False)

    for image_path in tqdm(image_paths, desc="Predict"):
        pred = predict_image(model, image_path, transform, device)
        color_mask = class_indices_to_color(pred)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        stem = image_path.stem
        cv2.imwrite(str(output_dir / f"{stem}_mask.png"), pred.astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{stem}_color.png"), color_mask_bgr)

    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()
