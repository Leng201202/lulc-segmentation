import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "cuda", gpu_id: int = 0) -> torch.device:
    """Return a CUDA device for training on NVIDIA GPUs."""
    if device_name == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install PyTorch with CUDA support on your NVIDIA GPU machine:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU {gpu_id} but only {torch.cuda.device_count()} CUDA device(s) are available."
        )

    torch.backends.cudnn.benchmark = True
    return torch.device(f"cuda:{gpu_id}")


def save_checkpoint(path, model, optimizer, epoch, metrics, config) -> None:
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


def load_checkpoint(path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
