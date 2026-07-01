import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str = "cuda", gpu_id: int = 0) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available on this machine.")

    if device_name == "cuda":
        if torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested GPU {gpu_id} but only {torch.cuda.device_count()} device(s) exist."
                )
            torch.backends.cudnn.benchmark = True
            return torch.device(f"cuda:{gpu_id}")
        raise RuntimeError(
            "CUDA requested but not available. Install PyTorch with CUDA support."
        )

    raise ValueError(f"Unknown device: {device_name}")
