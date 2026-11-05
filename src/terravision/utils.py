"""Reproducibility and device helpers shared by training and inference."""

from __future__ import annotations

import os
import platform
import random
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    ``deterministic=True`` also forces deterministic cuDNN kernels. That makes
    GPU runs repeatable bit-for-bit but slower, so it is off by default: seeds
    alone already make data splits, initial weights and augmentation repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch  # imported here so data-only code works without torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None):
    """Return ``torch.device('cuda')`` when available, else CPU.

    ``prefer="cpu"`` forces CPU (useful to check CPU loading on Colab).
    """
    import torch

    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def environment_info() -> dict[str, Any]:
    """Hardware/software summary to store with every training run."""
    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info
