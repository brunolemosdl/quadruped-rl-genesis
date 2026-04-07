"""Deterministic seeding for Python, NumPy, and Torch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed the main random number generators used by the project.

    The function synchronizes Python's ``random`` module, NumPy, the process
    hash seed, and Torch when it is installed.

    Args:
        seed (int): Seed value applied consistently across supported libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass
