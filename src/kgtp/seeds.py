"""Deterministic runtime helpers."""

from __future__ import annotations

import importlib
import os
import random


def set_seed(seed: int = 13) -> None:
    """Set deterministic seeds for Python and optional numerical libraries."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
