"""Utility functions for trading metrics and small helpers."""

import numpy as np
from typing import Sequence


def sharpe(mean: float, std: float, eps: float = 1e-9) -> float:
    if std <= 0:
        return mean if mean > 0 else 0.0
    return float(mean / (std + eps))


def mean_and_std(xs: Sequence[float]):
    arr = np.asarray(xs)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)
