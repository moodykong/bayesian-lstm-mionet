"""Relative error metrics and the prediction interval coverage probability."""

from __future__ import annotations

import numpy as np

#: Half width of a two sided 95% Gaussian prediction interval.
Z_95 = 1.9600


def l2_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative L2 error in percent."""
    return float(100 * np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))


def l1_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative L1 error in percent."""
    return float(
        100 * np.linalg.norm(y_true - y_pred, ord=1) / np.linalg.norm(y_true, ord=1)
    )


def relative_error_per_trajectory(
    y_true: np.ndarray, y_pred: np.ndarray, order: int | None = None
) -> np.ndarray:
    """Row-wise relative error of ``[N_trajs, N_search]`` arrays (as a fraction)."""
    if order is None:
        return np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
    return np.linalg.norm(y_true - y_pred, ord=order, axis=1) / np.linalg.norm(
        y_true, ord=order, axis=1
    )


def picp(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = Z_95
) -> float:
    """Prediction interval coverage probability of the ``mean +/- z * std`` band.

    Returns the fraction of true values inside the interval (1.0 = full
    coverage); with the default ``z`` this is the 95% interval.
    """
    lower = mean - z * std
    upper = mean + z * std
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))
