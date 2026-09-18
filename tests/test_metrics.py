"""Relative error metrics and the prediction interval coverage probability."""

from __future__ import annotations

import numpy as np
import pytest

from blstm_mionet.evaluation.metrics import (
    Z_95,
    l1_relative_error,
    l2_relative_error,
    picp,
    relative_error_per_trajectory,
)


# --------------------------------------------------------------------------- #
# L2 / L1 relative errors (both in percent)
# --------------------------------------------------------------------------- #
def test_l2_relative_error_on_known_vectors() -> None:
    y_true = np.array([3.0, 4.0])
    # ||y_true|| = 5, ||y_true - y_pred|| = 1 -> 20 %.
    y_pred = np.array([3.0, 3.0])
    assert l2_relative_error(y_true, y_pred) == pytest.approx(20.0)


def test_l2_relative_error_is_a_ratio_of_norms() -> None:
    y_true = np.array([1.0, 2.0, 2.0])  # norm 3
    y_pred = np.array([1.0, 2.0, 0.0])  # difference norm 2
    assert l2_relative_error(y_true, y_pred) == pytest.approx(100 * 2 / 3)


def test_l1_relative_error_on_known_vectors() -> None:
    y_true = np.array([1.0, 2.0, 2.0])  # sum |.| = 5
    y_pred = np.array([1.0, 2.0, 0.0])  # sum |difference| = 2
    assert l1_relative_error(y_true, y_pred) == pytest.approx(40.0)


def test_errors_vanish_for_a_perfect_prediction() -> None:
    y_true = np.array([0.5, -1.5, 2.0])
    assert l2_relative_error(y_true, y_true) == pytest.approx(0.0)
    assert l1_relative_error(y_true, y_true) == pytest.approx(0.0)


def test_errors_are_hundred_percent_for_a_zero_prediction() -> None:
    y_true = np.array([2.0, -3.0, 6.0])
    zeros = np.zeros_like(y_true)
    assert l2_relative_error(y_true, zeros) == pytest.approx(100.0)
    assert l1_relative_error(y_true, zeros) == pytest.approx(100.0)


def test_errors_return_python_floats() -> None:
    y_true = np.array([1.0, 1.0])
    assert isinstance(l2_relative_error(y_true, y_true * 2), float)
    assert isinstance(l1_relative_error(y_true, y_true * 2), float)


# --------------------------------------------------------------------------- #
# Row wise errors (fractions, not percent)
# --------------------------------------------------------------------------- #
def test_relative_error_per_trajectory_l2() -> None:
    y_true = np.array([[3.0, 4.0], [1.0, 0.0]])
    y_pred = np.array([[3.0, 3.0], [0.0, 0.0]])
    errors = relative_error_per_trajectory(y_true, y_pred)
    assert errors.shape == (2,)
    assert errors == pytest.approx([0.2, 1.0])


def test_relative_error_per_trajectory_l1() -> None:
    y_true = np.array([[1.0, 2.0, 2.0], [4.0, 0.0, 0.0]])
    y_pred = np.array([[1.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    errors = relative_error_per_trajectory(y_true, y_pred, order=1)
    assert errors == pytest.approx([2 / 5, 2 / 4])


def test_relative_error_per_trajectory_matches_the_scalar_metric() -> None:
    y_true = np.array([[3.0, 4.0]])
    y_pred = np.array([[3.0, 3.0]])
    assert relative_error_per_trajectory(y_true, y_pred)[0] * 100 == pytest.approx(
        l2_relative_error(y_true[0], y_pred[0])
    )


# --------------------------------------------------------------------------- #
# PICP
# --------------------------------------------------------------------------- #
def test_picp_full_and_empty_coverage() -> None:
    mean = np.zeros(10)
    std = np.ones(10)
    assert picp(mean, mean, std) == pytest.approx(1.0)
    assert picp(np.full(10, 10.0), mean, std) == pytest.approx(0.0)


def test_picp_on_a_constructed_interval() -> None:
    """Six of ten values sit inside mean +/- 1.96 std."""
    mean = np.zeros(10)
    std = np.ones(10)
    inside = [0.0, 1.0, -1.0, 1.95, -1.95, 0.5]
    outside = [1.97, -1.97, 5.0, -5.0]
    y_true = np.array(inside + outside)
    assert picp(y_true, mean, std) == pytest.approx(0.6)


def test_picp_uses_closed_interval_boundaries() -> None:
    mean = np.zeros(2)
    std = np.ones(2)
    y_true = np.array([Z_95, -Z_95])
    assert picp(y_true, mean, std) == pytest.approx(1.0)


def test_picp_honours_a_custom_z() -> None:
    mean = np.zeros(4)
    std = np.ones(4)
    y_true = np.array([0.5, 0.9, 1.5, 3.0])
    assert picp(y_true, mean, std, z=1.0) == pytest.approx(0.5)
    assert picp(y_true, mean, std, z=2.0) == pytest.approx(0.75)


def test_picp_works_on_matrices_and_varying_std() -> None:
    y_true = np.array([[0.0, 10.0], [0.0, 0.0]])
    mean = np.zeros((2, 2))
    std = np.array([[1.0, 1.0], [1.0, 0.0]])
    # (0, 0) and (1, 0) are covered, (0, 1) is far out, (1, 1) has a zero width
    # interval that still contains the exact value.
    assert picp(y_true, mean, std) == pytest.approx(0.75)


def test_z_95_constant() -> None:
    assert Z_95 == pytest.approx(1.96)
