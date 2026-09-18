"""Numerical routines: RK4 integration and a 1-D Gaussian random field.

The implementations are carried over verbatim from ``src/utils/math_utils.py``
because the published results depend on them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import cholesky


@dataclass
class Solution:
    """Container for an integrated trajectory (replaces the old ``dotdict``)."""

    x: Any = field(default_factory=list)
    t: Any = field(default_factory=list)
    u: Any = field(default_factory=list)


def grf_1d(a: float = 0.01, nu: float = 1.0) -> CubicSpline:
    """Draw a sample path of a 1-D Gaussian random field as a cubic spline.

    ``a`` is the correlation length and ``nu`` the smoothness exponent of the
    correlation function ``exp(-(h / a) ** (2 * nu))``.
    """

    # Correlation function
    def rho(h, a=a, nu=nu):
        return np.exp(-((h / a) ** (2 * nu)))

    # Space discretization
    x = np.linspace(-10, 10, 2000)

    # Distance matrix
    H = np.abs(x[:, np.newaxis] - x)

    # Covariance matrix
    sigma = rho(H)
    sigma += 1e-6 * np.eye(sigma.shape[0])

    # Cholesky factorization
    L = cholesky(sigma, lower=True)

    # Independent standard Gaussian random variables
    z = np.random.normal(size=x.size)

    # Gaussian random field
    y = np.dot(L, z)
    spline = CubicSpline(x, y)
    return spline


def runge_kutta(
    f: Callable[[np.ndarray, Any], np.ndarray], x: np.ndarray, u: Any, h: float
) -> np.ndarray:
    """One classical fourth order Runge-Kutta step of ``f(x, u)``."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * h * k1, u)
    k3 = f(x + 0.5 * h * k2, u)
    k4 = f(x + h * k3, u)
    next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6
    return next_x


def integrate(
    method: Callable[..., np.ndarray],
    f: Callable[[np.ndarray, Any], np.ndarray],
    control: Callable[[float, np.ndarray], Any],
    x0: np.ndarray,
    h: float,
    N: int,
) -> Solution:
    """Integrate ``f`` for ``N`` steps of size ``h`` under ``control``.

    ``method`` is kept for signature compatibility with the original code; the
    integration itself always uses :func:`runge_kutta`, exactly as before.
    """
    soln = Solution()
    soln.x = []
    soln.t = []
    soln.u = []

    x = x0

    t = 0 * h
    u = control(t, x)
    soln.x.append(x)

    for n in range(1, N):
        # log previous control
        soln.t.append(t)
        soln.u.append(u)
        # compute next state
        x_next = runge_kutta(f, x, u, h)
        # log next state
        soln.x.append(x_next)
        x = x_next
        t = n * h
        u = control(t, x)

    soln.x = np.vstack(soln.x)
    soln.t = np.vstack(soln.t)
    soln.u = np.vstack(soln.u)

    return soln
