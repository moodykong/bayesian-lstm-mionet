"""Vector fields and control functions of the benchmark systems.

The formulas are copied verbatim from ``src/config/data_config.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def pendulum(x: np.ndarray, u: float) -> np.ndarray:
    """Damped, driven pendulum: ``x = (theta, omega)``, control torque ``u``."""
    m = 1
    length = 1
    inertia = (1 / 12) * m * (length**2)
    b = 0.01
    g = 9.81
    theta, omega = x
    f_theta = omega
    f_omega = (1 / (0.25 * m * (length**2) + inertia)) * (
        u - b * omega - 0.5 * m * length * g * np.sin(theta)
    )
    return np.array([f_theta, f_omega])


def lorentz(x: np.ndarray, u: float) -> np.ndarray:
    """Lorenz system with the classical parameters (sigma, rho, beta).

    The system is autonomous; ``u`` is accepted and ignored so that the
    integrator can treat both benchmarks uniformly.
    """
    sigma = 10
    rho = 28
    beta = 8 / 3
    x, y, z = x
    f_x = sigma * (y - x)
    f_y = x * (rho - z) - y
    f_z = x * y - beta * z
    return np.array([f_x, f_y, f_z])


def control_formula(t: float, x: np.ndarray) -> float:
    """Deterministic out-of-distribution control ``u(t) = sin(t / 2)``."""
    return np.sin(t / 2)


def control_grf(func: Callable[[float], float]) -> Callable[[float, np.ndarray], float]:
    """Wrap a Gaussian random field sample into a state feedback control.

    The field is evaluated at the angular velocity ``x[1]``, not at the time
    ``t``: this is how the research code behind the paper generated the
    pendulum data, although the paper writes the control as ``u(t)``.
    """

    def control(t: float, x: np.ndarray) -> float:
        theta_dot = x[1]
        return func(theta_dot)

    return control


SYSTEMS = {"pendulum": pendulum, "lorentz": lorentz}
