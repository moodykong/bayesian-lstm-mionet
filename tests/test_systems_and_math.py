"""Vector fields, the RK4 integrator and the Gaussian random field."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from blstm_mionet.data.systems import (
    SYSTEMS,
    control_formula,
    control_grf,
    lorentz,
    pendulum,
)
from blstm_mionet.utils.math import Solution, grf_1d, integrate, runge_kutta
from blstm_mionet.utils.seed import set_seed


def _decay(x: np.ndarray, u: float) -> np.ndarray:
    """``dx/dt = -2 x``; the closed form solution is ``x(t) = x0 exp(-2 t)``."""
    return -2.0 * x


def _harmonic(x: np.ndarray, u: float) -> np.ndarray:
    """``x'' = -x`` written as a first order system (closed form: cos/-sin)."""
    return np.array([x[1], -x[0]])


def _zero_control(t: float, x: np.ndarray) -> float:
    return 0.0


# --------------------------------------------------------------------------- #
# runge_kutta
# --------------------------------------------------------------------------- #
def test_runge_kutta_matches_the_taylor_polynomial() -> None:
    """For a linear field one RK4 step is the 4th order Taylor polynomial."""
    h = 0.1
    step = runge_kutta(_decay, np.array([1.0]), 0.0, h)
    z = -2.0 * h
    expected = 1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    assert step[0] == pytest.approx(expected, rel=1e-14)
    # ... which is exp(-2h) up to the fifth order truncation error.
    assert step[0] == pytest.approx(np.exp(z), abs=1e-5)


def test_runge_kutta_is_fourth_order_accurate() -> None:
    """100 steps of h = 0.01 stay within the O(h^4) global error bound."""
    h = 0.01
    x = np.array([1.0])
    for _ in range(100):
        x = runge_kutta(_decay, x, 0.0, h)
    assert x[0] == pytest.approx(np.exp(-2.0), rel=1e-8)


def test_runge_kutta_error_shrinks_like_h_to_the_fourth() -> None:
    def error(h: float) -> float:
        x = np.array([1.0])
        for _ in range(int(round(1.0 / h))):
            x = runge_kutta(_decay, x, 0.0, h)
        return abs(float(x[0]) - np.exp(-2.0))

    coarse, fine = error(0.1), error(0.05)
    # Halving the step must divide the error by roughly 2**4 = 16.
    assert 10.0 < coarse / fine < 25.0


def test_runge_kutta_on_a_two_dimensional_system() -> None:
    h = 0.001
    x = np.array([1.0, 0.0])
    for _ in range(1000):
        x = runge_kutta(_harmonic, x, None, h)
    assert x[0] == pytest.approx(np.cos(1.0), abs=1e-10)
    assert x[1] == pytest.approx(-np.sin(1.0), abs=1e-10)


# --------------------------------------------------------------------------- #
# integrate
# --------------------------------------------------------------------------- #
def test_integrate_shapes_and_time_grid() -> None:
    n_steps = 25
    h = 0.01
    soln = integrate(runge_kutta, _decay, _zero_control, np.array([1.0]), h, n_steps)

    assert isinstance(soln, Solution)
    # ``x`` carries the initial state plus N - 1 integration steps ...
    assert soln.x.shape == (n_steps, 1)
    # ... while ``t`` and ``u`` are logged once per step.
    assert soln.t.shape == (n_steps - 1, 1)
    assert soln.u.shape == (n_steps - 1, 1)

    assert soln.t[0, 0] == pytest.approx(0.0)
    assert soln.t[-1, 0] == pytest.approx((n_steps - 2) * h)
    assert np.allclose(soln.u, 0.0)


def test_integrate_matches_the_closed_form_solution() -> None:
    n_steps = 101
    h = 0.01
    soln = integrate(runge_kutta, _decay, _zero_control, np.array([1.0]), h, n_steps)
    times = np.arange(n_steps) * h
    assert np.allclose(soln.x[:, 0], np.exp(-2.0 * times), atol=1e-10)


def test_integrate_multi_dimensional_state() -> None:
    soln = integrate(
        runge_kutta, lorentz, _zero_control, np.array([1.0, 1.0, 1.0]), 0.01, 40
    )
    assert soln.x.shape == (40, 3)
    assert np.isfinite(soln.x).all()


def test_integrate_records_the_control() -> None:
    soln = integrate(
        runge_kutta, _harmonic, control_formula, np.array([1.0, 0.0]), 0.1, 6
    )
    assert soln.u.shape == (5, 1)
    assert soln.u[0, 0] == pytest.approx(np.sin(0.0))
    assert soln.u[1, 0] == pytest.approx(np.sin(0.1 / 2))


# --------------------------------------------------------------------------- #
# Vector fields
# --------------------------------------------------------------------------- #
def test_lorentz_vector_field_at_a_point() -> None:
    """sigma (y - x), x (rho - z) - y, x y - beta z at (1, 2, 3)."""
    value = lorentz(np.array([1.0, 2.0, 3.0]), u=0.0)
    assert value.shape == (3,)
    assert value == pytest.approx([10.0, 23.0, -6.0])


def test_lorentz_is_autonomous() -> None:
    x = np.array([2.0, -1.0, 5.0])
    assert np.array_equal(lorentz(x, 0.0), lorentz(x, 1e6))


def test_lorentz_fixed_point_at_the_origin() -> None:
    assert lorentz(np.array([0.0, 0.0, 0.0]), 0.0) == pytest.approx([0.0, 0.0, 0.0])


def test_pendulum_vector_field_at_a_point() -> None:
    """(theta, omega) = (0.5, 1.0) under a torque of 2.0 N m."""
    value = pendulum(np.array([0.5, 1.0]), u=2.0)
    assert value.shape == (2,)
    assert value[0] == pytest.approx(1.0)
    assert value[1] == pytest.approx(-1.0847468005608472, rel=1e-12)


def test_pendulum_torque_enters_linearly() -> None:
    x = np.array([0.5, 1.0])
    delta = pendulum(x, u=3.0)[1] - pendulum(x, u=2.0)[1]
    # 1 / (0.25 m l^2 + inertia) with m = l = 1 and inertia = 1/12.
    assert delta == pytest.approx(1.0 / (0.25 + 1 / 12))


def test_pendulum_hanging_equilibrium() -> None:
    assert pendulum(np.array([0.0, 0.0]), u=0.0) == pytest.approx([0.0, 0.0])


def test_systems_registry() -> None:
    assert set(SYSTEMS) == {"pendulum", "lorentz"}
    assert SYSTEMS["lorentz"] is lorentz
    assert SYSTEMS["pendulum"] is pendulum


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
def test_control_formula_is_sin_of_half_t() -> None:
    assert control_formula(0.0, np.array([0.0, 0.0])) == pytest.approx(0.0)
    assert control_formula(np.pi, np.array([0.0, 0.0])) == pytest.approx(1.0)


def test_control_grf_reads_the_angular_velocity() -> None:
    control = control_grf(lambda value: 3.0 * value)
    assert control(0.0, np.array([10.0, 2.0])) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# Gaussian random field
# --------------------------------------------------------------------------- #
def test_grf_1d_is_callable_deterministic_and_seed_dependent() -> None:
    """One test for the whole GRF contract: each sample costs a 2000^2 Cholesky."""
    query = np.linspace(-5.0, 5.0, 11)

    set_seed(123)
    field = grf_1d(a=0.01, nu=1.0)
    set_seed(123)
    same_seed = grf_1d(a=0.01, nu=1.0)
    set_seed(124)
    other_seed = grf_1d(a=0.01, nu=1.0)

    # callable output
    assert isinstance(field, CubicSpline)
    values = field(query)
    assert values.shape == query.shape
    assert np.isfinite(values).all()
    assert float(field(0.0)) == pytest.approx(float(field(np.array(0.0))))

    # deterministic under set_seed, and a different seed gives a different field
    assert np.array_equal(values, same_seed(query))
    assert not np.allclose(values, other_seed(query))
