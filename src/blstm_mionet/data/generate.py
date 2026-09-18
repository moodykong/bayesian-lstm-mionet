"""Trajectory generation for the ODE benchmarks (Lorenz and pendulum).

This is the ODE branch of the old ``src/database_generator.py``; the Ausgrid
branch lives in :mod:`blstm_mionet.data.ausgrid`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from blstm_mionet.config import DataConfig
from blstm_mionet.data.systems import SYSTEMS, control_formula, control_grf
from blstm_mionet.utils.math import grf_1d, integrate, runge_kutta


def generate_trajectories(config: DataConfig) -> dict[str, Any]:
    """Integrate ``config.n_sample`` trajectories of the configured system.

    Returns a dictionary with ``x`` of shape ``[N_time, C, N_sample]``, ``t`` of
    shape ``[N_time, 1]`` and, for non-autonomous systems, ``u`` of shape
    ``[N_time, 1, N_sample]``.  Samples containing NaNs are re-drawn.
    """
    try:
        ode_system = SYSTEMS[config.system]
    except KeyError as exc:
        raise ValueError(f"no vector field for system {config.system!r}") from exc

    # Define the time parameters
    T = config.t_max
    h = config.step_size
    N_time = int(T / h)
    N_sample = config.n_sample

    # Define the initial states
    x_init_pts = np.array(config.x_init_pts, dtype=float).reshape(-1, 2)
    x_num = x_init_pts.shape[0]
    x_init = np.random.uniform(x_init_pts[:, 0], x_init_pts[:, 1], (N_sample, x_num))

    # Generate the data
    data: dict[str, Any] = {}
    x = u = None
    soln = None
    i = 0
    progress_bar = tqdm(
        total=N_sample,
        desc=f"Generating {N_sample} data ...",
        dynamic_ncols=True,
        disable=not config.verbose,
    )
    while i < N_sample:
        # Define the control
        if config.control == "designate":
            control = control_formula
        elif config.control == "gaussian":
            grf = grf_1d(a=config.grf.a, nu=config.grf.nu)
            control = control_grf(grf)
        elif config.control is None:
            control = control_formula
        else:
            raise ValueError(f"invalid control function {config.control!r}")

        soln = integrate(runge_kutta, ode_system, control, x_init[i], h, N_time)

        if np.isnan(soln.x).sum() + np.isnan(soln.u).sum() > 0:
            continue
        x = (
            np.dstack((x, soln.x[:-1, :]))
            if i > 0
            else np.expand_dims(soln.x[:-1, :], 2)
        )
        u = np.dstack((u, soln.u)) if i > 0 else np.expand_dims(soln.u, 2)
        i += 1
        progress_bar.update(1)
    progress_bar.close()

    data["x"] = x
    data["t"] = soln.t
    if config.control is not None:
        data["u"] = u
    return data


def save_dataset(data: dict[str, Any], path: str | Path) -> Path:
    """Save a dataset dictionary as a pickled ``.npy`` file."""
    output = Path(path)
    if output.parent != Path(""):
        output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.save(handle, data)
    return output


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load a dataset dictionary written by :func:`save_dataset`."""
    return np.load(str(path), allow_pickle=True).item()
