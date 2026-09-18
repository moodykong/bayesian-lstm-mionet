"""Sub-sequence sampling and masking of the length-variant input functions.

Both routines take a ``(u, x, t)`` split as produced by
:func:`blstm_mionet.data.datasets.split_dataset` and return the four arrays
consumed by the operators: the masked input trajectory, the current state
``x_n``, the target ``x_next`` and the time parameters
``(t_n, h, t_n + h)`` scaled back to physical time.

The sampling logic (including the internal re-seeding with 999) is carried
over verbatim from ``src/utils/data_utils.py``: the published results depend
on it.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.interpolate import interp1d

SEED = 999


def prepare_local_predict_dataset(
    data: tuple,
    search_len: int = 10,
    search_num: int = 5,
    search_random: bool = True,
    offset: float = 0,
    t_max: float | None = 10,
    state_component: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data set for local prediction problem based on history inputs."""
    ## Step 1: collect and copy data
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    u_data, x_data, t_data = data
    u = copy.deepcopy(u_data) if u_data is not None else None
    x = copy.deepcopy(x_data)
    x = x[:, :, [state_component]]
    t = copy.deepcopy(t_data)
    nData = x.shape[0]
    t = t.squeeze()
    t_s = t[1] - t[0]
    t_max = t_max / t_s if t_max is not None else t.size
    offset = offset / t_s

    if offset + search_len > t.size:
        raise ValueError(
            f"The offset {offset} + search length {search_len} is larger than the total time length {t.size}."
        )

    # Determine the maximum time index
    t_max = int(t_max)
    offset = int(offset)

    # Truncate the data
    u = u[:, 0:t_max, :] if u is not None else None
    x = x[:, 0:t_max, :]
    t = t[0:t_max]

    ## Step 2: interpolate and sample data

    # Interpolate the data
    splines_x = []
    for i in range(nData):
        spline_x_i = interp1d(
            np.arange(t.size),
            x[i, :, :].squeeze(),
            kind="cubic",
            fill_value=1e-6,
            bounds_error=False,
        )
        splines_x.append(spline_x_i)

    t_params = _sample_time_parameters(
        nData=nData,
        n_time=t.size,
        search_len=search_len,
        search_num=search_num,
        search_random=search_random,
        offset=offset,
    )

    ## Step 3: mask the data
    mask_idxs = np.ones((search_num, nData, t.size))  # [N_search, N_sample, N_time]
    mask_idxs *= np.arange(t.size)
    mask_idxs = mask_idxs > t_params[:, :, [0]]

    input_traj = u if u is not None else x  # [N_sample, N_time, C]
    input_traj_masked = np.repeat(
        np.expand_dims(input_traj, axis=0), search_num, axis=0
    )  # [N_search, N_sample, N_time, C]
    input_traj_masked[mask_idxs] = 0.0

    ## Step 4: unfold the data
    # Log the data index
    data_idxs = np.arange(nData, dtype=int)
    data_idxs = np.repeat(
        np.expand_dims(data_idxs, axis=0), search_num, axis=0
    )  # [N_search, N_sample]

    # Unfold the data
    input_traj_masked = input_traj_masked.reshape(
        -1, t.size, input_traj_masked.shape[-1]
    )  # [N_search * N_sample, N_time, C]
    t_params = t_params.reshape(-1, t_params.shape[-1])  # [N_search * N_sample, 3]
    data_idxs = data_idxs.reshape(-1)  # [N_search * N_sample]

    ## Step 5: get the current and next state
    x_n = np.zeros((t_params.shape[0], x.shape[-1]))
    x_next = np.zeros((t_params.shape[0], x.shape[-1]))

    for i in range(t_params.shape[0]):
        idxs_i = data_idxs[i]
        spline_x_i = splines_x[idxs_i]
        t_n = int(t_params[i, 0])
        t_next = t_params[i, 2]

        x_n[i, :] = x[idxs_i, t_n, :].reshape(-1, 1)
        x_next[i, :] = spline_x_i(t_next).reshape(-1, 1)

    if verbose:
        print(
            f"Shapes for training input={input_traj_masked.shape}, x_n={x_n.shape}, x_next={x_next.shape}"
        )

    return (input_traj_masked, x_n, x_next, t_params * t_s)


def prepare_future_local_predict_dataset(
    data: tuple,
    search_len: int = 10,
    search_num: int = 5,
    search_random: bool = True,
    offset: float = 0,
    t_max: float | None = 10,
    state_component: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare data set for local prediction problem based on near future inputs.

    Used by ``DeepONet_Local``, which sees the control at the prediction time
    instead of the past input function; it therefore requires a control ``u``.
    """
    ## Step 1: collect and copy data
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    u_data, x_data, t_data = data
    if u_data is None:
        raise ValueError(
            "DeepONet_Local requires an input function u; the dataset has none."
        )
    u = copy.deepcopy(u_data)
    x = copy.deepcopy(x_data)
    x = x[:, :, [state_component]]
    t = copy.deepcopy(t_data)
    nData = x.shape[0]
    t = t.squeeze()
    t_s = t[1] - t[0]
    t_max = t_max / t_s if t_max is not None else t.size
    offset = offset / t_s

    if offset + search_len > t.size:
        raise ValueError(
            f"The offset {offset} + search length {search_len} is larger than the total time length {t.size}."
        )

    # Determine the maximum time index
    t_max = int(t_max)
    offset = int(offset)

    # Truncate the data
    u = u[:, 0:t_max, :]
    x = x[:, 0:t_max, :]
    t = t[0:t_max]

    ## Step 2: interpolate and sample data

    # Interpolate the data
    splines_u = []
    splines_x = []
    for i in range(nData):
        spline_x_i = interp1d(
            np.arange(t.size),
            x[i, :, :].squeeze(),
            kind="cubic",
            fill_value=1e-6,
            bounds_error=False,
        )
        splines_x.append(spline_x_i)

        spline_u_i = interp1d(
            np.arange(t.size),
            u[i, :, :].squeeze(),
            kind="cubic",
            fill_value=1e-6,
            bounds_error=False,
        )
        splines_u.append(spline_u_i)

    t_params = _sample_time_parameters(
        nData=nData,
        n_time=t.size,
        search_len=search_len,
        search_num=search_num,
        search_random=search_random,
        offset=offset,
    )

    ## Step 3: repeat the data to match the search_num
    input_traj = u  # [N_sample, N_time, C]
    input_traj = np.repeat(
        np.expand_dims(input_traj, axis=0), search_num, axis=0
    )  # [N_search, N_sample, N_time, C]

    ## Step 4: unfold the data
    # Log the data index
    data_idxs = np.arange(nData, dtype=int)
    data_idxs = np.repeat(
        np.expand_dims(data_idxs, axis=0), search_num, axis=0
    )  # [N_search, N_sample]

    # Unfold the data
    input_traj = input_traj.reshape(
        -1, t.size, input_traj.shape[-1]
    )  # [N_search * N_sample, N_time, C]
    t_params = t_params.reshape(-1, t_params.shape[-1])  # [N_search * N_sample, 3]
    data_idxs = data_idxs.reshape(-1)  # [N_search * N_sample]

    ## Step 5: get the current and next state
    x_n = np.zeros((t_params.shape[0], x.shape[-1]))
    x_next = np.zeros((t_params.shape[0], x.shape[-1]))
    input_traj_next = np.zeros((t_params.shape[0], u.shape[-1]))

    for i in range(t_params.shape[0]):
        idxs_i = data_idxs[i]
        spline_x_i = splines_x[idxs_i]
        t_n = int(t_params[i, 0])
        t_next = t_params[i, 2]

        x_n[i, :] = x[idxs_i, t_n, :].reshape(-1, 1)
        x_next[i, :] = spline_x_i(t_next).reshape(-1, 1)

        spline_u_i = splines_u[idxs_i]
        input_traj_next[i, :] = spline_u_i(t_next).reshape(-1, 1)

    if verbose:
        print(
            f"Shapes for training input={input_traj.shape}, x_n={x_n.shape}, x_next={x_next.shape}"
        )

    return (input_traj_next, x_n, x_next, t_params * t_s)


def _sample_time_parameters(
    nData: int,
    n_time: int,
    search_len: int,
    search_num: int,
    search_random: bool,
    offset: int,
) -> np.ndarray:
    """Draw ``[start_index, interval_length, end_index]`` for every sub-sequence.

    Identical in both preparation routines; factored out without changing the
    arithmetic or the order of the random draws.
    """
    if search_random:
        t_params = np.random.rand(search_num, nData, 3)
        # Randomly select the starting index
        t_params[:, :, 0] = (
            offset + t_params[:, :, 0] * (n_time - offset - search_len * 2)
        ).astype(int)
        # Randomly select the length of the interval
        t_params[:, :, 1] = t_params[:, :, 1] * search_len
        # Compute the ending index
        t_params[:, :, 2] = t_params[:, :, 0] + t_params[:, :, 1]
    else:
        t_params = np.ones((search_num, nData, 3))
        idx_end = n_time - offset - search_len * 2
        # Select the starting index
        t_params[:, :, 0] = (
            ((offset + np.linspace(1, idx_end, search_num))).astype(int).reshape(-1, 1)
        )
        # Select the length of the interval
        t_params[:, :, 1] = t_params[:, :, 1] * 0.5 * search_len
        # Compute the ending index
        t_params[:, :, 2] = t_params[:, :, 0] + t_params[:, :, 1]
    return t_params
