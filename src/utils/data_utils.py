#!/usr/bin/env python
# coding: utf-8

import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import torch
import utils.torch_utils as torch_utils

from scipy import interpolate
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Any


# Dotdict
class dotdict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Dataset_Torch(Dataset):
    """
    Generate dataset for pytorch
    """

    def __init__(
        self,
        input_data: torch.Tensor,
        x_n: torch.Tensor,
        x_next: torch.Tensor,
        t_params: torch.Tensor,
    ) -> None:
        self.input_data = input_data
        self.x_n = x_n
        self.x_next = x_next
        self.t_params = t_params
        self.len = self.t_params.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.input_data[idx], self.x_n[idx], self.t_params[idx]), self.x_next[
            idx
        ]


class Dataset_Stat:
    """
    Generate dataset for statistics
    """

    def __init__(
        self,
        input_data: np.ndarray = np.array([]),
        x_n: np.ndarray = np.array([]),
        x_next: np.ndarray = np.array([]),
        t_params: np.ndarray = np.array([]),
    ) -> None:
        self.input_data = input_data
        self.x_n = x_n
        self.t_params = t_params
        self.x_next = x_next

    def update_statistics(self) -> None:
        assert self.input_data.shape[0]

        # For brach memory input
        self.input_data_mean = np.mean(self.input_data, axis=0)
        self.input_data_std = np.std(self.input_data, axis=0)
        self.input_data_min = np.min(self.input_data, axis=0)
        self.input_data_max = np.max(self.input_data, axis=0)

        # For branch state input
        self.x_n_mean = np.mean(self.x_n, axis=0)
        self.x_n_std = np.std(self.x_n, axis=0)
        self.x_n_min = np.min(self.x_n, axis=0)
        self.x_n_max = np.max(self.x_n, axis=0)

        # For trunk input
        self.t_params_mean = np.mean(self.t_params, axis=0)
        self.t_params_std = np.std(self.t_params, axis=0)
        self.t_params_min = np.min(self.t_params, axis=0)
        self.t_params_max = np.max(self.t_params, axis=0)

        # For output
        self.x_next_mean = np.mean(self.x_next, axis=0)
        self.x_next_std = np.std(self.x_next, axis=0)
        self.x_next_min = np.min(self.x_next, axis=0)
        self.x_next_max = np.max(self.x_next, axis=0)


# normalize data
def normalize(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    return (data - mean) / (std + eps)


# minmax scale data
def minmaxscale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


# prepare dataset
def split_dataset(
    database: dict, test_size: float = 0.2, verbose: bool = True, rng: int = 456
) -> Any:
    u = np.transpose(database["u"], (2, 0, 1)) if "u" in database else None
    x = np.transpose(database["x"], (2, 0, 1))  # [N_sample, N_time, C]
    t = database["t"].T  # [N_time,]

    if test_size > 0.0 and test_size < 1.0:
        all_indices = list(range(x.shape[0]))  # Number of trajectories
        train_ind, test_ind = train_test_split(
            all_indices, test_size=test_size, random_state=rng
        )

        # As the data index is on axis=0
        u_train = u[train_ind, :, :] if u is not None else None
        u_test = u[test_ind, :, :] if u is not None else None
        x_train = x[train_ind, :, :]
        x_test = x[test_ind, :, :]
    elif test_size == 0.0:
        u_train = u
        u_test = None
        x_train = x
        x_test = None
    elif test_size == 1.0:
        u_train = None
        u_test = u
        x_train = None
        x_test = x
    else:
        raise ValueError("Invalid test size {}".format(test_size))
    t_train = t
    t_test = t
    return (u_train, x_train, t_train), (u_test, x_test, t_test)


def prepare_local_predict_dataset(
    data: list,
    search_len: int = 10,
    search_num: int = 5,
    search_random: bool = True,
    offset: int = 0,
    t_max: Any = 10,
    state_component: int = 0,
    verbose: bool = True,
) -> list:
    """Prepare data set for local prediction problem based on history inputs"""
    ## Step 1: collect and copy data
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
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
            "The offset {} + search length {} is larger than the total time length {}.".format(
                offset, search_len, t.size
            )
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

    # Sample the data
    if search_random:
        t_params = np.random.rand(search_num, nData, 3)
        # Randomly select the starting index
        t_params[:, :, 0] = (
            offset + t_params[:, :, 0] * (t.size - offset - search_len * 2)
        ).astype(int)
        # Randomly select the length of the interval
        t_params[:, :, 1] = t_params[:, :, 1] * search_len
        # Compute the ending index
        t_params[:, :, 2] = t_params[:, :, 0] + t_params[:, :, 1]

    else:
        t_params = np.ones((search_num, nData, 3))
        idx_end = t.size - offset - search_len * 2
        # Select the starting index
        t_params[:, :, 0] = (
            ((offset + np.linspace(1, idx_end, search_num))).astype(int).reshape(-1, 1)
        )
        # Select the length of the interval
        t_params[:, :, 1] = t_params[:, :, 1] * 0.5 * search_len
        # Compute the ending index
        t_params[:, :, 2] = t_params[:, :, 0] + t_params[:, :, 1]

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
            "Shapes for training input={}, x_n={}, x_next={}".format(
                input_traj_masked.shape, x_n.shape, x_next.shape
            )
        )

    return (input_traj_masked, x_n, x_next, t_params * t_s)


def prepare_future_local_predict_dataset(
    data: list,
    search_len: int = 10,
    search_num: int = 5,
    search_random: bool = True,
    offset: int = 0,
    t_max: Any = 10,
    state_component: int = 0,
    verbose: bool = True,
) -> list:
    """Prepare data set for local prediction problem based on near future inputs"""
    ## Step 1: collect and copy data
    seed = 999
    np.random.seed(seed)
    torch.manual_seed(seed)
    u_data, x_data, t_data = data
    if u_data is None:
        raise ValueError("The input data u_data is None.")
    else:
        pass
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
            "The offset {} + search length {} is larger than the total time length {}.".format(
                offset, search_len, t.size
            )
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

    # Sample the data
    if search_random:
        t_params = np.random.rand(search_num, nData, 3)
        # Randomly select the starting index
        t_params[:, :, 0] = (
            offset + t_params[:, :, 0] * (t.size - offset - search_len * 2)
        ).astype(int)
        # Randomly select the length of the interval
        t_params[:, :, 1] = t_params[:, :, 1] * search_len
        # Compute the ending index
        t_params[:, :, 2] = t_params[:, :, 0] + t_params[:, :, 1]

    else:
        t_params = np.ones((search_num, nData, 3))
        idx_end = t.size - offset - search_len * 2
        # Select the starting index
        t_params[:, :, 0] = (
            ((offset + np.linspace(1, idx_end, search_num))).astype(int).reshape(-1, 1)
        )
        # Select the length of the interval
        t_params[:, :, 1] = t_params[:, :, 1] * 0.5 * search_len
        # Compute the ending index
        t_params[:, :, 2] = t_params[:, :, 0] + t_params[:, :, 1]

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
            "Shapes for training input={}, x_n={}, x_next={}".format(
                input_traj.shape, x_n.shape, x_next.shape
            )
        )

    return (input_traj_next, x_n, x_next, t_params * t_s)


# scale and move dataset to torch
def scale_and_to_tensor(
    dataset: Dataset_Stat,
    scale_mode: str = "normalize",
    device: torch.device = torch.device("cpu"),
) -> Any:
    ## step 1: copy dataset data
    input_data = copy.deepcopy(dataset.input_data)
    x_n = copy.deepcopy(dataset.x_n)
    x_next = copy.deepcopy(dataset.x_next)
    t_params = copy.deepcopy(dataset.t_params)

    ## step 2: scale the data
    if scale_mode == "normalize":
        input_data = normalize(
            input_data, dataset.input_data_mean, dataset.input_data_std
        )
        x_n = normalize(x_n, dataset.x_n_mean, dataset.x_n_std)
        x_next = normalize(x_next, dataset.x_next_mean, dataset.x_next_std)
        t_params = normalize(t_params, dataset.t_params_mean, dataset.t_params_std)

    elif scale_mode == "min-max":
        input_data = minmaxscale(
            input_data, dataset.input_data_min, dataset.input_data_max
        )
        x_n = minmaxscale(x_n, dataset.x_n_min, dataset.x_n_max)
        x_next = minmaxscale(x_next, dataset.x_next_min, dataset.x_next_max)
        t_params = minmaxscale(t_params, dataset.t_params_min, dataset.t_params_max)

    input_data = torch.from_numpy(input_data).float().to(device)
    x_n = torch.from_numpy(x_n).float().to(device)
    x_next = torch.from_numpy(x_next).float().to(device)
    t_params = torch.from_numpy(t_params).float().to(device)

    memory_size = (
        input_data.element_size() * input_data.nelement()
        + x_n.element_size() * x_n.nelement()
        + x_next.element_size() * x_next.nelement()
        + t_params.element_size() * t_params.nelement()
    ) / (1024**2)
    print("Data memory size: {} MB".format(int(memory_size)))

    return (input_data, x_n, x_next, t_params)


# unnormalize data
def unnormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return data * std + mean


# unminmax scale data
def unminmaxscale(data, min_val, max_val):
    return data * (max_val - min_val) + min_val
