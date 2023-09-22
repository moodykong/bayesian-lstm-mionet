#!/usr/bin/env python
# coding: utf-8

import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import utils.pytorch_utils as ptu

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


# Data generator
class Dataset_Torch(Dataset):
    def __init__(
        self,
        input_data: np.ndarray,
        x_n: np.ndarray,
        x_next: np.ndarray,
        t_params: np.ndarray,
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


# Operator dataset
class Dataset_Stat:
    def __init__(
        self,
        input_data: np.ndarray = np.array([]),
        x_n: np.ndarray = np.array([]),
        x_next: np.ndarray = np.array([]),
        t_params: np.ndarray = np.array([]),
    ) -> None:
        self.input_data = input_data
        self.x_n = x_n
        self.x_next = x_next
        self.t_params = t_params

    def update_statistics(self) -> None:
        assert self.input_data.shape[0]

        # For brach input
        self.input_data_mean = np.mean(self.input_data, axis=0)
        self.input_data_std = np.std(self.input_data, axis=0)
        self.input_data_min = np.min(self.input_data, axis=0)
        self.input_data_max = np.max(self.input_data, axis=0)

        # For local time input
        self.t_params_mean = np.mean(self.t_params, axis=0)
        self.t_params_std = np.std(self.t_params, axis=0)
        self.t_params_min = np.min(self.t_params, axis=0)
        self.t_params_max = np.max(self.t_params, axis=0)

        # For trunk input
        self.x_n_mean = np.mean(self.x_n, axis=0)
        self.x_n_std = np.std(self.x_n, axis=0)
        self.x_n_min = np.min(self.x_n, axis=0)
        self.x_n_max = np.max(self.x_n, axis=0)

        # For DeepONet output
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
    t = database["t"].T

    all_indices = list(range(x.shape[0]))  # Number of trajectories
    train_ind, test_ind = train_test_split(
        all_indices, test_size=test_size, random_state=rng
    )

    # As the data index is on axis=0
    train_database = database[train_ind]
    test_database = database[test_ind]

    u_train = u[train_ind, :, :] if u is not None else None
    u_test = u[test_ind, :, :] if u is not None else None
    x_train = x[train_ind, :, :]
    x_test = x[test_ind, :, :]
    t_train = t
    t_test = t
    return (u_train, x_train, t_train), (u_test, x_test, t_test)


# Prepare dataset for local prediction problem
def prepare_local_predict_dataset(
    data: list,
    search_len: int = 10,
    search_num: int = 5,
    search_random: bool = True,
    offset: int = 0,
    verbose: bool = True,
) -> list:
    ## Step 1: collect and copy data
    u_data, x_data, t_data = data
    u = copy.deepcopy(u_data) if u_data is not None else None
    x = copy.deepcopy(x_data)
    t = copy.deepcopy(t_data)
    nData = x.shape[0]

    ## Step 2: interpolate and sample data

    # Interpolate the data
    splines_x = []
    for i in range(nData):
        splines_x_i = []
        for j in range(x.shape[-1]):
            spline_x_ij = interp1d(
                np.arange(t.size),
                x[i, :, j],
                kind="cubic",
                fill_value=1e-6,
                bounds_error=False,
            )
            splines_x_i.append(spline_x_ij)
        splines_x.append(splines_x_i)

    # Sample the data
    if search_random:
        t_params = np.random.rand(search_num, nData, 3)
        t_params[:, :, 0] = (
            offset + t_params[:, :, 0] * (t.size - offset - search_len)
        ).astype(
            int
        )  # Randomly select the starting index
        t_params[:, :, 1] = (
            t_params[:, :, 1] * search_len
        )  # Randomly select the length of the interval
        t_params[:, :, 2] = (
            t_params[:, :, 0] + t_params[:, :, 1]
        )  # Compute the ending index
    else:
        t_params = np.ones((search_num, nData, 3))
        idx_end = t.size - search_len
        t_params[:, :, 0] = (
            (offset + np.linspace(1, idx_end, search_num)).reshape(-1, 1).astype(int)
        )  # Select the starting index
        t_params[:, :, 1] = (
            t_params[:, :, 1] * 0.5 * search_len
        )  # Select the length of the interval
        t_params[:, :, 2] = (
            t_params[:, :, 0] + t_params[:, :, 1]
        )  # Compute the ending index

    ## Step 3: mask the data
    mask_idxs = np.ones((search_num, nData, t.size))
    mask_idxs *= np.arange(t.size)
    mask_idxs = mask_idxs > t_params[:, :, [0]]

    input_traj = u if u is not None else x
    input_masked = np.repeat(np.expand_dims(input_traj, axis=0), search_num, axis=0)
    input_masked[mask_idxs] = 0.0

    ## Step 4: unfold the data
    # Log the data index
    data_idxs = np.arange(nData, dtype=int)
    data_idxs = np.repeat(np.expand_dims(data_idxs, axis=0), search_num, axis=0)

    # Unfold the data
    input_masked = input_masked.reshape(-1, t.size, input_masked.shape[-1])
    t_params = t_params.reshape(-1, t_params.shape[-1])
    data_idxs = data_idxs.reshape(-1)

    ## Step 5: get the current and next state
    x_n = np.zeros((t_params.shape[0], input_masked.shape[-1]))
    x_next = np.zeros((t_params.shape[0], input_masked.shape[-1]))

    for i in range(t_params.shape[0]):
        idxs_i = data_idxs[i]
        spline_x_i = splines_x[idxs_i]
        t_n = t_params[i, 0]
        t_next = t_params[i, 2]

        for j in range(input_masked.shape[-1]):
            x_n[i, j] = x[idxs_i, t_n, j]
            x_next[i, j] = spline_x_i[j](t_next)

    if verbose:
        print(
            "Shapes for LSTM-MIONet training input={}, x_n={}, x_next={}".format(
                input_masked.shape, x_n.shape, x_next.shape
            )
        )

    return (input_masked, x_n, x_next, t_params)


# scale and move dataset to torch
def scale_and_to_device(
    dataset: Dataset_Stat,
    scale_mode: str = "normalize",
    device: torch.device = torch.device("cuda"),
    requires_grad: bool = True,
) -> list:
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

    if requires_grad:
        input_data = torch.from_numpy(input_data).float().to(device).requires_grad_()
        x_n = torch.from_numpy(x_n).float().to(device).requires_grad_()
        x_next = torch.from_numpy(x_next).float().to(device).requires_grad_()
        t_params = torch.from_numpy(t_params).float().to(device).requires_grad_()

    else:
        input_data = torch.from_numpy(input_data).float().to(device)
        x_n = torch.from_numpy(x_n).float().to(device)
        x_next = torch.from_numpy(x_next).float().to(device)
        t_params = torch.from_numpy(t_params).float().to(device)

    return (input_data, x_n, x_next, t_params)


# unnormalize data
def unnormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return data * std + mean


# unminmax scale data
def unminmaxscale(data, min_val, max_val):
    return data * (max_val - min_val) + min_val


# prepare test data
def prepare_test_data(
    u_data: np.ndarray, y_data: np.ndarray, G_data: np.ndarray, nLocs: int = 300
) -> list:
    u = copy.deepcopy(u_data)
    y = copy.deepcopy(y_data)
    G = copy.deepcopy(G_data)

    f = interpolate.interp1d(y, G, kind="cubic", copy=False, assume_sorted=True)
    out_loc = np.linspace(y[0], y[-1], nLocs)
    G_interp = f(out_loc)

    u_vec = np.tile(u.reshape(1, 1), (nLocs, 1)).reshape(nLocs, -1)
    y_vec = out_loc.reshape(nLocs, 1)
    G_vec = G_interp.reshape(nLocs, 1)

    return (u_vec, y_vec, G_vec)


def preparing_DeepOnet_test_for_optimization(u_data, y_data, nLocs):
    N = nLocs
    u_list = []
    y_list = []

    u = copy.deepcopy(u_data)
    y = copy.deepcopy(y_data)

    out_loc = np.linspace(y[0], y[-1], N)

    u_list.append(np.tile(u.reshape(1, 20), (N, 1)))
    y_list.append(out_loc.reshape(N, 1))

    u_vec = np.stack(u_list).reshape(N, -1)
    y_vec = np.stack(y_list).reshape(N, -1)

    return u_vec, y_vec


def prepare_test_data_HF(
    u_data: np.ndarray,
    y_data: np.ndarray,
    G_data: np.ndarray,
    nLocs: int,
    scale_mode: str,
    datasetLF: Any,
    modelLF: Any,
) -> list:
    u = copy.deepcopy(u_data)
    y = copy.deepcopy(y_data)
    G = copy.deepcopy(G_data)

    f = interpolate.interp1d(y, G, kind="cubic", copy=False, assume_sorted=True)
    out_loc = np.linspace(y[0], y[-1], nLocs)
    G_HF = f(out_loc)

    uu = copy.deepcopy(u)
    u_test = np.tile(uu.reshape(1, 1), (nLocs, 1)).reshape(nLocs, -1)
    y_test = out_loc.reshape(nLocs, 1)
    # dataset.update_statistics() why we do not use here
    ## scale
    if scale_mode == "normalize":
        u_test = normalize(u_test, datasetLF.dataU_mean, datasetLF.dataU_std)
        y_test = normalize(y_test, datasetLF.dataY_mean, datasetLF.dataY_std)

    elif scale_mode == "min-max":
        u_test = minmaxscale(u_test, datasetLF.dataU_min, datasetLF.dataU_max)
        y_test = minmaxscale(y_test, datasetLF.dataY_min, datasetLF.dataY_max)

    ## move to torch
    U = torch.from_numpy(u_test).float().to(ptu.device)
    Y = torch.from_numpy(y_test).float().to(ptu.device)

    ## forward pass
    with torch.no_grad():
        GG = modelLF((U, Y))

    if scale_mode == "normalize":
        G_LF = unnormalize(
            GG.detach().cpu().numpy(), datasetLF.dataG_mean, datasetLF.dataG_std
        ).flatten()

    elif scale_mode == "min-max":
        G_LF = unminmaxscale(
            GG.detach().cpu().numpy(), datasetLF.dataG_min, datasetLF.dataG_max
        ).flatten()

    u_vec = np.tile(u.reshape(1, 1), (nLocs, 1)).reshape(nLocs, -1)
    y_vec = out_loc.reshape(nLocs, 1)
    G_vec = G_HF.reshape(nLocs, 1) - G_LF.reshape(nLocs, 1)

    return u_vec, y_vec, G_vec, G_HF.reshape(nLocs, 1), G_LF.reshape(nLocs, 1)
