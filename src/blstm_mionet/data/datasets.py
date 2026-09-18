"""Torch datasets, train/test splitting and feature scaling."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

SPLIT_SEED = 456


class TorchDataset(Dataset):
    """Tensor dataset yielding ``((input_data, x_n, t_params), x_next)``.

    Formerly ``Dataset_Torch``.
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

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.input_data[idx],
            self.x_n[idx],
            self.t_params[idx],
        ), self.x_next[idx]


class DatasetStatistics:
    """Per-feature mean/std/min/max of a prepared dataset.

    Formerly ``Dataset_Stat``.
    """

    def __init__(
        self,
        input_data: np.ndarray | None = None,
        x_n: np.ndarray | None = None,
        x_next: np.ndarray | None = None,
        t_params: np.ndarray | None = None,
    ) -> None:
        empty = np.array([])
        self.input_data = empty if input_data is None else input_data
        self.x_n = empty if x_n is None else x_n
        self.t_params = empty if t_params is None else t_params
        self.x_next = empty if x_next is None else x_next

    def update_statistics(self) -> None:
        assert self.input_data.shape[0]

        # For branch memory input
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


def normalize(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Zero mean / unit variance scaling."""
    return (data - mean) / (std + eps)


def unnormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize` (without the epsilon)."""
    return data * std + mean


def minmaxscale(
    data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray
) -> np.ndarray:
    """Scale to ``[0, 1]``.

    Columns with ``max_val == min_val`` (for example the always-masked tail of
    the branch-memory input) are mapped to 0 instead of producing NaNs.
    """
    span = np.asarray(max_val - min_val, dtype=float)
    span = np.where(span == 0, 1.0, span)
    return (data - min_val) / span


def unminmaxscale(
    data: np.ndarray, min_val: np.ndarray, max_val: np.ndarray
) -> np.ndarray:
    """Inverse of :func:`minmaxscale`."""
    return data * (max_val - min_val) + min_val


def split_dataset(
    database: dict, test_size: float = 0.2, verbose: bool = True, rng: int = SPLIT_SEED
) -> tuple[tuple, tuple]:
    """Split a raw dataset dictionary into ``(u, x, t)`` train and test halves.

    ``test_size == 1.0`` puts every trajectory in the test half (used by the
    inference entry points), ``test_size == 0.0`` in the train half.
    """
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
        raise ValueError(f"Invalid test size {test_size}")
    if verbose:
        n_train = 0 if x_train is None else x_train.shape[0]
        n_test = 0 if x_test is None else x_test.shape[0]
        print(f"Split dataset into {n_train} training and {n_test} test trajectories.")
    t_train = t
    t_test = t
    return (u_train, x_train, t_train), (u_test, x_test, t_test)


def scale_and_to_tensor(
    dataset: DatasetStatistics,
    scale_mode: str = "normalize",
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale a prepared dataset and move it to ``device`` as float tensors."""
    if device is None:
        device = torch.device("cpu")
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

    if verbose:
        memory_size = (
            input_data.element_size() * input_data.nelement()
            + x_n.element_size() * x_n.nelement()
            + x_next.element_size() * x_next.nelement()
            + t_params.element_size() * t_params.nelement()
        ) / (1024**2)
        print(f"Data memory size: {int(memory_size)} MB")

    return (input_data, x_n, x_next, t_params)


def prepare_torch_dataset(
    split: tuple,
    preparer: Callable[..., tuple],
    state_component: int = 0,
    search_len: int = 10,
    search_num: int = 10,
    search_random: bool = True,
    offset: float = 0.0,
    t_max: float | None = None,
    scale_mode: str = "",
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[TorchDataset, DatasetStatistics, int]:
    """Run ``preparer`` on a ``(u, x, t)`` split and scale it onto ``device``.

    Returns the torch dataset, the unscaled statistics object (needed to
    un-scale predictions) and the number of state features.
    """
    if device is None:
        device = torch.device("cpu")
    input_masked, x_n, x_next, t_params = preparer(
        data=split,
        state_component=state_component,
        search_len=search_len,
        search_num=search_num,
        search_random=search_random,
        offset=offset,
        t_max=t_max,
        verbose=verbose,
    )
    statistics = DatasetStatistics(input_masked, x_n, x_next, t_params)
    statistics.update_statistics()
    state_feature_num: int = x_n.shape[-1]

    tensors = scale_and_to_tensor(
        statistics, scale_mode=scale_mode, device=device, verbose=verbose
    )
    return TorchDataset(*tensors), statistics, state_feature_num
