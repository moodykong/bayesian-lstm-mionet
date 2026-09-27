"""Trajectory generation, splitting, sub-sequence masking and scaling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from blstm_mionet.config import DataConfig
from blstm_mionet.data.datasets import (
    DatasetStatistics,
    TorchDataset,
    minmaxscale,
    normalize,
    scale_and_to_tensor,
    split_dataset,
    unminmaxscale,
    unnormalize,
)
from blstm_mionet.data.generate import (
    generate_trajectories,
    load_dataset,
    save_dataset,
)
from blstm_mionet.data.masking import (
    prepare_future_local_predict_dataset,
    prepare_local_predict_dataset,
)
from blstm_mionet.utils.seed import set_seed
from conftest import TINY_N_TIME

SEARCH_LEN = 2
SEARCH_NUM = 3


# --------------------------------------------------------------------------- #
# generate_trajectories
# --------------------------------------------------------------------------- #
def test_generate_lorentz_shapes_and_finiteness(lorentz_data) -> None:
    assert set(lorentz_data) == {"x", "t"}  # autonomous: no control is stored
    assert lorentz_data["x"].shape == (TINY_N_TIME, 3, 6)
    assert lorentz_data["t"].shape == (TINY_N_TIME, 1)
    assert np.isfinite(lorentz_data["x"]).all()
    assert not np.isnan(lorentz_data["x"]).any()
    # The 6 trajectories start from different initial values.
    assert len(np.unique(lorentz_data["x"][0, 0, :])) == 6


def test_generate_pendulum_shapes_and_finiteness(pendulum_data) -> None:
    assert set(pendulum_data) == {"x", "t", "u"}  # non-autonomous
    assert pendulum_data["x"].shape == (TINY_N_TIME, 2, 4)
    assert pendulum_data["u"].shape == (TINY_N_TIME, 1, 4)
    assert pendulum_data["t"].shape == (TINY_N_TIME, 1)
    assert np.isfinite(pendulum_data["x"]).all()
    assert np.isfinite(pendulum_data["u"]).all()


def test_generate_time_grid_is_uniform(lorentz_data) -> None:
    times = lorentz_data["t"].squeeze()
    assert times[0] == pytest.approx(0.0)
    assert np.allclose(np.diff(times), 0.01)


def test_generate_is_deterministic_under_set_seed() -> None:
    config = DataConfig(
        system="lorentz",
        t_max=0.2,
        step_size=0.01,
        n_sample=2,
        x_init_pts=[[-17.0, 20.0], [-23.0, 28.0], [0.0, 50.0]],
        control=None,
        verbose=False,
    )
    set_seed(999)
    first = generate_trajectories(config)
    set_seed(999)
    second = generate_trajectories(config)
    assert np.array_equal(first["x"], second["x"])


def test_generate_rejects_an_unknown_system() -> None:
    config = DataConfig(system="duffing", x_init_pts=[[0.0, 1.0]], verbose=False)
    with pytest.raises(ValueError, match="no vector field for system 'duffing'"):
        generate_trajectories(config)


# --------------------------------------------------------------------------- #
# save / load round trip
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(tmp_path: Path, lorentz_data) -> None:
    path = save_dataset(lorentz_data, tmp_path / "nested" / "dataset.npy")
    assert path.is_file()
    assert path.parent.is_dir()  # the parent directory was created

    reloaded = load_dataset(path)
    assert set(reloaded) == set(lorentz_data)
    for key, value in lorentz_data.items():
        assert np.array_equal(reloaded[key], value)


def test_save_and_load_round_trip_with_control(tmp_path: Path, pendulum_data) -> None:
    path = save_dataset(pendulum_data, tmp_path / "pendulum.npy")
    reloaded = load_dataset(str(path))
    assert np.array_equal(reloaded["u"], pendulum_data["u"])


# --------------------------------------------------------------------------- #
# split_dataset
# --------------------------------------------------------------------------- #
def test_split_dataset_sizes_and_layout(lorentz_data) -> None:
    (u_train, x_train, t_train), (u_test, x_test, t_test) = split_dataset(
        lorentz_data, test_size=0.5, verbose=False
    )
    # [N_time, C, N_sample] is transposed to [N_sample, N_time, C].
    assert x_train.shape == (3, TINY_N_TIME, 3)
    assert x_test.shape == (3, TINY_N_TIME, 3)
    assert u_train is None and u_test is None  # no control in the Lorenz data
    assert t_train.shape == t_test.shape == (1, TINY_N_TIME)


def test_split_dataset_keeps_the_control(pendulum_data) -> None:
    (u_train, x_train, _), (u_test, x_test, _) = split_dataset(
        pendulum_data, test_size=0.25, verbose=False
    )
    assert x_train.shape == (3, TINY_N_TIME, 2)
    assert x_test.shape == (1, TINY_N_TIME, 2)
    assert u_train.shape == (3, TINY_N_TIME, 1)
    assert u_test.shape == (1, TINY_N_TIME, 1)


def test_split_dataset_is_deterministic(lorentz_data) -> None:
    first, _ = split_dataset(lorentz_data, test_size=0.5, verbose=False)
    second, _ = split_dataset(lorentz_data, test_size=0.5, verbose=False)
    assert np.array_equal(first[1], second[1])


def test_split_dataset_all_train(lorentz_data) -> None:
    (_, x_train, _), (_, x_test, _) = split_dataset(
        lorentz_data, test_size=0.0, verbose=False
    )
    assert x_train.shape[0] == 6
    assert x_test is None


def test_split_dataset_all_test(lorentz_data) -> None:
    (_, x_train, _), (_, x_test, _) = split_dataset(
        lorentz_data, test_size=1.0, verbose=False
    )
    assert x_train is None
    assert x_test.shape[0] == 6


def test_split_dataset_rejects_an_invalid_size(lorentz_data) -> None:
    with pytest.raises(ValueError, match="Invalid test size"):
        split_dataset(lorentz_data, test_size=1.5, verbose=False)


# --------------------------------------------------------------------------- #
# prepare_local_predict_dataset
# --------------------------------------------------------------------------- #
@pytest.fixture
def lorentz_split(lorentz_data):
    _, test_split = split_dataset(lorentz_data, test_size=1.0, verbose=False)
    return test_split


@pytest.fixture
def pendulum_split(pendulum_data):
    _, test_split = split_dataset(pendulum_data, test_size=1.0, verbose=False)
    return test_split


def test_prepare_local_predict_shapes(lorentz_split) -> None:
    n_traj = lorentz_split[1].shape[0]
    masked, x_n, x_next, t_params = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        state_component=0,
        verbose=False,
    )
    n_rows = SEARCH_NUM * n_traj
    assert masked.shape == (n_rows, TINY_N_TIME, 1)
    assert x_n.shape == (n_rows, 1)
    assert x_next.shape == (n_rows, 1)
    assert t_params.shape == (n_rows, 3)
    assert np.isfinite(masked).all()
    assert np.isfinite(x_next).all()


def test_prepare_local_predict_selects_the_state_component(lorentz_split) -> None:
    _, x_n, _, _ = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=1,
        search_random=False,
        t_max=None,
        state_component=2,
        verbose=False,
    )
    x = lorentz_split[1]
    # z(t) of the Lorenz system is drawn from [0, 50] and stays positive.
    assert np.all(x_n >= 0.0)
    assert x_n.max() <= x[:, :, 2].max() + 1e-9


def test_prepare_local_predict_masks_everything_after_the_cut(lorentz_split) -> None:
    masked, x_n, _, t_params = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    t_s = float(lorentz_split[2].squeeze()[1] - lorentz_split[2].squeeze()[0])

    for row in range(masked.shape[0]):
        t_n = int(round(float(t_params[row, 0]) / t_s))
        # Everything strictly after the cut index is zeroed ...
        assert np.count_nonzero(masked[row, t_n + 1 :, :]) == 0
        # ... and the history up to the cut is kept (the last kept entry is x_n).
        assert np.count_nonzero(masked[row, : t_n + 1, :]) > 0
        assert masked[row, t_n, 0] == pytest.approx(float(x_n[row, 0]))


def test_prepare_local_predict_time_parameters(lorentz_split) -> None:
    _, _, _, t_params = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    t_s = 0.01
    # t_next = t_n + h, and h never exceeds search_len time steps.
    assert np.allclose(t_params[:, 2], t_params[:, 0] + t_params[:, 1])
    assert np.all(t_params[:, 1] >= 0.0)
    assert np.all(t_params[:, 1] <= SEARCH_LEN * t_s + 1e-12)
    assert np.all(t_params[:, 0] >= 0.0)
    assert np.all(t_params[:, 2] <= (TINY_N_TIME - 1) * t_s)


def test_prepare_local_predict_x_next_lies_inside_the_trajectory(
    lorentz_split,
) -> None:
    x = lorentz_split[1]
    n_traj = x.shape[0]
    _, _, x_next, _ = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    for row in range(x_next.shape[0]):
        traj = x[row % n_traj, :, 0]
        span = float(traj.max() - traj.min())
        assert traj.min() - 0.05 * span <= x_next[row, 0] <= traj.max() + 0.05 * span


def test_prepare_local_predict_is_deterministic(lorentz_split) -> None:
    """The routine draws from its own seeded generator, so repeated calls agree."""
    kwargs = dict(
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    np.random.seed(0)
    first = prepare_local_predict_dataset(lorentz_split, **kwargs)
    np.random.seed(12345)
    second = prepare_local_predict_dataset(lorentz_split, **kwargs)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_prepare_local_predict_leaves_the_global_rng_alone(lorentz_split) -> None:
    """Masking must not re-seed NumPy or PyTorch behind the caller's back."""
    set_seed(2024)
    expected_np = np.random.rand(3)
    expected_torch = torch.rand(3)

    set_seed(2024)
    prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    assert np.array_equal(np.random.rand(3), expected_np)
    assert torch.equal(torch.rand(3), expected_torch)


def test_prepare_local_predict_deterministic_grid(lorentz_split) -> None:
    """``search_random=False`` walks a deterministic grid of start indices."""
    _, _, _, t_params = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=False,
        t_max=None,
        verbose=False,
    )
    t_s = 0.01
    # h is fixed at 0.5 * search_len steps, and the starts are increasing.
    assert np.allclose(t_params[:, 1], 0.5 * SEARCH_LEN * t_s)
    n_traj = lorentz_split[1].shape[0]
    starts = t_params[:, 0].reshape(SEARCH_NUM, n_traj)[:, 0]
    assert np.all(np.diff(starts) > 0)


def test_prepare_local_predict_uses_the_control_when_present(pendulum_split) -> None:
    """For a non-autonomous system the masked input is ``u``, not ``x``."""
    masked, _, _, _ = prepare_local_predict_dataset(
        pendulum_split,
        search_len=SEARCH_LEN,
        search_num=1,
        search_random=False,
        t_max=None,
        verbose=False,
    )
    u = pendulum_split[0]
    assert masked.shape[-1] == u.shape[-1] == 1
    non_zero = masked[0, :, 0] != 0.0
    assert np.allclose(masked[0, non_zero, 0], u[0, non_zero, 0])


def test_prepare_local_predict_truncates_with_t_max(lorentz_split) -> None:
    masked, _, _, _ = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=1,
        search_random=False,
        t_max=0.5,
        verbose=False,
    )
    assert masked.shape[1] == 50


def test_prepare_local_predict_rejects_a_too_large_offset(lorentz_split) -> None:
    with pytest.raises(ValueError, match="is larger than the total time length"):
        prepare_local_predict_dataset(
            lorentz_split,
            search_len=SEARCH_LEN,
            search_num=1,
            offset=100.0,
            t_max=None,
            verbose=False,
        )


# --------------------------------------------------------------------------- #
# prepare_future_local_predict_dataset
# --------------------------------------------------------------------------- #
def test_prepare_future_local_predict_shapes(pendulum_split) -> None:
    n_traj = pendulum_split[1].shape[0]
    u_next, x_n, x_next, t_params = prepare_future_local_predict_dataset(
        pendulum_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    n_rows = SEARCH_NUM * n_traj
    # The "future local" routine returns the control at t_n + h, not a sequence.
    assert u_next.shape == (n_rows, 1)
    assert x_n.shape == (n_rows, 1)
    assert x_next.shape == (n_rows, 1)
    assert t_params.shape == (n_rows, 3)
    assert np.isfinite(u_next).all()


def test_prepare_future_local_predict_u_lies_inside_the_control(
    pendulum_split,
) -> None:
    u = pendulum_split[0]
    n_traj = u.shape[0]
    u_next, _, _, _ = prepare_future_local_predict_dataset(
        pendulum_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    for row in range(u_next.shape[0]):
        control = u[row % n_traj, :, 0]
        span = float(control.max() - control.min())
        assert (
            control.min() - 0.05 * span <= u_next[row, 0] <= control.max() + 0.05 * span
        )


def test_prepare_future_local_predict_requires_a_control(lorentz_split) -> None:
    with pytest.raises(ValueError, match="DeepONet_Local requires an input function"):
        prepare_future_local_predict_dataset(
            lorentz_split,
            search_len=SEARCH_LEN,
            search_num=SEARCH_NUM,
            t_max=None,
            verbose=False,
        )


def test_prepare_future_local_predict_is_deterministic(pendulum_split) -> None:
    kwargs = dict(
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    np.random.seed(3)
    first = prepare_future_local_predict_dataset(pendulum_split, **kwargs)
    np.random.seed(77)
    second = prepare_future_local_predict_dataset(pendulum_split, **kwargs)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_both_preparers_share_the_time_parameters(pendulum_split) -> None:
    kwargs = dict(
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    _, x_n_a, x_next_a, t_a = prepare_local_predict_dataset(pendulum_split, **kwargs)
    _, x_n_b, x_next_b, t_b = prepare_future_local_predict_dataset(
        pendulum_split, **kwargs
    )
    assert np.array_equal(t_a, t_b)
    assert np.array_equal(x_n_a, x_n_b)
    assert np.array_equal(x_next_a, x_next_b)


# --------------------------------------------------------------------------- #
# Scaling helpers and TorchDataset
# --------------------------------------------------------------------------- #
@pytest.fixture
def statistics(lorentz_split) -> DatasetStatistics:
    prepared = prepare_local_predict_dataset(
        lorentz_split,
        search_len=SEARCH_LEN,
        search_num=SEARCH_NUM,
        search_random=True,
        t_max=None,
        verbose=False,
    )
    stats = DatasetStatistics(*prepared)
    stats.update_statistics()
    return stats


def test_scale_and_to_tensor_without_scaling(statistics: DatasetStatistics) -> None:
    input_data, x_n, x_next, t_params = scale_and_to_tensor(
        statistics, scale_mode="", verbose=False
    )
    for tensor in (input_data, x_n, x_next, t_params):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
    assert np.allclose(x_n.numpy(), statistics.x_n.astype(np.float32), atol=1e-6)
    assert np.allclose(
        t_params.numpy(), statistics.t_params.astype(np.float32), atol=1e-6
    )


def test_scale_and_to_tensor_normalize(statistics: DatasetStatistics) -> None:
    _, x_n, x_next, t_params = scale_and_to_tensor(
        statistics, scale_mode="normalize", verbose=False
    )
    assert x_n.numpy().mean() == pytest.approx(0.0, abs=1e-4)
    assert x_n.numpy().std() == pytest.approx(1.0, abs=1e-3)
    assert x_next.numpy().mean() == pytest.approx(0.0, abs=1e-4)
    # The interval column of t_params is not constant, so it normalises too.
    assert t_params.numpy()[:, 1].mean() == pytest.approx(0.0, abs=1e-4)


def test_scale_and_to_tensor_min_max(statistics: DatasetStatistics) -> None:
    _, x_n, x_next, _ = scale_and_to_tensor(
        statistics, scale_mode="min-max", verbose=False
    )
    for tensor in (x_n, x_next):
        assert float(tensor.min()) >= -1e-6
        assert float(tensor.max()) <= 1.0 + 1e-6


def test_scale_and_to_tensor_min_max_does_not_produce_nans(
    statistics: DatasetStatistics,
) -> None:
    input_data, _, _, _ = scale_and_to_tensor(
        statistics, scale_mode="min-max", verbose=False
    )
    assert np.isfinite(input_data.numpy()).all()


def test_normalize_and_minmax_round_trip() -> None:
    data = np.array([[1.0, 2.0], [3.0, 5.0], [10.0, -1.0]])
    mean, std = data.mean(axis=0), data.std(axis=0)
    assert np.allclose(
        unnormalize(normalize(data, mean, std), mean, std), data, atol=1e-6
    )

    low, high = data.min(axis=0), data.max(axis=0)
    scaled = minmaxscale(data, low, high)
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)
    assert np.allclose(unminmaxscale(scaled, low, high), data)


def test_torch_dataset_item_layout(statistics: DatasetStatistics) -> None:
    tensors = scale_and_to_tensor(statistics, scale_mode="", verbose=False)
    dataset = TorchDataset(*tensors)
    assert len(dataset) == dataset.len == tensors[0].shape[0]

    (input_data, x_n, t_params), x_next = dataset[0]
    assert input_data.shape == (TINY_N_TIME, 1)
    assert x_n.shape == (1,)
    assert t_params.shape == (3,)
    assert x_next.shape == (1,)


def test_prepare_torch_dataset_reports_the_feature_count(
    lorentz_dataset: Path, make_torch_dataset
) -> None:
    dataset, stats, state_feature_num = make_torch_dataset(
        lorentz_dataset, search_num=SEARCH_NUM, search_len=SEARCH_LEN
    )
    assert state_feature_num == 1
    assert dataset.len == SEARCH_NUM * 6
    assert isinstance(stats, DatasetStatistics)
    assert stats.x_n.shape == (SEARCH_NUM * 6, 1)
