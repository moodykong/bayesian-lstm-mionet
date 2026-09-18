"""Single step evaluation, recursive rollouts and the Bayesian ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from blstm_mionet.config import InferConfig
from blstm_mionet.data.generate import load_dataset, save_dataset
from blstm_mionet.evaluation.evaluate import (
    evaluate_ensemble,
    evaluate_recursive,
    evaluate_single_step,
)
from blstm_mionet.evaluation.plotting import ensure_directory
from blstm_mionet.training import tracking
from blstm_mionet.utils.seed import set_seed
from conftest import build_torch_dataset

SEARCH_NUM = 4
N_TRAJECTORIES = 6


def infer_config(**overrides) -> InferConfig:
    config = InferConfig(
        datafile="unused",
        search_len=2,
        search_num=SEARCH_NUM,
        search_random=False,
        scale_mode="",
        batch_size=8,
        plot_trajs=False,
        plot_idxs=[0],
        device="cpu",
        verbose=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def trained_model(adam_run, cpu_device):
    model = tracking.load_model(adam_run["model_uri"], cpu_device)
    model.eval()
    return model


@pytest.fixture
def test_dataset(lorentz_dataset: Path):
    dataset, _, _ = build_torch_dataset(
        lorentz_dataset, search_num=SEARCH_NUM, search_len=2, search_random=False
    )
    return dataset


@pytest.fixture
def single_trajectory_torch_dataset(single_trajectory_dataset: Path):
    dataset, _, _ = build_torch_dataset(
        single_trajectory_dataset,
        search_num=SEARCH_NUM,
        search_len=2,
        search_random=False,
    )
    return dataset


# --------------------------------------------------------------------------- #
# Single step
# --------------------------------------------------------------------------- #
def test_evaluate_single_step_returns_finite_errors(
    trained_model, test_dataset
) -> None:
    result = evaluate_single_step(infer_config(), trained_model, test_dataset)

    assert set(result) == {
        "L1_error_mat",
        "L2_error_mat",
        "t_next",
        "y_pred",
        "y_true",
        "figure_path",
    }
    assert result["y_pred"].shape == (N_TRAJECTORIES, SEARCH_NUM)
    assert result["y_true"].shape == (N_TRAJECTORIES, SEARCH_NUM)
    assert result["t_next"].shape == (N_TRAJECTORIES, SEARCH_NUM)
    assert result["L2_error_mat"].shape == (N_TRAJECTORIES,)
    assert result["L1_error_mat"].shape == (N_TRAJECTORIES,)
    assert np.isfinite(result["L2_error_mat"]).all()
    assert np.isfinite(result["L1_error_mat"]).all()
    assert (result["L2_error_mat"] >= 0.0).all()
    assert np.isfinite(result["y_pred"]).all()
    assert result["figure_path"] is None  # plotting was switched off


def test_evaluate_single_step_matches_a_manual_forward_pass(
    trained_model, test_dataset
) -> None:
    result = evaluate_single_step(infer_config(), trained_model, test_dataset)
    with torch.no_grad():
        expected = trained_model(
            [test_dataset.input_data, test_dataset.x_n, test_dataset.t_params]
        )
    assert np.allclose(
        result["y_pred"],
        expected.numpy().reshape(SEARCH_NUM, -1).T,
        atol=1e-5,
    )


def test_evaluate_single_step_writes_a_figure(
    trained_model, test_dataset, tmp_path: Path
) -> None:
    figure_dir = tmp_path / "figures"
    result = evaluate_single_step(
        infer_config(plot_trajs=True, figure_dir=str(figure_dir), plot_idxs=[0, 2]),
        trained_model,
        test_dataset,
    )
    assert figure_dir.is_dir()
    assert (figure_dir / "infer_trajs_0.png").is_file()
    assert (figure_dir / "infer_trajs_2.png").is_file()
    assert result["figure_path"].endswith("infer_trajs_2.png")


def test_evaluate_single_step_verbose_prints_the_error_tables(
    trained_model, test_dataset, capsys
) -> None:
    evaluate_single_step(infer_config(verbose=True), trained_model, test_dataset)
    printed = capsys.readouterr().out
    assert "L1-relative Error %" in printed
    assert "L2-relative Error %" in printed
    assert "L2-relative error: mean" in printed


# --------------------------------------------------------------------------- #
# Recursive rollout
# --------------------------------------------------------------------------- #
def test_evaluate_recursive_with_full_teacher_forcing(
    trained_model, single_trajectory_torch_dataset
) -> None:
    dataset = single_trajectory_torch_dataset
    config = infer_config(recursive=True, teacher_forcing_prob=1.0, autonomous=True)

    set_seed(999)
    recursive = evaluate_recursive(config, trained_model, dataset)
    single = evaluate_single_step(config, trained_model, dataset)

    assert recursive["y_pred"].shape == (1, SEARCH_NUM)
    assert np.isfinite(recursive["y_pred"]).all()
    assert np.isfinite(recursive["L2_error_mat"]).all()
    # With probability 1 every step is fed the true state, so the rollout
    # reproduces the single step predictions exactly.
    assert np.allclose(recursive["y_pred"], single["y_pred"], atol=1e-5)
    assert np.allclose(recursive["y_true"], single["y_true"], atol=1e-5)


def test_evaluate_recursive_without_teacher_forcing(
    trained_model, single_trajectory_torch_dataset
) -> None:
    dataset = single_trajectory_torch_dataset
    config = infer_config(recursive=True, teacher_forcing_prob=0.0, autonomous=True)

    set_seed(999)
    free_running = evaluate_recursive(config, trained_model, dataset)
    set_seed(999)
    forced = evaluate_recursive(
        infer_config(recursive=True, teacher_forcing_prob=1.0, autonomous=True),
        trained_model,
        dataset,
    )

    assert free_running["y_pred"].shape == (1, SEARCH_NUM)
    assert np.isfinite(free_running["y_pred"]).all()
    assert np.isfinite(free_running["L2_error_mat"]).all()
    # The untrained-ish model drifts once it consumes its own predictions.
    assert not np.allclose(free_running["y_pred"], forced["y_pred"], atol=1e-5)


def test_evaluate_recursive_non_autonomous_keeps_the_input(
    trained_model, single_trajectory_torch_dataset
) -> None:
    """``autonomous=False`` leaves the input function untouched."""
    config = infer_config(recursive=True, teacher_forcing_prob=0.0, autonomous=False)
    set_seed(999)
    result = evaluate_recursive(config, trained_model, single_trajectory_torch_dataset)
    assert result["y_pred"].shape == (1, SEARCH_NUM)
    assert np.isfinite(result["y_pred"]).all()


def test_evaluate_recursive_writes_a_figure(
    trained_model, single_trajectory_torch_dataset, tmp_path: Path
) -> None:
    figure_dir = tmp_path / "recursive_figures"
    set_seed(999)
    result = evaluate_recursive(
        infer_config(
            recursive=True,
            teacher_forcing_prob=1.0,
            plot_trajs=True,
            figure_dir=str(figure_dir),
        ),
        trained_model,
        single_trajectory_torch_dataset,
    )
    assert (figure_dir / "infer_trajs_recursive_0.png").is_file()
    assert result["figure_path"].endswith("infer_trajs_recursive_0.png")


def test_evaluate_recursive_with_two_trajectories_matches_single_step(
    trained_model, lorentz_dataset: Path, tmp_path: Path
) -> None:
    raw = load_dataset(lorentz_dataset)
    two = {"x": raw["x"][:, :, :2], "t": raw["t"]}
    path = save_dataset(two, tmp_path / "two_trajectories.npy")
    dataset, _, _ = build_torch_dataset(
        path, search_num=SEARCH_NUM, search_len=2, search_random=False
    )
    config = infer_config(recursive=True, teacher_forcing_prob=1.0, autonomous=True)

    set_seed(999)
    recursive = evaluate_recursive(config, trained_model, dataset)
    single = evaluate_single_step(config, trained_model, dataset)

    assert recursive["y_pred"].shape == (2, SEARCH_NUM)
    assert np.allclose(recursive["y_pred"], single["y_pred"], atol=1e-5)


# --------------------------------------------------------------------------- #
# Bayesian ensemble
# --------------------------------------------------------------------------- #
@pytest.fixture
def ensemble_members(resgld_run) -> list[Path]:
    local_dir = tracking.download_artifacts(
        resgld_run["run_id"], tracking.ENSEMBLE_ARTIFACT_PATH
    )
    return sorted(Path(local_dir).glob("member_*.pt"))


@pytest.fixture
def ensemble_model(resgld_run, cpu_device):
    return tracking.load_model(resgld_run["model_uri"], cpu_device)


def test_evaluate_ensemble_moments_and_coverage(
    ensemble_model, ensemble_members, test_dataset, cpu_device
) -> None:
    set_seed(999)
    result = evaluate_ensemble(
        infer_config(n_ensemble=2),
        ensemble_model,
        test_dataset,
        ensemble_members,
        cpu_device,
    )

    assert result["n_members"] == 2
    for key in ("mean", "std", "sample", "y_true", "t_next"):
        assert result[key].shape == (N_TRAJECTORIES, SEARCH_NUM)
        assert np.isfinite(result[key]).all()
    assert (result["std"] >= 0.0).all()

    assert 0.0 <= result["picp"] <= 1.0
    for key in (
        "L1_error_mat",
        "L2_error_mat",
        "L1_error_sample_mat",
        "L2_error_sample_mat",
    ):
        assert result[key].shape == (N_TRAJECTORIES,)
        assert np.isfinite(result[key]).all()
        assert (result[key] >= 0.0).all()


def test_evaluate_ensemble_is_deterministic_under_set_seed(
    ensemble_model, ensemble_members, test_dataset, cpu_device
) -> None:
    config = infer_config()
    set_seed(999)
    first = evaluate_ensemble(
        config, ensemble_model, test_dataset, ensemble_members, cpu_device
    )
    set_seed(999)
    second = evaluate_ensemble(
        config, ensemble_model, test_dataset, ensemble_members, cpu_device
    )
    assert np.array_equal(first["mean"], second["mean"])
    assert np.array_equal(first["sample"], second["sample"])
    assert first["picp"] == second["picp"]


def test_evaluate_ensemble_writes_the_uq_figure(
    ensemble_model, ensemble_members, test_dataset, cpu_device, tmp_path: Path
) -> None:
    figure_dir = tmp_path / "uq"
    set_seed(999)
    evaluate_ensemble(
        infer_config(plot_trajs=True, figure_dir=str(figure_dir), plot_idxs=[1]),
        ensemble_model,
        test_dataset,
        ensemble_members,
        cpu_device,
    )
    assert (figure_dir / "uq_trajs_1.png").is_file()
    assert not (figure_dir / "uq_trajs_0.png").exists()


def test_evaluate_ensemble_requires_members(
    ensemble_model, test_dataset, cpu_device
) -> None:
    with pytest.raises(ValueError, match="no ensemble members were found"):
        evaluate_ensemble(infer_config(), ensemble_model, test_dataset, [], cpu_device)


def test_ensure_directory_creates_parents(tmp_path: Path) -> None:
    target = ensure_directory(tmp_path / "a" / "b")
    assert target.is_dir()
    # Idempotent.
    assert ensure_directory(target) == target
