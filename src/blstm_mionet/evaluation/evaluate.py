"""Evaluation entry points: single step, recursive rollout and Bayesian ensemble."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from blstm_mionet.config import InferConfig
from blstm_mionet.data.datasets import TorchDataset
from blstm_mionet.evaluation.metrics import picp, relative_error_per_trajectory
from blstm_mionet.evaluation.plotting import (
    ensure_directory,
    plot_comparison,
    plot_comparison_uq,
)


def evaluate_single_step(
    config: InferConfig, model: torch.nn.Module, dataset: TorchDataset
) -> dict[str, Any]:
    """One-step-ahead evaluation (formerly ``execute_test``)."""
    if config.verbose:
        print(f"\n***** Testing with {dataset.len} data samples*****\n")

    ## Step 1: load the data
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    ## Step 2 and 3: infer at each time step
    t_next_list = []
    y_pred_list = []
    y_true_list = []

    model.eval()
    for x_test_batch, y_test_batch in test_loader:
        ## batch testing
        # step a: forward pass without computing gradients
        with torch.no_grad():
            y_pred_list.append(model(x_test_batch).detach().cpu().numpy())
            y_true_list.append(y_test_batch.detach().cpu().numpy())
            t_params = x_test_batch[-1].detach().cpu().numpy()
            t_next_list.append(t_params[:, [-1]])

    # Stack the results to assemble trajectories
    y_pred = np.vstack(y_pred_list).reshape(config.search_num, -1).T
    y_true = np.vstack(y_true_list).reshape(config.search_num, -1).T
    t_next = (
        np.vstack(t_next_list).reshape(config.search_num, -1).T
    )  # [N_sample, N_search]

    assert y_pred.shape == y_true.shape == t_next.shape

    return _summarise(config, y_true, y_pred, t_next, prefix="infer_trajs")


def evaluate_recursive(
    config: InferConfig, model: torch.nn.Module, dataset: TorchDataset
) -> dict[str, Any]:
    """Autoregressive rollout with teacher forcing (formerly ``execute_test_recursive``).

    With ``teacher_forcing_prob = 1`` every step is fed the true state; with 0
    the model consumes its own predictions.  When ``autonomous`` is set the
    input function is rebuilt from the predicted states as well.
    """
    if config.verbose:
        print(f"\n***** Testing with {dataset.len} data samples*****\n")

    ## Step 1: load the data
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # The prepared dataset is search-major: sample ``idx`` is search step
    # ``idx // n_traj`` of trajectory ``idx % n_traj``.  The rollout buffer
    # below is ``[trajectory, search step]``.
    n_traj = dataset.len // config.search_num
    _check_rollout_grid(config, dataset)

    ## Step 2: seed the rollout with the true initial states
    t_next_list = []
    y_pred_list = dataset.x_n.reshape(config.search_num, -1).T
    # Add the last output time step to the list
    y_pred_last = dataset.x_next.reshape(config.search_num, -1).T
    y_pred_last = y_pred_last[:, [-1]]
    y_pred_list = torch.hstack((y_pred_list, y_pred_last))
    del y_pred_last

    y_true_list = []

    ## Step 3: infer at each time step
    model.eval()
    progress_bar = tqdm(total=dataset.len, desc="Recursive rollout", dynamic_ncols=True)

    for idx, (x_test_batch, y_test_batch) in enumerate(test_loader):
        ## batch testing
        # step a: forward pass without computing gradients
        with torch.no_grad():
            y_pred = model(x_test_batch)

            traj, search = idx % n_traj, idx // n_traj
            x_n = y_pred_list[traj, search].view(-1, 1)

            x_test_batch[1] = x_n

            if config.autonomous:
                input_traj = y_pred_list[traj, :].clone()
                input_traj[(search + 1) :] = 0
                x_test_batch[0] = input_traj[None, :-1, None]

            y_pred_recursive = model(x_test_batch)

            assert torch.numel(y_pred) == 1

            teacher_forcing = torch.rand(1) < config.teacher_forcing_prob
            if search < config.search_num and not teacher_forcing:
                y_pred = y_pred_recursive

            y_pred_list[traj, search + 1] = y_pred.view(-1)

            y_true_list.append(y_test_batch.detach().cpu().numpy())

            t_params = x_test_batch[-1].detach().cpu().numpy()
            t_next_list.append(t_params[:, [-1]])
        progress_bar.update(1)
    progress_bar.close()

    # Stack the results to assemble trajectories
    y_pred = y_pred_list[:, 1:].detach().cpu().numpy()
    y_true = np.vstack(y_true_list).reshape(config.search_num, -1).T
    t_next = (
        np.vstack(t_next_list).reshape(config.search_num, -1).T
    )  # [N_sample, N_search]

    assert y_pred.shape == y_true.shape == t_next.shape

    return _summarise(config, y_true, y_pred, t_next, prefix="infer_trajs_recursive")


def evaluate_ensemble(
    config: InferConfig,
    model: torch.nn.Module,
    dataset: TorchDataset,
    member_paths: Sequence[Path],
    device: torch.device,
) -> dict[str, Any]:
    """Posterior predictive evaluation of a reSGLD ensemble.

    Replaces ``test_replica`` / ``test_one_replica`` / ``model_pred``.  The
    original reloaded every ensemble member from disk once per test point;
    here each member is loaded once and applied to the whole test set, which
    yields the same predictions orders of magnitude faster.
    """
    if not member_paths:
        raise ValueError("no ensemble members were found for this run")

    if config.verbose:
        print(
            f"\n***** Testing {len(member_paths)} ensemble members on {dataset.len} data samples*****\n"
        )

    predictions = []
    progress_bar = trange(len(member_paths), desc="Ensemble members")
    for i in progress_bar:
        # Members hold only a state dict, so the restricted (non-pickle) loader
        # suffices and cannot execute code from a tampered artifact.
        checkpoint = torch.load(member_paths[i], map_location=device, weights_only=True)
        state_dict = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )
        model.load_state_dict(state_dict)
        model.to(device)
        predictions.append(_predict(model, dataset, config.batch_size))
    progress_bar.close()

    G_preds = np.vstack(predictions)  # [N_members, N_points]
    G_true = dataset.x_next.detach().cpu().numpy().flatten()

    ## first and second moments; the standard deviation uses the 1 / (M - 1)
    ## normalisation of the paper (Sec. 3.5.2)
    G_mean = np.mean(G_preds, axis=0)
    G_std = np.std(G_preds, axis=0, ddof=1 if G_preds.shape[0] > 1 else 0)

    ## one draw from the Gaussian approximation of the posterior predictive
    G_sample = posterior_predictive_sample(G_mean, G_std)

    t_next = dataset.t_params[:, -1].detach().cpu().numpy()

    ## Stack the results and reshape to [N_trajs, N_search]
    print("Inference complete. Organizing results...")
    mean = G_mean.reshape(config.search_num, -1).T
    std = G_std.reshape(config.search_num, -1).T
    true = G_true.reshape(config.search_num, -1).T
    sample = G_sample.reshape(config.search_num, -1).T
    t_next = t_next.reshape(config.search_num, -1).T

    L2_error_mat = relative_error_per_trajectory(true, mean)
    L1_error_mat = relative_error_per_trajectory(true, mean, order=1)
    L2_error_sample_mat = relative_error_per_trajectory(true, sample)
    L1_error_sample_mat = relative_error_per_trajectory(true, sample, order=1)
    coverage = picp(true, mean, std)

    ## Plot UQ
    plot_idxs = _plot_idxs(config)
    if config.plot_trajs:
        figure_dir = ensure_directory(config.figure_dir)
        for idx in range(mean.shape[0]):
            if idx in plot_idxs:
                fig_filename = str(figure_dir / f"uq_trajs_{idx}.png")
                plot_comparison_uq(
                    (true[idx].flatten(), mean[idx].flatten(), sample[idx].flatten()),
                    std[idx].flatten(),
                    ("True", "Ensemble mean", "Ensemble sample"),
                    color_list=("red", "blue", "black"),
                    linestyle_list=("solid", "dashed", "dotted"),
                    fig_path=fig_filename,
                )
                print(f"Figure saved to {fig_filename}.")

    if config.verbose:
        _print_error_table("L1-relative Error %", L1_error_mat)
        _print_error_table("L2-relative Error %", L2_error_mat)
        _print_error_table("L1-relative Error Sample %", L1_error_sample_mat)
        _print_error_table("L2-relative Error Sample %", L2_error_sample_mat)

    print(
        f"Ensemble prediction: mean std of the posterior = {float(np.mean(G_std)):.6f}, "
        f"mean |mean - truth| = {float(np.mean(np.abs(G_mean - G_true))):.6f}"
    )
    print(f"PICP (95% interval) = {coverage:.4f}")

    return {
        "L1_error_mat": L1_error_mat,
        "L2_error_mat": L2_error_mat,
        "L1_error_sample_mat": L1_error_sample_mat,
        "L2_error_sample_mat": L2_error_sample_mat,
        "picp": coverage,
        "mean": mean,
        "std": std,
        "sample": sample,
        "y_true": true,
        "t_next": t_next,
        "n_members": len(member_paths),
    }


def _check_rollout_grid(config: InferConfig, dataset: TorchDataset) -> None:
    """Warn when a free-running rollout would feed states back at the wrong time.

    The prediction for ``t_n + h`` becomes the current state of the next
    evaluation point, which is only consistent when ``t_(n+1) = t_n + h``.  With
    ``search_random = False`` that holds for
    ``search_num = N_time - 2 * search_len``.
    """
    if config.teacher_forcing_prob >= 1.0 or config.scale_mode:
        return
    t_params = dataset.t_params.detach().cpu().numpy()
    t_params = t_params.reshape(config.search_num, -1, t_params.shape[-1])
    t_n, h = t_params[:, :, 0], t_params[:, :, 1]
    if not np.allclose(t_n[1:], t_n[:-1] + h[:-1], rtol=0.0, atol=1e-4):
        warnings.warn(
            "consecutive rollout points are not one step h apart, so predicted "
            "states are fed back at the wrong time; use "
            "inference.search_random=false with "
            "inference.search_num = N_time - 2 * search_len",
            stacklevel=3,
        )


def posterior_predictive_sample(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Draw once from ``N(mean, diag(std ** 2))``, point by point.

    The code released with the paper passed ``std`` where the covariance
    expects the variance, i.e. it sampled with standard deviation
    ``sqrt(std)``; see ``CHANGELOG.md``.
    """
    return np.random.normal(loc=mean, scale=std)


def _predict(
    model: torch.nn.Module, dataset: TorchDataset, batch_size: int
) -> np.ndarray:
    """Run ``model`` over the whole dataset and return a flat prediction vector."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    outputs = []
    with torch.no_grad():
        for x_batch, _ in loader:
            outputs.append(model(x_batch).detach().cpu().numpy())
    return np.vstack(outputs).flatten()


def _plot_idxs(config: InferConfig) -> list[int]:
    if isinstance(config.plot_idxs, int):
        return [config.plot_idxs]
    return list(config.plot_idxs)


def _summarise(
    config: InferConfig,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_next: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    """Compute the relative errors, plot and print, and return the metrics."""
    L2_error_mat = relative_error_per_trajectory(y_true, y_pred)
    L1_error_mat = relative_error_per_trajectory(y_true, y_pred, order=1)

    plot_idxs = _plot_idxs(config)
    figure_path: str | None = None
    if config.plot_trajs:
        figure_dir = ensure_directory(config.figure_dir)
        for idx in range(y_pred.shape[0]):
            if idx in plot_idxs:
                figure_path = str(figure_dir / f"{prefix}_{idx}.png")
                plot_comparison(
                    (t_next[idx].flatten(), t_next[idx].flatten()),
                    (y_true[idx].flatten(), y_pred[idx].flatten()),
                    ("True", "Pred"),
                    color_list=("red", "blue"),
                    linestyle_list=("solid", "dashed"),
                    fig_path=figure_path,
                    save_fig=True,
                )
                print(f"Figure saved to {figure_path}.")

    if config.verbose:
        _print_error_table("L1-relative Error %", L1_error_mat)
        _print_error_table("L2-relative Error %", L2_error_mat)

    print(
        f"L2-relative error: mean = {np.mean(L2_error_mat) * 100:9.4f} %, st. dev. = {np.std(L2_error_mat) * 100:9.4f} % "
        f"over {y_pred.shape[0]} trajectories"
    )

    return {
        "L1_error_mat": L1_error_mat,
        "L2_error_mat": L2_error_mat,
        "t_next": t_next,
        "y_pred": y_pred,
        "y_true": y_true,
        "figure_path": figure_path,
    }


def _print_error_table(title: str, error_mat: np.ndarray) -> None:
    """Print the mean/standard deviation table of the original supervisor."""
    print("\n=============================")
    print(f"     {title}      ")
    print("=============================")
    print("     mean     st. dev.  ")
    print("-----------------------------")
    print(f" {np.mean(error_mat) * 100:9.4f} {np.std(error_mat) * 100:9.4f}")
    print("-----------------------------")
