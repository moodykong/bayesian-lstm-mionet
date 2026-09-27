"""Replica exchange stochastic gradient Langevin dynamics (reSGLD).

Two chains are trained simultaneously: a low temperature *exploit* chain and a
high temperature *explore* chain.  After every epoch the chains may swap their
parameters with probability ``min(1, exchange_rate)``; once the burn-in is over
the exploit chain is sampled to build the Bayesian ensemble.

Each chain follows the momentum (SGHMC-style) Langevin update

    v     <- (1 - alpha) * v - eta * grad(L) / sigma + scale * xi,   xi ~ N(0, I)
    theta <- theta + v

with ``scale = sqrt(2 * (alpha - beta) * eta * tau)`` (see
:class:`blstm_mionet.config.LangevinConfig`).  The noise ``xi`` is drawn
independently for every parameter entry; the code released with the paper drew
a single scalar per parameter tensor, see ``CHANGELOG.md``.  The swap criterion
and the burn-in rule are those of the paper.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from blstm_mionet.config import BayesianConfig, LangevinConfig, TrainConfig
from blstm_mionet.training import tracking


def _zero_velocities(net: torch.nn.Module) -> list[torch.Tensor]:
    """Zero-initialised velocity buffers, one per parameter tensor."""
    return [torch.zeros_like(p, requires_grad=False) for p in net.parameters()]


@torch.no_grad()
def langevin_step(
    net: torch.nn.Module,
    velocities: list[torch.Tensor],
    chain: LangevinConfig,
    sigma: float,
) -> None:
    """Apply one momentum Langevin update to ``net`` in place.

    ``v <- (1 - alpha) v - eta * grad / sigma + scale * xi`` with an
    independent standard normal ``xi`` per parameter entry, then
    ``theta <- theta + v``.  Gradients must already be populated.
    """
    for p, v in zip(net.parameters(), velocities, strict=True):
        if p.grad is None:
            continue
        v.mul_(1.0 - chain.alpha)
        v.add_(p.grad, alpha=-chain.eta / sigma)
        v.add_(torch.randn_like(p), alpha=chain.scale)
        p.add_(v)


def _chain_step(
    net: torch.nn.Module,
    velocities: list[torch.Tensor],
    chain: LangevinConfig,
    bayesian: BayesianConfig,
    x_batch: Any,
    y_batch: torch.Tensor,
) -> float:
    """Forward, backward and Langevin update of one chain; returns the batch MSE."""
    net.zero_grad()
    loss = ((net(x_batch) - y_batch) ** 2).mean()
    loss.backward()
    if bayesian.use_grad_norm:
        torch.nn.utils.clip_grad_value_(net.parameters(), bayesian.grad_norm)
    langevin_step(net, velocities, chain, bayesian.sigma)
    return loss.item()


def train_resgld(
    config: TrainConfig,
    bayesian: BayesianConfig,
    model_exploit: torch.nn.Module,
    model_explore: torch.nn.Module,
    dataset: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Train the exploit/explore replicas and log the posterior ensemble.

    Requires an active MLflow run.  Ensemble members are logged as
    ``ensemble/member_XXXX.pt`` (``{"state_dict": ...}`` checkpoints) and the
    exploit model itself is logged with ``mlflow.pytorch.log_model`` under
    ``model`` so that the class can be recovered.
    """
    ## Step 1: initialize velocity parameters
    N = dataset.len
    vel_exploit = _zero_velocities(model_exploit)
    vel_explore = _zero_velocities(model_explore)

    sigma = bayesian.sigma
    exploit = bayesian.exploit
    explore = bayesian.explore
    burn_in = bayesian.burn_in(config.epochs)

    if config.verbose:
        print(
            f"\n***** Replica-Training for {config.epochs} epochs and using {dataset.len} data samples*****\n"
        )
        print(
            f"Collecting up to {bayesian.n_ensemble} ensemble members after "
            f"epoch {burn_in}."
        )

    ## Step 2: load the torch dataset
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model_exploit.to(device)
    model_explore.to(device)

    if config.resume_model:
        trained_model = tracking.load_model(config.resume_model, device)
        model_exploit.load_state_dict(trained_model.state_dict())
        model_explore.load_state_dict(trained_model.state_dict())
        if config.verbose:
            print(f"Warm started both chains from {config.resume_model}.")

    ## Step 3: define best values, logger and pbar
    best = {"ge": np.inf}
    logger: dict[str, list] = {
        "ge exploit": [],
        "ge explore": [],
        "exchange rate": [],
        "switches": [],
    }
    progress_bar = trange(config.epochs)
    num_ensemble = 0
    best_state_dict = copy.deepcopy(model_exploit.state_dict())

    with tempfile.TemporaryDirectory() as ensemble_dir:
        try:
            ## step 4: training loop
            for epoch in progress_bar:
                epoch_loss_exploit = 0
                epoch_loss_explore = 0

                for x_batch, y_batch in train_loader:
                    epoch_loss_exploit += _chain_step(
                        model_exploit, vel_exploit, exploit, bayesian, x_batch, y_batch
                    )
                    epoch_loss_explore += _chain_step(
                        model_explore, vel_explore, explore, bayesian, x_batch, y_batch
                    )

                if len(train_loader) == 0:
                    raise ValueError(
                        "batch size larger than the number of training examples"
                    )
                avg_epoch_loss_exploit = epoch_loss_exploit / len(train_loader)
                avg_epoch_loss_explore = epoch_loss_explore / len(train_loader)

                ## log average epoch losses
                logger["ge exploit"].append(avg_epoch_loss_exploit / sigma)
                logger["ge explore"].append(avg_epoch_loss_explore / sigma)

                ## swapping step
                exchange_rate = np.exp(
                    bayesian.tau_delta
                    * (
                        logger["ge exploit"][-1] * N
                        - logger["ge explore"][-1] * N
                        - bayesian.tau_delta
                        * (bayesian.replica.sigma_uniform**2)
                        / bayesian.replica.f_adj
                    )
                )
                logger["exchange rate"].append(exchange_rate)
                it_switches = False
                if np.random.uniform(0, 1) < min(1, exchange_rate):
                    for f, c in zip(
                        model_exploit.parameters(),
                        model_explore.parameters(),
                        strict=True,
                    ):
                        f.data, c.data = (c.data, f.data)

                    vel_exploit, vel_explore = (vel_explore, vel_exploit)
                    it_switches = True
                    tqdm.write(
                        f"Swapped the chains (exchange rate {exchange_rate:.4g})."
                    )
                logger["switches"].append(it_switches)

                _log_epoch_metrics(logger, epoch)

                ## save best and print
                if epoch % bayesian.print_every == 0 or epoch + 1 == config.epochs:
                    if (avg_epoch_loss_exploit / sigma) < best["ge"]:
                        best["ge"] = avg_epoch_loss_exploit / sigma
                        best_state_dict = copy.deepcopy(model_exploit.state_dict())

                    progress_bar.set_postfix(
                        {
                            "switch?": it_switches,
                            "rate": np.round(exchange_rate, 10),
                            "ge exploit": np.round(avg_epoch_loss_exploit / sigma, 10),
                            "ge explore": np.round(avg_epoch_loss_explore / sigma, 10),
                            "best-ge": np.round(best["ge"], 10),
                        }
                    )

                ## UQ: collect the posterior ensemble after the burn-in
                if epoch > burn_in:
                    member_path = Path(ensemble_dir) / f"member_{num_ensemble:04d}.pt"
                    torch.save({"state_dict": model_exploit.state_dict()}, member_path)
                    num_ensemble += 1

        except KeyboardInterrupt:
            print("Keyboard Interrupted.")
        finally:
            progress_bar.close()
            if num_ensemble:
                mlflow.log_artifacts(
                    ensemble_dir, artifact_path=tracking.ENSEMBLE_ARTIFACT_PATH
                )

    ## Step 5: log the best exploit model so that the class is recoverable.
    # Registered under "<name>-bayesian" so that ``models:/<name>/latest`` keeps
    # resolving to the Adam-trained model rather than to an ensemble snapshot.
    model_exploit.load_state_dict(best_state_dict)
    registered_name = (
        f"{config.registered_model_name}-bayesian"
        if config.registered_model_name
        else None
    )
    model_uri = tracking.log_model(
        model_exploit,
        tracking.MODEL_ARTIFACT_NAME,
        registered_model_name=registered_name,
    )
    tracking.log_state_dict(best_state_dict, "best_replica_model.pt", "checkpoints")

    mlflow.log_metric("n_ensemble_members", num_ensemble)
    logger["n_ensemble"] = num_ensemble
    logger["best_ge"] = best["ge"]
    logger["model_uri"] = model_uri
    return logger


def _log_epoch_metrics(logger: dict[str, list], epoch: int) -> None:
    """Log the swap/exchange diagnostics of one epoch as MLflow metrics."""
    exchange_rate = float(logger["exchange rate"][-1])
    mlflow.log_metric("ge_exploit", float(logger["ge exploit"][-1]), epoch)
    mlflow.log_metric("ge_explore", float(logger["ge explore"][-1]), epoch)
    mlflow.log_metric("swap_probability", float(min(1.0, exchange_rate)), epoch)
    mlflow.log_metric("swap", float(bool(logger["switches"][-1])), epoch)
    if np.isfinite(exchange_rate):
        mlflow.log_metric("exchange_rate", exchange_rate, epoch)
