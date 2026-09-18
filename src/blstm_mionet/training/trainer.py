"""Deterministic training with Adam (the maximum-likelihood baseline).

This is ``optim.supervisor.execute_train`` with the I/O turned into arguments:
the caller opens the MLflow run, this function only logs metrics and models
into the active run.
"""

from __future__ import annotations

import copy
from typing import Any

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm.auto import trange

from blstm_mionet.config import TrainConfig
from blstm_mionet.training import tracking


def train_adam(
    config: TrainConfig,
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Train ``model`` with Adam, ``ReduceLROnPlateau`` and early stopping.

    Requires an active MLflow run.  Returns the metric history together with
    ``best_metric`` and the URIs of the logged models.
    """
    if config.verbose:
        print(
            f"\n***** Training with Adam Optimizer for {config.epochs} epochs and using {dataset.len} data samples*****\n"
        )

    ## Step 1: use trained model if required
    if config.resume_model:
        try:
            trained_model = tracking.load_model(config.resume_model, device)
            model.load_state_dict(trained_model.state_dict())
            print(f"Warm started from {config.resume_model}.")
        except Exception as exc:  # noqa: BLE001 - mirrors the original fallback
            print(
                "Error: ",
                exc,
                "loading trained model failed and new model will be trained instead.",
            )

    ## Step 2: define the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = None
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )

    loss_fn = torch.nn.L1Loss() if config.loss_function == "MAE" else torch.nn.MSELoss()
    monitor = config.monitor_metric

    ## Step 3: split the dataset
    # ``int((1 - split) * len)`` reproduces the original 80/20 rounding exactly.
    num_train = int((1.0 - config.validation_split) * dataset.len)
    num_val = dataset.len - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    ## Step 4: load the dataset
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    ## Step 5: initialize best values and logger
    best_metric = {"train_loss": np.inf, "val_loss": np.inf}
    log_metric: dict[str, Any] = {"train_loss": [], "val_loss": []}
    best_state_dict: dict | None = None
    best_epoch: int | None = None
    model_uris: list[str] = []

    progress_bar = trange(config.epochs)
    stop_ctr = 0

    ## Step 6: training loop
    model.to(device)
    # Mixed precision is a CUDA feature; on CPU the scaler and autocast are
    # disabled so the arithmetic stays in float32.
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    avg_epoch_loss = np.inf
    avg_epoch_val_loss = np.inf

    try:
        for epoch in progress_bar:
            model.train()
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                ## batch training
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    # step a: forward pass
                    y_pred = model(x_batch)

                    # step b: compute loss
                    loss = loss_fn(y_pred, y_batch)
                    epoch_loss += loss.squeeze()

                # step c: compute gradients and backpropagate
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # step d: optimize
                scaler.step(optimizer)
                scaler.update()

            if len(train_loader) == 0:
                raise ValueError(
                    "batch size larger than the number of training examples"
                )
            avg_epoch_loss = epoch_loss.item() / len(train_loader)

            log_metric["train_loss"].append([epoch, avg_epoch_loss])
            mlflow.log_metric("train_loss", avg_epoch_loss, epoch)

            ## validate and print
            if epoch % config.validate_freq == 0 or (epoch + 1) == config.epochs:
                model.eval()
                with torch.no_grad():
                    epoch_val_loss = 0
                    for x_val_batch, y_val_batch in val_loader:
                        ## batch validation
                        with torch.amp.autocast(
                            device_type=device.type, enabled=amp_enabled
                        ):
                            # step a: forward pass without computing gradients
                            y_val_pred = model(x_val_batch)

                            # step b: compute validation loss
                            val_loss = loss_fn(y_val_pred, y_val_batch)
                            epoch_val_loss += val_loss.squeeze()

                    if len(val_loader) == 0:
                        raise ValueError(
                            "batch size larger than the number of validation examples"
                        )
                    avg_epoch_val_loss = epoch_val_loss.item() / len(val_loader)

                    log_metric["val_loss"].append([epoch, avg_epoch_val_loss])
                    mlflow.log_metric("val_loss", avg_epoch_val_loss, epoch)

                ## save best results
                stop_ctr += 1
                if avg_epoch_loss < best_metric["train_loss"]:
                    best_metric["train_loss"] = avg_epoch_loss
                    if monitor == "train_loss":
                        stop_ctr = 0
                        best_state_dict = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        if config.save_model:
                            model_uris.append(
                                tracking.log_model(model, f"best_model_epoch_{epoch}")
                            )

                if avg_epoch_val_loss < best_metric["val_loss"]:
                    best_metric["val_loss"] = avg_epoch_val_loss
                    if monitor == "val_loss":
                        stop_ctr = 0
                        best_state_dict = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        if config.save_model:
                            model_uris.append(
                                tracking.log_model(model, f"best_model_epoch_{epoch}")
                            )

                ## run scheduler
                if scheduler is not None:
                    scheduler.step(avg_epoch_val_loss)

                ## print results
                progress_bar.set_postfix(
                    {
                        "Train": avg_epoch_loss,
                        "Val": avg_epoch_val_loss,
                        "Best_train": best_metric["train_loss"],
                        "Best_Val": best_metric["val_loss"],
                    }
                )

                ## early stopping
                if stop_ctr > config.early_stopping_epochs:
                    print("Early Stopping.")
                    break

    except KeyboardInterrupt:
        print("Keyboard Interrupted.")

    progress_bar.close()

    ## Step 7: restore and log the best model under a stable artifact name
    final_uri = None
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    if config.save_model:
        final_uri = tracking.log_model(
            model,
            tracking.MODEL_ARTIFACT_NAME,
            registered_model_name=config.registered_model_name,
        )

    del optimizer, train_loader, val_loader
    log_metric["best_metric"] = best_metric
    log_metric["best_epoch"] = best_epoch
    log_metric["model_uris"] = model_uris
    log_metric["model_uri"] = final_uri
    return log_metric
