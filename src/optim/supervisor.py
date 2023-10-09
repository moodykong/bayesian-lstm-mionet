#!/usr/bin/env python
# coding: utf-8

import torch

from torch.utils.data import random_split, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import trange
from tqdm import tqdm
from typing import Any, Callable
import utils.torch_utils as torch_utils
from utils.data_utils import (
    normalize,
    minmaxscale,
    unnormalize,
    unminmaxscale,
)
from utils.math_utils import (
    init_params,
    plot_comparison,
)

import numpy as np
import mlflow


def execute_train(
    config: dict, model: torch.nn, dataset: Any, device: torch.device
) -> dict:
    if config["verbose"]:
        print(
            "\n***** Training with Adam Optimizer for {} epochs and using {} data samples*****\n".format(
                config["epochs"], dataset.len
            )
        )

    ## Step 1: use trained model if required
    if config["use_trained_model"]:
        try:
            trained_model = torch_utils.restore(config["trained_model_path"])
            state_dict = trained_model["state_dict"]
            model.load_state_dict(state_dict)
            print("Trained model loaded successfully")
        except Exception as e:
            print(
                "Error: ",
                e,
                "loading trained model failed and new model will be trained instead.",
            )

    ## Step 2: define the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if config["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
            verbose=config["verbose"],
        )

    loss_fn = (
        torch.nn.L1Loss() if config["loss_function"] == "MAE" else torch.nn.MSELoss()
    )
    monitor = config["monitor_metric"]

    ## Step 3: split the dataset
    num_train = int(0.8 * dataset.len)
    num_val = dataset.len - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    ## Step 4: load the dataset
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

    ## Step 5: initialize best values and logger
    best_metric = {}
    best_metric["train_loss"] = np.Inf
    best_metric["val_loss"] = np.Inf

    # Metric logger
    log_metric = {}
    log_metric["train_loss"] = []
    log_metric["val_loss"] = []

    progress_bar = trange(config["epochs"])
    stop_ctr = 0

    ## Step 6: training loop
    model.to(device)
    if device == "parallel":
        model = torch.nn.DataParallel(model)
    scalar = GradScaler()

    try:
        mlflow.set_tracking_uri("mlruns")
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.set_experiment(experiment_name=config["experiment_name"])

        with mlflow.start_run(run_name=config["run_name"]):
            mlflow.log_artifacts("config", artifact_path="src/config")
            mlflow.log_artifact("models/architectures.py", artifact_path="src/models")
            mlflow.log_artifact("optim/supervisor.py", artifact_path="src/optim")
            mlflow.log_artifacts("utils", artifact_path="src/utils")
            mlflow.log_artifact("database_generator.py", artifact_path="src")
            mlflow.log_artifact("train.py", artifact_path="src")
            mlflow.log_artifact("infer.py", artifact_path="src")
            mlflow.log_params(config)

            for epoch in progress_bar:
                model.train()
                epoch_loss = 0
                for x_batch, y_batch in train_loader:
                    ## batch training
                    with autocast():
                        # step a: forward pass
                        y_pred = model(x_batch)

                        # step b: compute loss
                        loss = loss_fn(y_pred, y_batch)
                        epoch_loss += loss.squeeze()

                    # step c: compute gradients and backpropagate
                    optimizer.zero_grad()
                    scalar.scale(loss).backward()

                    # step d: optimize
                    scalar.step(optimizer)
                    scalar.update()

                try:
                    avg_epoch_loss = epoch_loss.item() / len(train_loader)
                except ZeroDivisionError as e:
                    print(
                        "Error: ",
                        e,
                        "batch size larger than number of training examples",
                    )

                log_metric["train_loss"].append([epoch, avg_epoch_loss])

                mlflow.log_metric("train_loss", avg_epoch_loss, epoch)

                ## validate and print
                if (
                    epoch % config["validate_freq"] == 0
                    or (epoch + 1) == config["epochs"]
                ):
                    model.eval()
                    with torch.no_grad():
                        epoch_val_loss = 0
                        for x_val_batch, y_val_batch in val_loader:
                            ## batch validation
                            with autocast():
                                # step a: forward pass without computing gradients
                                y_val_pred = model(x_val_batch)

                                # step b: compute validation loss
                                val_loss = loss_fn(y_val_pred, y_val_batch)
                                epoch_val_loss += val_loss.squeeze()

                        try:
                            avg_epoch_val_loss = epoch_val_loss.item() / len(val_loader)
                        except ZeroDivisionError as e:
                            print(
                                "Error: ",
                                e,
                                "batch size larger than number of training examples",
                            )

                        log_metric["val_loss"].append([epoch, avg_epoch_val_loss])
                        mlflow.log_metric("val_loss", avg_epoch_val_loss, epoch)

                    ## save best results
                    stop_ctr += 1
                    if avg_epoch_loss < best_metric["train_loss"]:
                        best_metric["train_loss"] = avg_epoch_loss
                        if monitor == "train_loss":
                            checkpoint_path = (
                                config["checkpoint_path"]
                                + f"best_train_loss_{epoch}.pt"
                            )

                            stop_ctr = 0

                            if config["save_model"]:
                                # torch_utils.save(model, optimizer, save_path=checkpoint_path)
                                mlflow.pytorch.log_model(
                                    model, f"best_model_epoch_{epoch}"
                                )

                    if avg_epoch_val_loss < best_metric["val_loss"]:
                        best_metric["val_loss"] = avg_epoch_val_loss
                        if monitor == "val_loss":
                            checkpoint_path = (
                                config["checkpoint_path"] + f"best_val_loss_{epoch}.pt"
                            )

                            stop_ctr = 0

                            if config["save_model"]:
                                # torch_utils.save(model, optimizer, save_path=checkpoint_path)
                                mlflow.pytorch.log_model(
                                    model, f"best_model_epoch_{epoch}"
                                )

                    ## run scheduler
                    if config["use_scheduler"]:
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
                    if stop_ctr > config["early_stopping_epochs"]:
                        print("Early Stopping.")
                        break

    except KeyboardInterrupt:
        print("Keyboard Interrupted.")
        mlflow.end_run()

    progress_bar.close()
    del optimizer, train_loader, val_loader
    return log_metric


def execute_test(config: dict, model: torch.nn, dataset: Any) -> list:
    if config["verbose"]:
        print("\n***** Testing with {} data samples*****\n".format(dataset.len))

    ## Step 1: load the data
    test_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    ## Step 2: infer at each time step

    t_next_list = []
    y_pred_list = []
    y_true_list = []

    ## Step 3: infer at each time step
    model.eval()

    for idx, (x_test_batch, y_test_batch) in enumerate(test_loader):
        ## batch testing
        # step a: forward pass without computing gradients
        with torch.no_grad():
            y_pred = model(x_test_batch).detach().cpu().numpy()
            y_pred_list.append(y_pred)

            y_true = y_test_batch.detach().cpu().numpy()
            y_true_list.append(y_true)

            t_params = x_test_batch[-1].detach().cpu().numpy()
            t_next = t_params[:, [-1]]
            t_next_list.append(t_next)

            del y_pred, y_true, t_params, t_next

    # Stack the results to assemble trajectories
    y_pred = np.vstack(y_pred_list).reshape(config["search_num"], -1).T
    y_true = np.vstack(y_true_list).reshape(config["search_num"], -1).T
    t_next = (
        np.vstack(t_next_list).reshape(config["search_num"], -1).T
    )  # [N_sample, N_search]
    num_trajs = y_pred.shape[0]

    del y_pred_list, y_true_list, t_next_list

    assert y_pred.shape == y_true.shape == t_next.shape  # check the shape of the output

    # Compute validation loss and plot
    L2_error_mat = np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(
        y_true, axis=1
    )
    L1_error_mat = np.linalg.norm(y_true - y_pred, ord=1, axis=1) / np.linalg.norm(
        y_true, ord=1, axis=1
    )

    if isinstance(config["scale_mode"], int):
        config["plot_idxs"] = [config["plot_idxs"]]

    for idx in range(num_trajs):
        # Plot the trajectories
        if (idx in config["plot_idxs"]) and config["plot_trajs"]:
            fig_filename = config["figure_path"] + "infer_trajs.png"
            plot_comparison(
                (t_next[idx].flatten(), t_next[idx].flatten()),
                (y_true[idx].flatten(), y_pred[idx].flatten()),
                ("True", "Pred"),
                color_list=("red", "blue"),
                linestyle_list=("solid", "dashed"),
                fig_path=fig_filename,
                save_fig=True,
            )

    # Print errors
    L1_mean = np.mean(L1_error_mat)
    L1_std = np.std(L1_error_mat)

    L2_mean = np.mean(L2_error_mat)
    L2_std = np.std(L2_error_mat)

    if config["verbose"]:
        print("\n=============================")
        print("     L1-relative Error %      ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (L1_mean * 100, L1_std * 100))
        print("-----------------------------")

        print("\n=============================")
        print("     L2-relative Error %      ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (L2_mean * 100, L2_std * 100))
        print("-----------------------------")

        print("\n=============================")
        print("[L1-relative Error list, L2-relative Error list ] %     ")
        print("=============================")
        print(
            np.hstack((L1_error_mat.reshape(-1, 1), L2_error_mat.reshape(-1, 1))) * 100
        )

    log_metric = {}
    log_metric["L1_error_mat"] = L1_error_mat
    log_metric["L2_error_mat"] = L2_error_mat
    log_metric["t_next"] = t_next
    log_metric["y_pred"] = y_pred
    log_metric["y_true"] = y_true

    return log_metric


def execute_test_recursive(config: dict, model: torch.nn, dataset: Any) -> list:
    if config["verbose"]:
        print("\n***** Testing with {} data samples*****\n".format(dataset.len))

    ## Step 1: load the data
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    ## Step 2: infer at each time step
    t_next_list = []
    y_pred_list = dataset.x_n.reshape(config["search_num"], -1).T
    # Add the last output time step to the list
    y_pred_last = dataset.x_next[[-1], :]
    y_pred_list = torch.hstack((y_pred_list, y_pred_last))
    del y_pred_last

    y_true_list = []

    ## Step 3: infer at each time step
    model.eval()
    progress_bar = tqdm(total=dataset.len, desc="Infering ...", dynamic_ncols=True)

    for idx, (x_test_batch, y_test_batch) in enumerate(test_loader):
        ## batch testing
        # step a: forward pass without computing gradients
        with torch.no_grad():
            y_pred = model(x_test_batch)

            x_n = y_pred_list[
                idx // (config["search_num"]), idx % (config["search_num"])
            ].view(-1, 1)

            x_test_batch[1] = x_n

            if config["autonomous"]:
                input_traj = y_pred_list[idx // config["search_num"], :].clone()
                input_traj[((idx % config["search_num"]) + 1) :] = 0
                x_test_batch[0] = input_traj[None, :-1, None]

            y_pred_recursive = model(x_test_batch)

            assert torch.numel(y_pred) == 1

            teacher_forcing = torch.rand(1) < config["teacher_forcing_prob"]
            if (config["search_num"]) - idx % (
                config["search_num"]
            ) > 0 and not teacher_forcing:
                y_pred = y_pred_recursive

            y_pred_list[
                idx // (config["search_num"]),
                idx % (config["search_num"]) + 1,
            ] = y_pred.view(-1)

            y_true = y_test_batch.detach().cpu().numpy()
            y_true_list.append(y_true)

            t_params = x_test_batch[-1].detach().cpu().numpy()
            t_next = t_params[:, [-1]]
            t_next_list.append(t_next)

            del y_pred, y_true, t_params, t_next, teacher_forcing
        progress_bar.update(1)
    progress_bar.close()

    # Stack the results to assemble trajectories
    y_pred = y_pred_list[:, 1:].detach().cpu().numpy()
    y_true = np.vstack(y_true_list).reshape(config["search_num"], -1).T
    t_next = (
        np.vstack(t_next_list).reshape(config["search_num"], -1).T
    )  # [N_sample, N_search]
    num_trajs = y_pred.shape[0]

    del y_pred_list, y_true_list, t_next_list

    assert y_pred.shape == y_true.shape == t_next.shape  # check the shape of the output

    # Compute validation loss and plot
    L2_error_mat = np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(
        y_true, axis=1
    )
    L1_error_mat = np.linalg.norm(y_true - y_pred, ord=1, axis=1) / np.linalg.norm(
        y_true, ord=1, axis=1
    )

    if isinstance(config["scale_mode"], int):
        config["plot_idxs"] = [config["plot_idxs"]]

    for idx in range(num_trajs):
        # Plot the trajectories
        if (idx in config["plot_idxs"]) and config["plot_trajs"]:
            fig_filename = config["figure_path"] + "infer_trajs.png"
            plot_comparison(
                (t_next[idx].flatten(), t_next[idx].flatten()),
                (y_true[idx].flatten(), y_pred[idx].flatten()),
                ("True", "Pred"),
                color_list=("red", "blue"),
                linestyle_list=("solid", "dashed"),
                fig_path=fig_filename,
                save_fig=True,
            )

    # Print errors
    L1_mean = np.mean(L1_error_mat)
    L1_std = np.std(L1_error_mat)

    L2_mean = np.mean(L2_error_mat)
    L2_std = np.std(L2_error_mat)

    if config["verbose"]:
        print("\n=============================")
        print("     L1-relative Error %      ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (L1_mean * 100, L1_std * 100))
        print("-----------------------------")

        print("\n=============================")
        print("     L2-relative Error %      ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (L2_mean * 100, L2_std * 100))
        print("-----------------------------")

        print("\n=============================")
        print("[L1-relative Error list, L2-relative Error list ] %     ")
        print("=============================")
        print(
            np.hstack((L1_error_mat.reshape(-1, 1), L2_error_mat.reshape(-1, 1))) * 100
        )

    log_metric = {}
    log_metric["L1_error_mat"] = L1_error_mat
    log_metric["L2_error_mat"] = L2_error_mat
    log_metric["t_next"] = t_next
    log_metric["y_pred"] = y_pred
    log_metric["y_true"] = y_true

    return log_metric
