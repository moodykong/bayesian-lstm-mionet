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
from utils.math_utils import init_params, plot_comparison, plot_comparison_UQ

import numpy as np
import mlflow
import os


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

    if isinstance(config["plot_idxs"], int):
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
    y_pred_last = dataset.x_next.reshape(config["search_num"], -1).T
    y_pred_last = y_pred_last[:, [-1]]
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


## replica with UQ
def replica_train_UQ_V2(
    model_exploit: torch.nn,
    model_explore: torch.nn,
    dataset: Any,
    train_params: dict,
    replica_params: dict = None,
    exploit_params: dict = None,
    explore_params: dict = None,
    verbose: bool = True,
    device: torch.device = None,
    logging_filename: str = "./output/best-replica-model.pt",
    experiment: str = "default",
) -> Any:
    ## Step 0: clear the ensemble dir
    mypath = "./output/ensemble/" + experiment + "/"  # Enter your path here
    for root, dirs, files in os.walk(mypath, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))

    ## Step 1: initialize velocity parameters
    N = dataset.len
    vel_exploit = init_params(model_exploit)
    vel_explore = init_params(model_explore)

    if verbose:
        print(
            "\n***** Replica-Training for {} epochs and using {} data samples*****\n".format(
                train_params["epochs"], dataset.len
            )
        )

    ## Step 2: load the torch dataset
    train_loader = DataLoader(
        dataset, batch_size=train_params["batch_size"], shuffle=True
    )

    model_exploit.to(device)
    model_explore.to(device)

    if device == "parallel":
        model_exploit = torch.nn.DataParallel(model_exploit)
        model_explore = torch.nn.DataParallel(model_explore)

    ## Step 3: define best values, logger and pbar
    best = dict()
    best["ge"] = np.Inf

    # logger
    logger = {}
    logger["ge exploit"] = []
    logger["ge explore"] = []
    logger["exchange rate"] = []
    logger["switches"] = []
    progress_bar = trange(train_params["epochs"])
    num_ensemble = 1

    ## step 4: training loop
    for epoch in progress_bar:
        epoch_loss_exploit = 0
        epoch_loss_explore = 0

        for x_batch, y_batch in train_loader:
            ## batch training of exploit model

            # step a: forward pass
            model_exploit.zero_grad()
            ge_exploit = model_exploit(x_batch)

            # step b: compute loss
            loss_exploit = ((ge_exploit - y_batch) ** 2).mean()

            # step c: compute gradients and backpropagate
            loss_exploit.backward()
            if train_params["use grad norm"]:
                torch.nn.utils.clip_grad_value_(
                    model_exploit.parameters(), train_params["grad norm"]
                )

            # step d: update parameters
            for k, p in enumerate(model_exploit.parameters()):
                brownie_exploit = np.random.normal(0, 1, 1)[0]
                grad_exploit = (
                    p.grad.data.cpu().detach().numpy() / exploit_params["sigma"]
                )
                vel_exploit[k] = (
                    -grad_exploit * exploit_params["eta"]
                    + (1 - exploit_params["alpha"]) * vel_exploit[k]
                    + brownie_exploit * exploit_params["scale"]
                )
                p.data.add_(
                    torch.tensor(
                        vel_exploit[k],
                        requires_grad=False,
                        device=device,
                        dtype=torch.float32,
                    )
                )

            # step e: log batch loss
            epoch_loss_exploit += loss_exploit.detach().cpu().numpy().squeeze()

            ## batch training of explore model

            # step a: forward pass
            model_explore.zero_grad()
            ge_explore = model_explore(x_batch)

            # step b: compute loss
            loss_explore = ((ge_explore - y_batch) ** 2).mean()

            # step c: compute gradients and backpropagate
            loss_explore.backward()
            if train_params["use grad norm"]:
                torch.nn.utils.clip_grad_value_(
                    model_explore.parameters(), train_params["grad norm"]
                )

            # step d: update parameters
            for k, p in enumerate(model_explore.parameters()):
                brownie_explore = np.random.normal(0, 1, 1)[0]
                grad_explore = (
                    p.grad.data.cpu().detach().numpy() / explore_params["sigma"]
                )
                vel_explore[k] = (
                    -grad_explore * explore_params["eta"]
                    + (1 - explore_params["alpha"]) * vel_explore[k]
                    + brownie_explore * explore_params["scale"]
                )
                p.data.add_(
                    torch.tensor(
                        vel_explore[k],
                        requires_grad=False,
                        device=device,
                        dtype=torch.float32,
                    )
                )

            # step e: log batch loss
            epoch_loss_explore += loss_explore.detach().cpu().numpy().squeeze()

            ## compute likelihood - exploit
            likelihood_ge_exploit = (
                loss_exploit.cpu().detach().numpy() / exploit_params["sigma"]
            )
            ## compute likelihood - explore
            likelihood_ge_explore = (
                loss_explore.cpu().detach().numpy() / explore_params["sigma"]
            )

        try:
            avg_epoch_loss_exploit = epoch_loss_exploit / len(train_loader)
            avg_epoch_loss_explore = epoch_loss_explore / len(train_loader)
        except ZeroDivisionError as e:
            print("error: ", e, "batch size larger than number of training examples")

        ## log average epoch losses
        logger["ge exploit"].append(avg_epoch_loss_exploit / exploit_params["sigma"])
        logger["ge explore"].append(avg_epoch_loss_explore / explore_params["sigma"])

        ## swapping step
        exchange_rate = np.exp(
            replica_params["tau delta"]
            * (
                logger["ge exploit"][-1] * N
                - logger["ge explore"][-1] * N
                - replica_params["tau delta"]
                * (replica_params["sigma uniform"] ** 2)
                / replica_params["F adj"]
            )
        )
        logger["exchange rate"].append(exchange_rate)
        it_switches = False
        if np.random.uniform(0, 1) < min(1, exchange_rate):
            for f, c in zip(model_exploit.parameters(), model_explore.parameters()):
                (f.data, c.data) = (c.data, f.data)

            (vel_exploit, vel_explore) = (vel_explore, vel_exploit)
            it_switches = True
            print("Switches LDs with exchange rate of = {}".format(exchange_rate))

            logger["switches"].append(it_switches)

        ## save best and print
        if (
            epoch % train_params["print every"] == 0
            or epoch + 1 == train_params["epochs"]
        ):
            if (avg_epoch_loss_exploit / exploit_params["sigma"]) < best["ge"]:
                best["ge"] = avg_epoch_loss_exploit / exploit_params["sigma"]
                torch_utils.save_model(model_exploit, save_path=logging_filename)

            # add 10 test trajectories for checking

            progress_bar.set_postfix(
                {
                    "switch?": it_switches,
                    "rate": np.round(exchange_rate, 10),
                    "ge exploit": np.round(
                        avg_epoch_loss_exploit / exploit_params["sigma"], 10
                    ),
                    "ge explore": np.round(
                        avg_epoch_loss_explore / explore_params["sigma"], 10
                    ),
                    "best-ge": np.round(best["ge"], 10),
                }
            )

        ## UQ
        if epoch > train_params["burn in"]:
            ensemble_filename = (
                "./output/ensemble/"
                + experiment
                + "/model-"
                + experiment
                + "-"
                + str(num_ensemble)
                + ".pt"
            )
            torch_utils.save_model(model_exploit, save_path=ensemble_filename)
            num_ensemble += 1

    return logger


## test replica on a set of test trajectories
def test_replica(
    config: dict,
    model: torch.nn,
    data: list,
    dataset: Any,
    device: torch.device = torch.device("cpu"),
    unscaling: bool = False,
    experiment: str = "default",
) -> Any:
    ## unpack test data
    x_data, y_data = data
    input_masked, x_n, t_params = x_data
    n_test = y_data.shape[0]
    if config["verbose"]:
        print(
            "Shapes of data input={}, x_n={}, t_params={}, output={}".format(
                input_masked.shape, x_n.shape, t_params.shape, y_data.shape
            )
        )

    # Results
    G_mean_list = []
    G_std_list = []
    G_true_list = []
    G_sample_list = []

    ## Log metric

    log_metric = dict()

    ## Run for loop over all test data points
    progress_bar = trange(n_test)
    for i in progress_bar:
        x_data_i = (input_masked[i, :], x_n[i, :], t_params[i, :])
        y_data_i = y_data[i, :]
        (
            G_mean,
            G_std,
            G_true,
            G_sample,
        ) = test_one_replica(
            model,
            (x_data_i, y_data_i),
            dataset,
            unscaling=unscaling,
            scale_mode=config["scale_mode"],
            device=device,
            experiment=experiment,
            n_ensemble=300,
        )

        G_mean_list.append(G_mean.flatten())
        G_std_list.append(G_std.flatten())
        G_true_list.append(G_true.flatten())
        G_sample_list.append(G_sample.flatten())

        del G_mean, G_std, G_true, G_sample

    ## Stack the results and reshape to [N_trajs, N_search]
    print("Inference complete. Organizing results...")
    G_mean = np.vstack(G_mean_list).reshape(config["search_num"], -1).T
    G_std = np.vstack(G_std_list).reshape(config["search_num"], -1).T
    G_true = np.vstack(G_true_list).reshape(config["search_num"], -1).T
    G_sample = np.vstack(G_sample_list).reshape(config["search_num"], -1).T
    num_trajs = G_mean.shape[0]

    # Compute validation loss and plot
    L2_error_mat = np.linalg.norm(G_true - G_mean, axis=1) / np.linalg.norm(
        G_true, axis=1
    )
    L1_error_mat = np.linalg.norm(G_true - G_mean, ord=1, axis=1) / np.linalg.norm(
        G_true, ord=1, axis=1
    )
    L2_error_sample_mat = np.linalg.norm(G_true - G_sample, axis=1) / np.linalg.norm(
        G_true, axis=1
    )
    L1_error_sample_mat = np.linalg.norm(
        G_true - G_sample, ord=1, axis=1
    ) / np.linalg.norm(G_true, ord=1, axis=1)

    ## Plot UQ
    for idx in range(num_trajs):
        # Plot the trajectories
        if (idx in config["plot_idxs"]) and config["plot_trajs"]:
            fig_filename = "./output/UQ-plot-" + str(idx) + ".png"
            plot_comparison_UQ(
                (G_true[idx].flatten(), G_mean[idx].flatten(), G_sample[idx].flatten()),
                G_std[idx].flatten(),
                ("True", "DeepONet ensemble mean", "DeepONet ensemble sample"),
                color_list=("red", "blue", "black"),
                linestyle_list=("solid", "dashed", "dotted"),
                fig_path=fig_filename,
            )

    # Print Errors

    l1_mean = np.mean(L1_error_mat)
    l1_std = np.std(L1_error_mat)

    l1_mean_sample = np.mean(L1_error_sample_mat)
    l1_std_sample = np.std(L1_error_sample_mat)

    l2_mean = np.mean(L2_error_mat)
    l2_std = np.std(L2_error_mat)

    l2_mean_sample = np.mean(L2_error_sample_mat)
    l2_std_sample = np.std(L2_error_sample_mat)

    if config["verbose"]:
        print("\n=============================")
        print("     L1-relative Error %      ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (l1_mean, l1_std))
        print("-----------------------------")

        print("\n=============================")
        print("     L2-relative Error %      ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (l2_mean, l2_std))
        print("-----------------------------")

        print("\n=============================")
        print("[L1-relative Error list, L2-relative Error list ] %      ")
        print("=============================")
        print(
            np.hstack(
                (
                    L1_error_mat.reshape(-1, 1) * 100,
                    L2_error_mat.reshape(-1, 1) * 100,
                )
            )
        )

        print("\n=============================")
        print("     L1-relative Error Sample %      ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (l1_mean_sample, l1_std_sample))
        print("-----------------------------")

        print("\n=============================")
        print("     L2-relative Error Sample %  ")
        print("=============================")
        print("     mean     st. dev.  ")
        print("-----------------------------")
        print(" %9.4f %9.4f" % (l2_mean_sample, l2_std_sample))
        print("-----------------------------")

        print("\n=============================")
        print("[L1-relative Error list sample, L2-relative Error list sample] %      ")
        print("=============================")
        print(
            np.hstack(
                (
                    L1_error_sample_mat.reshape(-1, 1) * 100,
                    L2_error_sample_mat.reshape(-1, 1) * 100,
                )
            )
        )

    log_metric["L1_error_mat"] = L1_error_mat
    log_metric["L2_error_mat"] = L2_error_mat
    log_metric["L1_error_sample_mat"] = L1_error_sample_mat
    log_metric["L2_error_sample_mat"] = L2_error_sample_mat

    return log_metric


def model_pred(
    model: torch.nn,
    x: np.ndarray,
    y: np.ndarray,
    dataset: Any,
    scale_mode: str,
    device: torch.device = torch.device("cpu"),
    unscaling=False,
) -> np.ndarray:
    with torch.no_grad():
        ## prepare DeepONet data
        x_test, y_true = x, y

        ## scale
        if scale_mode == "normalize":
            x_test = normalize(x_test, dataset.dataX_mean, dataset.dataX_std)
        elif scale_mode == "min-max":
            x_test = minmaxscale(x_test, dataset.dataX_min, dataset.dataX_max)

        ## move to torch
        x_tensor = [torch.from_numpy(x).float().unsqueeze(0).to(device) for x in x_test]

        ## forward pass
        y_pred = model(x_tensor)

        if unscaling:
            if scale_mode == "normalize":
                return (
                    unnormalize(
                        y_pred.detach().cpu().numpy(),
                        dataset.dataY_mean,
                        dataset.dataY_std,
                    ).flatten(),
                    y_true,
                )
            elif scale_mode == "min-max":
                return (
                    unminmaxscale(
                        y_pred.detach().cpu().numpy(),
                        dataset.dataY_min,
                        dataset.dataY_max,
                    ).flatten(),
                    y_true,
                )

        else:
            if scale_mode == "normalize":
                y_true = normalize(y_true, dataset.dataY_mean, dataset.dataY_std)
            elif scale_mode == "min-max":
                y_true = minmaxscale(G_true, dataset.dataY_min, dataset.dataY_max)
            return y_pred.detach().cpu().numpy().flatten(), y_true


## test one trajectory with replica-exchage
def test_one_replica(
    model: torch.nn,
    data: Any,
    dataset: Any,
    unscaling: bool = False,
    scale_mode: str = "normalize",
    device: torch.device = torch.device("cpu"),
    fig_filename: str = "./output/UQ-plot.png",
    experiment: str = "default",
    n_ensemble: int = 100,
) -> Any:
    ## unpack trajectory data
    x, y = data

    ## allocate memory for ensemble prediction
    G_preds = []
    G_trues = []

    ## initialize ensemble
    for i in range(n_ensemble):
        ensemble_log_file = (
            "./output/ensemble/"
            + experiment
            + "/model-"
            + experiment
            + "-"
            + str(i + 1)
            + ".pt"
        )

        ## upload ensemble member
        try:
            check_point = torch_utils.restore(ensemble_log_file)
        except Exception as e:
            print("Error: ", e, "loading ensemble member failed.")
            continue  # skip to next ensemble member
        state_dict = check_point["state_dict"]
        model.load_state_dict(state_dict)
        model.to(torch_utils.device)

        ## predict with ensemble model
        G_pred, G_true = model_pred(
            model, x, y, dataset, scale_mode, device=device, unscaling=unscaling
        )

        ## save predictions
        G_preds.append(G_pred)
        G_trues.append(G_true)

        del G_pred, G_true

    G_preds_vec = np.vstack(G_preds)

    G_true = G_trues[0]

    ## compute first and second moments
    G_mean = np.mean(G_preds_vec, axis=0)
    G_std = np.std(G_preds_vec, axis=0)

    ## sample from normal distribution
    G_sample = np.random.multivariate_normal(G_mean.flatten(), np.diag(G_std.flatten()))

    return G_mean, G_std, G_true, G_sample
