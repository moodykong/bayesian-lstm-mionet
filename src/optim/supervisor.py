#!/usr/bin/env python
# coding: utf-8

import torch

from torch.utils.data import random_split, DataLoader
from tqdm.auto import trange
from typing import Any, Callable
import utils.torch_utils as torch_utils
from utils.data_utils import normalize, minmaxscale, unnormalize, unminmaxscale, prepare_test_data, prepare_test_data_HF
from utils.math_utils import init_params,L2_relative_error, L1_relative_error, plot_comparison

import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (16.0, 12.0)

import numpy as np
import matplotlib.pyplot as plt
import scienceplots

def train(config: dict, model: torch.nn, dataset: Any) -> dict:

    if config["verbose"]:
        print("\n***** Training with Adam Optimizer for {} epochs and using {} data samples*****\n".format(config["epochs"], dataset.len))

    ## Step 1: use checkpoint if required
    if config["use_trained_model"]:
        trained_model = torch_utils.restore(config["trained_model_path"])
        state_dict = trained_model['state_dict']
        model.load_state_dict(state_dict)
        model.to(device)

    ## Step 2: define the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if config["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["scheduler_factor"], patience=config["scheduler_patience"], verbose=config["verbose"])

    loss_fn = torch.nn.L1Loss() if config['loss_function'] == 'MAE' else torch.nn.MSELoss()
    monitor = config["monitor_metric"]

    ## Step 3: split the dataset
    num_train = int(0.8 * dataset.len)
    num_val = dataset.len - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    ## Step 4: load the dataset
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=num_val, shuffle=True)

    ## Step 5: initialize best values and logger
    best_metric = {}
    best_metric["train_loss"] = np.Inf
    best_metric["val_loss"] = np.Inf

    # logger
    logger = {}
    logger["train_loss"] = []
    logger["val_loss"] = []

    progress = trange(config["epochs"])

    ## Step 6: training loop
    for epoch in progress:
        epoch_loss = 0
        model.train()

        for x_batch, y_batch in train_loader:
            ## batch training

            # step a: forward pass
            y_pred = model(x_batch)

            # step b: compute loss
            loss = loss_fn(y_pred, y_batch)

            # step c: compute gradients and backpropagate
            optimizer.zero_grad()
            loss.backward()

            # step d: optimize
            epoch_loss += loss.detach().cpu().numpy().squeeze()
            optimizer.step()

        try:
            avg_epoch_loss = epoch_loss / len(train_loader)
        except ZeroDivisionError as e:
            print("error: ", e, "batch size larger than number of training examples")

        logger["train_loss"].append(avg_epoch_loss)

        ## validate and print
        if epoch % config["validate_freq"] == 0 or (epoch + 1) == config["epochs"]:
            with torch.no_grad():
                epoch_val_loss = 0
                model.eval()
                for x_val_batch, y_val_batch in val_loader:
                    ## batch validation

                    # step a: forward pass without computing gradients
                    y_val_pred = model(x_val_batch)

                    # step b: compute validation loss
                    val_loss = loss_fn(y_val_pred, y_val_batch)
                    epoch_val_loss += val_loss.detach().cpu().numpy().squeeze()

                try:
                    avg_epoch_val_loss = epoch_val_loss / len(val_loader)
                except ZeroDivisionError as e:
                    print("error: ", e, "batch size larger than number of training examples")

                logger["val_loss"].append(avg_epoch_val_loss)

            ## save best results
            if avg_epoch_loss < best_metric["train_loss"]:
                best_metric["train_loss"] = avg_epoch_loss
                if monitor == "train_loss":
                    checkpoint_path = config['checkpoint_path'] + f"best_train_loss_{epoch}.pt"
                    torch_utils.save(model, optimizer, save_path=checkpoint_path)

            if avg_epoch_val_loss < best_metric["val_loss"]:
                best_metric["val_loss"] = avg_epoch_val_loss
                if monitor == "val_loss":
                    checkpoint_path = config['checkpoint_path'] + f"best_val_loss_{epoch}.pt"
                    torch_utils.save(model, optimizer, save_path=checkpoint_path)

            ## run scheduler
            if config["use_scheduler"]:
                scheduler.step(avg_epoch_val_loss)

            ## print results
            progress.set_postfix(
                {
                    'Train_Loss': avg_epoch_loss,
                    'Val_Loss': avg_epoch_val_loss,
                    'Best_train': best_metric["train_loss"],
                    'Best_Val': best_metric["val_loss"],
                 }
                 )

    progress.close()
    del optimizer, train_loader, val_loader
    return logger

def test(config: dict, model: torch.nn, dataset: Any) -> list:

    if config["verbose"]:
        print("\n***** Testing with {} data samples*****\n".format(dataset.len))

    ## Step 1: load the data
    test_loader = DataLoader(dataset, batch_size=dataset.len, shuffle=False)

    ## Step 2: infer at each time step
    # Errors
    L1_error_list = []
    L2_error_list = []
    t_next_list = []
    y_pred_list = []
    y_true_list = []

    for idx, (x_test_batch, y_test_batch) in enumerate(test_loader):
        ## batch testing
        # step a: forward pass without computing gradients
        y_pred = model(x_test_batch).detach().cpu().numpy()
        y_pred_list.append(y_pred)
        
        y_true = y_test_batch.detach().cpu().numpy()
        y_true_list.append(y_true)
        
        t_params = x_test_batch[-1].detach().cpu().numpy()
        t_next = t_params[-1] 
        t_next_list.append(t_next)
        
        del y_pred, y_true, t_params, t_next

    # Stack the results to assemble the trajectory
    y_pred = np.stack(y_pred_list).reshape(-1, config["search_num"])
    y_true = np.stack(y_true_list).reshape(-1, config["search_num"])
    t_next = np.stack(t_next_list).reshape(-1, config["search_num"])
    num_trajs = y_pred.shape[0]

    del y_pred_list, y_true_list, t_next_list
    
    assert y_pred.shape == y_true.shape == t_next.shape # check the shape of the output

    # Compute validation loss and plot
    for idx in range(num_trajs):
        # Compute validation loss
        L1_error_list.append(L1_relative_error(y_true, y_pred))
        L2_error_list.append(L2_relative_error(y_true, y_pred))

        if (idx in config["plot_idxs"]) and config["plot_trajs"]:
            fig_filename = config["figure_path"] +"infer_trajs.png"
            plot_comparison(
                (t_next.flatten(), t_next.flatten()),
                (y_true.flatten(), y_pred.flatten()),
                ("True", "Pred"),
                color_list=("red", "blue"),
                linestyle_list=("solid", "dashed"),
                fig_path=fig_filename,
                save_fig=True
                )


    # printing Errors    
    l1_error = np.stack(L1_error_list)
    l1_mean = np.mean(l1_error)
    l1_std = np.std(l1_error)

    l2_error = np.stack(L2_error_list)
    l2_mean = np.mean(l2_error)
    l2_std = np.std(l2_error)

    if verbose:
        print('\n=============================')
        print('     L1-relative Error %      ')
        print('=============================')
        print('     mean     st. dev.  ')
        print('-----------------------------')
        print(' %9.4f %9.4f' % (l1_mean, l1_std))
        print('-----------------------------')

        print('\n=============================')
        print('     L2-relative Error %      ')
        print('=============================')
        print('     mean     st. dev.  ')
        print('-----------------------------')
        print(' %9.4f %9.4f' % (l2_mean, l2_std))
        print('-----------------------------')

        print('\n=============================')
        print('[L1-relative Error list, L2-relative Error list ] %      ')
        print('=============================')
        print(np.hstack((l1_error.reshape(len(L1_error_list),1),l2_error.reshape(len(L2_error_list),1))))

    output={}
    output["L1_error_list"] = L1_error_list
    output["L2_error_list"] = L2_error_list
    output["t_next"] = t_next
    output["y_pred"] = y_pred
    output["y_true"] = y_true

    return output
