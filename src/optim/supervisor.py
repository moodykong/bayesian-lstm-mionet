#!/usr/bin/env python
# coding: utf-8

import torch

from torch.utils.data import random_split, DataLoader
from tqdm.auto import trange
from typing import Any, Callable
import utils.pytorch_utils as ptu
from utils.data_structures import normalize, minmaxscale, unnormalize, unminmaxscale, prepare_test_data, prepare_test_data_HF
from utils.math_utils import init_params,L2_relative_error, L1_relative_error, plot_comparison

import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (16.0, 12.0)

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import config.train_config as config # TODO: delete this line
from utils.math_utils import MAE_fn, MSE_fn

def train(config: dict, model: torch.nn, dataset: Any) -> dict:

    if config["verbose"]:
        print("\n***** Training with Adam Optimizer for {} epochs and using {} data samples*****\n".format(config["epochs"], dataset.len))

    ## step 1: use checkpoint if required
    if config["use_trained_model"]:
        trained_model = ptu.restore(config["trained_model_path"])
        state_dict = trained_model['state_dict']
        model.load_state_dict(state_dict)
        model.to(device)

    ## step 2: define the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    if config["use_scheduler"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["scheduler_factor"], patience=config["scheduler_patience"], verbose=config["verbose"])

    loss_fn = MAE_fn if config['loss_function'] == 'MAE' else MSE_fn
    monitor = config["monitor_metric"]

    ## step 3: split the dataset
    num_train = int(0.8 * dataset.len)
    num_val = dataset.len - num_train
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    ## step 4: load the dataset
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=num_val, shuffle=True)

    ## step 5: initialize best values and logger
    best_metric = {}
    best_metric["train_loss"] = np.Inf
    best_metric["val_loss"] = np.Inf

    # logger
    logger = {}
    logger["train_loss"] = []
    logger["val_loss"] = []

    progress = trange(config["epochs"])

    ## step 6: training loop
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
                    ptu.save(model, optimizer, save_path=checkpoint_path)

            if avg_epoch_val_loss < best_metric["val_loss"]:
                best_metric["val_loss"] = avg_epoch_val_loss
                if monitor == "val_loss":
                    checkpoint_path = config['checkpoint_path'] + f"best_val_loss_{epoch}.pt"
                    ptu.save(model, optimizer, save_path=checkpoint_path)

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

    ## Step 2: run for loop over all test trajectories

    # Errors
    L1_error_list = []
    L2_error_list = []

    t_list = []

    for idx, (x_test_batch, y_test_batch) in enumerate(test_loader):
        ## batch testing
        # step a: forward pass without computing gradients
        y_pred = model(x_test_batch)
        t_local = x_test_batch[-1].detach().cpu().numpy()
        t_list.append(t_local[-1])
        # step b: compute validation loss
        L1_error_list.append(L1_relative_error(y_test_batch.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))
        L2_error_list.append(L2_relative_error(y_test_batch.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))

        if (idx in config["plot_idxs"]) and config["plot_trajs"]:
            fig_filename = config["figure_path"] +"infer.png"
            plot_comparison(
                (y_plot.flatten(), y_plot.flatten()),
                (G_true.flatten(), G_pred.flatten()),
                ("True", "DeepONet predicted"),
                color_list=("red", "blue"),
                linestyle_list=("solid", "dashed"),
                fig_path=fig_filename,
                k=idx
                )
        
        del G_pred, G_true
                      

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
        print(np.hstack((l1_error.reshape(len(L1_error_list),1),l2_error.reshape(len(L1_error_list),1))))

    return y_data

def DeepONet_pred_LF(
        model: torch.nn,
        u: np.ndarray,
        y: np.ndarray,
        G: np.ndarray,
        datasetLF: Any,
        scale_mode: str,
        device: torch.device=torch.device('cpu'),
        nLocs: int=300,
        unscaling=False,
        field:str='lif',
        k: int=0,
        index: int=0) -> np.ndarray:

    with torch.no_grad():
        ## prepare DeepONet data
        u_test, y_test, G_true = prepare_test_data(u, y, G, nLocs=nLocs)

        # dataset.update_statistics() why we do not use here
        ## scale
        if scale_mode == "normalize":
            u_test = normalize(u_test, datasetLF.dataU_mean, datasetLF.dataU_std)
            y_test = normalize(y_test, datasetLF.dataY_mean, datasetLF.dataY_std)
            G_true = normalize(G_true, datasetLF.dataG_mean, datasetLF.dataG_std)

        elif scale_mode == "min-max":
            u_test = minmaxscale(u_test, datasetLF.dataU_min, datasetLF.dataU_max)
            y_test = minmaxscale(y_test, datasetLF.dataY_min, datasetLF.dataY_max)
            G_true  = minmaxscale(G_true, datasetLF.dataG_min, datasetLF.dataG_max)

        ## move to torch
        U = torch.from_numpy(u_test).float().to(device)
        Y = torch.from_numpy(y_test).float().to(device)

        ## forward pass
        G = model((U, Y)).detach().cpu().numpy()

        if scale_mode == "normalize":
        
            # G = unnormalize(G.detach().cpu().numpy(), dataset.dataG_mean, dataset.dataG_std).flatten()  # we comment this line because we  want to make the output scale sp we can calculate the error more realisticly
            Y = unnormalize(Y.detach().cpu().numpy(), datasetLF.dataY_mean, datasetLF.dataY_std).flatten() 
           
            
            
        elif scale_mode == "min-max":
            # G  = unminmaxscale(G.detach().cpu().numpy(), dataset.dataG_min, dataset.dataG_max).flatten() # we comment this line because we  want to make the output scale sp we can calculate the error more realisticly
            Y =  unminmaxscale(Y.detach().cpu().numpy(), datasetLF.dataY_min, datasetLF.dataY_max).flatten()
   
            
            
        if k==index: 
          LPredic = np.c_[Y[1:].reshape(-1,), G[1:].reshape(-1,)]
          np.savetxt('LFPredic_'+ field + '.txt',LPredic)   # X is an array  
          
          LTrue = np.c_[Y[1:].reshape(-1,), G_true[1:].reshape(-1,)]
          np.savetxt('LFTrue_' + field +'.txt',LTrue)   # X is an array   
#          
        return (G[1:], G_true[1:], Y[1:])


def DeepONet_pred_HF(
        modelLF: torch.nn,
        modelHF:torch.nn,
        u: np.ndarray,
        y: np.ndarray,
        G: np.ndarray,
        datasetLF: Any,
        datasetRes: Any,
        datasetHF: Any,
        scale_mode: str,
        device: torch.device=torch.device('cpu'),
        nLocs: int=300,
        unscaling=False,
        field:str='lif',
        k: int=0,
        index: int=0) -> np.ndarray:

    with torch.no_grad():
        ## prepare DeepONet data
        u_test, y_test, G_res, GG_HF, GG_LF_pred = prepare_test_data_HF(u, y, G, nLocs, scale_mode, datasetLF, modelLF)
        
        ## scale
        if scale_mode == "normalize":
            u_test = normalize(u_test, datasetRes.dataU_mean, datasetRes.dataU_std)
            y_test = normalize(y_test, datasetRes.dataY_mean, datasetRes.dataY_std)

        elif scale_mode == "min-max":
            u_test = minmaxscale(u_test, datasetRes.dataU_min, datasetRes.dataU_max)
            y_test = minmaxscale(y_test, datasetRes.dataY_min, datasetRes.dataY_max)

        ## move to torch
        U = torch.from_numpy(u_test).float().to(device)
        Y = torch.from_numpy(y_test).float().to(device)

        ## forward pass of HF model
        G = modelHF((U, Y))

        
        if scale_mode == "normalize":
            G=unnormalize(G.detach().cpu().numpy(), datasetRes.dataG_mean, datasetRes.dataG_std).flatten()
            Y = unnormalize(Y.datasetHF().cpu().numpy(), datasetRes.dataY_mean, datasetRes.dataY_std).flatten()
        elif scale_mode == "min-max":
            G=unminmaxscale(G.detach().cpu().numpy(), datasetRes.dataG_min, datasetRes.dataG_max).flatten()
            Y =  unminmaxscale(Y.detach().cpu().numpy(), datasetRes.dataY_min, datasetRes.dataY_max).flatten()

        res_pred=G.reshape(nLocs, 1)
        
        GG_HF_pred= GG_LF_pred + res_pred
        
        
        
         # scale the output
        if scale_mode == "normalize":
            GG_HF_pred=normalize(GG_HF_pred, datasetHF.dataG_mean, datasetHF.dataG_std).flatten()
            GG_HF=normalize(GG_HF, datasetHF.dataG_mean, datasetHF.dataG_std).flatten()            
           
        elif scale_mode == "min-max":
            GG_HF_pred=minmaxscale(GG_HF_pred, datasetHF.dataG_min, datasetHF.dataG_max).flatten()
            GG_HF=minmaxscale(GG_HF, datasetHF.dataG_min, datasetHF.dataG_max).flatten()


        if k==index: 
          LPredic = np.c_[Y[1:].reshape(-1,), GG_HF_pred[1:].reshape(-1,)]
          np.savetxt('HFPredic_'+ field + '.txt',LPredic)   # X is an array  
          
          LTrue = np.c_[Y[1:].reshape(-1,), GG_HF[1:].reshape(-1,)]
          np.savetxt('HFTrue_'+ field + '.txt',LTrue)   # X is an array   
          
          
            
        return (GG_HF_pred[1:], GG_HF[1:],  Y[1:])        