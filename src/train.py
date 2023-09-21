#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import torch
import utils.pytorch_utils as ptu
from models.nns import LSTM_MIONet
from optim.supervisor import train, test
from utils.data_utils import \
    Dataset_Torch, Dataset_Stat, \
    prepare_local_predict_dataset, \
    scale_and_to_device, split_dataset
from utils.math_utils import MAE_fn, MSE_fn
from config import train_config

seed = 1234

def main(args):

    ###################################
    # Step 0:
    
    ###################################

    ###################################
    # Step 1: initialize the gpu
    ###################################
    ptu.init_gpu(verbose=args.verbose)
    
    ###################################
    # Step 2: initialize the device
    ###################################
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    ###################################
    # Step 3: collect the dataset
    ###################################
    config = {}
    config.update(train_config.get_config())
    train_dataset = config['train_dataset']
    train_dataset = np.load(train_dataset).item()

    ###################################
    # Step 4: prepare training and test datasets
    ###################################
    
    train_data, test_data = split_dataset(train_dataset, test_size=0.1, verbose=config['verbose'])
    input_masked, x_n, x_next, t_local = prepare_local_predict_dataset(
        data = train_data,
        search_len= config['search_len'],
        search_num= config['search_num'],
        search_random= config['search_random'],
        offset= config['offset'],
        verbose= config['verbose']
        )
    train_data_stat = Dataset_Stat(input_masked, x_n, x_next, t_local)
    
    ###################################
    # Step 5: update dataset statistics
    ###################################
    train_data_stat.update_statistics()

    ###################################
    # Step 6: scale and move data to torch
    ###################################
    input_masked, x_n, x_next, t_local = scale_and_to_device(train_data_stat, scale_mode=config['scale_mode'], device=ptu.device, requires_grad=False)
    train_data_torch = Dataset_Torch(input_masked, x_n, x_next, t_local)

    ###################################
    # Step 7: define the model
    ###################################
    branch_state = {}
    branch_state['layer_size_list'] = [config['branch_state']['width']] * config['branch_state']['depth']
    branch_state['activation'] = config['branch_state']['activation']

    branch_memory = {}
    branch_memory['layer_size_list'] = [config['branch_memory']['width']] * config['branch_memory']['depth']
    branch_memory['lstm_size'] = config['branch_memory']['lstm_size']
    branch_memory['lstm_layer_num'] = config['branch_memory']['lstm_layer_num']
    branch_memory['activation'] = config['branch_memory']['activation']

    trunk = {}
    trunk['layer_size_list'] = [config['trunk']['width']] * config['trunk']['depth']
    trunk['activation'] = config['trunk']['activation']

    model = LSTM_MIONet(branch_state, branch_memory, trunk)
    
    
    if config['verbose']:
        print(model)  
              
    

    ###################################
    # Step 10: define loss function
    ###################################
    

    ###################################
    # Step 11: define logging filename
    ###################################
    checkpoint_path = config['checkpoint_path']

    ###################################
    # Step 12: train the LF model
    ################################### 

    train(model=model,
          dataset=train_data_torch,
          train_params=config,
          scheduler_params=scheduler_params,
          loss_fn=loss_fn,
          checkpoint_path=checkpoint_path,
          device=ptu.device,
          monitor=config['monitor_metric'],
          verbose=config['verbose'],
          )
    
    ###################################
    # Step 13: restore best saved LF model
    ###################################
    ckptl = ptu.restore(log_filename)
    state_dict = ckptl['state_dict']
    modelLF.load_state_dict(state_dict)
    modelLF.to(ptu.device) 
    
    ###################################
    # Step 14: test the LF model
    ###################################

    models=[modelLF]
    datasets=[Cylinder_datasetLF]
    
    test(models,
         test_data,
         datasets,
         scale_mode=args.scale_mode,
         nLocs=300,
         device=ptu.device,
         plot_trajs=True,
         plot_idxs=[0, 1, 2, 3, 4, 5],
         verbose=args.verbose,
         feild=feild,
         index=1,
         )
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training Cylinder DeepONet")
    
    ## general args
    parser.add_argument('--verbose', action='store_false', default=True, help="printing during training")
    parser.add_argument('--scale-mode', type=str, default="min-max", help="normalize or min-max")

    ## DeepONet args
    parser.add_argument('--LF-branch-type', type=str, default="MLP", help="modified, MLP or FF-MLP ")
    parser.add_argument('--HF-branch-type', type=str, default="MLP", help="modified, MLP or FF-MLP ")  
    parser.add_argument('--nSensors', type=int, default=1, help="number of design points")
    parser.add_argument('--LF-trunk-type', type=str, default="FF-MLP", help="modified, MLP or FF-MLP ")
    parser.add_argument('--HF-trunk-type', type=str, default="FF-MLP", help="modified, MLP or FF-MLP ")    
    parser.add_argument('--width', type=int, default=200, help="width of nns")
    parser.add_argument('--depth', type=int, default=5, help="depth of nns")
    parser.add_argument('--nBasis', type=int, default=100, help="number of basis")
    parser.add_argument('--activation', type=str, default="relu", help="leaky, silu, Rrelu, Mish, sin, relu, tanh, selu, rsin, or gelu")
    parser.add_argument('--deeponet-type', type=str, default="vanilla", help="vanilla or modified") 
    parser.add_argument('--nLocs', type=int, default=500, help="number of sampling locations")
    parser.add_argument('--LF-nmode', type=int, default=50, help="number of mode in FF_mlp locations")
    parser.add_argument('--HF-nmode', type=int, default=50, help="number of mode in FF_mlp locations")
    
    ## training args
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--batch-size', type=int, default=256, help="training batch size")
    parser.add_argument('--nEpochs', type=int, default=200, help="number of training epochs")
    parser.add_argument('--loss', type=str, default="MSE", help="MAE or MSE")
    parser.add_argument('--filename', type=str, default="train-cylinder-01", help="logging filename") 
    parser.add_argument('--monitor', type=str, default="train-loss", help="train-loss or val-loss")
    parser.add_argument('--rng', type=int, default=456, help="random state")

    args = parser.parse_args()
    main(args)