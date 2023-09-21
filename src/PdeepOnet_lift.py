#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import torch
import utils.pytorch_utils as ptu
from models.nns import DeepONet, modified_DeepONet, Physics_DeepONet
from optim.supervisor import train, test
from utils.data_structures import DataGenerator, Dataset, \
                                  prepare_dataset, prepare_dataset_HF,prepare_DeepONet_dataset,prepare_DeepONet_dataset_HF, \
                                  scale_and_move_data_to_torch
from utils.utils import MAE_fn, MSE_fn

seed = 1234

def main(args):

    ###################################
    # Step 0:
    field = args.field
    method = args.method
    cutting = args.cutting
       # LF-> low fidelity,
       # HF-> high fidelity
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
    # Step 3: collect the LF dataset
    ###################################
    data_pathLF ="./data/" +"Coarse_Database.npy"
    Coarse_Database = np.load(data_pathLF)            # field points and measurement/sampling locations

    ###################################
    # Step 4: prepare training and test datasets
    ###################################
    
    train_data, test_data = prepare_dataset(Coarse_Database, field=feild, test_size=0.1, verbose=args.verbose, rng=args.rng, cutting=cutting)
    u_train, y_train, g_train = prepare_DeepONet_dataset(train_data, nLocs=args.nLocs, verbose=args.verbose)
    Cylinder_datasetLF = Dataset(u_train, y_train, g_train)
    

    ###################################
    # Step 5: update dataset statistics
    ###################################
    Cylinder_datasetLF.update_statistics()

    ###################################
    # Step 6: scale and move data to torch
    ###################################
    U_train, Y_train, G_train = scale_and_move_data_to_torch(Cylinder_datasetLF, scale_mode=args.scale_mode, device=ptu.device, requires_grad=False)
    Train_datasetLF = DataGenerator(U_train, Y_train, G_train)

    ###################################
    # Step 7: define DeepONet LF model
    ###################################
    branch = {}
    branch["type"] = args.LF_branch_type
    branch["layer_size"] = [args.nSensors] + [args.width] * args.depth + [args.nBasis]
    branch["activation"] = args.activation
    branch["Nmode"] = args.LF_nmode
    
    trunk = {}
    trunk["type"] = args.LF_trunk_type
    trunk["layer_size"] = [1] + [args.width] * args.depth + [args.nBasis] # Create a list
    trunk["activation"] = args.activation
    trunk["Nmode"] = args.LF_nmode
    

    if args.deeponet_type == "vanilla":
        A_Model = DeepONet(branch, trunk).to(ptu.device)
    elif args.deeponet_type == "modified":
        A_Model = modified_DeepONet(branch, trunk).to(ptu.device)
        
    
    if args.deeponet_type == "vanilla":
        B_Model = DeepONet(branch, trunk).to(ptu.device)
    elif args.deeponet_type == "modified":
        B_Model = modified_DeepONet(branch, trunk).to(ptu.device)
        
    if Method== "Physics" :     
      # ----- Model= A(u,t) * sin(B(u,t) * t) ---- #
      modelLF = Physics_DeepONet (A_Model , B_Model)
    elif: Method== "Data" :
      modelLF = A_Model
    
    
    if args.verbose:
        print(modelLF)  
              
    ###################################
    # Step 8: define training parameters
    ###################################
    train_params = {}
    train_params["learning rate"] = args.learning_rate
    train_params["batch size"] = args.batch_size
    train_params["epochs"] = args.nEpochs
    train_params["validate every"] = 1

    ###################################
    # Step 9: define scheduler parameters
    ###################################
    scheduler_params = {}
    scheduler_params["patience"] = 100
    scheduler_params["factor"] = 0.9

    ###################################
    # Step 10: define loss function
    ###################################
    loss_fn = MAE_fn if args.loss == "MAE" else MSE_fn

    ###################################
    # Step 11: define logging filename
    ###################################
    log_filenameLF = "./output/" + feild + '_'+ method + "_LF_best_model.pt"

    ###################################
    # Step 12: train the LF model
    ################################### 

    train(modelLF,
          Train_datasetLF,
          train_params,
          scheduler_params=scheduler_params,
          loss_fn=loss_fn,
          logging_filename=log_filenameLF,
          ckpt_filename=None,
          use_ckpt=False,
          device=ptu.device,
          monitor=args.monitor,
          verbose=args.verbose,
          )
    
    ###################################
    # Step 13: restore best saved LF model
    ###################################
    ckptl = ptu.restore(log_filenameLF)
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
         
         
    ##----------------------------------- High fidelity Part -------------------------------## 
    
     
    ###################################
    # Step 15: collect the HF dataset
    ###################################
    data_pathHF ="./data/" +"Fine_Database.npy"
    Fine_Database = np.load(data_pathHF)            # field points and measurement/sampling locations

    ###################################
    # Step 16: prepare training and test datasets
    ###################################
    train_data, test_datah = prepare_dataset_HF(Fine_Database, field=feild, test_size=0.12, verbose=args.verbose, rng=2*args.rng, cutting=cutting)
    u_train, y_train, g_train = prepare_DeepONet_dataset(train_data, nLocs=args.nLocs, verbose=args.verbose)
    Cylinder_datasetHF = Dataset(u_train, y_train, g_train)
    
    u_train, y_train, g_train = prepare_DeepONet_dataset_HF(train_data, modelLF, Cylinder_datasetLF, scale_mode=args.scale_mode,nLocs=args.nLocs, verbose=args.verbose)
    Cylinder_datasetRes = Dataset(u_train, y_train, g_train)
   

    ###################################
    # Step 17: update dataset statistics
    ###################################
    Cylinder_datasetHF.update_statistics()
    Cylinder_datasetRes.update_statistics()
    ###################################
    # Step 18: scale and move data to torch
    ###################################
    U_train, Y_train, G_train = scale_and_move_data_to_torch(Cylinder_datasetRes, scale_mode=args.scale_mode, device=ptu.device, requires_grad=False)
    Train_datasetHF = DataGenerator(U_train, Y_train, G_train)

    ###################################
    # Step 19: define DeepONet HF model
    ###################################
    branch = {}
    branch["type"] = args.HF_branch_type
    branch["layer_size"] = [args.nSensors] + [args.width]  * args.depth + [args.nBasis]
    branch["activation"] = args.activation
    branch["Nmode"] = args.HF_nmode
   
    trunk = {}
    trunk["type"] = args.HF_trunk_type
    trunk["layer_size"] = [1] + [args.width]  * args.depth + [args.nBasis]
    trunk["activation"] = args.activation
    trunk["Nmode"] = args.HF_nmode
   

    if args.deeponet_type == "vanilla":
        A_Model = DeepONet(branch, trunk).to(ptu.device)
    elif args.deeponet_type == "modified":
        A_Model = modified_DeepONet(branch, trunk).to(ptu.device)
        
    
    if args.deeponet_type == "vanilla":
        B_Model = DeepONet(branch, trunk).to(ptu.device)
    elif args.deeponet_type == "modified":
        B_Model = modified_DeepONet(branch, trunk).to(ptu.device)
        
        
    if Method== "Physics":    
      # ----- Model= A(u,t) * sin(B(u,t) * t) ---- #
      modelHF = Physics_DeepONet (A_Model , B_Model)
    elif: Method== "Data":
      modelHF = A_Model
    
    
    if args.verbose:
        print(modelHF)  

    ###################################
    # Step 20: define training parameters
    ###################################
    train_params = {}
    train_params["learning rate"] = args.learning_rate
    train_params["batch size"] = args.batch_size
    train_params["epochs"] = args.nEpochs
    train_params["validate every"] = 1

    ###################################
    # Step 21: define scheduler parameters
    ###################################
    scheduler_params = {}
    scheduler_params["patience"] = 100
    scheduler_params["factor"] = 0.9

    ###################################
    # Step 22: define loss function
    ###################################
    loss_fn = MAE_fn if args.loss == "MAE" else MSE_fn

    ###################################
    # Step 23: define logging filename
    ###################################
    log_filenameHF = "./output/"  +feild+'_'+ method + "_HF_best_model.pt"

    ###################################
    # Step 24: train the HF model
    ################################### 

    train(modelHF,
          Train_datasetHF,
          train_params,
          scheduler_params=scheduler_params,
          loss_fn=loss_fn,
          logging_filename=log_filenameHF,
          ckpt_filename=None,
          use_ckpt=False,
          device=ptu.device,
          monitor=args.monitor,
          verbose=args.verbose,
          )
#    
    ###################################
    # Step 25: restore best saved HF model
    ###################################
    ckpth = ptu.restore(log_filenameHF)
    state_dicth = ckpth['state_dict']
    modelHF.load_state_dict(state_dicth)
    modelHF.to(ptu.device) 
    
    ###################################
    # Step 26: test the HF model
    ###################################
    modelsh=[modelLF, modelHF]
    datasetsh=[Cylinder_datasetLF, Cylinder_datasetRes,Cylinder_datasetHF]
    info=[field, method]
    
    test(modelsh,
         test_datah,
         datasetsh,
         scale_mode=args.scale_mode,
         nLocs=300,
         device=ptu.device,
         plot_trajs=True,
         plot_idxs=[0, 1, 2, 3, 4, 5],
         verbose=args.verbose,
         info=info,
         index=1,
         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training Cylinder DeepONet")
    
    ## general args
    parser.add_argument('--field', type=str, default="lift", help="lift or drag")
    parser.add_argument('--method', type=str, default="Physics", help="Physics or Data")
    parser.add_argument('--cutting', action='cut', default=True, help="True or False")
    
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
    parser.add_argument('--nEpochs', type=int, default=3, help="number of training epochs")
    parser.add_argument('--loss', type=str, default="MSE", help="MAE or MSE")
    parser.add_argument('--filename', type=str, default="train-cylinder-01", help="logging filename") 
    parser.add_argument('--monitor', type=str, default="train-loss", help="train-loss or val-loss")
    parser.add_argument('--rng', type=int, default=456, help="random state")

    args = parser.parse_args()
    main(args)