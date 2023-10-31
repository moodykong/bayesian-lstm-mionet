#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from models.architectures import LSTM_MIONet
import mlflow
from utils import torch_utils
from optim.supervisor import execute_test, execute_test_recursive
from utils.data_utils import (
    Dataset_Torch,
    Dataset_Stat,
    prepare_local_predict_dataset,
    prepare_future_local_predict_dataset,
    scale_and_to_tensor,
    split_dataset,
)
from config import infer_config, architecture_config
from utils.args_parser import add_infer_args, add_architecture_args, args_to_config
import argparse, os

# Set current working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def run(config):
    ###################################
    # Step 1: initialize the gpu
    ###################################
    torch_utils.init_gpu(gpu_id=config["device"], verbose=config["verbose"])

    ###################################
    # Step 2: set the seed
    ###################################
    seed = 999
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    ###################################
    # Step 3: collect the dataset
    ###################################

    test_dataset = config["datafile_path"]
    test_dataset = np.load(test_dataset, allow_pickle=True).item()  # load the .npy file

    ###################################
    # Step 4: prepare training and test datasets
    ###################################

    _, test_data = split_dataset(test_dataset, test_size=1, verbose=config["verbose"])

    match config["architecture"]:
        case "DeepONet":
            prepare_nn_dataset = prepare_local_predict_dataset
        case "DeepONet_Local":
            prepare_nn_dataset = prepare_future_local_predict_dataset
        case "LSTM_MIONet":
            prepare_nn_dataset = prepare_local_predict_dataset
        case "LSTM_MIONet_Static":
            prepare_nn_dataset = prepare_local_predict_dataset
        case "LSTM_DeepONet":
            prepare_nn_dataset = prepare_local_predict_dataset
        case _:
            raise ValueError("Invalid architecture string.")

    input_masked, x_n, x_next, t_params = prepare_nn_dataset(
        data=test_data,
        state_component=config["state_component"],
        search_len=config["search_len"],
        search_num=config["search_num"],
        search_random=config["search_random"],
        offset=config["offset"],
        t_max=config["t_max"],
        verbose=config["verbose"],
    )
    test_data_stat = Dataset_Stat(input_masked, x_n, x_next, t_params)
    state_feature_num = x_n.shape[-1]

    ###################################
    # Step 5: update dataset statistics
    ###################################
    test_data_stat.update_statistics()

    ###################################
    # Step 6: scale and move numpy data to tensor
    ###################################

    input_masked, x_n, x_next, t_params = scale_and_to_tensor(
        test_data_stat, scale_mode=config["scale_mode"], device=torch_utils.device
    )
    test_data_torch = Dataset_Torch(input_masked, x_n, x_next, t_params)

    ###################################
    # Step 7: load the model
    ###################################
    mlflow.set_tracking_uri("mlruns")
    print(config["trained_model_path"])
    model = mlflow.pytorch.load_model(
        config["trained_model_path"], map_location=torch_utils.device
    )

    # Extract the model from the DataParallel wrapper
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = model.to(torch_utils.device)

    if config["verbose"]:
        print(model)

    if config["recursive"]:
        execute_test_recursive(config=config, model=model, dataset=test_data_torch)
    else:
        execute_test(config=config, model=model, dataset=test_data_torch)


def main():
    # Add the arguments
    parser = argparse.ArgumentParser()
    parser = add_infer_args(parser)

    # Parse the arguments
    config = args_to_config(parser)

    # Run the program
    run(config)


if __name__ == "__main__":
    main()
