#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from models import architectures
import mlflow
from utils import torch_utils
from optim.supervisor import test_replica
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

    test_data = ((input_masked, x_n, t_params), x_next)

    ###################################
    # Step 5: update dataset statistics
    ###################################
    test_data_stat.update_statistics()

    ###################################
    # Step 6: scale and move numpy data to tensor
    ###################################

    # input_masked, x_n, x_next, t_params = scale_and_to_tensor(
    #   test_data_stat, scale_mode=config["scale_mode"], device=torch_utils.device
    # )
    # test_data_torch = Dataset_Torch(input_masked, x_n, x_next, t_params)

    ###################################
    # Step 7: define the model
    ###################################
    branch_state = {}
    branch_state["layer_size_list"] = [config["branch_state_width"]] * config[
        "branch_state_depth"
    ]
    branch_state["activation"] = config["branch_state_activation"]
    branch_state["state_feature_num"] = state_feature_num

    branch_memory = {}
    branch_memory["layer_size_list"] = [config["branch_memory_width"]] * config[
        "branch_memory_depth"
    ]
    branch_memory["lstm_size"] = config["branch_memory_lstm_size"]
    branch_memory["lstm_layer_num"] = config["branch_memory_lstm_layer_num"]
    branch_memory["activation"] = config["branch_memory_activation"]

    trunk = {}
    trunk["layer_size_list"] = [config["trunk_width"]] * config["trunk_depth"]
    trunk["activation"] = config["trunk_activation"]

    match config["architecture"]:
        case "DeepONet":
            model = architectures.DeepONet(branch_state, trunk, use_bias=True)
        case "DeepONet_Local":
            model = architectures.DeepONet_Local(branch_state, trunk, use_bias=True)
        case "LSTM_MIONet":
            model = architectures.LSTM_MIONet(
                branch_state, branch_memory, trunk, use_bias=True
            )
        case "LSTM_MIONet_Static":
            model = architectures.LSTM_MIONet_Static(
                branch_state, branch_memory, trunk, use_bias=True
            )
        case "LSTM_DeepONet":
            model = architectures.LSTM_DeepONet(branch_memory, trunk, use_bias=True)
        case _:
            raise ValueError("Invalid architecture string.")

    ###################################
    # Step 8: load the model
    ###################################
    log_metric = test_replica(
        config=config,
        model=model,
        data=test_data,
        dataset=test_data_stat,
        device=torch_utils.device,
        unscaling=False,
        experiment="default",
    )


def main():
    # Add the arguments
    parser = argparse.ArgumentParser()
    parser = add_infer_args(parser)
    parser = add_architecture_args(parser)

    # Parse the arguments
    config = args_to_config(parser)

    # Run the program
    run(config)


if __name__ == "__main__":
    main()
