#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from utils import torch_utils
from models.architectures import LSTM_MIONet
from optim.supervisor import execute_train, execute_test
from utils.data_utils import (
    Dataset_Torch,
    Dataset_Stat,
    prepare_local_predict_dataset,
    scale_and_to_tensor,
    split_dataset,
)
from config import train_config, architecture_config
from utils.args_parser import add_train_args, add_architecture_args, args_to_config
import argparse, os

# Set current working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def run(config):
    ###################################
    # Step 1: initialize the gpu
    ###################################
    torch_utils.init_gpu(verbose=config["verbose"])

    ###################################
    # Step 2: set the seed
    ###################################
    seed = 9999
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    ###################################
    # Step 3: collect the dataset
    ###################################

    train_dataset = config["datafile_path"]
    train_dataset = np.load(
        train_dataset, allow_pickle=True
    ).item()  # load the .npy file

    ###################################
    # Step 4: prepare training and test datasets
    ###################################

    train_data, test_data = split_dataset(
        train_dataset, test_size=0.1, verbose=config["verbose"]
    )
    input_masked, x_n, x_next, t_params = prepare_local_predict_dataset(
        data=train_data,
        search_len=config["search_len"],
        search_num=config["search_num"],
        search_random=config["search_random"],
        offset=config["offset"],
        verbose=config["verbose"],
    )
    train_data_stat = Dataset_Stat(input_masked, x_n, x_next, t_params)
    state_feature_num = x_n.shape[-1]

    input_masked, x_n, x_next, t_params = prepare_local_predict_dataset(
        data=test_data,
        search_len=config["search_len"],
        search_num=config["search_num"],
        search_random=config["search_random"],
        offset=config["offset"],
        verbose=config["verbose"],
    )
    test_data_stat = Dataset_Stat(input_masked, x_n, x_next, t_params)

    ###################################
    # Step 5: update dataset statistics
    ###################################
    train_data_stat.update_statistics()
    test_data_stat.update_statistics()

    ###################################
    # Step 6: scale and move numpy data to tensor
    ###################################
    input_masked, x_n, x_next, t_params = scale_and_to_tensor(
        train_data_stat, scale_mode=config["scale_mode"], device=torch_utils.device
    )
    train_data_torch = Dataset_Torch(input_masked, x_n, x_next, t_params)

    input_masked, x_n, x_next, t_params = scale_and_to_tensor(
        test_data_stat, scale_mode=config["scale_mode"], device=torch_utils.device
    )
    test_data_torch = Dataset_Torch(input_masked, x_n, x_next, t_params)

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

    model = LSTM_MIONet(branch_state, branch_memory, trunk)

    if config["verbose"]:
        print(model)

    ###################################
    # Step 8: train the model
    ###################################

    execute_train(
        config=config, model=model, dataset=train_data_torch, device=torch_utils.device
    )

    ###################################
    # Step 9: test the model
    ###################################

    execute_test(config=config, model=model, dataset=test_data_torch)


def main():
    # Load the configurations
    config = dict()
    config.update(train_config.get_config())
    config.update(architecture_config.get_config())
    # Add the arguments
    parser = argparse.ArgumentParser()
    parser = add_train_args(parser)
    parser = add_architecture_args(parser)
    # Parse the arguments
    args_config = args_to_config(parser)
    # Update the configurations
    config.update(args_config)
    # Create the checkpoint path
    os.makedirs(config["checkpoint_path"], exist_ok=True)
    # Run the program
    run(config)


if __name__ == "__main__":
    main()
