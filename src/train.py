#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from utils import torch_utils
from models import architectures
from optim.supervisor import execute_train, execute_test
from optim.supervisor import replica_train_UQ_V2 as execute_train_reSGLD
from utils.data_utils import (
    Dataset_Torch,
    Dataset_Stat,
    prepare_local_predict_dataset,
    prepare_future_local_predict_dataset,
    scale_and_to_tensor,
    split_dataset,
)
from config import ensemble_config as ensemble
from utils.args_parser import add_train_args, add_architecture_args, args_to_config
import argparse, os, copy

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
        data=train_data,
        state_component=config["state_component"],
        t_max=config["t_max"],
        search_len=config["search_len"],
        search_num=config["search_num"],
        search_random=config["search_random"],
        offset=config["offset"],
        verbose=config["verbose"],
    )
    train_data_stat = Dataset_Stat(input_masked, x_n, x_next, t_params)
    state_feature_num = x_n.shape[-1]

    input_masked, x_n, x_next, t_params = prepare_nn_dataset(
        data=test_data,
        state_component=config["state_component"],
        offset=config["offset"],
        t_max=config["t_max"],
        search_len=config["search_len"],
        search_num=config["search_num"],
        search_random=config["search_random"],
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

    if config["verbose"]:
        print(model)

    ###################################
    # Step 8: train the model
    ###################################

    if config["use_ensemble"]:
        execute_train_reSGLD(
            model_exploit=model,
            model_explore=copy.deepcopy(model),
            dataset=train_data_torch,
            train_params=config["train_params"],
            replica_params=config["replica_params"],
            exploit_params=config["exploit_params"],
            explore_params=config["explore_params"],
            device=torch_utils.device,
        )
    else:
        execute_train(
            config=config,
            model=model,
            dataset=train_data_torch,
            device=torch_utils.device,
        )

    ###################################
    # Step 9: test the model
    ###################################
    if config["use_ensemble"]:
        execute_test(
            config=config,
            model=model,
            dataset=test_data_torch,
            device=torch_utils.device,
        )
    else:
        execute_test(config=config, model=model, dataset=test_data_torch)


def main():
    # Add the arguments
    parser = argparse.ArgumentParser()
    parser = add_train_args(parser)
    parser = add_architecture_args(parser)

    # Parse the arguments
    config = args_to_config(parser)

    # Load the ensemble config
    if config["use_ensemble"]:
        ensemble_config = ensemble.get_config()
        train_params = dict()
        train_params["epochs"] = config["epochs"]
        train_params["batch_size"] = config["batch_size"]
        train_params["burn in"] = config["epochs"] - (ensemble_config["n_ensemble"] + 1)
        ensemble_config["train_params"].update(train_params)
        config.update(ensemble_config)

    # Create the checkpoint path
    os.makedirs(config["checkpoint_path"], exist_ok=True)

    # Run the program
    run(config)


if __name__ == "__main__":
    main()
