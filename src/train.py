#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import torch
import utils.torch_utils as torch_utils
from models.architectures import LSTM_MIONet
from optim.supervisor import train, test
from utils.data_utils import (
    Dataset_Torch,
    Dataset_Stat,
    prepare_local_predict_dataset,
    scale_and_to_device,
    split_dataset,
)
from config import train_config, architecture_config
from utils.arg_parser import add_train_args, add_architecture_args, args_to_config

seed = 1234


def run(config):
    ###################################
    # Step 0:

    ###################################

    ###################################
    # Step 1: initialize the gpu
    ###################################
    torch_utils.init_gpu(verbose=config["verbose"])

    ###################################
    # Step 2: initialize the device
    ###################################
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    ###################################
    # Step 3: collect the dataset
    ###################################

    train_dataset = config["datafile_path"]
    train_dataset = np.load(train_dataset).item()

    ###################################
    # Step 4: prepare training and test datasets
    ###################################

    train_data, test_data = split_dataset(
        train_dataset, test_size=0.1, verbose=config["verbose"]
    )
    input_masked, x_n, x_next, t_local = prepare_local_predict_dataset(
        data=train_data,
        search_len=config["search_len"],
        search_num=config["search_num"],
        search_random=config["search_random"],
        offset=config["offset"],
        verbose=config["verbose"],
    )
    train_data_stat = Dataset_Stat(input_masked, x_n, x_next, t_local)

    ###################################
    # Step 5: update dataset statistics
    ###################################
    train_data_stat.update_statistics()

    ###################################
    # Step 6: scale and move data to torch
    ###################################
    input_masked, x_n, x_next, t_local = scale_and_to_device(
        train_data_stat,
        scale_mode=config["scale_mode"],
        device=torch_utils.device,
        requires_grad=False,
    )
    train_data_torch = Dataset_Torch(input_masked, x_n, x_next, t_local)

    ###################################
    # Step 7: define the model
    ###################################
    branch_state = {}
    branch_state["layer_size_list"] = [config["branch_state"]["width"]] * config[
        "branch_state"
    ]["depth"]
    branch_state["activation"] = config["branch_state"]["activation"]

    branch_memory = {}
    branch_memory["layer_size_list"] = [config["branch_memory"]["width"]] * config[
        "branch_memory"
    ]["depth"]
    branch_memory["lstm_size"] = config["branch_memory"]["lstm_size"]
    branch_memory["lstm_layer_num"] = config["branch_memory"]["lstm_layer_num"]
    branch_memory["activation"] = config["branch_memory"]["activation"]

    trunk = {}
    trunk["layer_size_list"] = [config["trunk"]["width"]] * config["trunk"]["depth"]
    trunk["activation"] = config["trunk"]["activation"]

    model = LSTM_MIONet(branch_state, branch_memory, trunk)

    if config["verbose"]:
        print(model)

    ###################################
    # Step 11: define logging filename
    ###################################
    checkpoint_path = config["checkpoint_path"]

    ###################################
    # Step 12: train the LF model
    ###################################

    train(config=config, model=model, dataset=train_data_torch)

    ###################################
    # Step 13: restore the trained model
    ###################################
    if config["use_trained_model"]:
        try:
            trained_model = torch_utils.restore(config["trained_model_path"])
            state_dict = trained_model["state_dict"]
            model.load_state_dict(state_dict)
            model.to(torch_utils.device)
        except:
            print("Failed to load the trained model, use the untrained model instead.")

    ###################################
    # Step 14: test the model
    ###################################

    test(config=config, model=model, dataset=test_data)


def main():
    config = dict()
    config.update(train_config.get_config())
    config.update(architecture_config.get_config())
    parser = argparse.ArgumentParser()
    parser = add_train_args(parser)
    parser = add_architecture_args(parser)
    args_config = args_to_config(parser)
    config.update(args_config)
    os.makedirs(config["checkpoint_path"], exist_ok=True)


if __name__ == "__main__":
    main()
