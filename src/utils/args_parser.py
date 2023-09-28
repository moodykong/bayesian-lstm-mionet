from argparse import ArgumentParser
from config import train_config, infer_config, architecture_config, data_config


def add_train_args(parser: ArgumentParser):
    config = train_config.get_config()
    parser.add_argument(
        "--datafile_path",
        type=str,
        default=config["datafile_path"],
        help="Path to the datafile.",
    )
    parser.add_argument(
        "--search_len",
        type=int,
        default=config["search_len"],
        help="Length of the search length (h_max).",
    )
    parser.add_argument(
        "--search_num",
        type=int,
        default=config["search_num"],
        help="Number of the search points on each trajectory (N_h).",
    )
    parser.add_argument(
        "--search_random",
        type=bool,
        default=config["search_random"],
        help="Whether to sample the search points randomly. If False, the search points are evenly sampled.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=config["offset"],
        help="Offset is the minimum length of trajectory memory.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config["verbose"],
        help="Whether to print the training process.",
    )
    parser.add_argument(
        "--scale_mode",
        type=str,
        default=config["scale_mode"],
        help="Whether to scale the data. If None, no scaling is applied. If 'standard', the data is scaled to have zero mean and unit variance. If 'minmax', the data is scaled to be in the range of [0, 1].",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=config["learning_rate"],
        help="Learning rate of the optimizer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["batch_size"],
        help="Batch size of the training process.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["epochs"],
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--validate_freq",
        type=int,
        default=config["validate_freq"],
        help="Frequency of validation.",
    )
    parser.add_argument(
        "--use_scheduler",
        type=bool,
        default=config["use_scheduler"],
        help="Whether to use the scheduler.",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=config["scheduler_patience"],
        help="Patience of the scheduler.",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=config["scheduler_factor"],
        help="Factor of the scheduler.",
    )
    parser.add_argument(
        "--loss_function",
        type=str,
        default=config["loss_function"],
        help="Loss function to use.",
    )
    parser.add_argument(
        "--early_stopping_epochs",
        type=int,
        default=config["early_stopping_epochs"],
        help="Number of epochs to early stop.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=config["checkpoint_path"],
        help="Path to save the checkpoints.",
    )
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default=config["monitor_metric"],
        help="Metric to monitor.",
        choices=["val_loss", "train_loss"],
    )
    parser.add_argument(
        "--use_trained_model",
        type=bool,
        default=config["use_trained_model"],
        help="Whether to use previously trained model.",
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default=config["trained_model_path"],
        help="Path to the previously trained model.",
    )
    return parser


def add_infer_args(parser: ArgumentParser):
    config = infer_config.get_config()
    parser.add_argument(
        "--datafile_path",
        type=str,
        default=config["datafile_path"],
        help="Path to the datafile.",
    )
    parser.add_argument(
        "--search_len",
        type=int,
        default=config["search_len"],
        help="Length of the search length (h_max).",
    )
    parser.add_argument(
        "--search_num",
        type=int,
        default=config["search_num"],
        help="Number of the search points on each trajectory (N_h).",
    )
    parser.add_argument(
        "--search_random",
        type=bool,
        default=config["search_random"],
        help="Whether to sample the search points randomly. If False, the search points are evenly sampled.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=config["offset"],
        help="Offset is the minimum length of trajectory memory.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config["verbose"],
        help="Whether to print the inferring process.",
    )
    parser.add_argument(
        "--scale_mode",
        type=str,
        default=config["scale_mode"],
        help="Whether to scale the data. If None, no scaling is applied. If 'standard', the data is scaled to have zero mean and unit variance. If 'minmax', the data is scaled to be in the range of [0, 1].",
    )

    parser.add_argument(
        "--plot_trajs",
        type=bool,
        default=config["plot_trajs"],
        help="Whether to plot the trajectories.",
    )
    parser.add_argument(
        "--plot_idxs",
        type=list,
        default=config["plot_idxs"],
        help="Indices of the trajectories to plot.",
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default=config["trained_model_path"],
        help="Path to the previously trained model.",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default=config["figure_path"],
        help="Path to save the figures.",
    )
    return parser


def add_architecture_args(parser: ArgumentParser):
    config = architecture_config.get_config()
    parser.add_argument(
        "--branch_state_width",
        type=int,
        default=config["branch_state_width"],
        help="Width of the state branch.",
    )
    parser.add_argument(
        "--branch_state_depth",
        type=int,
        default=config["branch_state_depth"],
        help="Depth of the state branch.",
    )
    parser.add_argument(
        "--branch_state_activation",
        type=str,
        default=config["branch_state_activation"],
        help="Activation function of the state branch.",
    )
    parser.add_argument(
        "--branch_memory_width",
        type=int,
        default=config["branch_memory_width"],
        help="Width of the memory branch.",
    )
    parser.add_argument(
        "--branch_memory_depth",
        type=int,
        default=config["branch_memory_depth"],
        help="Depth of the memory branch.",
    )
    parser.add_argument(
        "--branch_memory_lstm_size",
        type=int,
        default=config["branch_memory_lstm_size"],
        help="Size of the LSTM in the memory branch.",
    )
    parser.add_argument(
        "--branch_memory_lstm_layer_num",
        type=int,
        default=config["branch_memory_lstm_layer_num"],
        help="Number of the LSTM layers in the memory branch.",
    )
    parser.add_argument(
        "--branch_memory_activation",
        type=str,
        default=config["branch_memory_activation"],
        help="Activation function of the memory branch.",
    )
    parser.add_argument(
        "--trunk_width",
        type=int,
        default=config["trunk_width"],
        help="Width of the trunk.",
    )
    parser.add_argument(
        "--trunk_depth",
        type=int,
        default=config["trunk_depth"],
        help="Depth of the trunk.",
    )
    parser.add_argument(
        "--trunk_activation",
        type=str,
        default=config["trunk_activation"],
        help="Activation function of the trunk.",
    )
    return parser


def add_data_args(parser: ArgumentParser):
    config = data_config.get_config()
    parser.add_argument(
        "--ode_system",
        type=str,
        default=config["ode_system"],
        choices=["pendulum", "lorentz"],
        help="ODE system to use.",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=config["t_max"],
        help="Maximum time (seconds) of the trajectory.",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=config["step_size"],
        help="Step size (seconds) of the trajectory.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=config["n_sample"],
        help="Number of the trajectories to generate.",
    )
    parser.add_argument(
        "--x_init_pts",
        type=float,
        default=config["x_init_pts"],
        nargs="*",
        help="The range of initial points of the trajectories. Two floats define the range of each state component.",
    )
    parser.add_argument(
        "--ctr_func",
        type=str,
        default=config["ctr_func"],
        choices=["designate", "gaussian"],
        help="Control function to use.",
    )
    parser.add_argument(
        "--datafile_path",
        type=str,
        default=config["datafile_path"],
        help="Path to save the data.",
    )
    return parser


def args_to_config(parser: ArgumentParser):
    args = parser.parse_args()
    config = dict()
    config["args"] = vars(args)
    for arg in vars(args):
        config[arg] = getattr(args, arg, None)
    return config
