from argparse import ArgumentParser
from config import train_config, infer_config, architecture_config, data_config


def add_train_args(parser: ArgumentParser):
    config = train_config.get_config()
    parser.add_argument("--datafile_path", type=str, default=config["datafile_path"])
    parser.add_argument("--search_len", type=int, default=config["search_len"])
    parser.add_argument("--search_num", type=int, default=config["search_num"])
    parser.add_argument("--search_random", type=bool, default=config["search_random"])
    parser.add_argument("--offset", type=int, default=config["offset"])
    parser.add_argument("--verbose", type=bool, default=config["verbose"])
    parser.add_argument("--scale_mode", type=str, default=config["scale_mode"])
    parser.add_argument("--learning_rate", type=float, default=config["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--epochs", type=int, default=config["epochs"])
    parser.add_argument("--validate_freq", type=int, default=config["validate_freq"])
    parser.add_argument("--use_scheduler", type=bool, default=config["use_scheduler"])
    parser.add_argument(
        "--scheduler_patience", type=int, default=config["scheduler_patience"]
    )
    parser.add_argument(
        "--scheduler_factor", type=float, default=config["scheduler_factor"]
    )
    parser.add_argument("--loss_function", type=str, default=config["loss_function"])
    parser.add_argument(
        "--early_stopping_epochs", type=int, default=config["early_stopping_epochs"]
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=config["checkpoint_path"]
    )
    parser.add_argument("--monitor_metric", type=str, default=config["monitor_metric"])
    parser.add_argument(
        "--use_trained_model", type=bool, default=config["use_trained_model"]
    )
    parser.add_argument(
        "--trained_model_path", type=str, default=config["trained_model_path"]
    )
    return parser


def add_infer_args(parser: ArgumentParser):
    config = infer_config.get_config()
    parser.add_argument("--datafile_path", type=str, default=config["datafile_path"])
    parser.add_argument("--search_len", type=int, default=config["search_len"])
    parser.add_argument("--search_num", type=int, default=config["search_num"])
    parser.add_argument("--search_random", type=bool, default=config["search_random"])
    parser.add_argument("--offset", type=int, default=config["offset"])
    parser.add_argument("--verbose", type=bool, default=config["verbose"])
    parser.add_argument("--scale_mode", type=str, default=config["scale_mode"])
    parser.add_argument("--loss_function", type=str, default=config["loss_function"])
    parser.add_argument("--plot_trajs", type=bool, default=config["plot_trajs"])
    parser.add_argument("--plot_idxs", type=list, default=config["plot_idxs"])
    parser.add_argument(
        "--trained_model_path", type=str, default=config["trained_model_path"]
    )
    parser.add_argument("--figure_path", type=str, default=config["figure_path"])
    return parser


def add_architecture_args(parser: ArgumentParser):
    config = architecture_config.get_config()
    parser.add_argument(
        "--branch_state_width", type=int, default=config["branch_state_width"]
    )
    parser.add_argument(
        "--branch_state_depth", type=int, default=config["branch_state_depth"]
    )
    parser.add_argument(
        "--branch_state_activation", type=str, default=config["branch_state_activation"]
    )
    parser.add_argument(
        "--branch_memory_width", type=int, default=config["branch_memory_width"]
    )
    parser.add_argument(
        "--branch_memory_depth", type=int, default=config["branch_memory_depth"]
    )
    parser.add_argument(
        "--branch_memory_lstm_size", type=int, default=config["branch_memory_lstm_size"]
    )
    parser.add_argument(
        "--branch_memory_lstm_layer_num",
        type=int,
        default=config["branch_memory_lstm_layer_num"],
    )
    parser.add_argument(
        "--branch_memory_activation",
        type=str,
        default=config["branch_memory_activation"],
    )
    parser.add_argument("--trunk_width", type=int, default=config["trunk_width"])
    parser.add_argument("--trunk_depth", type=int, default=config["trunk_depth"])
    parser.add_argument(
        "--trunk_activation", type=str, default=config["trunk_activation"]
    )
    return parser


def add_data_args(parser: ArgumentParser):
    config = data_config.get_config()
    parser.add_argument(
        "--ode_system",
        type=str,
        default=config["ode_system"],
        choices=["pendulum", "lorentz"],
    )
    parser.add_argument("--t_max", type=float, default=config["t_max"])
    parser.add_argument("--step_size", type=float, default=config["step_size"])
    parser.add_argument("--n_sample", type=int, default=config["n_sample"])
    parser.add_argument("--x_init_pts", type=list, default=config["x_init_pts"])
    parser.add_argument(
        "--ctr_func",
        type=str,
        default=config["ctr_func"],
        choices=["designate", "gaussian"],
    )
    parser.add_argument("--datafile_path", type=str, default=config["datafile_path"])
    return parser


def args_to_config(parser: ArgumentParser):
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg, None)
    return config
