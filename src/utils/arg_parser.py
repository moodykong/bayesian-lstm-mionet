from argparse import ArgumentParser


def train_args(parser: ArgumentParser):
    parser.add_argument("--datafile_path", type=str)
    parser.add_argument("--search_len", type=int)
    parser.add_argument("--search_num", type=int)
    parser.add_argument("--search_random", type=bool)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--verbose", type=bool)
    parser.add_argument("--scale_mode", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validate_freq", type=int)
    parser.add_argument("--use_scheduler", type=bool)
    parser.add_argument("--scheduler_patience", type=int)
    parser.add_argument("--scheduler_factor", type=float)
    parser.add_argument("--loss_function", type=str)
    parser.add_argument("--early_stopping_epochs", type=int)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--monitor_metric", type=str)
    parser.add_argument("--use_trained_model", type=bool)
    parser.add_argument("--trained_model_path", type=str)
    return parser


def infer_args(parser: ArgumentParser):
    parser.add_argument("--datafile_path", type=str)
    parser.add_argument("--search_len", type=int)
    parser.add_argument("--search_num", type=int)
    parser.add_argument("--search_random", type=bool)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--verbose", type=bool)
    parser.add_argument("--scale_mode", type=str)
    parser.add_argument("--loss_function", type=str)
    parser.add_argument("--plot_trajs", type=bool)
    parser.add_argument("--plot_idxs", type=list)
    parser.add_argument("--trained_model_path", type=str)
    parser.add_argument("--figure_path", type=str)
    return parser


def architecture_args(parser: ArgumentParser):
    parser.add_argument("--branch_state_width", type=int)
    parser.add_argument("--branch_state_depth", type=int)
    parser.add_argument("--branch_state_activation", type=str)
    parser.add_argument("--branch_memory_width", type=int)
    parser.add_argument("--branch_memory_depth", type=int)
    parser.add_argument("--branch_memory_lstm_size", type=int)
    parser.add_argument("--branch_memory_lstm_layer_num", type=int)
    parser.add_argument("--branch_memory_activation", type=str)
    parser.add_argument("--trunk_width", type=int)
    parser.add_argument("--trunk_depth", type=int)
    parser.add_argument("--trunk_activation", type=str)
    return parser


def args_to_config(args):
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg, None)
    return config
