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
        "--state_component",
        type=int,
        default=config["state_component"],
        help="Index of the state component to train.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=config["architecture"],
        choices=[
            "LSTM_MIONet",
            "LSTM_DeepONet",
            "LSTM_MIONet_Static",
            "DeepONet",
            "DeepONet_Local",
        ],
        help="Architecture to use.",
    )
    parser.add_argument(
        "--search_len",
        type=int,
        default=config["search_len"],
        help="Length of the search length (h_max). It is an integer of points.",
    )
    parser.add_argument(
        "--search_num",
        type=int,
        default=config["search_num"],
        help="Number of the search points on each trajectory (N_h). It is an integer of points.",
    )
    parser.add_argument(
        "--search_random",
        type=bool,
        default=config["search_random"],
        help="Whether to sample the search points randomly. If False, the search points are evenly sampled. Empty string means False. Otherwise, the value is True.",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=config["t_max"],
        help="Maximum time (float) of the trajectory.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=config["offset"],
        help="Offset is the minimum length (integer of points) of trajectory memory.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config["verbose"],
        help="Whether to print the training process. Empty string means False. Otherwise, the value is True.",
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
        help="Whether to use the scheduler. Empty string means False. Otherwise, the value is True.",
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
        "--save_model",
        type=bool,
        default=config["save_model"],
        help="Whether to save the model. Empty string means False. Otherwise, the value is True.",
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
        help="Whether to use previously trained model. Empty string means False. Otherwise, the value is True.",
    )
    parser.add_argument(
        "--trained_model_path",
        type=str,
        default=config["trained_model_path"],
        help="Path to the previously trained model.",
    )
    parser.add_argument(
        "--device",
        type=device_type,
        default=config["device"],
        help="Device to use. Enter an integer to use a specific GPU ID. Enter 'parallel' to use all available GPUs. Enter 'cpu' to use CPU.",
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "parallel", "cpu"],
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=config["experiment_name"],
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=config["run_name"],
        help="Name of the run for display.",
    )
    parser.add_argument(
        "--plot_trajs",
        type=bool,
        default=config["plot_trajs"],
        help="Whether to plot the test trajectories. Empty string means False. Otherwise, the value is True.",
    )
    parser.add_argument(
        "--plot_idxs",
        type=int,
        nargs="*",
        default=config["plot_idxs"],
        help="Indices of the test trajectories to plot.",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default=config["figure_path"],
        help="Path to save the figures.",
    )
    parser.add_argument(
        "--use_ensemble",
        type=bool,
        default=config["use_ensemble"],
        help="Whether to use ensemble. Empty string means False. Otherwise, the value is True.",
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
        "--architecture",
        type=str,
        default=config["architecture"],
        choices=[
            "LSTM_MIONet",
            "LSTM_DeepONet",
            "LSTM_MIONet_Static",
            "DeepONet",
            "DeepONet_Local",
        ],
        help="Architecture to use.",
    )
    parser.add_argument(
        "--recursive",
        type=bool,
        default=config["recursive"],
        help="Whether to recursively infer the trajectories. Empty string means False. Otherwise, the value is True.",
    )
    parser.add_argument(
        "--teacher_forcing_prob",
        type=float,
        default=config["teacher_forcing_prob"],
        help="Probability of using teacher forcing.",
    )
    parser.add_argument(
        "--autonomous",
        type=bool,
        default=config["autonomous"],
        help="Whether to use autonomous system. Empty string means False. Otherwise, the value is True.",
    )
    parser.add_argument(
        "--state_component",
        type=int,
        default=config["state_component"],
        help="Index of the state component to infer.",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=config["t_max"],
        help="Maximum time (float) of the trajectory.",
    )
    parser.add_argument(
        "--search_len",
        type=int,
        default=config["search_len"],
        help="Length of the search length (h_max). It is an integer of points.",
    )
    parser.add_argument(
        "--search_num",
        type=int,
        default=config["search_num"],
        help="Number of the search points on each trajectory (N_h). It is an integer of points.",
    )
    parser.add_argument(
        "--search_random",
        type=bool,
        default=config["search_random"],
        help="Whether to sample the search points randomly. If False, the search points are evenly sampled. Empty string means False. Otherwise, the value is True.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=config["offset"],
        help="Offset is the minimum length (integer of points) of trajectory memory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["batch_size"],
        help="Batch size of the training process.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config["verbose"],
        help="Whether to print the inferring process. Empty string means False. Otherwise, the value is True.",
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
        help="Whether to plot the trajectories. Empty string means False. Otherwise, the value is True.",
    )
    parser.add_argument(
        "--plot_idxs",
        type=int,
        nargs="*",
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
    parser.add_argument(
        "--device",
        type=device_type,
        default=config["device"],
        help="Device to use. Enter an integer to use a specific GPU ID. Enter 'parallel' to use all available GPUs. Enter 'cpu' to use CPU.",
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "parallel", "cpu"],
    )
    parser.add_argument(
        "--use_ensemble",
        type=bool,
        default=config["use_ensemble"],
        help="Whether to use ensemble. Empty string means False. Otherwise, the value is True.",
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
        choices=["pendulum", "lorentz", "Ausgrid"],
        help="ODE system to use.",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=config["t_max"],
        help="Maximum time (float) of the trajectory.",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=config["step_size"],
        help="Step size (float) of the trajectory.",
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
        "--Ausgrid_customer_id",
        type=int,
        default=config["Ausgrid_customer_id"],
        nargs="*",
        help="Customer ID of the Ausgrid dataset.",
    )
    parser.add_argument(
        "--Ausgrid_start_date",
        type=str,
        default=config["Ausgrid_start_date"],
        help="Start date of the Ausgrid dataset, e.g., 2010-7-1.",
    )
    parser.add_argument(
        "--Ausgrid_end_date",
        type=str,
        default=config["Ausgrid_end_date"],
        help="End date of the Ausgrid dataset, e.g., 2011-6-30.",
    )
    parser.add_argument(
        "--Ausgrid_category",
        type=str,
        default=config["Ausgrid_category"],
        choices=["GG", "CG"],
        help="Category of the Ausgrid dataset.",
    )
    parser.add_argument(
        "--Ausgrid_delta_t_idxs",
        type=float,
        default=config["Ausgrid_delta_t_idxs"],
        help="Time interval of the Ausgrid dataset.",
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
    config = vars(args)
    return config


def device_type(value):
    if value in ["parallel", "cpu"]:
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid device choice: {value}")
