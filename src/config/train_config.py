import numpy as np

## Training options ##
DATAFILE_PATH = "data/Ausgrid_select.npy"
CHECKPOINT_PATH = "models/checkpoints/"
SAVE_MODEL = True
MONITOR_METRIC = "val_loss"  # 'val_loss' or 'train_loss'
USE_TRAINED_MODEL = False
TRAINED_MODEL_PATH = ""
VERBOSE = True
VALIDATE_FREQ = 1
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
LOSS_FUNCTION = "MSE"


## Data options ##
OFFSET = 0.0
T_MAX = None
STATE_COMPONENT = 0
SEARCH_LEN = 10
SEARCH_NUM = 10
SEARCH_RANDOM = True
SCALE_MODE = ""

## Hyperparameters ##
LEARNING_RATE = 1e-3
BATCH_SIZE = 200
EPOCHS = 1000
EARLY_STOPPING_EPOCHS = 80

## Test options ##
PLOT_TRAJS = True
PLOT_IDXS = [0]
FIGURE_PATH = "figures/"

## Device options ##
DEVICE = "parallel"  # int,"parallel","cpu"
EXPERIMENT_NAME = "LSTM_MIONet_Experiment"
RUN_NAME = "Default_Single_Run"


def get_config():
    config = {
        "datafile_path": DATAFILE_PATH,
        "state_component": STATE_COMPONENT,
        "search_len": SEARCH_LEN,
        "search_num": SEARCH_NUM,
        "search_random": SEARCH_RANDOM,
        "offset": OFFSET,
        "t_max": T_MAX,
        "verbose": VERBOSE,
        "scale_mode": SCALE_MODE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "validate_freq": VALIDATE_FREQ,
        "use_scheduler": USE_SCHEDULER,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "scheduler_factor": SCHEDULER_FACTOR,
        "loss_function": LOSS_FUNCTION,
        "early_stopping_epochs": EARLY_STOPPING_EPOCHS,
        "save_model": SAVE_MODEL,
        "checkpoint_path": CHECKPOINT_PATH,
        "monitor_metric": MONITOR_METRIC,
        "use_trained_model": USE_TRAINED_MODEL,
        "trained_model_path": TRAINED_MODEL_PATH,
        "plot_trajs": PLOT_TRAJS,
        "plot_idxs": PLOT_IDXS,
        "figure_path": FIGURE_PATH,
        "device": DEVICE,
        "experiment_name": EXPERIMENT_NAME,
        "run_name": RUN_NAME,
    }
    return config
