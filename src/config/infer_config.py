import numpy as np

## Infer options ##
DATAFILE_PATH = "data/pendulum_ctr_grf_N_100_h001_T10.npy"
PLOT_TRAJS = True
PLOT_IDXS = [0]
TRAINED_MODEL_PATH = "runs:/868411c1066b445cbfd244ad13102813/best_model_epoch_488"
FIGURE_PATH = "figures/"
VERBOSE = True

## Data options ##
OFFSET = 0.02
T_MAX = 10
STATE_COMPONENT = 0
SEARCH_LEN = 2
SEARCH_NUM = 200
BATCH_SIZE = 100
SEARCH_RANDOM = False
SCALE_MODE = ""


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
        "batch_size": BATCH_SIZE,
        "plot_trajs": PLOT_TRAJS,
        "plot_idxs": PLOT_IDXS,
        "trained_model_path": TRAINED_MODEL_PATH,
        "figure_path": FIGURE_PATH,
    }
    return config
