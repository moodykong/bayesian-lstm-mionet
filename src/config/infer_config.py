import numpy as np

## Infer options ##
DATAFILE_PATH = "data/pendulum_ctr_grf_N_100_h001_T10_2.npy"
PLOT_TRAJS = True
PLOT_IDXS = [0]
TRAINED_MODEL_PATH = "runs:/fb566c054a3c4e83a76cca2f8c81a359/best_model_epoch_248"
FIGURE_PATH = "figures/"
VERBOSE = True
BATCH_SIZE = 100

## Data options ##
OFFSET = 0.02
T_MAX = 10
STATE_COMPONENT = 0
SEARCH_LEN = 10
SEARCH_NUM = 200
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
