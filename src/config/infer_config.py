import numpy as np

## Infer options ##
DATAFILE_PATH = "data/lorentz_N_100_h001_T20.npy"
PLOT_TRAJS = True
PLOT_IDXS = [0]
# TRAINED_MODEL_PATH = "runs:/031d4006d072418d9fefed87c9d64b23/best_model_epoch_299"
# TRAINED_MODEL_PATH = "runs:/76b3f35e9d3b4ad1b93076ee6df6dc9a/best_model_epoch_170"
TRAINED_MODEL_PATH = "runs:/09ea52ba93d54fc4b4efb715709e1ff7/best_model_epoch_24"
FIGURE_PATH = "figures/"
VERBOSE = True

## Data options ##
OFFSET = 0.02
T_MAX = 20
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
