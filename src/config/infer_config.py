import numpy as np

## Infer options ##
DATAFILE_PATH = "data/Ausgrid_cust_51-60.npy"
PLOT_TRAJS = True
PLOT_IDXS = [0]

## Load model ##
# TRAINED_MODEL_PATH = "models:/pendulum/latest"
# TRAINED_MODEL_PATH = "models:/lorentz/latest"
TRAINED_MODEL_PATH = "models:/Ausgrid/latest"

FIGURE_PATH = "figures/"
VERBOSE = True

## Data options ##
OFFSET = 0
T_MAX = None
STATE_COMPONENT = 0
SEARCH_LEN = 10
SEARCH_NUM = 50
BATCH_SIZE = 100
SEARCH_RANDOM = False
SCALE_MODE = ""

## Device options ##
DEVICE = 1  # int,"parallel","cpu"


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
        "device": DEVICE,
    }
    return config
