import numpy as np

## Infer options ##
DATAFILE_PATH = "data/Ausgrid_cust_51-60.npy"
PLOT_TRAJS = True
PLOT_IDXS = [0]
# TRAINED_MODEL_PATH = (
#    "runs:/031d4006d072418d9fefed87c9d64b23/best_model_epoch_299"  # pendulum h=0.01
# )
# TRAINED_MODEL_PATH = (
#    "runs:/76b3f35e9d3b4ad1b93076ee6df6dc9a/best_model_epoch_170"  # pendulum h=0.1
# )
# TRAINED_MODEL_PATH = (
#    "runs:/09ea52ba93d54fc4b4efb715709e1ff7/best_model_epoch_381"  # lorentz h=0.01
# )
# TRAINED_MODEL_PATH = (
#    "runs:/5e1fec5193a3437bb041dea1aa00fb80/best_model_epoch_280"  # pendulum h=0.2
# )
# TRAINED_MODEL_PATH = "runs:/dc2b156f846347289bcb1be4fe054f85/best_model_epoch_100"  # Ausgrid customer 1-50, h=0.25
TRAINED_MODEL_PATH = "runs:/f59b5c9580d24ef08aa32bd8cc77a9cb/best_model_epoch_140"  # Ausgrid customer 1-50, h=0.5

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
