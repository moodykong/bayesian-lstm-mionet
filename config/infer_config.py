import numpy as np

## Infer options ##
DATAFILE_PATH = ''
PLOT_TRAJS = True
PLOT_IDXS = [0]
TRAINED_MODEL_PATH = ''
FIGURE_PATH = 'figures/'
VERBOSE = True
LOSS_FUNCTION='MSE'

## Data options ##
SEARCH_LEN = 10
SEARCH_NUM = 5
SEARCH_RANDOM = False
OFFSET = 0
SCALE_MODE = ''

def get_config():
    config = {
        'datafile_path': DATAFILE_PATH,
        'search_len': SEARCH_LEN,
        'search_num': SEARCH_NUM,
        'search_random': SEARCH_RANDOM,
        'offset': OFFSET,
        'verbose': VERBOSE,
        'scale_mode': SCALE_MODE,
        'loss_function': LOSS_FUNCTION,
        'plot_trajs': PLOT_TRAJS,
        'plot_idxs': PLOT_IDXS,
        'trained_model_path': TRAINED_MODEL_PATH,
        'figure_path': FIGURE_PATH
        }
    return config