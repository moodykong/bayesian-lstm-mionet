import numpy as np

## Infer options ##
INFER_DATASET = ''
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

## Hyperparameters ##
LEARNING_RATE=1e-3
BATCH_SIZE=128

## Model options ##
BRANCH_STATE = {'width':100, 'depth':3, 'activation':'relu'}
BRANCH_MEMORY = {'width':100, 'depth':3, 'lstm_size':10, 'lstm_layer_num':2, 'activation':'relu'}
TRUNK = {'width':100, 'depth':3, 'activation':'relu'}

def get_config():
    config = {
        'infer_dataset': INFER_DATASET,
        'search_len': SEARCH_LEN,
        'search_num': SEARCH_NUM,
        'search_random': SEARCH_RANDOM,
        'offset': OFFSET,
        'verbose': VERBOSE,
        'scale_mode': SCALE_MODE,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'loss_function': LOSS_FUNCTION,
        'branch_state': BRANCH_STATE,
        'branch_memory': BRANCH_MEMORY,
        'trunk': TRUNK,
        'plot_trajs': PLOT_TRAJS,
        'plot_idxs': PLOT_IDXS,
        'trained_model_path': TRAINED_MODEL_PATH,
        'figure_path': FIGURE_PATH
        }
    return config