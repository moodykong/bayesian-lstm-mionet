import numpy as np

## Training options ##
DATAFILE_PATH = 'data/'
CHECKPOINT_PATH = "models/"
MONITOR_METRIC = 'val_loss' # 'val_loss' or 'train_loss'
USE_TRAINED_MODEL = False
TRAINED_MODEL_PATH = ''
VERBOSE = True
VALIDATE_FREQ=1
USE_SCHEDULER = False
SCHEDULER_PATIENCE=10
SCHEDULER_FACTOR=0.5
LOSS_FUNCTION='MSE'


## Data options ##
SEARCH_LEN = 10
SEARCH_NUM = 5
SEARCH_RANDOM = True
OFFSET = 0
SCALE_MODE = ''

## Hyperparameters ##
LEARNING_RATE=1e-3
BATCH_SIZE=128
EPOCHS=1000
EARLY_STOPPING_EPOCHS = 100

def get_config():
    config = {
        'datafile_path': DATAFILE_PATH,
        'search_len': SEARCH_LEN,
        'search_num': SEARCH_NUM,
        'search_random': SEARCH_RANDOM,
        'offset': OFFSET,
        'verbose': VERBOSE,
        'scale_mode': SCALE_MODE,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'validate_freq': VALIDATE_FREQ,
        'use_scheduler': USE_SCHEDULER, 
        'scheduler_patience': SCHEDULER_PATIENCE,
        'scheduler_factor': SCHEDULER_FACTOR,
        'loss_function': LOSS_FUNCTION,
        'early_stopping_epochs': EARLY_STOPPING_EPOCHS,
        'checkpoint_path': CHECKPOINT_PATH, 
        'monitor_metric': MONITOR_METRIC,
        'use_trained_model': USE_TRAINED_MODEL,
        'trained_model_path': TRAINED_MODEL_PATH

    }
    return config
