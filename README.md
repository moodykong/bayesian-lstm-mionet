# LSTM-MIONet

## How to set up the environment

The recommended environment tool for this repo is the devcontainder extension of Microsoft VS Code.

The configuration file of devcontainer is in `./.devcontainer` . 

Docker engine is required to run this container.

Alternatively, local environment tool like pip or conda can be used to set up the environment, and the dependencies are available at `./requirements.txt` .

## How to train and test the model

I provide three notebooks for the workflow of Lorentz, pendulum, and Ausgrid dataset under `./src` . 

Please refer to these notebooks for data generation, model training and inferring. 

## Data source

The pendulum and Lorentz data are generated from math formulas, so no additional data file is needed.

The Ausgrid data zip file is provided at [link](https://purdue0-my.sharepoint.com/:f:/g/personal/kong89_purdue_edu/EpzjocRBrE1Pgqi8CAdwL3ABWlN0nKbfFew7-jML6UOJvA?e=GUgUYn).

The Ausgrid problem requires raw data files in the locations below:

- src/data/Ausgrid/Solar home half-hour data - 1 July 2010 to 30 June 2011
- src/data/Ausgrid/Solar home half-hour data - 1 July 2011 to 30 June 2012
- src/data/Ausgrid/Solar home half-hour data - 1 July 2012 to 30 June 2013

## Mlfow tracked train data and artifacts

The mlflow tracking uri is set as `src/mlruns`.

To retrieve trained models and inspect the training process and arguments, refer to [link](https://purdue0-my.sharepoint.com/:f:/g/personal/kong89_purdue_edu/EpzjocRBrE1Pgqi8CAdwL3ABWlN0nKbfFew7-jML6UOJvA?e=GUgUYn).

Unzip the file then move the mlruns to `src/mlruns`.