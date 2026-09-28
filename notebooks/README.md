# Notebooks

Four tutorial notebooks for [`blstm_mionet`](../src/blstm_mionet), the reference implementation of
[*B-LSTM-MIONet: Bayesian LSTM-based Neural Operators for Learning the Response of Complex
Dynamical Systems to Length-Variant Multiple Input Functions*](https://arxiv.org/abs/2311.16519)
(arXiv:2311.16519).

Each notebook walks through one experiment with the Python API — load a configuration, look at a
masked training sample, build the operator, train it briefly, evaluate it, plot the result — and
shows the equivalent `blstm-mionet` command line call next to every step. The last cell of each
notebook collects the full-scale commands that reproduce the published numbers.

| Notebook | Paper | What it covers |
| --- | --- | --- |
| [`01_lorentz.ipynb`](01_lorentz.ipynb) | Sec. 4.1 | Lorenz 63, an **autonomous** chaotic system: the LSTM branch is fed the history of the state itself. Masking (Sec. 3.4), training, one-step evaluation, and a recursive rollout at teacher forcing 1.0 / 0.5 / 0.0. |
| [`02_pendulum.ipynb`](02_pendulum.ipynb) | Sec. 4.2, 5.1 | Pendulum swing-up, a **non-autonomous** system driven by a Gaussian-random-field torque. Adds an out-of-distribution test with `u = sin(t/2)` and a comparison against the `DeepONet_Local` baseline. |
| [`03_ausgrid.ipynb`](03_ausgrid.ipynb) | Sec. 4.3 | Rooftop-PV gross generation from the **licensed** Ausgrid solar-home dataset: where to get the CSV files, what layout the loader expects, and a synthetic stand-in so the notebook runs without them. |
| [`04_uncertainty.ipynb`](04_uncertainty.ipynb) | Sec. 3.5 | Bayesian UQ with the **replica-exchange SGLD** ensemble: exploit/explore chains and swaps, the posterior ensemble logged to MLflow, the 95 % confidence band and the PICP. |

## Install

The environment is managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).
PyTorch comes from an extra, so pick exactly one of `cpu`, `cu126` or `cu130`:

```bash
uv sync --extra cpu --group notebooks       # CPU wheels (enough for every QUICK run)
# uv sync --extra cu126 --group notebooks   # CUDA 12.6
# uv sync --extra cu130 --group notebooks   # CUDA 13.0
```

## Launch

```bash
uv run jupyter lab notebooks/
```

The setup cell of every notebook walks up from the kernel's working directory until it finds
`configs/`, so the notebooks work whether the kernel starts in `notebooks/` or at the repository
root. It then `chdir`s to the repository root, because every path in a configuration file is
resolved relative to the working directory, and points `MLFLOW_TRACKING_URI` at `<root>/mlruns`.

## The `QUICK` switch

Every notebook has a cell near the top with

```python
QUICK = True
```

followed by a list of dotted configuration overrides (`data.n_sample=40`, `training.epochs=5`, …)
and a `FULL_OVERRIDES` block. With `QUICK = True` the notebook runs end to end on a CPU in a few
minutes on a tiny dataset with a smaller network; the errors it prints are *demonstration* errors
and are much larger than the published ones. With `QUICK = False` the notebook uses
`configs/*.yaml` exactly as shipped, which is the paper's setting — that needs a GPU and hours, so
the command line (see the closing cell of each notebook) is usually the better route.

The markdown cell above the switch states both the `QUICK` value and the paper value of every
setting it changes.

## Where the outputs go

All relative to the repository root, and all git-ignored:

| Directory | Contents |
| --- | --- |
| `data/` | generated `.npy` datasets, and the synthetic Ausgrid CSV if notebook 03 has to build one |
| `mlruns/` | the MLflow file store: parameters, metrics, models and reSGLD ensemble artifacts |
| `figures/` | PNGs written by `blstm_mionet.evaluation.plotting` |

To browse the runs:

```bash
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui --backend-store-uri mlruns
```

then open <http://127.0.0.1:5000>. MLflow 3 refuses a file store unless `MLFLOW_ALLOW_FILE_STORE=true`
is set; the `blstm-mionet` commands and the notebooks set it for you, the bare `mlflow` CLI does not.

## Outputs are not committed

The notebooks are stored with their outputs cleared. The `nbstripout` hook in
[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) strips outputs and execution counts from
every notebook on commit, so re-running them locally does not create a diff. To clear them by hand:

```bash
uv run jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```
