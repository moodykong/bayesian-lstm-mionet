# Contributing

Thanks for your interest in B-LSTM-MIONet. Bug reports, reproductions and new dynamical systems are
all welcome.

## Setup

```bash
uv sync --extra cpu      # or --extra cu126 / --extra cu130 on a CUDA machine
uv run pre-commit install
```

`uv sync` installs the project in editable mode together with the `dev` dependency group (pytest,
ruff, black, pre-commit). Add `--group notebooks` when you need Jupyter.

## Tests, lint and formatting

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
```

Ruff and black read their settings from `pyproject.toml` (line length 88, target Python 3.10). Run
`uv run black src tests` to apply the formatting instead of only checking it. The pre-commit hooks in
`.pre-commit-config.yaml` run the same checks, plus whitespace and YAML/TOML validation, on every
commit.

## Notebooks

Notebook outputs are stripped by the `nbstripout` pre-commit hook, so commit notebooks with their
outputs cleared. Keep the figures a notebook produces out of git — `figures/`, `data/` and `mlruns/`
are all ignored. If a notebook needs a trained model, load it from an MLflow URI rather than from a
checkpoint committed to the repository.

## Adding a new dynamical system

1. Add the vector field to [`src/blstm_mionet/data/systems.py`](src/blstm_mionet/data/systems.py) as
   a function `f(x, u) -> dx/dt` and register it in the `SYSTEMS` dictionary. Add the name to
   `SYSTEM_CHOICES` in [`src/blstm_mionet/config.py`](src/blstm_mionet/config.py).
2. Add a YAML file under [`configs/`](configs/), copying the closest existing one — `configs/lorentz.yaml`
   for an autonomous system, `configs/pendulum.yaml` for one driven by an input function. Keep the
   inline comments that tie each value to the paper, and keep every path relative to the working
   directory.
3. Add a test under [`tests/`](tests/): generating a handful of short trajectories and checking their
   shapes and finiteness is enough, and keeps the suite fast on CPU.
4. Check it end to end with a tiny run before opening the pull request:

   ```bash
   uv run blstm-mionet generate --config configs/<system>.yaml --n-sample 20 --set data.t_max=2.0 \
       --output data/<system>_demo.npy
   uv run blstm-mionet train --config configs/<system>.yaml --data data/<system>_demo.npy \
       --epochs 2 --device cpu
   ```

## Reporting bugs

Open an issue at
[github.com/moodykong/bayesian-lstm-mionet/issues](https://github.com/moodykong/bayesian-lstm-mionet/issues)
with the command you ran, the config (or the `--set` overrides), the full traceback, and your OS,
Python and torch versions.
