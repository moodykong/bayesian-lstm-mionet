# `scripts/`

Shell entry points that drive the `blstm-mionet` command line interface: one
script per experiment of the paper, one Bayesian (reSGLD) script, a smoke test
and a data downloader.  Everything here is a thin wrapper around the CLI and the
YAML files in `configs/`; the scripts never contain hyper-parameters that are
not also in a configuration file.

## Prerequisites

Install the environment once, from the repository root:

```bash
uv sync --extra cpu        # or --extra cu126 / --extra cu130 for CUDA
```

The scripts look for `blstm-mionet` on `PATH` first (an activated virtual
environment), then fall back to `uv run blstm-mionet`; set
`BLSTM_MIONET_CMD="uv run --no-sync blstm-mionet"` to override that choice.
Every script can be started from anywhere -- each one `cd`s to the repository
root first -- and each one understands `--help`.

## Conventions

**`--quick`.** Every pipeline script takes `--quick`, which swaps the paper
scale settings for a tiny CPU run: a handful of short trajectories instead of
5000 long ones, 2 Adam epochs instead of 1000 (8 reSGLD epochs with a 3 member
ensemble instead of 400 epochs with 360), `--device cpu`, a small `search_num`,
and no MLflow model registration.  The steps themselves are
unchanged, so `--quick` exercises exactly the same code path in about a minute.
The numbers it prints are meaningless as science; they only prove the pipeline
runs.  Without `--quick` the scripts run the paper configuration and default to
`--device 0`.

**`--device DEVICE`.** A GPU index, `parallel` (all visible GPUs) or `cpu`.
Defaults to `0` for full runs and `cpu` in quick mode.

**`--workdir DIR`.** All paths inside `configs/*.yaml` are relative to the
current directory, so by default `data/`, `mlruns/`, `figures/` and `logs/` are
created in the repository root.  `--workdir DIR` runs the pipeline inside `DIR`
instead, which is how `smoke_test.sh` keeps the repository clean.

**Extra arguments.** Anything a pipeline script does not recognise is forwarded
verbatim to `blstm-mionet train`, for example
`scripts/reproduce_lorentz.sh --epochs 50 --set training.batch_size=500`.

**Runs and models.** Training prints `MLflow run uri: runs:/<id>`; the scripts
capture that id and evaluate `runs:/<id>/model`, so a run never picks up a
stale `models:/<name>/latest`.  The registered name of the configuration
(`lorentz`, `pendulum`, `Ausgrid`) is printed in the final summary of a full
run.  Runs land in `./mlruns`; browse them with
`mlflow ui --backend-store-uri ./mlruns`.

## The scripts

**`reproduce_lorentz.sh`** runs the autonomous Lorenz 63 experiment (paper
Sec. 4.1) end to end: it generates the 5000 trajectory training set
(`T = 20 s`, `dt = 0.01`), a 100 trajectory test set and a single reference
trajectory, trains LSTM-MIONet with Adam from `configs/lorentz.yaml`, evaluates
one-step-ahead prediction on the test set, and then rolls the operator out
recursively on the single trajectory with teacher forcing probabilities 1.0,
0.5 and 0.0 (the three panels of the notebook).  The test sets use a different
`data.seed` than the training set so they really are unseen initial conditions.
Figures land in `figures/lorentz/{train,test,recursive_tf10,recursive_tf05,recursive_tf00}`.

**`reproduce_pendulum.sh`** runs the non-autonomous pendulum experiment
(Sec. 4.2): 5000 trajectories driven by Gaussian random field controls
(`T = 10 s`), an in-distribution GRF test set, and the out-of-distribution set
driven by the designated control `u = sin(t / 2)` (`data.control=designate`).
After Adam training it evaluates both test sets, so the generalisation gap of
the paper is visible in one run.  Figures land in
`figures/pendulum/{train,test_grf,test_designate}`.

**`reproduce_ausgrid.sh`** runs the Ausgrid solar home experiment (Sec. 4.3).
It first checks that the three CSV files listed in `configs/ausgrid.yaml` exist
and, if they do not, prints exactly where to get them and stops.  It then
selects the gross generation profiles of customers 1-50 for training and of
customers 51-60 and 61-70 for the two test groups (2010-07-01 to 2011-06-30),
trains, and evaluates both test groups.  `--quick` needs no download at all: it
writes a small synthetic CSV with the same layout (title row, header,
48 half-hour columns, day-first dates) into `data/Ausgrid_synthetic/` and uses
customers 1-3 / 4-5 / 6 as the three groups -- useful to check the pipeline,
useless as a result.  Figures land in
`figures/ausgrid/{train,test_group_a,test_group_b}`.

**`reproduce_bayesian.sh SYSTEM`** (`SYSTEM` is `lorentz`, `pendulum` or
`ausgrid`) is the B-LSTM-MIONet half of the paper.  It reuses the datasets
written by the matching `reproduce_SYSTEM.sh` (generating them only if they are
missing), trains two replica-exchange Langevin chains with
`configs/bayesian/SYSTEM.yaml` for 400 epochs -- 39 burn-in epochs followed by
360 collected posterior samples -- and then runs `blstm-mionet infer-bayesian`,
which evaluates the `M = 300` members named by `inference.n_ensemble` and
prints the PICP of the 95% credible interval; the script repeats that PICP line
in its summary.  Note that a full Bayesian run also registers its exploit chain
under the same model name as the deterministic run (`models:/lorentz`, ...),
which the summary points out.

**`smoke_test.sh`** is the fast sanity check: it creates a temporary directory
(removed by a trap on exit), runs `reproduce_lorentz.sh --quick`,
`reproduce_pendulum.sh --quick` and `reproduce_bayesian.sh lorentz --quick`
inside it -- by default the three at once (each process limited to a third of
the cores so they do not fight over them), `--sequential` for one after another
with live output -- and exits non-zero if any of them fails, printing the failing
pipeline's log.  Nothing is written into the repository.  Use `--keep` to
inspect the temporary directory afterwards.

**`download_data.sh`** fetches the two things that are not in git: the Ausgrid
CSV selection and the `mlruns` store with the pretrained registered models
(`lorentz`, `pendulum`, `Ausgrid`).  Both currently live in one OneDrive folder,
and OneDrive share links cannot be downloaded non-interactively, so the script
does not pretend otherwise: run without arguments it prints step-by-step manual
instructions (download the folder as a zip in a browser, re-run with
`--archive PATH`), the three CSV paths `configs/ausgrid.yaml` expects and the
original Ausgrid source.  Given `--archive PATH` (or `--url URL`, for a future
Zenodo or GitHub release link that curl can follow) it unpacks the archive into
a temporary directory, verifies SHA-256 checksums when `scripts/checksums.sha256`
lists any, merges the Ausgrid folders into `data/Ausgrid/` and the run store
into `./mlruns` without ever replacing or deleting existing files, and reports
what landed where.  It needs no Python environment.

**`checksums.sha256`** ships empty on purpose; the maintainer fills it in once
the archive has a stable published URL.  **`_common.sh`** is not a user facing
script: it is sourced by the others and holds the CLI discovery, the shared
argument parsing, the "generate only if missing" helper, the run-id capture and
the synthetic Ausgrid CSV writer.

## Runtimes

| command | measured here (CPU) | paper scale |
| --- | --- | --- |
| `reproduce_lorentz.sh --quick` | 55 s | -- |
| `reproduce_pendulum.sh --quick` | 44 s | -- |
| `reproduce_ausgrid.sh --quick` | 41 s | -- |
| `reproduce_bayesian.sh lorentz --quick` | 30 s | -- |
| `smoke_test.sh` (the three in parallel) | 55 s | -- |
| `reproduce_lorentz.sh` | -- | hours on one GPU, not measured here |
| `reproduce_pendulum.sh` | -- | hours on one GPU, not measured here |
| `reproduce_ausgrid.sh` | -- | hours on one GPU, not measured here |
| `reproduce_bayesian.sh SYSTEM` | -- | hours on one GPU, not measured here |

The quick numbers were measured on a 20 core CPU-only machine (torch 2.14, no
CUDA) that was busy with other work, so they are upper bounds; they are
dominated by interpreter start-up, MLflow model logging and -- for the pendulum
-- the 2000 x 2000 Cholesky factorisation behind every Gaussian random field
control, not by the training itself.  Running three pipelines at once only pays
off when each of them is kept from grabbing every core, which is why
`smoke_test.sh` caps `OMP_NUM_THREADS`/`MKL_NUM_THREADS` at a third of the
cores; without that cap the same three pipelines took five times longer here.

The full pipelines were not run in this environment: they need a GPU,
5000 trajectory datasets and up to 1000 Adam epochs (400 reSGLD epochs with two
chains each), which is hours per experiment.
