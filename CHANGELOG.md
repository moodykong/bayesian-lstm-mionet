# Changelog

## Unreleased

### Changed: results differ from the code the paper was produced with

- **reSGLD noise is drawn per parameter entry.** The Langevin update adds `scale * xi` with
  `xi ~ N(0, I)` drawn independently for every weight. The research code drew a single scalar per
  parameter tensor, so every entry of a weight matrix received the same kick. Bayesian ensembles
  trained now will therefore differ from the published ones (Table 2, Table 4). The update now
  runs in PyTorch on the training device and draws from the PyTorch generator, so `set_seed`
  makes it reproducible. The Adam path is unaffected.
- **The posterior predictive sample uses the right spread.** `infer-bayesian` draws its
  "Ensemble sample" from `N(mean, std^2)`. The research code passed the standard deviation where
  the covariance expects the variance, which sampled with standard deviation `sqrt(std)`. Only
  the sample curve and its two error tables change; the ensemble mean, standard deviation and
  PICP do not.
- **The ensemble standard deviation uses `1 / (M - 1)`**, as defined in Sec. 3.5.2 of the paper
  (it was `1 / M`). For M = 300 this widens the band by 0.17 %.
- **Ausgrid is tested on 100 sub-sequences per day**, as in Sec. 4.3 of the paper (the
  configuration and `scripts/reproduce_ausgrid.sh` used 200).

### Fixed

- **The published pretrained models did not load.** The archived `mlruns` store was written by
  the research code, which pickled its models against `models.architectures` and recorded
  absolute paths under the machine it ran on. `load_model` now maps those pickles onto the
  classes of this package, and the new `blstm-mionet relocate-mlruns` command points a moved store
  at its new location. `tests/test_legacy_models.py` writes such a store with a frozen copy of the
  research code's classes, moves it, and checks that the predictions are unchanged; the same
  check passed on a store written with MLflow 2.8.
- `LSTM_MLP` takes the length of a zero-padded history from the position of its last non-zero
  step instead of counting non-zero entries, so an exact zero *inside* a history no longer drops
  the most recent samples. Histories without interior zeros are encoded exactly as before, so
  existing trained models are unaffected.
- The masking routines no longer re-seed NumPy's and PyTorch's global generators. They draw from
  a private `RandomState(999)`, which yields exactly the same sub-sequences as before.
- The Adam loop accumulates the epoch loss from detached tensors instead of chaining every
  batch's autograd graph together for the rest of the epoch.
- A `training.resume_model` that cannot be loaded is an error; it used to print a message and
  silently train from scratch.
- `--device parallel` claimed to use every GPU but never replicated the model. It is now
  documented as a legacy alias of the current CUDA device, and the configurations use `0`.
- `evaluate_recursive` warns when a free-running rollout would feed a prediction for `t_n + h`
  back in at a different time, i.e. when consecutive evaluation points are not `h` apart.
- Continuous integration: the Python 3.12 job never ran a test, because `uv run` rebuilt the
  environment for Python 3.10 without the `cpu` extra. Every step now uses the matrix interpreter.

### Documentation

- The Ausgrid window (CSV columns 18-38) holds the half-hour readings from 07:00 to 17:00, not
  "9 am to 7 pm" as the paper puts it (the paper counts the columns from midnight and skips the
  five metadata columns). The column window itself is unchanged.
- The README states two properties of the pendulum data that the paper's wording does not: the
  Gaussian random field is evaluated at `theta_dot` (a state-feedback torque), and its kernel
  `exp(-(d / a)^2)` with `a = 0.01` is an RBF kernel with `l ≈ 0.007`. Both are kept as in the
  research code that produced the published numbers.
- The reSGLD epoch count (400, i.e. 40 burn-in epochs before 360 members) is attributed to the
  research code; the paper does not state it.
- The `mlflow ui` hints in the notebooks, scripts and READMEs set `MLFLOW_ALLOW_FILE_STORE=true`,
  without which MLflow 3 refuses the `mlruns` file store.
- `scripts/README.md`: a reSGLD run registers `<name>-bayesian`, and its burn-in is 40 epochs.

## 1.0.0 (2026-09-18)

First packaged release of the research code behind arXiv:2311.16519: the `blstm_mionet` package
and its `blstm-mionet` command line interface, one YAML configuration per experiment, a pytest
suite, CI, tutorial notebooks and reproduce scripts.
