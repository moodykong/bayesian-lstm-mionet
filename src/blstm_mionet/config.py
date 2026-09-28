"""Typed configuration objects and YAML loading.

Every experiment is described by a single YAML file with the sections
``data``, ``model``, ``training``, ``inference`` and ``tracking``; the
replica-exchange SGLD hyper-parameters live in a separate file that is merged
in under the ``bayesian`` key (``--bayesian configs/bayesian/lorentz.yaml``).

All paths appearing in a configuration file (datasets, figure directories, the
MLflow tracking URI) are interpreted **relative to the current working
directory**.  The package never changes the working directory.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "AusgridConfig",
    "BayesianConfig",
    "BranchMemoryConfig",
    "BranchStateConfig",
    "ConfigError",
    "DataConfig",
    "ExperimentConfig",
    "GRFConfig",
    "InferConfig",
    "LangevinConfig",
    "ModelConfig",
    "ReplicaConfig",
    "TrackingConfig",
    "TrainConfig",
    "TrunkConfig",
    "load_config",
]

ARCHITECTURE_CHOICES = (
    "LSTM_MIONet",
    "LSTM_DeepONet",
    "DeepONet",
    "DeepONet_Local",
)
SYSTEM_CHOICES = ("lorentz", "pendulum", "ausgrid")
CONTROL_CHOICES = ("designate", "gaussian")
SCALE_MODE_CHOICES = ("", "normalize", "min-max")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or incomplete."""


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class GRFConfig:
    """Parameters of the 1-D Gaussian random field used as a control signal."""

    a: float = 0.01
    nu: float = 1.0


@dataclass
class AusgridConfig:
    """Selection parameters for the (licensed) Ausgrid solar-home dataset.

    ``csv_paths`` must point at the CSV files released by Ausgrid; they are not
    redistributed with this repository.
    """

    csv_paths: list[str] = field(default_factory=list)
    customer_id: list[int] | None = None
    #: ``YYYY-MM-DD``; an unquoted YAML date is accepted as well.
    start_date: str | datetime.date | None = None
    end_date: str | datetime.date | None = None
    category: str | None = "GG"
    delta_t_idxs: float = 0.1
    # Half-hour columns kept from every daily record; 18:39 selects the columns
    # that are mostly non-zero (daylight hours for the "GG" category).
    column_start: int = 18
    column_end: int = 39
    # Daily records with fewer than this fraction of non-zero readings are dropped.
    min_nonzero_fraction: float = 0.8
    # Original sampling interval of the raw columns, in hours.
    sample_interval_hours: float = 0.5


@dataclass
class DataConfig:
    """Dataset generation options."""

    system: str = "lorentz"
    t_max: float = 10.0
    step_size: float = 0.01
    n_sample: int = 100
    #: ``[[lo, hi], ...]`` ranges the initial state components are drawn from.
    x_init_pts: list[list[float]] = field(default_factory=list)
    #: ``"designate"`` (u = sin(t / 2)), ``"gaussian"`` (GRF) or ``None``.
    control: str | None = None
    grf: GRFConfig = field(default_factory=GRFConfig)
    ausgrid: AusgridConfig = field(default_factory=AusgridConfig)
    output: str = "data/dataset.npy"
    seed: int = 999
    verbose: bool = True

    def validate(self) -> None:
        if self.system not in SYSTEM_CHOICES:
            raise ConfigError(
                f"data.system must be one of {SYSTEM_CHOICES}, got {self.system!r}"
            )
        if self.system == "ausgrid":
            if not self.ausgrid.csv_paths:
                raise ConfigError(
                    "data.ausgrid.csv_paths is required for the Ausgrid system; "
                    "point it at the CSV files distributed by Ausgrid."
                )
            return
        if self.control is not None and self.control not in CONTROL_CHOICES:
            raise ConfigError(
                f"data.control must be one of {CONTROL_CHOICES} or null, "
                f"got {self.control!r}"
            )
        if not self.x_init_pts:
            raise ConfigError("data.x_init_pts is required for ODE systems")
        for pair in self.x_init_pts:
            if len(pair) != 2:
                raise ConfigError(
                    "every entry of data.x_init_pts must be a [low, high] pair, "
                    f"got {pair!r}"
                )
        if self.step_size <= 0:
            raise ConfigError("data.step_size must be positive")
        if self.n_sample <= 0:
            raise ConfigError("data.n_sample must be positive")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class BranchStateConfig:
    """Fully connected branch acting on the current state ``x_n``."""

    width: int = 200
    depth: int = 3
    activation: str = "relu"


@dataclass
class BranchMemoryConfig:
    """LSTM branch acting on the (masked, length-variant) input function."""

    width: int = 200
    depth: int = 3
    activation: str = "relu"
    lstm_size: int = 100
    lstm_layer_num: int = 2


@dataclass
class TrunkConfig:
    """Fully connected trunk acting on the time spacing ``h``."""

    width: int = 200
    depth: int = 3
    activation: str = "relu"


@dataclass
class ModelConfig:
    architecture: str = "LSTM_MIONet"
    branch_state: BranchStateConfig = field(default_factory=BranchStateConfig)
    branch_memory: BranchMemoryConfig = field(default_factory=BranchMemoryConfig)
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    use_bias: bool = True

    def validate(self) -> None:
        if self.architecture not in ARCHITECTURE_CHOICES:
            raise ConfigError(
                f"model.architecture must be one of {ARCHITECTURE_CHOICES}, "
                f"got {self.architecture!r}"
            )


# --------------------------------------------------------------------------- #
# Training / inference
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Options for :func:`blstm_mionet.training.trainer.train_adam`."""

    datafile: str = ""
    state_component: int = 0
    #: ``h_max``: maximum sub-sequence length, in time steps.
    search_len: int = 10
    #: ``N_h``: number of sub-sequences sampled from every trajectory.
    search_num: int = 10
    search_random: bool = True
    offset: float = 0.0
    t_max: float | None = None
    scale_mode: str = ""
    #: fraction of trajectories held out for the post-training evaluation.
    holdout_size: float = 0.1
    #: fraction of the training samples used for validation / early stopping.
    validation_split: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 200
    epochs: int = 1000
    validate_freq: int = 1
    use_scheduler: bool = True
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    loss_function: str = "MSE"
    early_stopping_epochs: int = 80
    monitor_metric: str = "val_loss"
    save_model: bool = True
    registered_model_name: str | None = None
    #: optional MLflow URI to warm start from (replaces the old
    #: ``use_trained_model`` / ``trained_model_path`` pair of local checkpoints).
    resume_model: str | None = None
    plot_trajs: bool = True
    plot_idxs: list[int] = field(default_factory=lambda: [0])
    figure_dir: str = "figures"
    device: int | str = "cpu"
    experiment_name: str = "blstm_mionet"
    run_name: str = "run"
    verbose: bool = True

    def validate(self) -> None:
        if not self.datafile:
            raise ConfigError(
                "training.datafile is required (use --data to override it)"
            )
        if self.monitor_metric not in ("val_loss", "train_loss"):
            raise ConfigError(
                "training.monitor_metric must be 'val_loss' or 'train_loss', "
                f"got {self.monitor_metric!r}"
            )
        if self.loss_function not in ("MSE", "MAE"):
            raise ConfigError(
                f"training.loss_function must be 'MSE' or 'MAE', got {self.loss_function!r}"
            )
        if self.scale_mode not in SCALE_MODE_CHOICES:
            raise ConfigError(
                f"training.scale_mode must be one of {SCALE_MODE_CHOICES}, "
                f"got {self.scale_mode!r}"
            )
        if self.epochs <= 0:
            raise ConfigError("training.epochs must be positive")


@dataclass
class InferConfig:
    """Options for the ``infer`` and ``infer-bayesian`` sub-commands."""

    datafile: str = ""
    #: MLflow model URI, e.g. ``models:/lorentz/latest`` or ``runs:/<id>/<path>``.
    model: str = ""
    #: MLflow run URI or bare run id holding a reSGLD ensemble.
    run: str = ""
    recursive: bool = False
    teacher_forcing_prob: float = 1.0
    autonomous: bool = True
    state_component: int = 0
    search_len: int = 10
    search_num: int = 50
    search_random: bool = False
    offset: float = 0.0
    t_max: float | None = None
    scale_mode: str = ""
    batch_size: int = 100
    #: number of ensemble members to read back; ``None`` uses every member found.
    n_ensemble: int | None = None
    plot_trajs: bool = True
    plot_idxs: list[int] = field(default_factory=lambda: [0])
    figure_dir: str = "figures"
    device: int | str = "cpu"
    verbose: bool = True

    def validate(self) -> None:
        if not self.datafile:
            raise ConfigError(
                "inference.datafile is required (use --data to override it)"
            )
        if self.scale_mode not in SCALE_MODE_CHOICES:
            raise ConfigError(
                f"inference.scale_mode must be one of {SCALE_MODE_CHOICES}, "
                f"got {self.scale_mode!r}"
            )


# --------------------------------------------------------------------------- #
# Bayesian (replica exchange SGLD)
# --------------------------------------------------------------------------- #
@dataclass
class LangevinConfig:
    """Parameters of one stochastic gradient Langevin dynamics chain.

    ``tau`` is the temperature (the injected noise grows like ``sqrt(tau)``, so
    the explore chain uses the larger value), ``eta`` the step size, ``alpha``
    the friction and ``v`` enters the gradient-noise estimate ``beta``.
    """

    tau: float = 1e-7
    eta: float = 1e-4
    alpha: float = 0.1
    v: float = 0.1

    @property
    def beta(self) -> float:
        """``0.5 * v * eta`` (as in the original ``ensemble_config.py``)."""
        return 0.5 * self.v * self.eta

    @property
    def scale(self) -> float:
        """Per-entry standard deviation of the injected Gaussian noise,
        ``sqrt(2 * (alpha - beta) * eta * tau)``."""
        return math.sqrt(2.0 * (self.alpha - self.beta) * self.eta) * math.sqrt(
            self.tau
        )


@dataclass
class ReplicaConfig:
    """Parameters of the replica exchange (swap) criterion."""

    sigma_uniform: float = 1.0
    f_adj: float = 9e7


@dataclass
class BayesianConfig:
    """Replica exchange SGLD hyper-parameters."""

    n_ensemble: int = 360
    use_grad_norm: bool = True
    grad_norm: float = 50.0
    print_every: int = 1
    #: noise level; every chain uses ``sigma = 2 * level ** 2``.
    level: float = 15.0
    exploit: LangevinConfig = field(default_factory=LangevinConfig)
    explore: LangevinConfig = field(
        default_factory=lambda: LangevinConfig(tau=2e-7, eta=1e-4, alpha=0.1, v=0.1)
    )
    replica: ReplicaConfig = field(default_factory=ReplicaConfig)

    @property
    def sigma(self) -> float:
        return 2 * (self.level**2)

    @property
    def tau_delta(self) -> float:
        """``1 / tau_exploit - 1 / tau_explore``."""
        return (1 / self.exploit.tau) - (1 / self.explore.tau)

    def burn_in(self, epochs: int) -> int:
        """Epoch after which ensemble members start being collected.

        Derived exactly as in the original ``train.py``:
        ``epochs - (n_ensemble + 1)``.
        """
        return epochs - (self.n_ensemble + 1)

    def validate(self) -> None:
        if self.n_ensemble <= 0:
            raise ConfigError("bayesian.n_ensemble must be positive")
        for name in ("exploit", "explore"):
            chain: LangevinConfig = getattr(self, name)
            if chain.tau <= 0:
                raise ConfigError(f"bayesian.{name}.tau must be positive")
            if chain.alpha - chain.beta < 0:
                raise ConfigError(
                    f"bayesian.{name}: alpha ({chain.alpha}) must exceed "
                    f"beta = 0.5 * v * eta ({chain.beta})"
                )
        if self.exploit.tau == self.explore.tau:
            raise ConfigError(
                "bayesian.exploit.tau and bayesian.explore.tau must differ, "
                "otherwise the two replicas share a temperature"
            )


# --------------------------------------------------------------------------- #
# Tracking + bundle
# --------------------------------------------------------------------------- #
@dataclass
class TrackingConfig:
    """MLflow tracking options.

    ``uri`` is overridden by the ``MLFLOW_TRACKING_URI`` environment variable
    when that variable is set.
    """

    uri: str = "mlruns"


@dataclass
class ExperimentConfig:
    """Everything needed to generate data, train and evaluate one experiment."""

    name: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    inference: InferConfig = field(default_factory=InferConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    bayesian: BayesianConfig | None = None

    def validate(self) -> None:
        self.model.validate()
        if self.bayesian is not None:
            self.bayesian.validate()

    def to_flat_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flatten to ``{"training.epochs": 10, ...}`` for ``mlflow.log_params``."""
        return _flatten(self, prefix)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    bayesian_path: str | Path | None = None,
) -> ExperimentConfig:
    """Read ``path`` (and optionally a Bayesian file) into an :class:`ExperimentConfig`.

    Parameters
    ----------
    path:
        YAML file with the ``data``/``model``/``training``/``inference``/
        ``tracking`` sections.
    overrides:
        ``["training.epochs=10", "data.ausgrid.csv_paths=[a.csv, b.csv]"]``.
        Values are parsed as YAML scalars, so ``true``, ``null``, numbers and
        inline lists all work.
    bayesian_path:
        YAML file with the replica exchange SGLD parameters.  Its contents are
        merged under the ``bayesian`` key (a top level ``bayesian:`` section in
        that file is unwrapped automatically).
    """
    raw = _read_yaml(path)
    if bayesian_path is not None:
        bayesian_raw = _read_yaml(bayesian_path)
        if set(bayesian_raw) == {"bayesian"}:
            bayesian_raw = bayesian_raw["bayesian"] or {}
        merged = _deep_merge(raw.get("bayesian") or {}, bayesian_raw)
        # An empty file (or one holding only an empty ``bayesian:`` key) must
        # not silently start reSGLD with every default; leave the section unset
        # so the CLI can report it.
        raw["bayesian"] = merged or None

    for override in overrides or []:
        _apply_override(raw, override)

    config = _from_mapping(ExperimentConfig, raw, "")
    config.validate()
    return config


def _read_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{file_path}: expected a YAML mapping at the top level")
    return loaded


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply a single ``section.key=value`` override in place."""
    if "=" not in override:
        raise ConfigError(
            f"invalid override {override!r}: expected the form KEY=VALUE, "
            "for example training.epochs=10"
        )
    key, _, value = override.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"invalid override {override!r}: empty key")
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid override {override!r}: {exc}") from exc

    node: Any = raw
    parts = key.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        elif not isinstance(child, dict):
            raise ConfigError(
                f"invalid override {override!r}: {part!r} is not a section"
            )
        node = child
    node[parts[-1]] = parsed


#: ``typing.Union[...]`` and the PEP 604 ``X | Y`` form have different origins.
UNION_ORIGINS = (Union, UnionType)


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) in UNION_ORIGINS:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _from_mapping(cls: type, data: Any, prefix: str) -> Any:
    """Recursively build the dataclass ``cls`` from a nested mapping."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        section = prefix.rstrip(".") or "<root>"
        raise ConfigError(f"{section}: expected a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        section = prefix.rstrip(".") or "<root>"
        raise ConfigError(
            f"{section}: unknown key(s) {unknown}; allowed keys are {sorted(known)}"
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        inner = _unwrap_optional(hints[f.name])
        if is_dataclass(inner):
            if value is None and _is_optional(hints[f.name]):
                # An optional section that is absent or null stays unset.
                kwargs[f.name] = None
            elif value is not None and not isinstance(value, dict):
                raise ConfigError(
                    f"{prefix}{f.name}: expected a mapping, got {type(value).__name__}"
                )
            else:
                kwargs[f.name] = _from_mapping(inner, value, f"{prefix}{f.name}.")
        else:
            kwargs[f.name] = _coerce(value, hints[f.name], f"{prefix}{f.name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:  # pragma: no cover - defensive
        section = prefix.rstrip(".") or "<root>"
        raise ConfigError(f"{section}: {exc}") from exc


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) in UNION_ORIGINS and type(None) in get_args(
        annotation
    )


def _coerce(value: Any, annotation: Any, key: str) -> Any:
    """Check a scalar/sequence value against its annotation, with clear errors."""
    if value is None:
        if _is_optional(annotation):
            return None
        raise ConfigError(f"{key}: must not be null")

    inner = _unwrap_optional(annotation)
    origin = get_origin(inner)

    if inner is Any:
        return value
    if origin in (list, tuple):
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{key}: expected a list, got {value!r}")
        return list(value)
    if origin in UNION_ORIGINS:
        allowed = tuple(arg for arg in get_args(inner) if arg is not type(None))
        if isinstance(value, allowed):
            return value
        names = " or ".join(getattr(arg, "__name__", str(arg)) for arg in allowed)
        raise ConfigError(f"{key}: expected {names}, got {value!r}")
    if inner is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{key}: expected true or false, got {value!r}")
    if inner is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{key}: expected a number, got {value!r}")
        return float(value)
    if inner is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        return value
    if inner is str:
        if not isinstance(value, str):
            raise ConfigError(f"{key}: expected a string, got {value!r}")
        return value
    return value


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        key = f"{prefix}{f.name}"
        if is_dataclass(value) and not isinstance(value, type):
            flat.update(_flatten(value, f"{key}."))
        elif dataclasses.is_dataclass(value):  # pragma: no cover - defensive
            flat[key] = str(value)
        else:
            flat[key] = value
    return flat
