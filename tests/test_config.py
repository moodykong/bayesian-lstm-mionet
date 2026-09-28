"""Loading, overriding and validating the YAML experiment configurations."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from blstm_mionet.config import (
    ARCHITECTURE_CHOICES,
    BayesianConfig,
    ConfigError,
    DataConfig,
    ExperimentConfig,
    InferConfig,
    ModelConfig,
    TrackingConfig,
    TrainConfig,
    load_config,
)
from conftest import BAYESIAN_CONFIG_DIR, CONFIG_DIR

EXPERIMENT_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
BAYESIAN_CONFIGS = sorted(BAYESIAN_CONFIG_DIR.glob("*.yaml"))


def _ids(paths: list[Path]) -> list[str]:
    return [path.stem for path in paths]


# --------------------------------------------------------------------------- #
# Every shipped configuration file loads
# --------------------------------------------------------------------------- #
def test_configs_are_shipped() -> None:
    assert _ids(EXPERIMENT_CONFIGS) == ["ausgrid", "lorentz", "pendulum"]
    assert _ids(BAYESIAN_CONFIGS) == ["ausgrid", "lorentz", "pendulum"]


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=_ids(EXPERIMENT_CONFIGS))
def test_load_shipped_experiment_config(path: Path) -> None:
    config = load_config(path)

    assert isinstance(config, ExperimentConfig)
    assert config.name == path.stem
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainConfig)
    assert isinstance(config.inference, InferConfig)
    assert isinstance(config.tracking, TrackingConfig)
    # No --bayesian file was given, so the reSGLD section stays empty.
    assert config.bayesian is None

    assert config.model.architecture in ARCHITECTURE_CHOICES
    assert config.data.system == path.stem
    assert config.training.datafile
    assert config.inference.datafile
    assert config.training.epochs > 0
    assert config.tracking.uri == "mlruns"


@pytest.mark.parametrize("path", BAYESIAN_CONFIGS, ids=_ids(BAYESIAN_CONFIGS))
def test_load_shipped_bayesian_config(path: Path) -> None:
    config = load_config(CONFIG_DIR / path.name, bayesian_path=path)

    assert isinstance(config.bayesian, BayesianConfig)
    bayesian = config.bayesian
    assert bayesian.n_ensemble == 360
    assert bayesian.level == pytest.approx(15.0)
    assert bayesian.sigma == pytest.approx(2 * 15.0**2)
    # The explore chain is the hot one.
    assert bayesian.explore.tau > bayesian.exploit.tau
    assert bayesian.tau_delta == pytest.approx(
        1 / bayesian.exploit.tau - 1 / bayesian.explore.tau
    )
    # Derived quantities of the Langevin chains.
    assert bayesian.exploit.beta == pytest.approx(0.5 * 0.1 * 1e-4)
    assert bayesian.exploit.scale > 0.0
    # burn_in = epochs - (n_ensemble + 1)
    assert bayesian.burn_in(400) == 400 - 361


def test_bayesian_file_is_merged_without_its_top_level_key(
    lorentz_config_path: Path, tmp_path: Path
) -> None:
    """A bayesian file may or may not wrap its keys in a ``bayesian:`` block."""
    unwrapped = tmp_path / "bayesian_flat.yaml"
    unwrapped.write_text("n_ensemble: 5\nlevel: 3.0\n", encoding="utf-8")

    config = load_config(lorentz_config_path, bayesian_path=unwrapped)
    assert config.bayesian is not None
    assert config.bayesian.n_ensemble == 5
    assert config.bayesian.sigma == pytest.approx(18.0)


def test_train_without_bayesian_file_keeps_none(lorentz_config_path: Path) -> None:
    assert load_config(lorentz_config_path, bayesian_path=None).bayesian is None


def test_empty_bayesian_file_defines_no_bayesian_settings(
    lorentz_config_path: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "empty_bayesian.yaml"
    empty.write_text("# nothing here\n", encoding="utf-8")
    assert load_config(lorentz_config_path, bayesian_path=empty).bayesian is None


# --------------------------------------------------------------------------- #
# Dotted overrides
# --------------------------------------------------------------------------- #
def test_override_scalar_types(lorentz_config_path: Path) -> None:
    config = load_config(
        lorentz_config_path,
        [
            "training.epochs=3",
            "training.learning_rate=0.05",
            "training.use_scheduler=false",
            "training.t_max=null",
            "training.registered_model_name=tiny",
            "data.t_max=2.0",
            "data.n_sample=7",
        ],
    )
    assert config.training.epochs == 3
    assert isinstance(config.training.epochs, int)
    assert config.training.learning_rate == pytest.approx(0.05)
    assert config.training.use_scheduler is False
    assert config.training.t_max is None
    assert config.training.registered_model_name == "tiny"
    assert config.data.t_max == pytest.approx(2.0)
    assert config.data.n_sample == 7


def test_override_list_values(ausgrid_config_path: Path) -> None:
    config = load_config(
        ausgrid_config_path,
        [
            "data.ausgrid.csv_paths=[a.csv, b.csv]",
            "data.ausgrid.customer_id=[1, 2, 3]",
            "inference.plot_idxs=[0, 2]",
        ],
    )
    assert config.data.ausgrid.csv_paths == ["a.csv", "b.csv"]
    assert config.data.ausgrid.customer_id == [1, 2, 3]
    assert config.inference.plot_idxs == [0, 2]


def test_override_yaml_typed_date(ausgrid_config_path: Path) -> None:
    """An unquoted YAML date is parsed as ``datetime.date`` and accepted."""
    config = load_config(ausgrid_config_path, ["data.ausgrid.end_date=2010-08-10"])
    assert config.data.ausgrid.end_date == datetime.date(2010, 8, 10)


def test_override_creates_missing_section(lorentz_config_path: Path) -> None:
    config = load_config(lorentz_config_path, ["bayesian.n_ensemble=4"])
    assert config.bayesian is not None
    assert config.bayesian.n_ensemble == 4


def test_override_of_nested_dataclass_keeps_siblings(
    lorentz_config_path: Path,
) -> None:
    config = load_config(lorentz_config_path, ["model.branch_memory.lstm_size=7"])
    assert config.model.branch_memory.lstm_size == 7
    assert config.model.branch_memory.width == 200
    assert config.model.branch_memory.lstm_layer_num == 2


# --------------------------------------------------------------------------- #
# Error messages
# --------------------------------------------------------------------------- #
def test_missing_file_message(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError) as excinfo:
        load_config(missing)
    assert "configuration file not found" in str(excinfo.value)
    assert "does_not_exist.yaml" in str(excinfo.value)


def test_wrong_type_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["training.epochs=not-a-number"])
    message = str(excinfo.value)
    assert message == "training.epochs: expected an integer, got 'not-a-number'"


def test_wrong_bool_type_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["training.save_model=yeah"])
    assert (
        str(excinfo.value) == "training.save_model: expected true or false, got 'yeah'"
    )


def test_wrong_list_type_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["data.x_init_pts=3"])
    assert str(excinfo.value) == "data.x_init_pts: expected a list, got 3"


def test_null_for_non_optional_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["training.epochs=null"])
    assert str(excinfo.value) == "training.epochs: must not be null"


def test_unknown_key_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["training.epoch_count=5"])
    message = str(excinfo.value)
    assert message.startswith("training: unknown key(s) ['epoch_count']")
    assert "allowed keys are" in message
    assert "'epochs'" in message


def test_unknown_top_level_key_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["optimiser=adam"])
    assert str(excinfo.value).startswith("<root>: unknown key(s) ['optimiser']")


def test_removed_architecture_message(lorentz_config_path: Path) -> None:
    """``LSTM_MIONet_Static`` was dropped during the migration."""
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["model.architecture=LSTM_MIONet_Static"])
    message = str(excinfo.value)
    assert "model.architecture must be one of" in message
    assert "'LSTM_MIONet_Static'" in message
    for choice in ARCHITECTURE_CHOICES:
        assert choice in message


def test_malformed_override_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["training.epochs"])
    assert "expected the form KEY=VALUE" in str(excinfo.value)


def test_override_into_non_section_message(lorentz_config_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(lorentz_config_path, ["training.epochs.value=3"])
    assert "'epochs' is not a section" in str(excinfo.value)


def test_section_is_not_a_mapping_message(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("training: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert str(excinfo.value) == "training: expected a mapping, got int"


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "expected a YAML mapping at the top level" in str(excinfo.value)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("training.monitor_metric=accuracy", "monitor_metric must be"),
        ("training.loss_function=huber", "loss_function must be"),
        ("training.scale_mode=zscore", "scale_mode must be one of"),
        ("training.epochs=0", "epochs must be positive"),
        ("training.datafile=''", "training.datafile is required"),
    ],
)
def test_train_validation_messages(
    lorentz_config_path: Path, override: str, fragment: str
) -> None:
    config = load_config(lorentz_config_path, [override])
    with pytest.raises(ConfigError) as excinfo:
        config.training.validate()
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("data.system=duffing", "data.system must be one of"),
        ("data.control=random", "data.control must be one of"),
        ("data.x_init_pts=[]", "data.x_init_pts is required"),
        ("data.step_size=-1.0", "data.step_size must be positive"),
        ("data.n_sample=0", "data.n_sample must be positive"),
        ("data.x_init_pts=[[1.0, 2.0, 3.0]]", "[low, high] pair"),
    ],
)
def test_data_validation_messages(
    lorentz_config_path: Path, override: str, fragment: str
) -> None:
    config = load_config(lorentz_config_path, [override])
    with pytest.raises(ConfigError) as excinfo:
        config.data.validate()
    assert fragment in str(excinfo.value)


def test_ausgrid_requires_csv_paths(ausgrid_config_path: Path) -> None:
    config = load_config(ausgrid_config_path, ["data.ausgrid.csv_paths=[]"])
    with pytest.raises(ConfigError) as excinfo:
        config.data.validate()
    assert "data.ausgrid.csv_paths is required" in str(excinfo.value)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("bayesian.n_ensemble=0", "n_ensemble must be positive"),
        ("bayesian.exploit.tau=-1.0", "exploit.tau must be positive"),
        ("bayesian.explore.tau=1.0e-7", "must differ"),
        ("bayesian.exploit.alpha=0.0", "must exceed"),
    ],
)
def test_bayesian_validation_messages(
    lorentz_config_path: Path,
    bayesian_lorentz_config_path: Path,
    override: str,
    fragment: str,
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(
            lorentz_config_path,
            [override],
            bayesian_path=bayesian_lorentz_config_path,
        )
    assert fragment in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Flattening
# --------------------------------------------------------------------------- #
def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        node = nested
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def test_to_flat_dict_keys_and_values(lorentz_config_path: Path) -> None:
    flat = load_config(lorentz_config_path).to_flat_dict()

    assert flat["name"] == "lorentz"
    assert flat["model.architecture"] == "LSTM_MIONet"
    assert flat["model.branch_memory.lstm_size"] == 10
    assert flat["training.epochs"] == 1000
    assert flat["data.x_init_pts"] == [[-17.0, 20.0], [-23.0, 28.0], [0.0, 50.0]]
    assert flat["tracking.uri"] == "mlruns"
    # Nothing nested survives the flattening.
    for key, value in flat.items():
        assert "." in key or key in {"name", "bayesian"}
        assert not hasattr(value, "__dataclass_fields__")


def test_to_flat_dict_round_trip(
    lorentz_config_path: Path, bayesian_lorentz_config_path: Path, tmp_path: Path
) -> None:
    """flatten -> YAML -> load_config reproduces the same configuration."""
    original = load_config(
        lorentz_config_path, bayesian_path=bayesian_lorentz_config_path
    )
    flat = original.to_flat_dict()
    assert flat["bayesian.n_ensemble"] == 360

    round_trip_path = tmp_path / "round_trip.yaml"
    round_trip_path.write_text(
        yaml.safe_dump(_unflatten(flat), sort_keys=True), encoding="utf-8"
    )
    reloaded = load_config(round_trip_path)

    assert reloaded.to_flat_dict() == flat
    assert reloaded == original


def test_to_flat_dict_accepts_a_prefix(lorentz_config_path: Path) -> None:
    flat = load_config(lorentz_config_path).to_flat_dict(prefix="cfg.")
    assert flat["cfg.training.epochs"] == 1000
    assert all(key.startswith("cfg.") for key in flat)
