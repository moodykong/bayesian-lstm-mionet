"""End-to-end exercise of the ``blstm-mionet`` command line interface.

Every command runs in-process through ``blstm_mionet.cli.main.main`` with the
working directory moved into ``tmp_path``, so the relative paths inside the
shipped configuration files resolve into the temporary directory and the MLflow
store lives there too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blstm_mionet import __version__
from blstm_mionet.cli.main import main
from conftest import parse_model_uri, parse_run_id

LORENTZ_DATA = "data/lorentz_smoke.npy"
LORENTZ_ONE = "data/lorentz_one.npy"
PENDULUM_DATA = "data/pendulum_smoke.npy"
PENDULUM_ONE = "data/pendulum_one.npy"
AUSGRID_DATA = "data/ausgrid_smoke.npy"


# --------------------------------------------------------------------------- #
# --help / --version
# --------------------------------------------------------------------------- #
def test_version(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"blstm-mionet {__version__}"


def test_root_help_lists_every_command(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    printed = capsys.readouterr().out
    for command in ("generate", "train", "infer", "infer-bayesian"):
        assert command in printed
    assert "B-LSTM-MIONet" in printed
    assert "--set" in printed


@pytest.mark.parametrize(
    ("command", "fragments"),
    [
        ("generate", ("--output", "--n-sample", "--config")),
        ("train", ("--bayesian", "--epochs", "--device", "--no-plot", "--run-name")),
        (
            "infer",
            ("--model", "--recursive", "--teacher-forcing-prob", "--figure-dir"),
        ),
        ("infer-bayesian", ("--run", "--device", "--no-plot", "--data")),
    ],
)
def test_subcommand_help(command: str, fragments: tuple[str, ...], capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0
    printed = capsys.readouterr().out
    for fragment in fragments:
        assert fragment in printed
    # Every sub-command documents the --set override syntax.
    assert "training.epochs=10" in printed


def test_no_command_prints_the_help(capsys) -> None:
    assert main([]) == 1
    assert "usage: blstm-mionet" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_missing_config_file_is_reported(run_cli, capsys) -> None:
    run_cli("generate", "--config", "no_such_config.yaml", expected_code=2)
    assert "configuration file not found" in capsys.readouterr().err


def test_unknown_override_is_reported(
    run_cli, lorentz_config_path: Path, capsys
) -> None:
    run_cli(
        "generate",
        "--config",
        lorentz_config_path,
        "--set",
        "data.samples=10",
        expected_code=2,
    )
    assert "unknown key(s) ['samples']" in capsys.readouterr().err


def test_infer_without_a_model_uri_is_reported(
    run_cli, lorentz_config_path: Path, capsys
) -> None:
    run_cli(
        "infer",
        "--config",
        lorentz_config_path,
        "--set",
        "inference.model=''",
        expected_code=2,
    )
    assert "inference.model is required" in capsys.readouterr().err


def test_infer_bayesian_without_a_run_is_reported(
    run_cli, lorentz_config_path: Path, capsys
) -> None:
    run_cli("infer-bayesian", "--config", lorentz_config_path, expected_code=2)
    assert "inference.run is required" in capsys.readouterr().err


def test_missing_dataset_is_reported(
    run_cli, lorentz_config_path: Path, capsys
) -> None:
    run_cli(
        "train",
        "--config",
        lorentz_config_path,
        "--data",
        "data/not_generated.npy",
        "--device",
        "cpu",
        expected_code=2,
    )
    assert "file not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def test_generate_writes_the_dataset(
    run_cli, lorentz_config_path: Path, tmp_path: Path, capsys
) -> None:
    run_cli(
        "generate",
        "--config",
        lorentz_config_path,
        "--n-sample",
        "3",
        "--set",
        "data.t_max=0.5",
        "--output",
        LORENTZ_DATA,
    )
    assert (tmp_path / LORENTZ_DATA).is_file()
    assert "Data saved in" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Full pipelines
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:consecutive rollout points")
def test_lorentz_end_to_end(
    run_cli, lorentz_config_path: Path, tmp_path: Path, capsys
) -> None:
    """generate -> train -> infer -> recursive infer, the documented recipe."""
    run_cli(
        "generate",
        "--config",
        lorentz_config_path,
        "--n-sample",
        "20",
        "--set",
        "data.t_max=2.0",
        "--output",
        LORENTZ_DATA,
    )
    assert (tmp_path / LORENTZ_DATA).is_file()

    run_cli(
        "train",
        "--config",
        lorentz_config_path,
        "--data",
        LORENTZ_DATA,
        "--epochs",
        "2",
        "--device",
        "cpu",
        "--no-plot",
    )
    training_output = capsys.readouterr().out
    run_id = parse_run_id(training_output)
    model_uri = parse_model_uri(training_output)
    assert model_uri == f"runs:/{run_id}/model"
    assert "L2-relative error" in training_output  # the held out trajectories
    assert not (tmp_path / "figures").exists()  # --no-plot

    run_cli(
        "infer",
        "--config",
        lorentz_config_path,
        "--data",
        LORENTZ_DATA,
        "--model",
        model_uri,
        "--set",
        "inference.search_num=20",
        "--device",
        "cpu",
        "--no-plot",
    )
    assert "L2-relative error" in capsys.readouterr().out

    # The recursive rollout is defined for a single trajectory.
    run_cli(
        "generate",
        "--config",
        lorentz_config_path,
        "--n-sample",
        "1",
        "--set",
        "data.t_max=2.0",
        "--output",
        LORENTZ_ONE,
    )
    capsys.readouterr()
    run_cli(
        "infer",
        "--config",
        lorentz_config_path,
        "--data",
        LORENTZ_ONE,
        "--model",
        model_uri,
        "--set",
        "inference.search_num=20",
        "--device",
        "cpu",
        "--recursive",
        "--teacher-forcing-prob",
        "0.5",
        "--figure-dir",
        "figures/recursive",
    )
    assert "L2-relative error" in capsys.readouterr().out
    assert (
        tmp_path / "figures" / "recursive" / "infer_trajs_recursive_0.png"
    ).is_file()


@pytest.mark.slow
def test_pendulum_end_to_end(
    run_cli, pendulum_config_path: Path, tmp_path: Path, capsys
) -> None:
    """The non-autonomous benchmark: the control u travels with the dataset."""
    run_cli(
        "generate",
        "--config",
        pendulum_config_path,
        "--n-sample",
        "4",
        "--set",
        "data.t_max=1.0",
        "--output",
        PENDULUM_DATA,
    )
    assert (tmp_path / PENDULUM_DATA).is_file()

    run_cli(
        "train",
        "--config",
        pendulum_config_path,
        "--data",
        PENDULUM_DATA,
        "--epochs",
        "2",
        "--device",
        "cpu",
        "--no-plot",
        "--set",
        "training.search_num=4",
    )
    model_uri = parse_model_uri(capsys.readouterr().out)

    run_cli(
        "infer",
        "--config",
        pendulum_config_path,
        "--data",
        PENDULUM_DATA,
        "--model",
        model_uri,
        "--set",
        "inference.search_num=10",
        "--device",
        "cpu",
        "--no-plot",
    )
    assert "L2-relative error" in capsys.readouterr().out

    run_cli(
        "generate",
        "--config",
        pendulum_config_path,
        "--n-sample",
        "1",
        "--set",
        "data.t_max=1.0",
        "--output",
        PENDULUM_ONE,
    )
    capsys.readouterr()
    run_cli(
        "infer",
        "--config",
        pendulum_config_path,
        "--data",
        PENDULUM_ONE,
        "--model",
        model_uri,
        "--set",
        "inference.search_num=10",
        "--device",
        "cpu",
        "--recursive",
        "--teacher-forcing-prob",
        "1.0",
        "--no-plot",
    )
    assert "L2-relative error" in capsys.readouterr().out


@pytest.mark.slow
def test_bayesian_end_to_end(
    run_cli,
    lorentz_config_path: Path,
    bayesian_lorentz_config_path: Path,
    tmp_path: Path,
    capsys,
) -> None:
    """train --bayesian -> infer-bayesian on the reSGLD posterior ensemble."""
    run_cli(
        "generate",
        "--config",
        lorentz_config_path,
        "--n-sample",
        "6",
        "--set",
        "data.t_max=1.0",
        "--output",
        LORENTZ_DATA,
    )

    run_cli(
        "train",
        "--config",
        lorentz_config_path,
        "--bayesian",
        bayesian_lorentz_config_path,
        "--data",
        LORENTZ_DATA,
        "--epochs",
        "8",
        "--set",
        "bayesian.n_ensemble=3",
        "--device",
        "cpu",
        "--no-plot",
    )
    training_output = capsys.readouterr().out
    run_id = parse_run_id(training_output)
    assert "Logged 3 ensemble members" in training_output

    run_cli(
        "infer-bayesian",
        "--config",
        lorentz_config_path,
        "--data",
        LORENTZ_DATA,
        "--run",
        f"runs:/{run_id}",
        "--set",
        "inference.search_num=5",
        "--device",
        "cpu",
        "--no-plot",
    )
    inference_output = capsys.readouterr().out
    assert "Found 3 ensemble members" in inference_output
    assert "PICP (95% interval)" in inference_output
    assert "Ensemble prediction: mean std of the posterior" in inference_output


@pytest.mark.slow
def test_ausgrid_end_to_end(
    run_cli, ausgrid_config_path: Path, ausgrid_csv: Path, tmp_path: Path, capsys
) -> None:
    """The licensed CSV files are replaced by the synthetic fixture."""
    csv_override = f'data.ausgrid.csv_paths=["{ausgrid_csv}"]'
    run_cli(
        "generate",
        "--config",
        ausgrid_config_path,
        "--set",
        csv_override,
        "--set",
        "data.ausgrid.customer_id=[1, 2]",
        "--set",
        "data.ausgrid.end_date=2010-07-10",
        "--output",
        AUSGRID_DATA,
    )
    assert (tmp_path / AUSGRID_DATA).is_file()
    assert "num_trajs= 20" in capsys.readouterr().out

    run_cli(
        "train",
        "--config",
        ausgrid_config_path,
        "--data",
        AUSGRID_DATA,
        "--epochs",
        "2",
        "--device",
        "cpu",
        "--no-plot",
        "--set",
        "training.search_num=2",
    )
    model_uri = parse_model_uri(capsys.readouterr().out)

    run_cli(
        "infer",
        "--config",
        ausgrid_config_path,
        "--data",
        AUSGRID_DATA,
        "--model",
        model_uri,
        "--set",
        "inference.search_num=5",
        "--device",
        "cpu",
        "--no-plot",
    )
    assert "L2-relative error" in capsys.readouterr().out
