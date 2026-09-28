"""Command line interface of ``blstm-mionet``."""

from __future__ import annotations

import argparse
import os

# Command-line runs are headless; pick a non-interactive backend before any
# sub-command imports pyplot.  Notebooks and scripts that import the package
# directly keep whatever backend they have chosen.
os.environ.setdefault("MPLBACKEND", "Agg")

from blstm_mionet.config import ExperimentConfig, InferConfig, TrainConfig  # noqa: E402
from blstm_mionet.utils.device import DeviceSpec, parse_device_spec

DESCRIPTION = (
    "B-LSTM-MIONet: Bayesian LSTM-based neural operators for length-variant "
    "multiple input functions."
)

CONFIG_EPILOG = (
    "Values from the YAML file can be overridden on the command line with "
    "repeated --set options, e.g. --set training.epochs=10 "
    "--set data.ausgrid.csv_paths='[a.csv, b.csv]'. Paths inside a "
    "configuration file are relative to the current working directory."
)


def device_spec(value: str) -> DeviceSpec:
    """``argparse`` type for ``--device`` (a GPU index or "cpu")."""
    try:
        return parse_device_spec(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``--config`` / ``--set`` pair shared by every sub-command."""
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="experiment YAML file (see configs/).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override a configuration entry using its dotted path; may be "
            "repeated. The value is parsed as YAML."
        ),
    )
    return parser


def infer_config_from_training(config: ExperimentConfig) -> InferConfig:
    """Build the evaluation settings used for the post-training test split."""
    training: TrainConfig = config.training
    return InferConfig(
        datafile=training.datafile,
        state_component=training.state_component,
        search_len=training.search_len,
        search_num=training.search_num,
        search_random=training.search_random,
        offset=training.offset,
        t_max=training.t_max,
        scale_mode=training.scale_mode,
        batch_size=training.batch_size,
        plot_trajs=training.plot_trajs,
        plot_idxs=training.plot_idxs,
        figure_dir=training.figure_dir,
        device=training.device,
        verbose=training.verbose,
    )


def apply_optional(target: object, attribute: str, value: object | None) -> None:
    """Assign ``value`` to ``target.attribute`` unless it is ``None``."""
    if value is not None:
        setattr(target, attribute, value)
