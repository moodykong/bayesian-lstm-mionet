"""``blstm-mionet generate``: build a training/test dataset."""

from __future__ import annotations

import argparse

from blstm_mionet.cli import add_config_arguments, apply_optional
from blstm_mionet.config import load_config
from blstm_mionet.data.ausgrid import select_ausgrid_data
from blstm_mionet.data.generate import generate_trajectories, save_dataset
from blstm_mionet.utils.seed import set_seed

HELP = "generate a dataset (ODE trajectories, or a selection of Ausgrid profiles)"


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_config_arguments(parser)
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="where to write the .npy dataset (overrides data.output).",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=None,
        help="number of trajectories to generate (overrides data.n_sample).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides)
    data = config.data
    apply_optional(data, "output", args.output)
    apply_optional(data, "n_sample", args.n_sample)
    data.validate()

    set_seed(data.seed)

    if data.system == "ausgrid":
        dataset = select_ausgrid_data(
            csv_paths=data.ausgrid.csv_paths,
            cust_id=data.ausgrid.customer_id,
            start_date=data.ausgrid.start_date,
            end_date=data.ausgrid.end_date,
            category=data.ausgrid.category,
            delta_t_idxs=data.ausgrid.delta_t_idxs,
            column_start=data.ausgrid.column_start,
            column_end=data.ausgrid.column_end,
            min_nonzero_fraction=data.ausgrid.min_nonzero_fraction,
            sample_interval_hours=data.ausgrid.sample_interval_hours,
            verbose=data.verbose,
        )
    else:
        dataset = generate_trajectories(data)

    path = save_dataset(dataset, data.output)
    print(f"Data saved in {path}.")
    return 0
