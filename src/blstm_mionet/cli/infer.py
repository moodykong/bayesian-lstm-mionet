"""``blstm-mionet infer``: evaluate a single trained model."""

from __future__ import annotations

import argparse

from blstm_mionet.cli import add_config_arguments, apply_optional, device_spec
from blstm_mionet.config import load_config
from blstm_mionet.data.datasets import prepare_torch_dataset, split_dataset
from blstm_mionet.data.generate import load_dataset
from blstm_mionet.evaluation.evaluate import evaluate_recursive, evaluate_single_step
from blstm_mionet.models import dataset_preparer_for
from blstm_mionet.training import tracking
from blstm_mionet.utils.device import resolve_device
from blstm_mionet.utils.seed import set_seed

HELP = "evaluate a trained model on a dataset (single step or recursive rollout)"


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_config_arguments(parser)
    parser.add_argument(
        "--data",
        metavar="PATH",
        default=None,
        help="test dataset .npy file (overrides inference.datafile).",
    )
    parser.add_argument(
        "--model",
        metavar="URI",
        default=None,
        help=(
            "MLflow model URI, e.g. models:/lorentz/latest or "
            "runs:/<run id>/model (overrides inference.model)."
        ),
    )
    parser.add_argument(
        "--device",
        type=device_spec,
        default=None,
        help="GPU index or 'cpu' (overrides inference.device).",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="roll the model out autoregressively (overrides inference.recursive).",
    )
    parser.add_argument(
        "--teacher-forcing-prob",
        type=float,
        default=None,
        help=(
            "probability of feeding the true state during a recursive rollout "
            "(overrides inference.teacher_forcing_prob)."
        ),
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="do not write comparison figures."
    )
    parser.add_argument(
        "--figure-dir",
        metavar="PATH",
        default=None,
        help="directory for the figures (overrides inference.figure_dir).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides)
    inference = config.inference
    apply_optional(inference, "datafile", args.data)
    apply_optional(inference, "model", args.model)
    apply_optional(inference, "device", args.device)
    apply_optional(inference, "recursive", args.recursive)
    apply_optional(inference, "teacher_forcing_prob", args.teacher_forcing_prob)
    apply_optional(inference, "figure_dir", args.figure_dir)
    if args.no_plot:
        inference.plot_trajs = False
    inference.validate()
    if not inference.model:
        raise ValueError(
            "inference.model is required; pass --model runs:/<run id>/model or "
            "set it in the configuration file"
        )

    set_seed()
    device = resolve_device(inference.device, verbose=inference.verbose)

    raw = load_dataset(inference.datafile)
    _, test_split = split_dataset(raw, test_size=1.0, verbose=inference.verbose)

    preparer = dataset_preparer_for(config.model.architecture)
    test_dataset, _, _ = prepare_torch_dataset(
        test_split,
        preparer,
        state_component=inference.state_component,
        search_len=inference.search_len,
        search_num=inference.search_num,
        search_random=inference.search_random,
        offset=inference.offset,
        t_max=inference.t_max,
        scale_mode=inference.scale_mode,
        device=device,
        verbose=inference.verbose,
    )

    tracking.configure_tracking(config.tracking.uri)
    print(f"Loading model from {inference.model}")
    model = tracking.load_model(inference.model, device)
    if inference.verbose:
        print(model)

    if inference.recursive:
        evaluate_recursive(inference, model, test_dataset)
    else:
        evaluate_single_step(inference, model, test_dataset)
    return 0
