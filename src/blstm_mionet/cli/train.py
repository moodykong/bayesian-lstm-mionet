"""``blstm-mionet train``: Adam training, or replica exchange SGLD with --bayesian."""

from __future__ import annotations

import argparse
import copy

from blstm_mionet.cli import (
    add_config_arguments,
    apply_optional,
    device_spec,
    infer_config_from_training,
)
from blstm_mionet.config import load_config
from blstm_mionet.data.datasets import prepare_torch_dataset, split_dataset
from blstm_mionet.data.generate import load_dataset
from blstm_mionet.evaluation.evaluate import evaluate_single_step
from blstm_mionet.models import build_model, dataset_preparer_for
from blstm_mionet.training import tracking
from blstm_mionet.training.resgld import train_resgld
from blstm_mionet.training.trainer import train_adam
from blstm_mionet.utils.device import resolve_device
from blstm_mionet.utils.seed import set_seed

HELP = "train an operator with Adam, or with replica exchange SGLD (--bayesian)"


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_config_arguments(parser)
    parser.add_argument(
        "--bayesian",
        metavar="PATH",
        default=None,
        help=(
            "replica exchange SGLD configuration (configs/bayesian/*.yaml); "
            "training switches from Adam to reSGLD and logs a posterior ensemble."
        ),
    )
    parser.add_argument(
        "--data",
        metavar="PATH",
        default=None,
        help="training dataset .npy file (overrides training.datafile).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="overrides training.epochs."
    )
    parser.add_argument(
        "--device",
        type=device_spec,
        default=None,
        help="GPU index, 'parallel' or 'cpu' (overrides training.device).",
    )
    parser.add_argument("--run-name", default=None, help="overrides training.run_name.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="do not write the post-training comparison figures.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides, bayesian_path=args.bayesian)
    training = config.training
    apply_optional(training, "datafile", args.data)
    apply_optional(training, "epochs", args.epochs)
    apply_optional(training, "device", args.device)
    apply_optional(training, "run_name", args.run_name)
    if args.no_plot:
        training.plot_trajs = False
    training.validate()
    if args.bayesian is not None and config.bayesian is None:
        raise ValueError(f"{args.bayesian} does not define any bayesian settings")

    set_seed()
    device = resolve_device(training.device, verbose=training.verbose)

    ## Collect the dataset and split off the held out trajectories
    raw = load_dataset(training.datafile)
    train_split, test_split = split_dataset(
        raw, test_size=training.holdout_size, verbose=training.verbose
    )

    preparer = dataset_preparer_for(config.model.architecture)
    prepare_kwargs = dict(
        state_component=training.state_component,
        search_len=training.search_len,
        search_num=training.search_num,
        search_random=training.search_random,
        offset=training.offset,
        t_max=training.t_max,
        scale_mode=training.scale_mode,
        device=device,
        verbose=training.verbose,
    )
    train_dataset, _, state_feature_num = prepare_torch_dataset(
        train_split, preparer, **prepare_kwargs
    )
    test_dataset = None
    if test_split[1] is not None:
        test_dataset, _, _ = prepare_torch_dataset(
            test_split, preparer, **prepare_kwargs
        )

    ## Build the model
    model = build_model(config.model, state_feature_num)
    if training.verbose:
        print(model)

    tracking.configure_tracking(config.tracking.uri)
    with tracking.start_run(training.experiment_name, training.run_name) as run_context:
        run_id = run_context.info.run_id
        tracking.log_config(config)
        tracking.log_source_snapshot()

        if config.bayesian is not None:
            history = train_resgld(
                config=training,
                bayesian=config.bayesian,
                model_exploit=model,
                model_explore=copy.deepcopy(model),
                dataset=train_dataset,
                device=device,
            )
            print(
                f"Logged {history['n_ensemble']} ensemble members to "
                f"runs:/{run_id}/{tracking.ENSEMBLE_ARTIFACT_PATH}"
            )
        else:
            history = train_adam(
                config=training, model=model, dataset=train_dataset, device=device
            )

        if test_dataset is not None:
            evaluate_single_step(
                infer_config_from_training(config), model, test_dataset
            )

    print(f"MLflow run id: {run_id}")
    print(f"MLflow run uri: runs:/{run_id}")
    if history.get("model_uri"):
        print(f"Model uri: {history['model_uri']}")
        print(f"Model uri (run scoped): runs:/{run_id}/{tracking.MODEL_ARTIFACT_NAME}")
    return 0
