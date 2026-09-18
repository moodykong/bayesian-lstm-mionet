"""``blstm-mionet infer-bayesian``: evaluate the reSGLD posterior ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

from blstm_mionet.cli import add_config_arguments, apply_optional, device_spec
from blstm_mionet.config import load_config
from blstm_mionet.data.datasets import prepare_torch_dataset, split_dataset
from blstm_mionet.data.generate import load_dataset
from blstm_mionet.evaluation.evaluate import evaluate_ensemble
from blstm_mionet.models import build_model, dataset_preparer_for
from blstm_mionet.training import tracking
from blstm_mionet.utils.device import resolve_device
from blstm_mionet.utils.seed import set_seed

HELP = "evaluate the Bayesian (reSGLD) ensemble logged by a training run"


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_config_arguments(parser)
    parser.add_argument(
        "--data",
        metavar="PATH",
        default=None,
        help="test dataset .npy file (overrides inference.datafile).",
    )
    parser.add_argument(
        "--run",
        metavar="URI",
        default=None,
        help=(
            "MLflow run holding the ensemble: runs:/<run id> or a bare run id "
            "(overrides inference.run)."
        ),
    )
    parser.add_argument(
        "--device",
        type=device_spec,
        default=None,
        help="GPU index, 'parallel' or 'cpu' (overrides inference.device).",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="do not write the UQ figures."
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
    apply_optional(inference, "run", args.run)
    apply_optional(inference, "device", args.device)
    apply_optional(inference, "figure_dir", args.figure_dir)
    if args.no_plot:
        inference.plot_trajs = False
    inference.validate()
    if not inference.run:
        raise ValueError(
            "inference.run is required; pass --run runs:/<run id> or set it in "
            "the configuration file"
        )

    set_seed()
    device = resolve_device(inference.device, verbose=inference.verbose)

    raw = load_dataset(inference.datafile)
    _, test_split = split_dataset(raw, test_size=1.0, verbose=inference.verbose)

    preparer = dataset_preparer_for(config.model.architecture)
    test_dataset, _, state_feature_num = prepare_torch_dataset(
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
    run_id = tracking.run_id_from_uri(inference.run)

    ## Recover the model class from the run, falling back to the configuration
    model_uri = f"runs:/{run_id}/{tracking.MODEL_ARTIFACT_NAME}"
    try:
        model = tracking.load_model(model_uri, device)
        print(f"Loaded the ensemble model class from {model_uri}")
    except Exception as exc:  # noqa: BLE001 - the config is a valid fallback
        print(f"Could not load {model_uri} ({exc}); rebuilding from the configuration.")
        model = build_model(config.model, state_feature_num).to(device)

    ## Download the ensemble members
    local_dir = tracking.download_artifacts(run_id, tracking.ENSEMBLE_ARTIFACT_PATH)
    member_paths = sorted(Path(local_dir).glob("member_*.pt"))
    if inference.n_ensemble is not None:
        member_paths = member_paths[: inference.n_ensemble]
    print(f"Found {len(member_paths)} ensemble members in {local_dir}")

    evaluate_ensemble(inference, model, test_dataset, member_paths, device)
    return 0
