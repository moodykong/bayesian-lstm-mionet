"""``blstm-mionet relocate-mlruns``: make a copied MLflow store usable in place."""

from __future__ import annotations

import argparse

from blstm_mionet.training.tracking import relocate_file_store

HELP = "point a copied or downloaded mlruns store at its new location"
EPILOG = None  # takes no configuration file


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "store",
        nargs="?",
        default="mlruns",
        metavar="PATH",
        help=(
            "MLflow file store to fix. MLflow records absolute paths, so a store "
            "unpacked from an archive still points at the machine it was written on."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    rewritten = relocate_file_store(args.store)
    if rewritten:
        print(
            f"Rewrote the paths in {len(rewritten)} meta.yaml files under {args.store}."
        )
    else:
        print(f"{args.store} already points at its own location; nothing to do.")
    return 0
