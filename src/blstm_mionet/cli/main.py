"""Root parser of the ``blstm-mionet`` command line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from blstm_mionet import __version__
from blstm_mionet.cli import CONFIG_EPILOG, DESCRIPTION
from blstm_mionet.cli import generate as generate_cmd
from blstm_mionet.cli import infer as infer_cmd
from blstm_mionet.cli import infer_bayesian as infer_bayesian_cmd
from blstm_mionet.cli import train as train_cmd
from blstm_mionet.config import ConfigError

COMMANDS = (
    ("generate", generate_cmd),
    ("train", train_cmd),
    ("infer", infer_cmd),
    ("infer-bayesian", infer_bayesian_cmd),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the root parser with one sub-parser per sub-command."""
    parser = argparse.ArgumentParser(
        prog="blstm-mionet",
        description=DESCRIPTION,
        epilog=CONFIG_EPILOG,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, module in COMMANDS:
        subparser = subparsers.add_parser(
            name,
            help=module.HELP,
            description=module.HELP[0].upper() + module.HELP[1:] + ".",
            epilog=CONFIG_EPILOG,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(func=module.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``blstm-mionet`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
