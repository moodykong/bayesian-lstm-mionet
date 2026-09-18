# shellcheck shell=bash
#
# scripts/_common.sh -- helpers shared by the reproduction scripts.
#
# This file is *sourced*, never executed.  Every script in this directory
# starts with
#
#     set -euo pipefail
#     cd "$(dirname "${BASH_SOURCE[0]}")/.."
#     REPO_ROOT="$PWD"
#     source scripts/_common.sh
#
# so that relative paths in the YAML configuration files (data/, mlruns/,
# figures/) resolve against the repository root, or against --workdir when one
# is given.
#
# Conventions provided here:
#   * ``blstm`` runs the command line interface, echoing the full command first.
#   * ``common_parse_args`` implements the shared --quick / --device / --workdir
#     / --help options and collects everything else into PASSTHROUGH.
#   * ``ensure_dataset`` generates a dataset only when it does not exist yet.
#   * ``train_and_capture_run`` runs ``blstm-mionet train`` and recovers
#     ``runs:/<id>`` from its output, so inference never has to rely on
#     ``models:/<name>/latest``.

## --------------------------------------------------------------------------
## Locating the command line interface
## --------------------------------------------------------------------------
## BLSTM_MIONET_CMD can override the invocation, e.g.
##   BLSTM_MIONET_CMD="uv run --no-sync blstm-mionet" scripts/smoke_test.sh
if [[ -n "${BLSTM_MIONET_CMD:-}" ]]; then
    read -r -a BLSTM <<<"${BLSTM_MIONET_CMD}"
elif command -v blstm-mionet >/dev/null 2>&1; then
    BLSTM=(blstm-mionet)
elif command -v uv >/dev/null 2>&1; then
    # The reproduce scripts change into their --workdir before running the CLI,
    # so pin uv to the repository's project (REPO_ROOT is set by the caller;
    # fall back to the directory above this file).
    _blstm_project="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    BLSTM=(uv run --project "$_blstm_project" blstm-mionet)
    unset _blstm_project
else
    echo "error: neither 'blstm-mionet' nor 'uv' is on PATH." >&2
    echo "       install the environment with 'uv sync --extra cpu' (or" >&2
    echo "       --extra cu126 / --extra cu130) and re-run this script." >&2
    exit 127
fi

## MLflow 3 prints a multi-line hint about a tracing skill on every command.
export MLFLOW_DISABLE_AGENT_HINT="${MLFLOW_DISABLE_AGENT_HINT:-1}"

## The interpreter that runs the package.  Two helpers below need it: reading
## data.ausgrid.csv_paths out of a YAML file and writing the synthetic CSV of
## scripts/reproduce_ausgrid.sh --quick.
PYTHON_CMD=()
if [[ "${BLSTM[0]}" == "uv" ]]; then
    PYTHON_CMD=("${BLSTM[@]:0:$((${#BLSTM[@]} - 1))}" python)
else
    _blstm_script="$(command -v "${BLSTM[0]}" 2>/dev/null || true)"
    if [[ -n "$_blstm_script" ]]; then
        _blstm_shebang="$(head -n 1 "$_blstm_script")"
        if [[ "$_blstm_shebang" == '#!'* ]]; then
            _blstm_interp="${_blstm_shebang#\#!}"
            if [[ -x "${_blstm_interp%% *}" ]]; then
                # shellcheck disable=SC2206  # the shebang may carry arguments
                PYTHON_CMD=($_blstm_interp)
            fi
        fi
    fi
    [[ ${#PYTHON_CMD[@]} -eq 0 ]] && PYTHON_CMD=(python3)
    unset _blstm_script _blstm_shebang _blstm_interp
fi

## --------------------------------------------------------------------------
## Printing
## --------------------------------------------------------------------------
STEP_INDEX=0
SCRIPT_START=$SECONDS

hr() { printf '%s\n' "------------------------------------------------------------------------"; }

## Wall clock since the script started, as mm:ss.
elapsed() {
    local total=$((SECONDS - SCRIPT_START))
    printf '%02d:%02d' $((total / 60)) $((total % 60))
}

step() {
    STEP_INDEX=$((STEP_INDEX + 1))
    printf '\n'
    hr
    printf '[%d/%s] (t+%s) %s\n' "$STEP_INDEX" "${STEP_TOTAL:-?}" "$(elapsed)" "$*"
    hr
}

note() { printf '  %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

## Echo a command and run it.
blstm() {
    printf '\n$ %s %s\n\n' "${BLSTM[*]}" "$*"
    "${BLSTM[@]}" "$@"
}

## --------------------------------------------------------------------------
## Argument parsing
## --------------------------------------------------------------------------
## Sets QUICK (0/1), DEVICE, WORKDIR and the PASSTHROUGH array.  The calling
## script must define a ``usage`` function, which -h / --help then prints.
QUICK=0
DEVICE=""
WORKDIR=""
PASSTHROUGH=()

common_parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --quick)
            QUICK=1
            shift
            ;;
        --device)
            [[ $# -ge 2 ]] || die "--device needs a value (a GPU index, 'parallel' or 'cpu')"
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#--device=}"
            shift
            ;;
        --workdir)
            [[ $# -ge 2 ]] || die "--workdir needs a directory"
            WORKDIR="$2"
            shift 2
            ;;
        --workdir=*)
            WORKDIR="${1#--workdir=}"
            shift
            ;;
        --)
            shift
            PASSTHROUGH+=("$@")
            break
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
        esac
    done

    ## The GPU index of the paper runs; --quick is a CPU smoke test.
    if [[ -z "$DEVICE" ]]; then
        if [[ $QUICK -eq 1 ]]; then DEVICE="cpu"; else DEVICE="0"; fi
    fi
}

## Move into --workdir (created on demand) so that data/, mlruns/ and figures/
## are written there instead of into the repository.
enter_workdir() {
    if [[ -n "$WORKDIR" ]]; then
        mkdir -p "$WORKDIR"
        cd "$WORKDIR" || die "cannot enter $WORKDIR"
    fi
    printf 'Working directory : %s\n' "$PWD"
    printf 'Repository        : %s\n' "$REPO_ROOT"
    printf 'CLI               : %s\n' "${BLSTM[*]}"
    printf 'Device            : %s\n' "$DEVICE"
    printf 'Mode              : %s\n' "$([[ $QUICK -eq 1 ]] && echo 'quick (tiny CPU smoke run)' || echo 'full (paper scale)')"
    if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
        printf 'Extra train args  : %s\n' "${PASSTHROUGH[*]}"
    fi
}

## --------------------------------------------------------------------------
## Datasets
## --------------------------------------------------------------------------
## ensure_dataset <path> <generate arguments...>
## Generates <path> unless it already exists.
ensure_dataset() {
    local path="$1"
    shift
    if [[ -f "$path" ]]; then
        note "Reusing the existing dataset $path (delete it to regenerate)."
        return 0
    fi
    blstm generate "$@" --output "$path"
}

## --------------------------------------------------------------------------
## Training
## --------------------------------------------------------------------------
## train_and_capture_run <logfile> <train arguments...>
## Runs ``blstm-mionet train`` and exports RUN_URI / RUN_ID / MODEL_URI from the
## "MLflow run uri:" line it prints.
RUN_URI=""
RUN_ID=""
MODEL_URI=""

train_and_capture_run() {
    local log="$1"
    shift
    mkdir -p "$(dirname "$log")"
    printf '\n$ %s train %s\n\n' "${BLSTM[*]}" "$*"
    "${BLSTM[@]}" train "$@" 2>&1 | tee "$log"

    RUN_URI="$(sed -n 's/^MLflow run uri: //p' "$log" | tail -n 1)"
    [[ -n "$RUN_URI" ]] || die "could not find the 'MLflow run uri:' line in $log"
    RUN_ID="${RUN_URI#runs:/}"
    MODEL_URI="runs:/${RUN_ID}/model"
    printf '\n'
    note "Captured $RUN_URI"
    note "Inference will use --model $MODEL_URI"
}

## --------------------------------------------------------------------------
## Reading values back out of the YAML configuration files
## --------------------------------------------------------------------------
## Report training.registered_model_name of a configuration file, if any.
registered_model_name() {
    "${PYTHON_CMD[@]}" - "$1" <<'YAML_READER'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
print((config.get("training") or {}).get("registered_model_name") or "")
YAML_READER
}

## Write a synthetic Ausgrid-shaped CSV (only used by --quick).
write_synthetic_csv() {
    local path="$1"
    mkdir -p "$(dirname "$path")"
    "${PYTHON_CMD[@]}" - "$path" <<'SYNTHETIC_CSV'
"""Write a small CSV with the layout of the Ausgrid solar-home files.

Row 0 is the title row skipped by ``pd.read_csv(..., header=1)``; row 1 is the
header.  The five leading columns are followed by the 48 half-hour readings and
a "Row Quality" column, so ``iloc[:, 18:39]`` selects the daylight readings the
configuration expects.  Values are a smooth generation bump plus a little
noise; they are NOT real measurements.
"""

import csv
import math
import random
import sys
from datetime import date, timedelta

PATH = sys.argv[1]
CUSTOMERS = range(1, 7)
START = date(2010, 7, 1)
DAYS = 41
HALF_HOURS = [
    f"{(i + 1) // 2 % 24}:{'30' if (i + 1) % 2 else '00'}" for i in range(48)
]

random.seed(20231116)


def profile(day_index: int) -> list[float]:
    """Gross generation over one day: zero at night, a bump around noon."""
    amplitude = 0.9 + 0.2 * math.sin(day_index / 7.0)
    readings = []
    for i in range(48):
        hour = (i + 1) * 0.5
        if 5.0 <= hour <= 19.0:
            bump = math.sin(math.pi * (hour - 5.0) / 14.0) ** 2
            value = amplitude * bump + random.uniform(-0.03, 0.03)
            readings.append(round(max(value, 1e-3), 4))
        else:
            readings.append(0.0)
    return readings


with open(PATH, "w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["SYNTHETIC test data with the layout of the Ausgrid solar home files"]
    )
    writer.writerow(
        ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"]
        + HALF_HOURS
        + ["Row Quality"]
    )
    for customer in CUSTOMERS:
        for day in range(DAYS):
            stamp = START + timedelta(days=day)
            # Ausgrid writes the day first ("1/07/2010").
            stamp_text = f"{stamp.day}/{stamp.month:02d}/{stamp.year}"
            writer.writerow(
                [customer, 2.0, 2076, "GG", stamp_text] + profile(day + customer) + [""]
            )
            # A second category, so the "GG" filter has something to drop.
            writer.writerow(
                [customer, 2.0, 2076, "GC", stamp_text] + [0.1] * 48 + [""]
            )
print(f"Synthetic Ausgrid CSV written to {PATH}")
SYNTHETIC_CSV
}

## Print the entries of data.ausgrid.csv_paths, one per line.
ausgrid_csv_paths() {
    "${PYTHON_CMD[@]}" - "$1" <<'YAML_READER'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
ausgrid = (config.get("data") or {}).get("ausgrid") or {}
for path in ausgrid.get("csv_paths") or []:
    print(path)
YAML_READER
}
