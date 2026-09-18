#!/usr/bin/env bash
#
# smoke_test.sh -- fast local sanity check of the whole command line interface.
#
#   scripts/smoke_test.sh [--sequential] [--device DEVICE] [--keep]
#
# Runs three --quick pipelines back to back (by default all three at once, one
# process each) inside a temporary directory that is removed on exit:
#
#   * scripts/reproduce_lorentz.sh  --quick   generate / train / infer / rollout
#   * scripts/reproduce_pendulum.sh --quick   generate / train / infer (+ OOD)
#   * scripts/reproduce_bayesian.sh lorentz --quick   reSGLD + infer-bayesian
#
# Nothing is written into the repository: every pipeline gets its own --workdir
# under the temporary directory, so data/, mlruns/ and figures/ live there.
# The script exits non-zero as soon as one of the pipelines fails and prints
# that pipeline's log.  It takes roughly a minute and a half on an idle CPU.
#
# Requires the environment of the repository:  uv sync --extra cpu
# (or --extra cu126 / --extra cu130 for CUDA).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
    cat <<'EOF'
Options:
  --sequential     run the pipelines one after another with live output
  --device DEVICE  GPU index, "parallel" or "cpu" (default: cpu)
  --keep           keep the temporary directory and print its path
  -h, --help       show this message
EOF
}

SEQUENTIAL=0
KEEP=0
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --sequential)
        SEQUENTIAL=1
        shift
        ;;
    --keep)
        KEEP=1
        shift
        ;;
    --device)
        DEVICE="$2"
        shift 2
        ;;
    --device=*)
        DEVICE="${1#--device=}"
        shift
        ;;
    *)
        usage
        printf '\nerror: unknown argument %s\n' "$1" >&2
        exit 2
        ;;
    esac
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/blstm-mionet-smoke.XXXXXXXX")"
cleanup() {
    if [[ $KEEP -eq 1 ]]; then
        printf '\nKept the temporary directory: %s\n' "$WORK_DIR"
    else
        rm -rf "$WORK_DIR"
    fi
}
trap cleanup EXIT

LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

NAMES=(lorentz pendulum bayesian)
## Referenced through namerefs below, which shellcheck cannot follow.
# shellcheck disable=SC2034
COMMAND_lorentz=("$REPO_ROOT/scripts/reproduce_lorentz.sh" --quick)
# shellcheck disable=SC2034
COMMAND_pendulum=("$REPO_ROOT/scripts/reproduce_pendulum.sh" --quick)
# shellcheck disable=SC2034
COMMAND_bayesian=("$REPO_ROOT/scripts/reproduce_bayesian.sh" lorentz --quick)

START=$SECONDS
printf 'B-LSTM-MIONet smoke test\n'
printf '  repository      : %s\n' "$REPO_ROOT"
printf '  temporary dir   : %s\n' "$WORK_DIR"
printf '  device          : %s\n' "$DEVICE"
printf '  mode            : %s\n' "$([[ $SEQUENTIAL -eq 1 ]] && echo sequential || echo 'parallel (3 processes)')"

## Three interpreters at once would each start as many BLAS/OpenMP threads as
## there are cores, which mostly makes them fight each other; give each process
## a fair share instead.
if [[ $SEQUENTIAL -eq 0 ]]; then
    CORES="$(nproc 2>/dev/null || echo 4)"
    THREADS=$((CORES / 3))
    ((THREADS < 1)) && THREADS=1
    ((THREADS > 8)) && THREADS=8
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
    printf '  threads/process : %s\n' "$OMP_NUM_THREADS"
fi

declare -A STATUS=()
declare -A DURATION=()
declare -A PID=()
declare -A PID_START=()

launch() {
    local name="$1"
    local -n command_ref="COMMAND_$name"
    "${command_ref[@]}" --device "$DEVICE" --workdir "$WORK_DIR/$name" \
        >"$LOG_DIR/$name.log" 2>&1
}

if [[ $SEQUENTIAL -eq 1 ]]; then
    for name in "${NAMES[@]}"; do
        printf '\n=== %s (quick) ===\n' "$name"
        declare -n command_ref="COMMAND_$name"
        started=$SECONDS
        set +e
        "${command_ref[@]}" --device "$DEVICE" --workdir "$WORK_DIR/$name" 2>&1 |
            tee "$LOG_DIR/$name.log"
        STATUS["$name"]=${PIPESTATUS[0]}
        set -e
        DURATION["$name"]=$((SECONDS - started))
        unset -n command_ref
    done
else
    for name in "${NAMES[@]}"; do
        printf '  starting        : %s\n' "$name"
        PID_START["$name"]=$SECONDS
        launch "$name" &
        PID["$name"]=$!
    done
    printf '\nWaiting for the three pipelines (logs in %s) ...\n' "$LOG_DIR"
    for name in "${NAMES[@]}"; do
        set +e
        wait "${PID[$name]}"
        STATUS["$name"]=$?
        set -e
        DURATION["$name"]=$((SECONDS - PID_START[$name]))
        printf '  finished        : %-9s exit %s after %ss\n' \
            "$name" "${STATUS[$name]}" "${DURATION[$name]}"
    done
fi

## A leading "--" would be read as an option by the printf builtin, so the rule
## is always printed as an argument.
hr() { printf '%s\n' "------------------------------------------------------------------------"; }

FAILED=0
printf '\n'
hr
printf 'Summary\n'
hr
for name in "${NAMES[@]}"; do
    if [[ "${STATUS[$name]}" -eq 0 ]]; then
        printf '  PASS  %-9s %3ss\n' "$name" "${DURATION[$name]}"
    else
        printf '  FAIL  %-9s %3ss  (exit %s)\n' \
            "$name" "${DURATION[$name]}" "${STATUS[$name]}"
        FAILED=1
    fi
done
printf '  total %ss\n' "$((SECONDS - START))"

## Show the result lines of the successful pipelines, the whole log of a failure.
for name in "${NAMES[@]}"; do
    printf '\n--- %s ---\n' "$name"
    if [[ "${STATUS[$name]}" -eq 0 ]]; then
        grep -a -E 'L2-relative error|^PICP|pipeline finished' "$LOG_DIR/$name.log" || true
    else
        cat "$LOG_DIR/$name.log"
    fi
done

if [[ $FAILED -ne 0 ]]; then
    printf '\nsmoke test FAILED\n' >&2
    exit 1
fi
printf '\nsmoke test passed\n'
