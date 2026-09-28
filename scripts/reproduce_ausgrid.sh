#!/usr/bin/env bash
#
# reproduce_ausgrid.sh -- full Ausgrid solar-home pipeline of the paper (Sec. 4.3).
#
#   scripts/reproduce_ausgrid.sh [--quick] [--device DEVICE] [--workdir DIR]
#                                [extra arguments passed to `blstm-mionet train`]
#
# Steps
#   1. check that the Ausgrid CSV files listed in configs/ausgrid.yaml are there
#      (they are licensed data and are not redistributed; see
#      scripts/download_data.sh)
#   2. select the training profiles   (customers 1-50,  2010-07-01 to 2011-06-30)
#   3. select the first test group    (customers 51-60, same period)
#   4. select the second test group   (customers 61-70, same period)
#   5. train LSTM-MIONet with Adam (configs/ausgrid.yaml)
#   6. one-step-ahead inference on customers 51-60
#   7. one-step-ahead inference on customers 61-70
#
# --quick does not need the Ausgrid data at all: it writes a small synthetic CSV
# with the same layout (title row, header, 48 half-hour columns), trains for 2
# epochs on the CPU and finishes in about a minute.  The full pipeline takes
# hours on one GPU.  --workdir DIR runs everything inside DIR, so data/, mlruns/
# and figures/ are written there instead of into the repository; in that case
# the Ausgrid CSV files must be reachable from DIR as well.
#
# Requires the environment of the repository:  uv sync --extra cpu
# (or --extra cu126 / --extra cu130 for CUDA).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"
# shellcheck source=scripts/_common.sh
source scripts/_common.sh

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
    cat <<'EOF'
Options:
  --quick            synthetic CSV, tiny CPU settings, about a minute end to end
  --device DEVICE    GPU index or "cpu" (default: 0, cpu with --quick)
  --workdir DIR      run inside DIR instead of the repository root
  -h, --help         show this message
  ...                every other argument is forwarded to `blstm-mionet train`
EOF
}

## Comma separated integer range, e.g. customer_range 51 60 -> 51,52,...,60
customer_range() {
    seq -s, "$1" "$2"
}

STEP_TOTAL=7
common_parse_args "$@"
enter_workdir

CONFIG="$REPO_ROOT/configs/ausgrid.yaml"
FIGURES="figures/ausgrid"

step "Checking the Ausgrid CSV files listed in configs/ausgrid.yaml"
mapfile -t CSV_PATHS < <(ausgrid_csv_paths "$CONFIG")
MISSING=()
for path in "${CSV_PATHS[@]}"; do
    if [[ -f "$path" ]]; then
        note "found   : $path"
    else
        note "MISSING : $path"
        MISSING+=("$path")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    if [[ $QUICK -eq 0 ]]; then
        printf '\n'
        hr
        printf 'The Ausgrid data is missing.\n'
        hr
        cat <<EOF
The solar home half-hour CSV files are licensed by Ausgrid and are not
redistributed with this repository.  To run the full pipeline:

  1. fetch the authors' archive with scripts/download_data.sh (Ausgrid no
     longer hosts the "Solar home electricity data" download page),
  2. extract them so that these paths exist, relative to $PWD:

EOF
        for path in "${CSV_PATHS[@]}"; do printf '       %s\n' "$path"; done
        cat <<EOF

  3. re-run this script.

Alternatively run 'scripts/reproduce_ausgrid.sh --quick', which builds a small
synthetic CSV with the same layout and needs no download at all.  If your copy
of the data lives elsewhere, point this directory at it

  ln -s /path/to/your/Ausgrid $PWD/data/Ausgrid

or edit data.ausgrid.csv_paths in configs/ausgrid.yaml (arguments given to this
script are forwarded to 'blstm-mionet train', not to the data selection).
EOF
        exit 1
    fi
    note "Not needed in --quick mode: a synthetic CSV is generated instead."
fi

if [[ $QUICK -eq 1 ]]; then
    SYNTHETIC_CSV="data/Ausgrid_synthetic/synthetic_solar_home_data.csv"
    write_synthetic_csv "$SYNTHETIC_CSV"
    ## Customers 1-3 stand in for 1-50, 4-5 for 51-60 and 6 for 61-70.
    AUSGRID_COMMON=(
        --set "data.ausgrid.csv_paths=[\"$SYNTHETIC_CSV\"]"
        --set data.ausgrid.end_date=2010-08-10
    )
    TRAIN_DATA="data/ausgrid_quick_cust_1-3.npy"
    TEST_A_DATA="data/ausgrid_quick_cust_4-5.npy"
    TEST_B_DATA="data/ausgrid_quick_cust_6.npy"
    TRAIN_GROUP="$(customer_range 1 3)"
    TEST_A_GROUP="$(customer_range 4 5)"
    TEST_B_GROUP="6"
    TRAIN_ARGS=(--epochs 2 --set training.registered_model_name=null)
    TEST_SEARCH_NUM=20
    RUN_NAME="lstm_mionet_ausgrid_quick"
else
    AUSGRID_COMMON=()
    ## The names below are the ones the configuration file expects.
    TRAIN_DATA="data/ausgrid_cust_1-50.npy"
    TEST_A_DATA="data/ausgrid_cust_51-60.npy"
    TEST_B_DATA="data/ausgrid_cust_61-70.npy"
    TRAIN_GROUP="$(customer_range 1 50)"
    TEST_A_GROUP="$(customer_range 51 60)"
    TEST_B_GROUP="$(customer_range 61 70)"
    TRAIN_ARGS=()
    ## 100 sub-sequences per daily profile (configs/ausgrid.yaml, paper Sec. 4.3).
    TEST_SEARCH_NUM=100
    RUN_NAME="lstm_mionet_ausgrid"
fi

step "Training profiles: customers $TRAIN_GROUP -> $TRAIN_DATA"
ensure_dataset "$TRAIN_DATA" --config "$CONFIG" \
    --set "data.ausgrid.customer_id=[$TRAIN_GROUP]" \
    "${AUSGRID_COMMON[@]}"

step "First test group: customers $TEST_A_GROUP -> $TEST_A_DATA"
ensure_dataset "$TEST_A_DATA" --config "$CONFIG" \
    --set "data.ausgrid.customer_id=[$TEST_A_GROUP]" \
    "${AUSGRID_COMMON[@]}"

step "Second test group: customers $TEST_B_GROUP -> $TEST_B_DATA"
ensure_dataset "$TEST_B_DATA" --config "$CONFIG" \
    --set "data.ausgrid.customer_id=[$TEST_B_GROUP]" \
    "${AUSGRID_COMMON[@]}"

step "Adam training of LSTM-MIONet on $TRAIN_DATA"
train_and_capture_run "logs/ausgrid_train.log" \
    --config "$CONFIG" \
    --data "$TRAIN_DATA" \
    --device "$DEVICE" \
    --run-name "$RUN_NAME" \
    --set training.figure_dir="$FIGURES/train" \
    "${TRAIN_ARGS[@]}" \
    ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

step "One-step-ahead inference on customers $TEST_A_GROUP"
blstm infer \
    --config "$CONFIG" \
    --data "$TEST_A_DATA" \
    --model "$MODEL_URI" \
    --device "$DEVICE" \
    --no-recursive \
    --set inference.search_num="$TEST_SEARCH_NUM" \
    --figure-dir "$FIGURES/test_group_a"

step "One-step-ahead inference on customers $TEST_B_GROUP"
blstm infer \
    --config "$CONFIG" \
    --data "$TEST_B_DATA" \
    --model "$MODEL_URI" \
    --device "$DEVICE" \
    --no-recursive \
    --set inference.search_num="$TEST_SEARCH_NUM" \
    --figure-dir "$FIGURES/test_group_b"

REGISTERED="$(registered_model_name "$CONFIG" || true)"

printf '\n'
hr
printf 'Ausgrid pipeline finished in %s (mm:ss).\n' "$(elapsed)"
hr
note "MLflow run          : $RUN_URI"
note "Model used for infer: $MODEL_URI"
note "MLflow tracking dir : $PWD/mlruns  (MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri $PWD/mlruns)"
note "Training log        : $PWD/logs/ausgrid_train.log"
note "Figures             : $PWD/$FIGURES/{train,test_group_a,test_group_b}"
if [[ $QUICK -eq 1 ]]; then
    note "Synthetic data was used; the numbers above are NOT the paper's."
    note "Model registration was disabled in --quick mode."
else
    note "Registered model    : models:/${REGISTERED:-Ausgrid}/latest (configs/ausgrid.yaml)"
fi
