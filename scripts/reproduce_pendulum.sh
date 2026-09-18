#!/usr/bin/env bash
#
# reproduce_pendulum.sh -- full pendulum pipeline of the paper (Sec. 4.2).
#
#   scripts/reproduce_pendulum.sh [--quick] [--device DEVICE] [--workdir DIR]
#                                 [extra arguments passed to `blstm-mionet train`]
#
# Steps
#   1. generate the training set (5000 initial states driven by Gaussian random
#      field controls, T = 10 s, dt = 0.01)
#   2. generate the in-distribution test set (100 GRF controls, another seed)
#   3. generate the out-of-distribution test set (100 trajectories driven by the
#      designated control u = sin(t / 2))
#   4. train LSTM-MIONet with Adam (configs/pendulum.yaml)
#   5. one-step-ahead inference on the GRF test set
#   6. one-step-ahead inference on the out-of-distribution set
#
# --quick replaces every step with tiny CPU settings (20 short trajectories,
# 2 epochs, --device cpu, a small search_num) and finishes in about a minute.
# The full pipeline takes hours on one GPU.  --workdir DIR runs everything
# inside DIR, so data/, mlruns/ and figures/ are written there instead of into
# the repository.
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
  --quick            tiny CPU settings, about a minute end to end
  --device DEVICE    GPU index, "parallel" or "cpu" (default: 0, cpu with --quick)
  --workdir DIR      run inside DIR instead of the repository root
  -h, --help         show this message
  ...                every other argument is forwarded to `blstm-mionet train`
EOF
}

STEP_TOTAL=6
common_parse_args "$@"
enter_workdir

CONFIG="$REPO_ROOT/configs/pendulum.yaml"
FIGURES="figures/pendulum"

if [[ $QUICK -eq 1 ]]; then
    TRAIN_DATA="data/pendulum_quick_grf_N10_T2.npy"
    TEST_DATA="data/pendulum_quick_grf_N5_T2.npy"
    OOD_DATA="data/pendulum_quick_designate_N5_T2.npy"
    ## Every GRF control costs a 2000 x 2000 Cholesky factorisation, so the
    ## quick run draws as few of them as possible.
    GEN_COMMON=(--set data.t_max=2.0)
    N_TRAIN=10
    N_TEST=5
    TRAIN_ARGS=(--epochs 2 --set training.registered_model_name=null)
    TEST_SEARCH_NUM=20
    RUN_NAME="lstm_mionet_pendulum_quick"
else
    ## The names below are the ones the configuration file expects.
    TRAIN_DATA="data/pendulum_ctr_grf_N_5000_h001_T10.npy"
    TEST_DATA="data/pendulum_ctr_grf_N_100_h001_T10.npy"
    OOD_DATA="data/pendulum_ctr_designate_N_100_h001_T10.npy"
    GEN_COMMON=()
    N_TRAIN=5000
    N_TEST=100
    TRAIN_ARGS=()
    ## 200 sub-sequences per test trajectory (configs/pendulum.yaml).
    TEST_SEARCH_NUM=200
    RUN_NAME="lstm_mionet_pendulum"
fi

step "Training data: $N_TRAIN pendulum trajectories with GRF controls -> $TRAIN_DATA"
ensure_dataset "$TRAIN_DATA" --config "$CONFIG" --n-sample "$N_TRAIN" \
    "${GEN_COMMON[@]}"

step "Test data: $N_TEST held out GRF controls -> $TEST_DATA"
ensure_dataset "$TEST_DATA" --config "$CONFIG" --n-sample "$N_TEST" \
    --set data.seed=1000 "${GEN_COMMON[@]}"

step "Out-of-distribution data: $N_TEST trajectories with u = sin(t / 2) -> $OOD_DATA"
ensure_dataset "$OOD_DATA" --config "$CONFIG" --n-sample "$N_TEST" \
    --set data.control=designate --set data.seed=1001 "${GEN_COMMON[@]}"

step "Adam training of LSTM-MIONet on $TRAIN_DATA"
train_and_capture_run "logs/pendulum_train.log" \
    --config "$CONFIG" \
    --data "$TRAIN_DATA" \
    --device "$DEVICE" \
    --run-name "$RUN_NAME" \
    --set training.figure_dir="$FIGURES/train" \
    "${TRAIN_ARGS[@]}" \
    ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

step "One-step-ahead inference on the GRF test set $TEST_DATA"
blstm infer \
    --config "$CONFIG" \
    --data "$TEST_DATA" \
    --model "$MODEL_URI" \
    --device "$DEVICE" \
    --no-recursive \
    --set inference.search_num="$TEST_SEARCH_NUM" \
    --figure-dir "$FIGURES/test_grf"

step "One-step-ahead inference on the out-of-distribution set $OOD_DATA"
blstm infer \
    --config "$CONFIG" \
    --data "$OOD_DATA" \
    --model "$MODEL_URI" \
    --device "$DEVICE" \
    --no-recursive \
    --set inference.search_num="$TEST_SEARCH_NUM" \
    --figure-dir "$FIGURES/test_designate"

REGISTERED="$(registered_model_name "$CONFIG" || true)"

printf '\n'
hr
printf 'Pendulum pipeline finished in %s (mm:ss).\n' "$(elapsed)"
hr
note "MLflow run          : $RUN_URI"
note "Model used for infer: $MODEL_URI"
note "MLflow tracking dir : $PWD/mlruns  (mlflow ui --backend-store-uri $PWD/mlruns)"
note "Training log        : $PWD/logs/pendulum_train.log"
note "Figures             : $PWD/$FIGURES/{train,test_grf,test_designate}"
if [[ $QUICK -eq 1 ]]; then
    note "Model registration was disabled in --quick mode."
else
    note "Registered model    : models:/${REGISTERED:-pendulum}/latest (configs/pendulum.yaml)"
fi
