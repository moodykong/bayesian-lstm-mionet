#!/usr/bin/env bash
#
# reproduce_lorentz.sh -- full Lorenz 63 pipeline of the paper (Sec. 4.1).
#
#   scripts/reproduce_lorentz.sh [--quick] [--device DEVICE] [--workdir DIR]
#                                [extra arguments passed to `blstm-mionet train`]
#
# Steps
#   1. generate the training set   (5000 initial conditions, T = 20 s, dt = 0.01)
#   2. generate the test set       (100 initial conditions, a different seed)
#   3. generate a single reference trajectory for the recursive rollout
#   4. train LSTM-MIONet with Adam (configs/lorentz.yaml)
#   5. one-step-ahead inference on the 100-trajectory test set
#   6. recursive rollouts on the single trajectory with teacher forcing
#      1.0, 0.5 and 0.0, as in the original notebook
#
# --quick replaces every step with the tiny CPU settings used by
# scripts/smoke_test.sh (20 short trajectories, 2 epochs, --device cpu, a small
# search_num); it finishes in about a minute.  The full pipeline takes hours on
# one GPU.  --workdir DIR runs everything inside DIR, so data/, mlruns/ and
# figures/ are written there instead of into the repository.
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

STEP_TOTAL=8
common_parse_args "$@"
enter_workdir

CONFIG="$REPO_ROOT/configs/lorentz.yaml"
FIGURES="figures/lorentz"

if [[ $QUICK -eq 1 ]]; then
    TRAIN_DATA="data/lorentz_quick_N20_T2.npy"
    TEST_DATA="data/lorentz_quick_N10_T2.npy"
    SINGLE_DATA="data/lorentz_quick_N1_T2.npy"
    GEN_COMMON=(--set data.t_max=2.0)
    N_TRAIN=20
    N_TEST=10
    TRAIN_ARGS=(--epochs 2 --set training.registered_model_name=null)
    TEST_SEARCH_NUM=20
    ## Fewer steps than there are time steps, so the quick rollout strides
    ## through the trajectory instead of walking it step by step.
    ROLLOUT_SEARCH_NUM=50
    RUN_NAME="lstm_mionet_lorentz_quick"
else
    ## The names below are the ones the configuration file expects.
    TRAIN_DATA="data/lorentz_N_5000_h001_T20.npy"
    TEST_DATA="data/lorentz_N_100_h001_T20.npy"
    SINGLE_DATA="data/lorentz_N_1_h001_T20.npy"
    GEN_COMMON=()
    N_TRAIN=5000
    N_TEST=100
    TRAIN_ARGS=()
    ## 200 sub-sequences per test trajectory (configs/lorentz.yaml).
    TEST_SEARCH_NUM=200
    ## T / dt - 2 * search_len - 1 consecutive steps, as in the notebook.
    ROLLOUT_SEARCH_NUM=1995
    RUN_NAME="lstm_mionet_lorentz"
fi

step "Training data: $N_TRAIN Lorenz trajectories -> $TRAIN_DATA"
ensure_dataset "$TRAIN_DATA" --config "$CONFIG" --n-sample "$N_TRAIN" \
    "${GEN_COMMON[@]}"

step "Test data: $N_TEST held out Lorenz trajectories -> $TEST_DATA"
ensure_dataset "$TEST_DATA" --config "$CONFIG" --n-sample "$N_TEST" \
    --set data.seed=1000 "${GEN_COMMON[@]}"

step "Recursive rollout data: 1 trajectory -> $SINGLE_DATA"
ensure_dataset "$SINGLE_DATA" --config "$CONFIG" --n-sample 1 \
    --set data.seed=1001 "${GEN_COMMON[@]}"

step "Adam training of LSTM-MIONet on $TRAIN_DATA"
train_and_capture_run "logs/lorentz_train.log" \
    --config "$CONFIG" \
    --data "$TRAIN_DATA" \
    --device "$DEVICE" \
    --run-name "$RUN_NAME" \
    --set training.figure_dir="$FIGURES/train" \
    "${TRAIN_ARGS[@]}" \
    ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

step "One-step-ahead inference on $TEST_DATA"
blstm infer \
    --config "$CONFIG" \
    --data "$TEST_DATA" \
    --model "$MODEL_URI" \
    --device "$DEVICE" \
    --no-recursive \
    --set inference.search_num="$TEST_SEARCH_NUM" \
    --figure-dir "$FIGURES/test"

for TFP in 1.0 0.5 0.0; do
    TAG="tf${TFP/./}"
    step "Recursive rollout on $SINGLE_DATA, teacher forcing probability $TFP"
    blstm infer \
        --config "$CONFIG" \
        --data "$SINGLE_DATA" \
        --model "$MODEL_URI" \
        --device "$DEVICE" \
        --recursive \
        --teacher-forcing-prob "$TFP" \
        --set inference.autonomous=true \
        --set inference.search_num="$ROLLOUT_SEARCH_NUM" \
        --figure-dir "$FIGURES/recursive_$TAG"
done

REGISTERED="$(registered_model_name "$CONFIG" || true)"

printf '\n'
hr
printf 'Lorenz pipeline finished in %s (mm:ss).\n' "$(elapsed)"
hr
note "MLflow run          : $RUN_URI"
note "Model used for infer: $MODEL_URI"
note "MLflow tracking dir : $PWD/mlruns  (mlflow ui --backend-store-uri $PWD/mlruns)"
note "Training log        : $PWD/logs/lorentz_train.log"
note "Figures             : $PWD/$FIGURES/{train,test,recursive_tf10,recursive_tf05,recursive_tf00}"
if [[ $QUICK -eq 1 ]]; then
    note "Model registration was disabled in --quick mode."
else
    note "Registered model    : models:/${REGISTERED:-lorentz}/latest (configs/lorentz.yaml)"
fi
