#!/usr/bin/env bash
#
# reproduce_bayesian.sh -- B-LSTM-MIONet (replica exchange SGLD) for one system.
#
#   scripts/reproduce_bayesian.sh SYSTEM [--quick] [--device DEVICE]
#                                 [--workdir DIR]
#                                 [extra arguments passed to `blstm-mionet train`]
#
# SYSTEM is lorentz, pendulum or ausgrid.
#
# Steps
#   1. generate (or reuse) the training set of that system
#   2. generate (or reuse) the test set of that system
#   3. train two Langevin chains with configs/bayesian/SYSTEM.yaml; after the
#      burn-in the exploit chain is sampled once per epoch, giving a 360
#      member posterior ensemble (400 epochs - 40 burn-in, as in the research
#      code; the paper evaluates M = 300 of them)
#   4. evaluate the ensemble with `blstm-mionet infer-bayesian`, which reads the
#      M = 300 members named by inference.n_ensemble and prints the PICP of the
#      95% credible interval
#
# The datasets are the ones written by scripts/reproduce_SYSTEM.sh, so running
# both scripts in the same directory generates the data only once.
#
# --quick shrinks everything to a tiny CPU run (a 3 member ensemble over 8
# epochs) that finishes in well under a minute; Ausgrid then uses a synthetic
# CSV instead of the licensed data.  The full pipeline takes hours on one GPU.
# --workdir DIR runs everything inside DIR, so data/, mlruns/ and figures/ are
# written there instead of into the repository.
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
  --quick            tiny CPU settings, well under a minute end to end
  --device DEVICE    GPU index or "cpu" (default: 0, cpu with --quick)
  --workdir DIR      run inside DIR instead of the repository root
  -h, --help         show this message
  ...                every other argument is forwarded to `blstm-mionet train`
EOF
}

## Pull the SYSTEM positional out of the argument list, then parse the rest.
SYSTEM=""
ARGS=()
PREVIOUS=""
for arg in "$@"; do
    case "$PREVIOUS" in
    --device | --workdir)
        ARGS+=("$arg")
        PREVIOUS=""
        continue
        ;;
    esac
    case "$arg" in
    lorentz | pendulum | ausgrid)
        if [[ -z "$SYSTEM" ]]; then SYSTEM="$arg"; else ARGS+=("$arg"); fi
        ;;
    *)
        ARGS+=("$arg")
        ;;
    esac
    PREVIOUS="$arg"
done

STEP_TOTAL=4
common_parse_args ${ARGS[@]+"${ARGS[@]}"}

if [[ -z "$SYSTEM" ]]; then
    usage
    printf '\n'
    die "SYSTEM is required and must be one of: lorentz, pendulum, ausgrid"
fi

enter_workdir
printf 'System            : %s\n' "$SYSTEM"

CONFIG="$REPO_ROOT/configs/$SYSTEM.yaml"
BAYESIAN_CONFIG="$REPO_ROOT/configs/bayesian/$SYSTEM.yaml"
[[ -f "$CONFIG" ]] || die "no configuration file $CONFIG"
[[ -f "$BAYESIAN_CONFIG" ]] || die "no configuration file $BAYESIAN_CONFIG"
FIGURES="figures/bayesian_$SYSTEM"

if [[ $QUICK -eq 1 ]]; then
    ## 8 epochs - (3 + 1) = 4 burn-in epochs, then a 3 member ensemble.
    TRAIN_ARGS=(--epochs 8 --set bayesian.n_ensemble=3
        --set training.registered_model_name=null)
    TEST_SEARCH_NUM=20
else
    ## 400 epochs - (360 + 1) = 39 burn-in epochs, then 360 members of which
    ## inference.n_ensemble = 300 are evaluated (the paper's M = 300).
    TRAIN_ARGS=(--epochs 400)
    TEST_SEARCH_NUM=200
fi

case "$SYSTEM" in
lorentz)
    if [[ $QUICK -eq 1 ]]; then
        TRAIN_DATA="data/lorentz_quick_N20_T2.npy"
        TEST_DATA="data/lorentz_quick_N10_T2.npy"
        GEN_TRAIN=(--n-sample 20 --set data.t_max=2.0)
        GEN_TEST=(--n-sample 10 --set data.seed=1000 --set data.t_max=2.0)
    else
        TRAIN_DATA="data/lorentz_N_5000_h001_T20.npy"
        TEST_DATA="data/lorentz_N_100_h001_T20.npy"
        GEN_TRAIN=(--n-sample 5000)
        GEN_TEST=(--n-sample 100 --set data.seed=1000)
    fi
    TRAIN_LABEL="Lorenz trajectories"
    TEST_LABEL="held out Lorenz trajectories"
    ;;
pendulum)
    if [[ $QUICK -eq 1 ]]; then
        TRAIN_DATA="data/pendulum_quick_grf_N10_T2.npy"
        TEST_DATA="data/pendulum_quick_grf_N5_T2.npy"
        GEN_TRAIN=(--n-sample 10 --set data.t_max=2.0)
        GEN_TEST=(--n-sample 5 --set data.seed=1000 --set data.t_max=2.0)
    else
        TRAIN_DATA="data/pendulum_ctr_grf_N_5000_h001_T10.npy"
        TEST_DATA="data/pendulum_ctr_grf_N_100_h001_T10.npy"
        GEN_TRAIN=(--n-sample 5000)
        GEN_TEST=(--n-sample 100 --set data.seed=1000)
    fi
    TRAIN_LABEL="pendulum trajectories with GRF controls"
    TEST_LABEL="held out GRF controls"
    ;;
ausgrid)
    if [[ $QUICK -eq 1 ]]; then
        SYNTHETIC_CSV="data/Ausgrid_synthetic/synthetic_solar_home_data.csv"
        write_synthetic_csv "$SYNTHETIC_CSV"
        AUSGRID_COMMON=(
            --set "data.ausgrid.csv_paths=[\"$SYNTHETIC_CSV\"]"
            --set data.ausgrid.end_date=2010-08-10
        )
        TRAIN_DATA="data/ausgrid_quick_cust_1-3.npy"
        TEST_DATA="data/ausgrid_quick_cust_4-5.npy"
        GEN_TRAIN=(--set "data.ausgrid.customer_id=[1,2,3]" "${AUSGRID_COMMON[@]}")
        GEN_TEST=(--set "data.ausgrid.customer_id=[4,5]" "${AUSGRID_COMMON[@]}")
    else
        TRAIN_DATA="data/ausgrid_cust_1-50.npy"
        TEST_DATA="data/ausgrid_cust_51-60.npy"
        GEN_TRAIN=(--set "data.ausgrid.customer_id=[$(seq -s, 1 50)]")
        GEN_TEST=(--set "data.ausgrid.customer_id=[$(seq -s, 51 60)]")
        if [[ ! -f "$TRAIN_DATA" || ! -f "$TEST_DATA" ]]; then
            while read -r path; do
                [[ -f "$path" ]] || die "missing Ausgrid CSV file '$path'; run scripts/download_data.sh or scripts/reproduce_ausgrid.sh --quick"
            done < <(ausgrid_csv_paths "$CONFIG")
        fi
    fi
    TRAIN_LABEL="daily Ausgrid profiles"
    TEST_LABEL="daily Ausgrid profiles of the first test group"
    ;;
esac

step "Training data: $TRAIN_LABEL -> $TRAIN_DATA"
ensure_dataset "$TRAIN_DATA" --config "$CONFIG" "${GEN_TRAIN[@]}"

step "Test data: $TEST_LABEL -> $TEST_DATA"
ensure_dataset "$TEST_DATA" --config "$CONFIG" "${GEN_TEST[@]}"

step "Replica exchange SGLD training ($BAYESIAN_CONFIG)"
train_and_capture_run "logs/bayesian_${SYSTEM}_train.log" \
    --config "$CONFIG" \
    --bayesian "$BAYESIAN_CONFIG" \
    --data "$TRAIN_DATA" \
    --device "$DEVICE" \
    --run-name "resgld_$SYSTEM" \
    --set training.figure_dir="$FIGURES/train" \
    "${TRAIN_ARGS[@]}" \
    ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

step "Posterior predictive evaluation of the ensemble in $RUN_URI"
INFER_LOG="logs/bayesian_${SYSTEM}_infer.log"
mkdir -p logs
printf '\n$ %s infer-bayesian --config %s --data %s --run %s\n\n' \
    "${BLSTM[*]}" "$CONFIG" "$TEST_DATA" "$RUN_URI"
"${BLSTM[@]}" infer-bayesian \
    --config "$CONFIG" \
    --data "$TEST_DATA" \
    --run "$RUN_URI" \
    --device "$DEVICE" \
    --set inference.search_num="$TEST_SEARCH_NUM" \
    --figure-dir "$FIGURES/uq" 2>&1 | tee "$INFER_LOG"

PICP_LINE="$(grep -a '^PICP' "$INFER_LOG" | tail -n 1 || true)"
REGISTERED="$(registered_model_name "$CONFIG" || true)"

printf '\n'
hr
printf 'Bayesian %s pipeline finished in %s (mm:ss).\n' "$SYSTEM" "$(elapsed)"
hr
note "${PICP_LINE:-PICP line not found in $INFER_LOG}"
note "MLflow run          : $RUN_URI"
note "Ensemble artifacts  : $RUN_URI/ensemble  (member_XXXX.pt)"
note "MLflow tracking dir : $PWD/mlruns  (MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri $PWD/mlruns)"
note "Training log        : $PWD/logs/bayesian_${SYSTEM}_train.log"
note "Inference log       : $PWD/$INFER_LOG"
note "Figures             : $PWD/$FIGURES/{train,uq}"
if [[ $QUICK -eq 1 ]]; then
    note "Model registration was disabled in --quick mode."
elif [[ -n "$REGISTERED" ]]; then
    note "Registered model    : the exploit chain is also registered as"
    note "                      models:/$REGISTERED-bayesian/latest, so the Adam"
    note "                      model models:/$REGISTERED/latest is left untouched."
fi
