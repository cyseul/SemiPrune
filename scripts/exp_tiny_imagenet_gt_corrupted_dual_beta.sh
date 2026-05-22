#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_SCRIPT="train_imagenet.py"
DATA_DIR="${DATA_DIR:-../data/tiny-imagenet-200-C}"
BASE_DIR="${BASE_DIR:-data-model/tiny-imagenet-c}"
SCORE_PATH="${BASE_DIR}/gt-corrupted/tiny_imagenet_score_multiT.pt"

GPUS=(0 1 2 3 4 5 6 7)
T_VALUE=50
BETA_CD=3
RATIOS_STR="${RATIOS_STR:-0.7 0.5 0.3 0.2 0.1}"
RUNS_STR="${RUNS_STR:-1 2 3 4 5}"

read -r -a RATIOS <<< "${RATIOS_STR}"
read -r -a RUNS <<< "${RUNS_STR}"

get_batch_size() {
    case "$1" in
        0.1) echo 64 ;;
        0.2) echo 128 ;;
        0.3|0.5|0.7) echo 256 ;;
        *) echo 256 ;;
    esac
}

build_jobs() {
    local ratio run
    for ratio in "${RATIOS[@]}"; do
        for run in "${RUNS[@]}"; do
            echo "${ratio} ${run}"
        done
    done
}

validate_config() {
    if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
        echo "Python script does not exist: ${PYTHON_SCRIPT}" >&2
        exit 1
    fi

    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python executable is not available: ${PYTHON_BIN}" >&2
        exit 1
    fi

    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "Data directory does not exist: ${DATA_DIR}" >&2
        exit 1
    fi

    if [[ ! -f "${SCORE_PATH}" ]]; then
        echo "Score file does not exist: ${SCORE_PATH}" >&2
        exit 1
    fi

    if [[ ${#RATIOS[@]} -eq 0 ]]; then
        echo "RATIOS is empty. Set RATIOS_STR, for example: RATIOS_STR='0.7 0.1'" >&2
        exit 1
    fi

    if [[ ${#RUNS[@]} -eq 0 ]]; then
        echo "RUNS is empty. Set RUNS_STR, for example: RUNS_STR='1 2 3 4 5'" >&2
        exit 1
    fi
}

run_job() {
    local gpu="$1"
    local ratio="$2"
    local run="$3"
    local batch_size
    batch_size="$(get_batch_size "${ratio}")"
    local task_name="gt-corrupted-rn34-90ep-dual-beta-T${T_VALUE}-r${ratio}-p1-cd${BETA_CD}-run${run}"

    echo "[launch] gpu=${gpu} ratio=${ratio} T=${T_VALUE} cd=${BETA_CD} run=${run} batch=${batch_size} task=${task_name}"

    "${PYTHON_BIN}" "${PYTHON_SCRIPT}" \
        --epochs 90 \
        --iterations-per-testing 800 \
        --lr 0.1 \
        --scheduler cosine \
        --task-name "${task_name}" \
        --data-dir "${DATA_DIR}" \
        --base-dir "${BASE_DIR}" \
        --dataset tiny-imagenet \
        --coreset \
        --coreset-mode dual_beta \
        --data-score-path "${SCORE_PATH}" \
        --network resnet34 \
        --batch-size "${batch_size}" \
        --coreset-ratio "${ratio}" \
        --pseudo-dual-T "${T_VALUE}" \
        --pseudo-dual-p 1 \
        --beta-cd "${BETA_CD}" \
        --gpuid "${gpu}" \
        --ignore-td
}

validate_config
mapfile -t JOBS < <(build_jobs)

for gpu_idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gpu_idx]}"
    (
        for ((job_idx=gpu_idx; job_idx<${#JOBS[@]}; job_idx+=${#GPUS[@]})); do
            read -r ratio run <<< "${JOBS[$job_idx]}"
            run_job "${gpu}" "${ratio}" "${run}"
        done
    ) &
done

wait
