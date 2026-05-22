#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/train_imagenet.py}"

DATA_DIR="${DATA_DIR:-../data/imagenet}"
BASE_DIR="${BASE_DIR:-data-model/imagenet}"
SCORE_PATH="${SCORE_PATH:-data-model/imagenet/fixmatch-ps/imagenet_score.pt}"

GPUS_STR="${GPUS_STR:-0 1 2 3 4}"
NETWORK="${NETWORK:-resnet34}"
EPOCHS="${EPOCHS:-90}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.1}"
SCHEDULER="${SCHEDULER:-cosine}"
ITERATIONS_PER_TESTING="${ITERATIONS_PER_TESTING:-5000}"

# Pair order:
# coreset-ratio: 0.7 0.5 0.3 0.2 0.1
# mis-ratio:     0.0 0.1 0.2 0.4 0.5
PAIRS=(
    "0.7 0.0"
    "0.5 0.1"
    "0.3 0.2"
    "0.2 0.4"
    "0.1 0.5"
)

read -r -a GPUS <<< "${GPUS_STR}"

ratio_tag() {
    local x="$1"
    echo "${x//./p}"
}

validate_config() {
    if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
        echo "Missing train script: ${TRAIN_SCRIPT}" >&2
        exit 1
    fi
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python executable is not available: ${PYTHON_BIN}" >&2
        exit 1
    fi
    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "Data directory does not exist: ${DATA_DIR}" >&2
        echo "Override with DATA_DIR=/path/to/imagenet" >&2
        exit 1
    fi
    if [[ ! -f "${SCORE_PATH}" ]]; then
        echo "Score file does not exist: ${SCORE_PATH}" >&2
        exit 1
    fi
    mkdir -p "${BASE_DIR}"
}

run_job() {
    local gpu="$1"
    local coreset_ratio="$2"
    local mis_ratio="$3"
    local task_name log_path

    task_name="fixmatch-ps-budget-${NETWORK}-${EPOCHS}ep-r$(ratio_tag "${coreset_ratio}")-mr$(ratio_tag "${mis_ratio}")"
    log_path="${BASE_DIR}/${task_name}.launch.log"

    echo "[launch] gpu=${gpu} ratio=${coreset_ratio} mis=${mis_ratio} task=${task_name}"

    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        --dataset imagenet \
        --network "${NETWORK}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --scheduler "${SCHEDULER}" \
        --iterations-per-testing "${ITERATIONS_PER_TESTING}" \
        --data-dir "${DATA_DIR}" \
        --base-dir "${BASE_DIR}" \
        --task-name "${task_name}" \
        --coreset \
        --coreset-mode budget \
        --data-score-path "${SCORE_PATH}" \
        --coreset-key accumulated_margin \
        --coreset-ratio "${coreset_ratio}" \
        --mis-ratio "${mis_ratio}" \
        --gpuid "${gpu}" \
        > "${log_path}" 2>&1
}

validate_config

declare -A PID_TO_GPU
declare -A GPU_BUSY
declare -A PID_TO_JOB

for gpu in "${GPUS[@]}"; do
    GPU_BUSY["${gpu}"]=0
done

launch_job() {
    local gpu="$1"
    local coreset_ratio="$2"
    local mis_ratio="$3"

    run_job "${gpu}" "${coreset_ratio}" "${mis_ratio}" &
    local pid=$!
    PID_TO_GPU["${pid}"]="${gpu}"
    PID_TO_JOB["${pid}"]="ratio=${coreset_ratio},mis=${mis_ratio}"
    GPU_BUSY["${gpu}"]=1
}

next_free_gpu() {
    local gpu
    for gpu in "${GPUS[@]}"; do
        if [[ "${GPU_BUSY[${gpu}]}" -eq 0 ]]; then
            echo "${gpu}"
            return 0
        fi
    done
    return 1
}

wait_one() {
    local pid gpu
    while true; do
        for pid in "${!PID_TO_GPU[@]}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                wait "${pid}"
                gpu="${PID_TO_GPU[${pid}]}"
                echo "[done] gpu=${gpu} ${PID_TO_JOB[${pid}]}"
                GPU_BUSY["${gpu}"]=0
                unset "PID_TO_GPU[${pid}]"
                unset "PID_TO_JOB[${pid}]"
                return 0
            fi
        done
        sleep 10
    done
}

for pair in "${PAIRS[@]}"; do
    read -r coreset_ratio mis_ratio <<< "${pair}"

    until gpu="$(next_free_gpu)"; do
        wait_one
    done
    launch_job "${gpu}" "${coreset_ratio}" "${mis_ratio}"
done

while [[ "${#PID_TO_GPU[@]}" -gt 0 ]]; do
    wait_one
done

echo "All ImageNet fixmatch budget pair jobs finished."
