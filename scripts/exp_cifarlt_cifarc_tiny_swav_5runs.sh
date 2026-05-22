#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_IMAGENET_SCRIPT="${TRAIN_IMAGENET_SCRIPT:-${REPO_DIR}/train_imagenet.py}"

GPUS_STR="${GPUS_STR:-0 1 2 3 4 5 6 7}"
RUNS_STR="${RUNS_STR:-1 2 3 4 5}"
RATIOS_STR="${RATIOS_STR:-0.7 0.5 0.3 0.2 0.1}"

TINY_IMAGENET_DATA_DIR="${TINY_IMAGENET_DATA_DIR:-../data/tiny-imagenet-200-C}"
TINY_NETWORK="${TINY_NETWORK:-resnet34}"
TINY_EPOCHS="${TINY_EPOCHS:-90}"
TINY_LR="${TINY_LR:-0.1}"
TINY_SCHEDULER="${TINY_SCHEDULER:-cosine}"
TINY_IMAGENET_SCORE_PATH="${TINY_IMAGENET_SCORE_PATH:-data-model/tiny-imagenet-c_swav/data-score-tiny-imagenet-c-swav.pickle}"
TINY_IMAGENET_BASE_DIR="${TINY_IMAGENET_BASE_DIR:-data-model/tiny-imagenet-c_swav}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a RUNS <<< "${RUNS_STR}"
read -r -a RATIOS <<< "${RATIOS_STR}"

ratio_tag() {
    local x="$1"
    echo "${x//./p}"
}

get_batch_size() {
    case "$1" in
        0.7|0.5|0.3) echo 128 ;;
        0.2) echo 64 ;;
        0.1) echo 32 ;;
        *)
            echo "Unsupported ratio for batch size: $1" >&2
            exit 1
            ;;
    esac
}

validate_config() {
    if [[ ! -f "${TRAIN_IMAGENET_SCRIPT}" ]]; then
        echo "Missing train script: ${TRAIN_IMAGENET_SCRIPT}" >&2
        exit 1
    fi

    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python executable is not available: ${PYTHON_BIN}" >&2
        exit 1
    fi

    if [[ ! -d "${TINY_IMAGENET_DATA_DIR}" ]]; then
        echo "Data directory does not exist: ${TINY_IMAGENET_DATA_DIR}" >&2
        exit 1
    fi

    mkdir -p "${TINY_IMAGENET_BASE_DIR}"

    if [[ ! -f "${TINY_IMAGENET_SCORE_PATH}" ]]; then
        echo "Score file does not exist: ${TINY_IMAGENET_SCORE_PATH}" >&2
        exit 1
    fi
}

build_jobs() {
    local ratio run
    for ratio in "${RATIOS[@]}"; do
        for run in "${RUNS[@]}"; do
            echo "tiny-imagenet ${ratio} ${run}"
        done
    done
}

run_job() {
    local gpu="$1"
    local job_key="$2"
    local ratio="$3"
    local run="$4"
    local seed=$((run - 1))
    local batch_size task_name log_path
    local -a cmd

    batch_size="$(get_batch_size "${ratio}")"
    task_name="${job_key}-prototype-${TINY_NETWORK}-${TINY_EPOCHS}ep-r$(ratio_tag "${ratio}")-run${run}"
    log_path="${TINY_IMAGENET_BASE_DIR}/${task_name}.launch.log"

    echo "[launch] key=${job_key} dataset=tiny-imagenet gpu=${gpu} run=${run} seed=${seed} ratio=${ratio} batch=${batch_size} task=${task_name}"

    cmd=(
        "${PYTHON_BIN}" "${TRAIN_IMAGENET_SCRIPT}"
        --dataset tiny-imagenet
        --network "${TINY_NETWORK}"
        --epochs "${TINY_EPOCHS}"
        --batch-size "${batch_size}"
        --lr "${TINY_LR}"
        --scheduler "${TINY_SCHEDULER}"
        --data-dir "${TINY_IMAGENET_DATA_DIR}"
        --base-dir "${TINY_IMAGENET_BASE_DIR}"
        --task-name "${task_name}"
        --coreset
        --coreset-mode swav
        --data-score-path "${TINY_IMAGENET_SCORE_PATH}"
        --coreset-ratio "${ratio}"
        --gpuid "${gpu}"
    )

    "${cmd[@]}" > "${log_path}" 2>&1
}

validate_config
mapfile -t JOBS < <(build_jobs)

declare -A PID_TO_GPU
declare -A GPU_BUSY
declare -A PID_TO_JOB

for gpu in "${GPUS[@]}"; do
    GPU_BUSY["$gpu"]=0
done

launch_job() {
    local gpu="$1"
    local job_key="$2"
    local ratio="$3"
    local run="$4"

    run_job "$gpu" "$job_key" "$ratio" "$run" &
    local pid=$!

    PID_TO_GPU["$pid"]="$gpu"
    PID_TO_JOB["$pid"]="${job_key} ratio=${ratio} run=${run}"
    GPU_BUSY["$gpu"]=1
}

release_finished_gpus() {
    local pid gpu
    for pid in "${!PID_TO_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu="${PID_TO_GPU[$pid]}"
            GPU_BUSY["$gpu"]=0
            unset PID_TO_GPU["$pid"]
            unset PID_TO_JOB["$pid"]
        fi
    done
}

wait_for_any() {
    local pid finished_pid="" status=0

    while :; do
        for pid in "${!PID_TO_GPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                finished_pid="$pid"
                set +e
                wait "$finished_pid"
                status=$?
                set -e
                break 2
            fi
        done
        sleep 1
    done

    if [[ ${status} -ne 0 ]]; then
        echo "[warn] job failed: pid=${finished_pid:-unknown} ${PID_TO_JOB[${finished_pid:-0}]-unknown}" >&2
    fi
    release_finished_gpus
}

job_ptr=0
num_jobs=${#JOBS[@]}

while (( job_ptr < num_jobs )); do
    for gpu in "${GPUS[@]}"; do
        if (( job_ptr >= num_jobs )); then
            break
        fi

        if [[ "${GPU_BUSY[$gpu]}" -eq 0 ]]; then
            read -r job_key ratio run <<< "${JOBS[$job_ptr]}"
            launch_job "$gpu" "$job_key" "$ratio" "$run"
            ((job_ptr+=1))
        fi
    done

    if (( job_ptr < num_jobs )); then
        wait_for_any
    fi
done

while ((${#PID_TO_GPU[@]} > 0)); do
    wait_for_any
done

echo "All Tiny ImageNet SWAV prototype jobs finished."