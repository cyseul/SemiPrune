#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS=(0 1 2 3)
BASE_DIR="${SCRIPT_DIR}/data-model/food101"
SCORE_PATH="${BASE_DIR}/fixmatch-ps-rn18-200ep/food101_score_multiT.pt"
DATA_DIR="../data/food101"

RATIOS=(0.7 0.5 0.3 0.2 0.1)
MIS_RATIOS=(0.1 0.3 0.5 0.6 0.8)
RUNS=(2 3 4 5)

get_batch_size() {
    case "$1" in
        0.2) echo 64 ;;
        0.1) echo 32 ;;
        *) echo 128 ;;
    esac
}

build_jobs() {
    local run ratio mis_ratio
    for run in "${RUNS[@]}"; do
        for ratio in "${RATIOS[@]}"; do
            for mis_ratio in "${MIS_RATIOS[@]}"; do
                python - <<PY
ratio = float("${ratio}")
mis = float("${mis_ratio}")
if mis < 1.0 - ratio:
    print("${run} ${ratio} ${mis_ratio}")
PY
            done
        done
    done
}

run_job() {
    local gpu="$1"
    local run="$2"
    local ratio="$3"
    local mis_ratio="$4"
    local batch_size
    batch_size="$(get_batch_size "${ratio}")"

    local task_name="fixmatch-ps-rn18-200ep-aum-r${ratio}-mr${mis_ratio}-run${run}"

    echo "[START] gpu=${gpu} run=${run} ratio=${ratio} mis_ratio=${mis_ratio}"

    python "${SCRIPT_DIR}/train_imagenet.py" \
        --epochs 200 \
        --iterations-per-testing 800 \
        --lr 0.1 \
        --scheduler cosine \
        --task-name "${task_name}" \
        --data-dir "${DATA_DIR}" \
        --base-dir "${BASE_DIR}" \
        --dataset food101 \
        --coreset \
        --coreset-mode budget \
        --data-score-path "${SCORE_PATH}" \
        --coreset-key accumulated_margin \
        --network resnet18 \
        --batch-size "${batch_size}" \
        --coreset-ratio "${ratio}" \
        --mis-ratio "${mis_ratio}" \
        --gpuid "${gpu}" \
        --ignore-td \
        --run "${run}"

    echo "[DONE]  gpu=${gpu} run=${run} ratio=${ratio} mis_ratio=${mis_ratio}"
}

mapfile -t JOBS < <(build_jobs)

declare -A PID_TO_GPU
declare -A GPU_BUSY

for gpu in "${GPUS[@]}"; do
    GPU_BUSY["$gpu"]=0
done

launch_job() {
    local gpu="$1"
    local run="$2"
    local ratio="$3"
    local mis_ratio="$4"

    run_job "$gpu" "$run" "$ratio" "$mis_ratio" &
    local pid=$!

    PID_TO_GPU["$pid"]="$gpu"
    GPU_BUSY["$gpu"]=1
}

job_ptr=0
num_jobs=${#JOBS[@]}
num_gpus=${#GPUS[@]}

while (( job_ptr < num_jobs )); do
    for gpu in "${GPUS[@]}"; do
        if (( job_ptr >= num_jobs )); then
            break
        fi

        if [[ "${GPU_BUSY[$gpu]}" -eq 0 ]]; then
            read -r run ratio mis_ratio <<< "${JOBS[$job_ptr]}"
            launch_job "$gpu" "$run" "$ratio" "$mis_ratio"
            ((job_ptr+=1))
        fi
    done

    if (( job_ptr < num_jobs )); then
        finished_pid=""
        wait -n -p finished_pid

        finished_gpu="${PID_TO_GPU[$finished_pid]}"
        GPU_BUSY["$finished_gpu"]=0
        unset PID_TO_GPU["$finished_pid"]
    fi
done

while ((${#PID_TO_GPU[@]} > 0)); do
    finished_pid=""
    wait -n -p finished_pid
    finished_gpu="${PID_TO_GPU[$finished_pid]}"
    GPU_BUSY["$finished_gpu"]=0
    unset PID_TO_GPU["$finished_pid"]
done