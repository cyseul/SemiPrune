#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_SCRIPT="${REPO_DIR}/train_imagenet.py"

DATA_DIR="${DATA_DIR:-../data/tiny-imagenet-200-C}"
BASE_DIR="${BASE_DIR:-data-model/tiny-imagenet-c}"

FIXMATCH_SCORE_PATH="${FIXMATCH_SCORE_PATH:-${BASE_DIR}/fixmatch-ps/tiny_imagenet_score_multiT.pt}"
ELFS_SCORE_PATH="${ELFS_SCORE_PATH:-${BASE_DIR}/elfs-ps/tiny_imagenet_score_multiT.pt}"

GPUS_STR="${GPUS_STR:-0 1 2 3 4 5 6 7}"
RATIOS_STR="${RATIOS_STR:-0.7 0.5 0.3 0.2 0.1}"

# 이미 run1을 돌렸고 "4개 더"라면 2 3 4 5 추천
# 완전히 새로 4개 run이면 RUNS_STR="1 2 3 4"로 바꾸면 됨
RUNS_STR="${RUNS_STR:-2 3 4 5}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a RATIOS <<< "${RATIOS_STR}"
read -r -a RUNS <<< "${RUNS_STR}"

# ============================================================
# Search 결과로 얻은 pair를 여기에 넣으면 됨
# 형식: ["coreset_ratio"]="mis_ratio"
# 예시는 임시값이므로 반드시 실제 search 결과로 교체
# ============================================================

declare -A FIX_MIS_BY_RATIO=(
    ["0.7"]="0.2"
    ["0.5"]="0.4"
    ["0.3"]="0.6"
    ["0.2"]="0.7"
    ["0.1"]="0.8"
)

declare -A ELFS_MIS_BY_RATIO=(
    ["0.7"]="0.0"
    ["0.5"]="0.1"
    ["0.3"]="0.1"
    ["0.2"]="0.1"
    ["0.1"]="0.1"
)

get_batch_size() {
    case "$1" in
        0.1) echo 64 ;;
        0.2) echo 128 ;;
        *) echo 256 ;;
    esac
}

get_score_path() {
    case "$1" in
        fixmatch-ps) echo "${FIXMATCH_SCORE_PATH}" ;;
        elfs-ps) echo "${ELFS_SCORE_PATH}" ;;
        *)
            echo "Unsupported score source: $1" >&2
            exit 1
            ;;
    esac
}

get_mis_ratio() {
    local score_tag="$1"
    local ratio="$2"

    case "${score_tag}" in
        fixmatch-ps)
            if [[ -z "${FIX_MIS_BY_RATIO[$ratio]+x}" ]]; then
                echo "Missing fixmatch-ps mis_ratio for coreset ratio ${ratio}" >&2
                exit 1
            fi
            echo "${FIX_MIS_BY_RATIO[$ratio]}"
            ;;
        elfs-ps)
            if [[ -z "${ELFS_MIS_BY_RATIO[$ratio]+x}" ]]; then
                echo "Missing elfs-ps mis_ratio for coreset ratio ${ratio}" >&2
                exit 1
            fi
            echo "${ELFS_MIS_BY_RATIO[$ratio]}"
            ;;
        *)
            echo "Unsupported score source: ${score_tag}" >&2
            exit 1
            ;;
    esac
}

build_jobs() {
    local score_tag ratio mis_ratio run

    for score_tag in fixmatch-ps elfs-ps; do
        for ratio in "${RATIOS[@]}"; do
            mis_ratio="$(get_mis_ratio "${score_tag}" "${ratio}")"

            for run in "${RUNS[@]}"; do
                echo "${score_tag} ${ratio} ${mis_ratio} ${run}"
            done
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

    if [[ ! -d "${BASE_DIR}" ]]; then
        echo "Base directory does not exist: ${BASE_DIR}" >&2
        exit 1
    fi

    if [[ ! -f "${FIXMATCH_SCORE_PATH}" ]]; then
        echo "fixmatch-ps score file does not exist: ${FIXMATCH_SCORE_PATH}" >&2
        echo "Override with FIXMATCH_SCORE_PATH=/path/to/score.pt" >&2
        exit 1
    fi

    if [[ ! -f "${ELFS_SCORE_PATH}" ]]; then
        echo "elfs-ps score file does not exist: ${ELFS_SCORE_PATH}" >&2
        echo "Override with ELFS_SCORE_PATH=/path/to/score.pt" >&2
        exit 1
    fi

    if [[ ${#GPUS[@]} -eq 0 ]]; then
        echo "GPUS is empty. Set GPUS_STR, for example: GPUS_STR='0 1 2 3'" >&2
        exit 1
    fi

    if [[ ${#RATIOS[@]} -eq 0 ]]; then
        echo "RATIOS is empty. Set RATIOS_STR, for example: RATIOS_STR='0.7 0.5 0.3 0.2 0.1'" >&2
        exit 1
    fi

    if [[ ${#RUNS[@]} -eq 0 ]]; then
        echo "RUNS is empty. Set RUNS_STR, for example: RUNS_STR='2 3 4 5'" >&2
        exit 1
    fi
}

run_job() {
    local gpu="$1"
    local score_tag="$2"
    local ratio="$3"
    local mis_ratio="$4"
    local run="$5"

    local batch_size score_path task_name seed

    batch_size="$(get_batch_size "${ratio}")"
    score_path="$(get_score_path "${score_tag}")"

    # run마다 seed를 다르게 줌
    # run=2면 seed=1, run=3이면 seed=2 ...
    seed=$((run - 1))

    task_name="${score_tag}-budget-r${ratio}-mr${mis_ratio}-run${run}"

    echo "[launch] score=${score_tag} gpu=${gpu} run=${run} seed=${seed} ratio=${ratio} mis_ratio=${mis_ratio} batch=${batch_size} task=${task_name}"

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
        --coreset-mode budget \
        --data-score-path "${score_path}" \
        --network resnet34 \
        --batch-size "${batch_size}" \
        --coreset-ratio "${ratio}" \
        --mis-ratio "${mis_ratio}" \
        --gpuid "${gpu}" \
        --ignore-td
}

validate_config
mapfile -t JOBS < <(build_jobs)

echo "Total jobs: ${#JOBS[@]}"
echo "GPUs: ${GPUS[*]}"
echo "Runs: ${RUNS[*]}"

for gpu_idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gpu_idx]}"
    (
        for ((job_idx = gpu_idx; job_idx < ${#JOBS[@]}; job_idx += ${#GPUS[@]})); do
            read -r score_tag ratio mis_ratio run <<< "${JOBS[$job_idx]}"
            run_job "${gpu}" "${score_tag}" "${ratio}" "${mis_ratio}" "${run}"
        done
    ) &
done

wait
echo "All jobs finished."