#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_PY="${TRAIN_PY:-${REPO_DIR}/train_imagenet.py}"

BASE_DIR="${BASE_DIR:-data-model/tiny-imagenet-c}"
SCORE_SET_NAME="${SCORE_SET_NAME:-fixmatch-ps}"
DATA_DIR="${DATA_DIR:-../data/tiny-imagenet-200-C}"
NETWORK="${NETWORK:-resnet34}"
EPOCHS="${EPOCHS:-90}"
LR="${LR:-0.1}"
SCHEDULER="${SCHEDULER:-cosine}"

CORESET_RATIO="${1:-0.7}"
PSEUDO_DUAL_P="${2:-1.0}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"

GPUS_STR="${GPUS_STR:-0 1 2 3 4 5 6 7}"
T_VALUES_STR="${T_VALUES_STR:-30 40 50 60 70 80 90 100}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a T_VALUES <<< "${T_VALUES_STR}"

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

SCORE_PATH="${BASE_DIR}/${SCORE_SET_NAME}/tiny_imagenet_score_multiT.pt"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "Train script does not exist: ${TRAIN_PY}" >&2
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

if [[ ! -f "${SCORE_PATH}" ]]; then
  echo "Score file does not exist: ${SCORE_PATH}" >&2
  exit 1
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPUS is empty. Set GPUS_STR, for example: GPUS_STR='0 1 2 3'" >&2
  exit 1
fi

if [[ ${#T_VALUES[@]} -eq 0 ]]; then
  echo "T_VALUES is empty. Set T_VALUES_STR, for example: T_VALUES_STR='30 40 50'" >&2
  exit 1
fi

launch_job() {
  local gpu_id="$1"
  local t="$2"

  local task_name="${SCORE_SET_NAME}-dual-hard-r$(ratio_tag "${CORESET_RATIO}")-t${t}"
  local log_path="${BASE_DIR}/${task_name}.launch.log"

  echo "[launch] score=${SCORE_SET_NAME} dataset=tiny-imagenet gpu=${gpu_id} seed=${SEED} ratio=${CORESET_RATIO} p=${PSEUDO_DUAL_P} T=${t} batch=${BATCH_SIZE} task=${task_name}"

  "${PYTHON_BIN}" "${TRAIN_PY}" \
    --dataset tiny-imagenet \
    --network "${NETWORK}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --scheduler "${SCHEDULER}" \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --coreset \
    --coreset-mode dual_hard \
    --data-score-path "${SCORE_PATH}" \
    --coreset-ratio "${CORESET_RATIO}" \
    --pseudo-dual-p "${PSEUDO_DUAL_P}" \
    --pseudo-dual-T "${t}" \
    --gpuid "${gpu_id}" \
    --seed "${SEED}" \
    --ignore-td \
    > "${log_path}" 2>&1
}

NUM_GPUS=${#GPUS[@]}
pids=()
gpu_slots=()

for idx in "${!T_VALUES[@]}"; do
  t="${T_VALUES[$idx]}"
  slot=$((idx % NUM_GPUS))
  gpu_id="${GPUS[$slot]}"

  if [[ -n "${pids[$slot]:-}" ]]; then
    echo "[wait] gpu=${gpu_id} previous pid=${pids[$slot]} finished, launching next job..."
    wait "${pids[$slot]}"
  fi

  launch_job "${gpu_id}" "${t}" &
  pids[$slot]=$!
  gpu_slots[$slot]="${gpu_id}"
done

for slot in "${!pids[@]}"; do
  if [[ -n "${pids[$slot]:-}" ]]; then
    echo "[final wait] gpu=${gpu_slots[$slot]} pid=${pids[$slot]}"
    wait "${pids[$slot]}"
  fi
done

echo "All Tiny-ImageNet fixmatch-ps dual_hard T-sweep jobs finished."
