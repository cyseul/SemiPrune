#!/usr/bin/env bash
set -uo pipefail

TRAIN_PY="${TRAIN_PY:-train.py}"

BASE_DIR="${BASE_DIR:-data-model-0504/food101}"
SCORE_SET_NAME="${SCORE_SET_NAME:-gnn-du-hard}"
DATA_DIR="${DATA_DIR:-../data/food-101}"
SCORE_PATH="${SCORE_PATH:-../score_extrapolation/scores/extrapolation/extrapolated/gnn_du_FOOD101_resnet18_k_10_seed_7575_euclidean_5_5.npy}"

NETWORK="${NETWORK:-resnet18}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"

CORESET_RATIOS_STR="${CORESET_RATIOS_STR:-0.7 0.5 0.3 0.2 0.1}"
BATCH_SIZES_STR="${BATCH_SIZES_STR:-128 128 128 64 32}"
SEEDS_STR="${SEEDS_STR:-0 1 2 3 4}"
GPUS_STR="${GPUS_STR:-0 1 2 3 4 5 6 7}"

read -r -a CORESET_RATIOS <<< "${CORESET_RATIOS_STR}"
read -r -a BATCH_SIZES <<< "${BATCH_SIZES_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"
read -r -a GPUS <<< "${GPUS_STR}"

mkdir -p "${BASE_DIR}"
LOG_DIR="${BASE_DIR}/launch_logs"
mkdir -p "${LOG_DIR}"

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[error] train script does not exist: ${TRAIN_PY}" >&2
  exit 1
fi

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[error] data directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi

if [[ ! -f "${SCORE_PATH}" ]]; then
  echo "[error] score file does not exist: ${SCORE_PATH}" >&2
  exit 1
fi

if [[ "${#CORESET_RATIOS[@]}" -ne "${#BATCH_SIZES[@]}" ]]; then
  echo "[error] CORESET_RATIOS_STR and BATCH_SIZES_STR must have the same number of values." >&2
  exit 1
fi

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[error] GPUS_STR is empty. Example: GPUS_STR='0 1 2 3'" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
JOB_QUEUE="${TMP_DIR}/jobs.txt"
LOCK_DIR="${TMP_DIR}/lock"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

for seed in "${SEEDS[@]}"; do
  for i in "${!CORESET_RATIOS[@]}"; do
    echo "${seed} ${CORESET_RATIOS[$i]} ${BATCH_SIZES[$i]}" >> "${JOB_QUEUE}"
  done
done

get_next_job() {
  local __resultvar="$1"
  local line=""

  while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
    sleep 0.1
  done

  if [[ ! -s "${JOB_QUEUE}" ]]; then
    rmdir "${LOCK_DIR}"
    return 1
  fi

  IFS= read -r line < "${JOB_QUEUE}" || true
  tail -n +2 "${JOB_QUEUE}" > "${JOB_QUEUE}.tmp" || true
  mv "${JOB_QUEUE}.tmp" "${JOB_QUEUE}"

  rmdir "${LOCK_DIR}"

  printf -v "${__resultvar}" '%s' "${line}"
  return 0
}

launch_job() {
  local gpu_id="$1"
  local seed="$2"
  local coreset_ratio="$3"
  local batch_size="$4"

  local ratio_name
  ratio_name="$(ratio_tag "${coreset_ratio}")"

  local task_name="${SCORE_SET_NAME}-r${ratio_name}-seed${seed}"
  local log_path="${LOG_DIR}/${task_name}.gpu${gpu_id}.log"

  echo "[launch] gpu=${gpu_id} seed=${seed} ratio=${coreset_ratio} batch=${batch_size} task=${task_name}"
  echo "[score] ${SCORE_PATH}"
  echo "[log] ${log_path}"

  python "${TRAIN_PY}" \
    --dataset food101 \
    --network "${NETWORK}" \
    --epochs "${EPOCHS}" \
    --batch-size "${batch_size}" \
    --lr "${LR}" \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --coreset \
    --coreset-mode hard \
    --data-score-path "${SCORE_PATH}" \
    --coreset-ratio "${coreset_ratio}" \
    --gpuid "${gpu_id}" \
    --seed "${seed}" \
    --ignore-td \
    --wandb-project food101-extrapolated \
    > "${log_path}" 2>&1
}

worker() {
  local gpu_id="$1"
  local job_line=""
  local seed=""
  local coreset_ratio=""
  local batch_size=""

  echo "[worker-start] gpu=${gpu_id}"

  while get_next_job job_line; do
    if [[ -z "${job_line}" ]]; then
      continue
    fi

    read -r seed coreset_ratio batch_size <<< "${job_line}"
    echo "[worker] gpu=${gpu_id} picked seed=${seed} ratio=${coreset_ratio} batch=${batch_size}"

    if launch_job "${gpu_id}" "${seed}" "${coreset_ratio}" "${batch_size}"; then
      echo "[done] gpu=${gpu_id} seed=${seed} ratio=${coreset_ratio}"
    else
      echo "[failed] gpu=${gpu_id} seed=${seed} ratio=${coreset_ratio}. Check log under ${LOG_DIR}"
    fi
  done

  echo "[worker-exit] gpu=${gpu_id} no more jobs"
}

pids=()

for gpu_id in "${GPUS[@]}"; do
  worker "${gpu_id}" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}" || true
done

echo "All FOOD101 extrapolated hard coreset jobs finished."
