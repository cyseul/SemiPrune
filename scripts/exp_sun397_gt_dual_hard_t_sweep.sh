#!/usr/bin/env bash
set -uo pipefail

TRAIN_PY="${TRAIN_PY:-train.py}"

BASE_DIR="${BASE_DIR:-data-model-0504/sun397}"
SCORE_SET_NAME="${SCORE_SET_NAME:-gt-dual-hard}"
DATA_DIR="${DATA_DIR:-../data/sun397_full_split}"

NETWORK="${NETWORK:-resnet18}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"

CORESET_RATIO="${1:-0.7}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"


GPUS_STR="${GPUS_STR:-0 1 2 3 4 5 6 7}"
T_VALUES_STR="${T_VALUES_STR:-30 40 50 60}"

SCORE_PATH="${SCORE_PATH:-data-model-0504/sun397/gt-full-train/data-score-gt-full-train.pickle}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a T_VALUES <<< "${T_VALUES_STR}"

mkdir -p "${BASE_DIR}"
LOG_DIR="${BASE_DIR}/launch_logs"
mkdir -p "${LOG_DIR}"

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

TMP_DIR="$(mktemp -d)"
JOB_QUEUE="${TMP_DIR}/jobs.txt"
LOCK_DIR="${TMP_DIR}/lock"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

for t in "${T_VALUES[@]}"; do
  echo "${t}" >> "${JOB_QUEUE}"
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
  local t="$2"

  local ratio_name
  ratio_name="$(ratio_tag "${CORESET_RATIO}")"

  local task_name="${SCORE_SET_NAME}-dual-hard-r${ratio_name}-t${t}-seed${SEED}"
  local log_path="${LOG_DIR}/${task_name}.gpu${gpu_id}.log"

  echo "[launch] gpu=${gpu_id} seed=${SEED} ratio=${CORESET_RATIO} T=${t} batch=${BATCH_SIZE} task=${task_name}"
  echo "[log] ${log_path}"

  python "${TRAIN_PY}" \
    --dataset sun397 \
    --network "${NETWORK}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --coreset \
    --coreset-mode dual_hard \
    --data-score-path "${SCORE_PATH}" \
    --coreset-ratio "${CORESET_RATIO}" \
    --pseudo-dual-T "${t}" \
    --gpuid "${gpu_id}" \
    --seed "${SEED}" \
    --ignore-td \
    > "${log_path}" 2>&1
}

worker() {
  local gpu_id="$1"
  local t=""

  echo "[worker-start] gpu=${gpu_id}"

  while get_next_job t; do
    if [[ -z "${t}" ]]; then
      continue
    fi

    echo "[worker] gpu=${gpu_id} picked T=${t}"

    if launch_job "${gpu_id}" "${t}"; then
      echo "[done] gpu=${gpu_id} T=${t}"
    else
      echo "[failed] gpu=${gpu_id} T=${t}. Check log under ${LOG_DIR}"
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

echo "All sun397 dual_hard T-sweep jobs finished."