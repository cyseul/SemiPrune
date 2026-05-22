#!/usr/bin/env bash
set -uo pipefail

TRAIN_PY="${TRAIN_PY:-train.py}"

BASE_DIR="${BASE_DIR:-data-model-0504/sun397}"
SCORE_SET_NAME="${SCORE_SET_NAME:-semi-dual-beta}"
DATA_DIR="${DATA_DIR:-../data/sun397_full_split_resized_512}"

NETWORK="${NETWORK:-resnet18}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"

CORESET_RATIOS=(0.7 0.5 0.3 0.2 0.1)
BATCH_SIZES=(128 128 128 64 32)
SEEDS=(0)

GPUS_STR="${GPUS_STR:-0 1 2 3 4 5 6 7}"
T_VALUE="${T_VALUE:-20 30}"
BETA_CD_VALUES_STR="${BETA_CD_VALUES_STR:-4.0}"
PSEUDO_DUAL_P="${PSEUDO_DUAL_P:-1}"

SCORE_PATH="${SCORE_PATH:-data-model-0504/sun397/fixmatch-ps/data-score-fixmatch-ps.pickle}"

read -r -a GPUS <<< "${GPUS_STR}"
read -r -a BETA_CD_VALUES <<< "${BETA_CD_VALUES_STR}"

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

if [[ "${#CORESET_RATIOS[@]}" -ne "${#BATCH_SIZES[@]}" ]]; then
  echo "[error] CORESET_RATIOS and BATCH_SIZES must have the same length." >&2
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  for i in "${!CORESET_RATIOS[@]}"; do
    coreset_ratio="${CORESET_RATIOS[$i]}"
    batch_size="${BATCH_SIZES[$i]}"

    for beta_cd in "${BETA_CD_VALUES[@]}"; do
      echo "${seed} ${coreset_ratio} ${batch_size} ${beta_cd}" >> "${JOB_QUEUE}"
    done
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
  local beta_cd="$5"

  local ratio_name
  ratio_name="$(ratio_tag "${coreset_ratio}")"

  local beta_cd_name
  beta_cd_name="$(ratio_tag "${beta_cd}")"

  local task_name="dual-beta-r${ratio_name}-T${T_VALUE}-cd${beta_cd_name}-seed${seed}"
  local log_path="${LOG_DIR}/${task_name}.gpu${gpu_id}.log"

  echo "[launch] gpu=${gpu_id} seed=${seed} ratio=${coreset_ratio} T=${T_VALUE} p=${PSEUDO_DUAL_P} cd=${beta_cd} batch=${batch_size} task=${task_name}"
  echo "[log] ${log_path}"

  python "${TRAIN_PY}" \
    --dataset sun397 \
    --network "${NETWORK}" \
    --epochs "${EPOCHS}" \
    --batch-size "${batch_size}" \
    --lr "${LR}" \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --coreset \
    --coreset-mode dual_beta \
    --data-score-path "${SCORE_PATH}" \
    --coreset-ratio "${coreset_ratio}" \
    --pseudo-dual-T "${T_VALUE}" \
    --pseudo-dual-p "${PSEUDO_DUAL_P}" \
    --beta-cd "${beta_cd}" \
    --gpuid "${gpu_id}" \
    --seed "${seed}" \
    --ignore-td \
    > "${log_path}" 2>&1
}

worker() {
  local gpu_id="$1"
  local job_line=""
  local seed=""
  local coreset_ratio=""
  local batch_size=""
  local beta_cd=""

  echo "[worker-start] gpu=${gpu_id}"

  while get_next_job job_line; do
    if [[ -z "${job_line}" ]]; then
      continue
    fi

    read -r seed coreset_ratio batch_size beta_cd <<< "${job_line}"

    echo "[worker] gpu=${gpu_id} picked seed=${seed} ratio=${coreset_ratio} batch=${batch_size} cd=${beta_cd}"

    if launch_job "${gpu_id}" "${seed}" "${coreset_ratio}" "${batch_size}" "${beta_cd}"; then
      echo "[done] gpu=${gpu_id} seed=${seed} ratio=${coreset_ratio} cd=${beta_cd}"
    else
      echo "[failed] gpu=${gpu_id} seed=${seed} ratio=${coreset_ratio} cd=${beta_cd}. Check log under ${LOG_DIR}"
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

echo "All sun397 dual_beta c_d-sweep jobs finished."