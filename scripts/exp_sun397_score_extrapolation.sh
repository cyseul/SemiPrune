#!/usr/bin/env bash
set -uo pipefail

TRAIN_PY="${TRAIN_PY:-train.py}"

BASE_DIR="${BASE_DIR:-data-model-0504/sun397}"
SCORE_SET_NAME="${SCORE_SET_NAME:-scoreext}"
DATA_DIR="${DATA_DIR:-../data/sun397_full_split_resized_512}"

NETWORK="${NETWORK:-resnet18}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"

CORESET_RATIOS=(0.7 0.5 0.3 0.2 0.1)
BATCH_SIZES=(128 128 128 64 32)
SEEDS=(0 1 2 3 4)

GPUS=(0 1 2 3 4 5 6 7)

SCORE_PATH="${SCORE_PATH:-gnn_du_SUN397_resnet18_k_10_seed_7575_euclidean_5_6.npy}"

mkdir -p "${BASE_DIR}"
LOG_DIR="${BASE_DIR}/launch_logs"
mkdir -p "${LOG_DIR}"

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

run_job() {
  local gpu_id="$1"
  local ratio="$2"
  local batch_size="$3"
  local seed="$4"

  local ratio_name
  ratio_name="$(ratio_tag "${ratio}")"

  local task_name="${SCORE_SET_NAME}-r${ratio_name}-seed${seed}"
  local log_path="${LOG_DIR}/${task_name}.gpu${gpu_id}.log"

  echo "[launch] gpu=${gpu_id} seed=${seed} ratio=${ratio} batch=${batch_size} task=${task_name}"
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
    --coreset-mode hard \
    --data-score-path "${SCORE_PATH}" \
    --coreset-ratio "${ratio}" \
    --gpuid "${gpu_id}" \
    --seed "${seed}" \
    --ignore-td \
    --wandb-project SUN397_extrapolation_subset
    > "${log_path}" 2>&1
}

job_id=0
active_jobs=0
num_gpus=${#GPUS[@]}

for i in "${!CORESET_RATIOS[@]}"; do
  ratio="${CORESET_RATIOS[$i]}"
  batch_size="${BATCH_SIZES[$i]}"

  for seed in "${SEEDS[@]}"; do
    gpu_id="${GPUS[$((job_id % num_gpus))]}"

    run_job "${gpu_id}" "${ratio}" "${batch_size}" "${seed}" &

    job_id=$((job_id + 1))
    active_jobs=$((active_jobs + 1))

    if (( active_jobs >= num_gpus )); then
      wait -n || true
      active_jobs=$((active_jobs - 1))
    fi
  done
done

while (( active_jobs > 0 )); do
  wait -n || true
  active_jobs=$((active_jobs - 1))
done

echo "All sun397 extrap-hard jobs finished."