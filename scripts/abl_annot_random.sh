#!/usr/bin/env bash
set -uo pipefail

TRAIN_PY="${TRAIN_PY:-train.py}"
DATA_DIR="${DATA_DIR:-../data}"

NETWORK="${NETWORK:-resnet18}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"

BASE_DIR="./data-model-0504/caltech101/random"

CALTECH_SPLIT_PATH="data/embeddings/Caltech101-dino_vitb16/caltech101_split_seed0.pt"

SEEDS=(0 1 2)

# Same coreset ratios
CORESET_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6)

# Physical GPUs
DEVICES=(0 1 2 3 4 5 6 7)

# GPU 1개당 동시에 올릴 job 수
JOBS_PER_GPU=1

GPU_SLOTS=()
for gpu in "${DEVICES[@]}"; do
  for ((k=0; k<JOBS_PER_GPU; k++)); do
    GPU_SLOTS+=("${gpu}")
  done
done

num_slots=${#GPU_SLOTS[@]}
active_jobs=0
job_id=0

mkdir -p logs

get_batch_size() {
  case "$1" in
    0.2) echo 64 ;;
    0.1) echo 32 ;;
    *) echo 128 ;;
  esac
}

run_job() {
  local coreset_ratio="$1"
  local batch_size="$2"
  local seed="$3"
  local device="$4"
  local task_name="$5"

  local log_path="logs/${task_name}.log"

  echo "[START] ratio=${coreset_ratio}, seed=${seed}, batch=${batch_size}, gpu=${device}, task=${task_name}"
  echo "[LOG] ${log_path}"

  if [ ! -f "${CALTECH_SPLIT_PATH}" ]; then
    echo "[FAIL/SKIP] caltech split path not found: ${CALTECH_SPLIT_PATH}"
    exit 0
  fi

  set +e

  python "${TRAIN_PY}" \
    --dataset caltech101 \
    --data-dir "${DATA_DIR}" \
    --caltech-split-path "${CALTECH_SPLIT_PATH}" \
    --base-dir "${BASE_DIR}" \
    --batch-size "${batch_size}" \
    --task-name "${task_name}" \
    --network "${NETWORK}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --coreset \
    --coreset-mode random \
    --coreset-ratio "${coreset_ratio}" \
    --seed "${seed}" \
    --gpuid "${device}" \
    --wandb-project "Caltech101_Random" \
    > "${log_path}" 2>&1

  status=$?

  if (( status == 0 )); then
    echo "[DONE] ratio=${coreset_ratio}, seed=${seed}, gpu=${device}, task=${task_name}"
  else
    echo "[FAIL/SKIP] ratio=${coreset_ratio}, seed=${seed}, gpu=${device}, status=${status}, task=${task_name}"
  fi

  exit 0
}

for coreset_ratio in "${CORESET_RATIOS[@]}"; do
  batch_size="$(get_batch_size "${coreset_ratio}")"
  ratio_tag="${coreset_ratio/./p}"

  for seed in "${SEEDS[@]}"; do
    device="${GPU_SLOTS[$((job_id % num_slots))]}"
    task_name="random_subset${ratio_tag}_seed${seed}"

    run_job \
      "${coreset_ratio}" \
      "${batch_size}" \
      "${seed}" \
      "${device}" \
      "${task_name}" &

    job_id=$((job_id + 1))
    active_jobs=$((active_jobs + 1))

    if (( active_jobs >= num_slots )); then
      wait -n || true
      active_jobs=$((active_jobs - 1))
    fi
  done
done

while (( active_jobs > 0 )); do
  wait -n || true
  active_jobs=$((active_jobs - 1))
done

echo "All jobs finished."