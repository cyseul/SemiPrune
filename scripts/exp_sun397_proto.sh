#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CORESET_RATIOS=(0.7 0.5 0.3 0.2 0.1)
BATCH_SIZES=(128 128 128 64 32)
GPUS=(0 1 2 3 4 5 6 7)
SEEDS=(0 1 2 3 4)

DATA_SCORE_PATH="data-model-0505/sun-proto/prototype_score.pt"
DATA_DIR="../data/sun397_full_split_resized_512"
BASE_DIR="data-model-0505/sun-proto"
WANDB_PROJECT="SUN397_Baselines_0506"

run_job() {
  local gpu="$1"
  local coreset_ratio="$2"
  local batch_size="$3"
  local seed="$4"

  local ratio_tag="${coreset_ratio/./p}"
  local task_name="sun397-proto-r${ratio_tag}-s${seed}"

  echo "[START] gpu=${gpu} ratio=${coreset_ratio} batch=${batch_size} seed=${seed} task=${task_name}"

  python train.py \
    --epochs 200 \
    --dataset sun397 \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --data-score-path "${DATA_SCORE_PATH}" \
    --network resnet18 \
    --ignore-td \
    --coreset \
    --coreset-mode swav \
    --coreset-ratio "${coreset_ratio}" \
    --batch-size "${batch_size}" \
    --seed "${seed}" \
    --gpuid "${gpu}" \
    --wandb-project "${WANDB_PROJECT}"

  echo "[DONE] gpu=${gpu} ratio=${coreset_ratio} seed=${seed} task=${task_name}"
}

job_id=0
active_jobs=0
num_gpus=${#GPUS[@]}

for i in "${!CORESET_RATIOS[@]}"; do
  coreset_ratio="${CORESET_RATIOS[$i]}"
  batch_size="${BATCH_SIZES[$i]}"

  for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$((job_id % num_gpus))]}"

    run_job "${gpu}" "${coreset_ratio}" "${batch_size}" "${seed}" &

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

echo "All SUN397 proto jobs finished."
