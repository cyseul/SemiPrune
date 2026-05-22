#!/usr/bin/env bash
set -uo pipefail

SLEEP_HOURS="${SLEEP_HOURS:-6}"
echo "[sleep] waiting ${SLEEP_HOURS} hours before launching jobs..."
sleep "${SLEEP_HOURS}h"

DATA_DIR="${DATA_DIR:-../data}"
BASE_ROOT="${BASE_ROOT:-./data-model/food101/fixmatch-ps}"

LABEL_BUDGETS=(757)
RUNS=(1)
GPUS=(0 1 2 3 4 5 6 7)

RATIOS=(0.7 0.5 0.3 0.2 0.1)

NETWORK="${NETWORK:-resnet18}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"
SCHEDULER="${SCHEDULER:-cosine}"
ITERATIONS_PER_TESTING="${ITERATIONS_PER_TESTING:-800}"
WANDB_PROJECT="${WANDB_PROJECT:-label_ablation_food101_lb1515_resnet18}"

ratio_tag() {
  echo "${1//./p}"
}

get_batch_size() {
  case "$1" in
    0.2) echo 64 ;;
    0.1) echo 32 ;;
    *) echo 128 ;;
  esac
}

get_mis_ratios() {
  local ratio="$1"

  case "${ratio}" in
    0.7) echo "0.0 0.1 0.2 0.3" ;;
    0.5) echo "0.1 0.2 0.3 0.4 0.5" ;;
    0.3) echo "0.2 0.3 0.4 0.5 0.6" ;;
    0.2) echo "0.3 0.4 0.5 0.6 0.7" ;;
    0.1) echo "0.4 0.5 0.6 0.7 0.8" ;;
    *) echo "0.0" ;;
  esac
}

run_job() {
  local gpu="$1"
  local label_budget="$2"
  local run="$3"
  local ratio="$4"
  local mis_ratio="$5"

  local seed batch_size base_dir score_path task_name log_path

  seed="$((run - 1))"
  batch_size="$(get_batch_size "${ratio}")"

  base_dir="${BASE_ROOT}/lb${label_budget}"
  score_path="${base_dir}/full-train/data-score-full-train.pickle"

  mkdir -p "${base_dir}"

  task_name="fixmatch-lb${label_budget}-aum-budget-r$(ratio_tag "${ratio}")-mr$(ratio_tag "${mis_ratio}")-run${run}"
  log_path="${base_dir}/${task_name}.launch.log"

  echo "[launch] gpu=${gpu} lb=${label_budget} run=${run} ratio=${ratio} mis=${mis_ratio}"

  python train.py \
    --dataset food101 \
    --network "${NETWORK}" \
    --epochs "${EPOCHS}" \
    --batch-size "${batch_size}" \
    --lr "${LR}" \
    --scheduler "${SCHEDULER}" \
    --iterations-per-testing "${ITERATIONS_PER_TESTING}" \
    --data-dir "${DATA_DIR}" \
    --base-dir "${base_dir}" \
    --task-name "${task_name}" \
    --coreset \
    --coreset-mode budget \
    --data-score-path "${score_path}" \
    --coreset-ratio "${ratio}" \
    --coreset-key accumulated_margin \
    --mis-key accumulated_margin \
    --mis-ratio "${mis_ratio}" \
    --gpuid "${gpu}" \
    --seed "${seed}" \
    --ignore-td \
    --wandb-project "${WANDB_PROJECT}" \
    > "${log_path}" 2>&1
}

job_id=0
active_jobs=0
num_gpus=${#GPUS[@]}

for label_budget in "${LABEL_BUDGETS[@]}"; do
  for run in "${RUNS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      read -r -a MIS_RATIOS <<< "$(get_mis_ratios "${ratio}")"

      for mis_ratio in "${MIS_RATIOS[@]}"; do
        gpu="${GPUS[$((job_id % num_gpus))]}"

        run_job "${gpu}" "${label_budget}" "${run}" "${ratio}" "${mis_ratio}" &

        job_id=$((job_id + 1))
        active_jobs=$((active_jobs + 1))

        if (( active_jobs >= num_gpus )); then
          wait -n || true
          active_jobs=$((active_jobs - 1))
        fi
      done
    done
  done
done

while (( active_jobs > 0 )); do
  wait -n || true
  active_jobs=$((active_jobs - 1))
done

echo "All jobs finished."