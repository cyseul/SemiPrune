#!/usr/bin/env bash
set -euo pipefail
echo "Sleep for 1 hour..."
sleep 1h

DATA_DIR="../data"

# Same GPU can appear multiple times if you want multiple workers per GPU.
GPUS=(0 1 2 3 4 5 6 7)

RATIOS=(0.7 0.5 0.3 0.2 0.1)

SEEDS=(0)
SCORE_METRICS=("forgetting" "el2n")

BASE_DIR="./data-model/cifar100/elfs-ps/full-train"

SCORE_PATH="data-model/cifar100/elfs-ps/data-score-all-data.pickle"

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
    0.5) echo "0.1 0.2 0.3 0.4" ;;
    0.3) echo "0.2 0.3 0.4 0.5 0.6" ;;
    0.2) echo "0.3 0.4 0.5 0.6 0.7" ;;
    0.1) echo "0.4 0.5 0.6 0.7 0.8" ;;
    *) echo "0.0" ;;
  esac
}

valid_mis_ratio() {
  local ratio="$1"
  local mis_ratio="$2"

  python - <<PY
ratio = float("${ratio}")
mis = float("${mis_ratio}")
raise SystemExit(0 if mis <= 1.0 - ratio + 1e-12 else 1)
PY
}

get_mis_descending() {
  local score_metric="$1"

  case "${score_metric}" in
    forgetting) echo "True" ;;
    el2n) echo "False" ;;
    *)
      echo "Unsupported score metric: ${score_metric}" >&2
      return 1
      ;;
  esac
}

build_jobs() {
  local ratio mis_ratio seed score_metric
  local -a mis_grid

  for score_metric in "${SCORE_METRICS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      read -r -a mis_grid <<< "$(get_mis_ratios "${ratio}")"

      for mis_ratio in "${mis_grid[@]}"; do
        # Safety check: valid only when mis_ratio <= 1 - coreset_ratio.
        if ! valid_mis_ratio "${ratio}" "${mis_ratio}"; then
          continue
        fi

        for seed in "${SEEDS[@]}"; do
          echo "${ratio} ${mis_ratio} ${seed} ${score_metric}"
        done
      done
    done
  done
}

run_job() {
  local gpu="$1"
  local ratio="$2"
  local mis_ratio="$3"
  local seed="$4"
  local score_metric="$5"

  local batch_size
  batch_size="$(get_batch_size "${ratio}")"

  local ratio_tag="${ratio/./p}"
  local mis_tag="${mis_ratio/./p}"

  local mis_descending
  mis_descending="$(get_mis_descending "${score_metric}")"

  local task_name="cifar100-elfs-ps-${score_metric}-r${ratio_tag}-mr${mis_tag}-md${mis_descending}-s${seed}"

  if [[ ! -f "${SCORE_PATH}" ]]; then
    echo "[SKIP] missing score file: ${SCORE_PATH}" >&2
    return 0
  fi

  echo "[START] gpu=${gpu} metric=${score_metric} ratio=${ratio} mis_ratio=${mis_ratio} mis_descending=${mis_descending} seed=${seed}"

  python train.py \
    --epochs 200 \
    --iterations-per-testing 800 \
    --lr 0.1 \
    --lr_policy cosine \
    --task-name "${task_name}" \
    --dataset cifar100 \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --coreset \
    --coreset-mode budget \
    --data-score-path "${SCORE_PATH}" \
    --mis-key "${score_metric}" \
    --mis-data-score-descending "${mis_descending}" \
    --coreset-key "${score_metric}" \
    --network resnet18 \
    --batch-size "${batch_size}" \
    --coreset-ratio "${ratio}" \
    --save-coreset \
    --mis-ratio "${mis_ratio}" \
    --seed "${seed}" \
    --gpuid "${gpu}" \
    --ignore-td \
    --wandb-project "cifar100_score_abl_elfs"

  echo "[DONE]  gpu=${gpu} metric=${score_metric} ratio=${ratio} mis_ratio=${mis_ratio} mis_descending=${mis_descending} seed=${seed}"
}

mapfile -t JOBS < <(build_jobs)

echo "Total jobs: ${#JOBS[@]}"

if (( ${#JOBS[@]} == 0 )); then
  echo "No jobs to run." >&2
  exit 1
fi

worker() {
  local gpu="$1"
  local worker_id="$2"
  local job_id=0
  local job
  local ratio mis_ratio seed score_metric

  for job in "${JOBS[@]}"; do
    if (( job_id % ${#GPUS[@]} == worker_id )); then
      read -r ratio mis_ratio seed score_metric <<< "${job}"
      run_job "${gpu}" "${ratio}" "${mis_ratio}" "${seed}" "${score_metric}"
    fi

    job_id=$((job_id + 1))
  done
}

for worker_id in "${!GPUS[@]}"; do
  worker "${GPUS[$worker_id]}" "${worker_id}" &
done

wait

echo "All jobs finished."