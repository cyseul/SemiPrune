#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="../data/sun397_full_split_resized_512"

# Same GPU can appear multiple times if you want multiple workers per GPU.
GPUS=(0 1 2 3 4 5 6 7)

RATIOS=(0.7 0.5 0.3 0.2 0.1)
SEEDS=(0)

# ELFS 계열만 실행
METHODS=("elfs" "elfs-self")

SCORE_METRIC="accumulated_margin"
MIS_DESCENDING="False"

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

build_jobs() {
  local method ratio mis_ratio seed
  local -a mis_grid

  for method in "${METHODS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
      read -r -a mis_grid <<< "$(get_mis_ratios "${ratio}")"

      for mis_ratio in "${mis_grid[@]}"; do
        if ! valid_mis_ratio "${ratio}" "${mis_ratio}"; then
          continue
        fi

        for seed in "${SEEDS[@]}"; do
          echo "budget ${method} ${ratio} ${mis_ratio} ${seed}"
        done
      done
    done
  done
}

run_budget_job() {
  local gpu="$1"
  local method="$2"
  local ratio="$3"
  local mis_ratio="$4"
  local seed="$5"

  local batch_size
  batch_size="$(get_batch_size "${ratio}")"

  local base_dir="data-model-0505/sun-${method}/full-train"
  local score_path="${base_dir}/data-score-full-train.pickle"

  local ratio_tag="${ratio/./p}"
  local mis_tag="${mis_ratio/./p}"

  local task_name="sun397-${method}-ps-${SCORE_METRIC}-r${ratio_tag}-mr${mis_tag}-md${MIS_DESCENDING}-s${seed}"

  if [[ ! -f "${score_path}" ]]; then
    echo "[SKIP] missing score file: ${score_path}" >&2
    return 0
  fi

  echo "[START] gpu=${gpu} method=${method} metric=${SCORE_METRIC} ratio=${ratio} mis_ratio=${mis_ratio} seed=${seed}"

  python train.py \
    --epochs 200 \
    --iterations-per-testing 800 \
    --lr 0.1 \
    --lr_policy cosine \
    --task-name "${task_name}" \
    --dataset sun397 \
    --data-dir "${DATA_DIR}" \
    --base-dir "${base_dir}" \
    --coreset \
    --coreset-mode budget \
    --data-score-path "${score_path}" \
    --mis-key "${SCORE_METRIC}" \
    --mis-data-score-descending "${MIS_DESCENDING}" \
    --coreset-key "${SCORE_METRIC}" \
    --network resnet18 \
    --batch-size "${batch_size}" \
    --coreset-ratio "${ratio}" \
    --save-coreset \
    --mis-ratio "${mis_ratio}" \
    --seed "${seed}" \
    --gpuid "${gpu}" \
    --ignore-td \
    --wandb-project "SUN397_Baselines_0506"

  echo "[DONE] gpu=${gpu} method=${method} ratio=${ratio} mis_ratio=${mis_ratio} seed=${seed}"
}

mapfile -t JOBS < <(build_jobs)

echo "Total ELFS jobs: ${#JOBS[@]}"

if (( ${#JOBS[@]} == 0 )); then
  echo "No jobs to run." >&2
  exit 1
fi

worker() {
  local gpu="$1"
  local worker_id="$2"
  local job_id=0
  local job
  local mode method ratio mis_ratio seed

  for job in "${JOBS[@]}"; do
    if (( job_id % ${#GPUS[@]} == worker_id )); then
      read -r mode method ratio mis_ratio seed <<< "${job}"
      run_budget_job "${gpu}" "${method}" "${ratio}" "${mis_ratio}" "${seed}"
    fi

    job_id=$((job_id + 1))
  done
}

for worker_id in "${!GPUS[@]}"; do
  worker "${GPUS[$worker_id]}" "${worker_id}" &
done

wait

echo "All ELFS jobs finished."