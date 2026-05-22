#!/usr/bin/env bash
set -euo pipefail
sleep 1h
DATA_DIR="../data/sun397_full_split_resized_512"

# Same GPU can appear multiple times if you want multiple workers per GPU.
GPUS=(0 1 2 3 4 5 6 7)

SEEDS=(1 2 3 4)

METHODS=("fixmatch")
SCORE_METRIC="accumulated_margin"
MIS_DESCENDING="False"

get_batch_size() {
  case "$1" in
    0.2) echo 64 ;;
    0.1) echo 32 ;;
    *) echo 128 ;;
  esac
}

# Format: "coreset_ratio:mis_ratio"
get_pairs_for_method() {
  local method="$1"

  case "${method}" in
    elfs)
      echo "0.7:0.2 0.5:0.3 0.3:0.4 0.2:0.5 0.1:0.6"
      ;;
    elfs-self)
      echo "0.7:0.1 0.5:0.2 0.3:0.3 0.2:0.4 0.1:0.5"
      ;;
    fixmatch)
      echo "0.5:0.2 0.3:0.2"
      ;;
    *)
      echo "[ERROR] unknown method: ${method}" >&2
      return 1
      ;;
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
  local method pair ratio mis_ratio seed
  local -a pairs

  for method in "${METHODS[@]}"; do
    read -r -a pairs <<< "$(get_pairs_for_method "${method}")"

    for pair in "${pairs[@]}"; do
      ratio="${pair%%:*}"
      mis_ratio="${pair##*:}"

      # Safety check: valid only when mis_ratio <= 1 - coreset_ratio.
      if ! valid_mis_ratio "${ratio}" "${mis_ratio}"; then
        echo "[SKIP] invalid pair for method=${method}: ratio=${ratio}, mis_ratio=${mis_ratio}" >&2
        continue
      fi

      for seed in "${SEEDS[@]}"; do
        echo "budget ${method} ${ratio} ${mis_ratio} ${seed}"
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

  echo "[START] mode=budget gpu=${gpu} method=${method} metric=${SCORE_METRIC} ratio=${ratio} mis_ratio=${mis_ratio} seed=${seed}"

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

  echo "[DONE] mode=budget gpu=${gpu} method=${method} ratio=${ratio} mis_ratio=${mis_ratio} seed=${seed}"
}

mapfile -t JOBS < <(build_jobs)

echo "Total budget jobs: ${#JOBS[@]}"

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

      case "${mode}" in
        budget)
          run_budget_job "${gpu}" "${method}" "${ratio}" "${mis_ratio}" "${seed}"
          ;;
        *)
          echo "[ERROR] unsupported mode: ${mode}" >&2
          return 1
          ;;
      esac
    fi

    job_id=$((job_id + 1))
  done
}

for worker_id in "${!GPUS[@]}"; do
  worker "${GPUS[$worker_id]}" "${worker_id}" &
done

wait

echo "All budget jobs finished."