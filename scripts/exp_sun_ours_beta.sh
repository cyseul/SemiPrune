#!/usr/bin/env bash
set -euo pipefail
sleep 5h

DATA_DIR="../data/sun397_full_split_resized_512"

GPUS=(0 1 2 3 4 5 6 7)
DUAL_BETA_RATIOS=(0.7 0.5 0.3 0.2 0.1)
SEEDS=(0 1 2)

METHOD="fixmatch"

DUAL_BETA_T=30
DUAL_BETA_CDS=(5 6)

get_batch_size() {
  case "$1" in
    0.2) echo 64 ;;
    0.1) echo 32 ;;
    *) echo 128 ;;
  esac
}

build_jobs() {
  local ratio seed cd

  for cd in "${DUAL_BETA_CDS[@]}"; do
    for ratio in "${DUAL_BETA_RATIOS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "dual_beta ${METHOD} ${ratio} ${seed} ${cd}"
      done
    done
  done
}

run_dual_beta_job() {
  local gpu="$1"
  local method="$2"
  local ratio="$3"
  local seed="$4"
  local beta_cd="$5"

  local batch_size
  batch_size="$(get_batch_size "${ratio}")"

  local base_dir="data-model-0505/sun-${method}/full-train"
  local score_path="${base_dir}/data-score-full-train.pickle"

  local ratio_tag="${ratio/./p}"
  local task_name="sun397-${method}-dual-beta-T${DUAL_BETA_T}-cd${beta_cd}-r${ratio_tag}-s${seed}"

  if [[ ! -f "${score_path}" ]]; then
    echo "[SKIP] missing score file: ${score_path}" >&2
    return 0
  fi

  echo "[START] mode=dual_beta gpu=${gpu} method=${method} ratio=${ratio} T=${DUAL_BETA_T} beta_cd=${beta_cd} seed=${seed}"

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
    --coreset-mode dual_beta \
    --data-score-path "${score_path}" \
    --network resnet18 \
    --batch-size "${batch_size}" \
    --coreset-ratio "${ratio}" \
    --save-coreset \
    --pseudo-dual-T "${DUAL_BETA_T}" \
    --beta-cd "${beta_cd}" \
    --seed "${seed}" \
    --gpuid "${gpu}" \
    --ignore-td \
    --wandb-project "SUN397_Baselines_0506"

  echo "[DONE] mode=dual_beta gpu=${gpu} method=${method} ratio=${ratio} T=${DUAL_BETA_T} beta_cd=${beta_cd} seed=${seed}"
}

mapfile -t JOBS < <(build_jobs)

echo "Total dual_beta jobs: ${#JOBS[@]}"

if (( ${#JOBS[@]} == 0 )); then
  echo "No jobs to run." >&2
  exit 1
fi

worker() {
  local gpu="$1"
  local worker_id="$2"
  local job_id=0
  local job
  local mode method ratio seed beta_cd

  for job in "${JOBS[@]}"; do
    if (( job_id % ${#GPUS[@]} == worker_id )); then
      read -r mode method ratio seed beta_cd <<< "${job}"

      case "${mode}" in
        dual_beta)
          run_dual_beta_job "${gpu}" "${method}" "${ratio}" "${seed}" "${beta_cd}"
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

echo "All dual_beta jobs finished."