#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="data-model/caltech101"
DATA_DIR="../data"
SPLIT_PATH="data/embeddings/Caltech101-dino_vitb16/caltech101_split_seed0.pt"

SCORE_SET_NAMES=(
  "fixmatch-ps-rn18-200ep-pseudo-full"
)

RATIOS=(0.7 0.5 0.3 0.2 0.1)
MIS_RATIOS=(0.0 0.2 0.5 0.6 0.7)
BATCHES=(128 128 128 64 32)
RUNS=(10)
GPUS=(0 1 2 3)

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

build_jobs() {
  local score_set_name="$1"
  local run ratio mis_ratio batch
  local i

  for run in "${RUNS[@]}"; do
    for i in "${!RATIOS[@]}"; do
      ratio="${RATIOS[$i]}"
      mis_ratio="${MIS_RATIOS[$i]}"
      batch="${BATCHES[$i]}"
      echo "${score_set_name} ${run} ${ratio} ${mis_ratio} ${batch}"
    done
  done
}

run_job() {
  local gpu="$1"
  local score_set_name="$2"
  local run="$3"
  local ratio="$4"
  local mis_ratio="$5"
  local batch="$6"

  local seed=$((run - 1))
  local score_path="${BASE_DIR}/${score_set_name}/data-score-${score_set_name}.pickle"

  if [[ ! -f "$score_path" ]]; then
    echo "Score file does not exist: $score_path"
    exit 1
  fi

  local task_name="${score_set_name}-aum-rn18-200ep-r$(ratio_tag "$ratio")-m$(ratio_tag "$mis_ratio")-run${run}"
  local log_path="${BASE_DIR}/${task_name}.launch.log"

  echo "[launch] score_set=${score_set_name} gpu=${gpu} run=${run} seed=${seed} ratio=${ratio} mis_ratio=${mis_ratio} batch=${batch} task=${task_name}"

  python train.py \
    --dataset caltech101 \
    --network resnet18 \
    --epochs 200 \
    --batch-size "${batch}" \
    --lr 0.1 \
    --lr_policy cosine \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --caltech-split-path "${SPLIT_PATH}" \
    --coreset \
    --coreset-mode budget \
    --data-score-path "${score_path}" \
    --coreset-ratio "${ratio}" \
    --mis-ratio "${mis_ratio}" \
    --gpuid "${gpu}" \
    --seed "${seed}" \
    > "${log_path}" 2>&1
}

mapfile -t JOBS < <(
  for score_set_name in "${SCORE_SET_NAMES[@]}"; do
    build_jobs "${score_set_name}"
  done
)

declare -A PID_TO_GPU
declare -A GPU_BUSY

for gpu in "${GPUS[@]}"; do
  GPU_BUSY["$gpu"]=0
done

launch_job() {
  local gpu="$1"
  local score_set_name="$2"
  local run="$3"
  local ratio="$4"
  local mis_ratio="$5"
  local batch="$6"

  run_job "$gpu" "$score_set_name" "$run" "$ratio" "$mis_ratio" "$batch" &
  local pid=$!

  PID_TO_GPU["$pid"]="$gpu"
  GPU_BUSY["$gpu"]=1
}

job_ptr=0
num_jobs=${#JOBS[@]}

while (( job_ptr < num_jobs )); do
  for gpu in "${GPUS[@]}"; do
    if (( job_ptr >= num_jobs )); then
      break
    fi

    if [[ "${GPU_BUSY[$gpu]}" -eq 0 ]]; then
      read -r score_set_name run ratio mis_ratio batch <<< "${JOBS[$job_ptr]}"
      launch_job "$gpu" "$score_set_name" "$run" "$ratio" "$mis_ratio" "$batch"
      ((job_ptr+=1))
    fi
  done

  if (( job_ptr < num_jobs )); then
    finished_pid=""
    wait -n -p finished_pid
    finished_gpu="${PID_TO_GPU[$finished_pid]}"
    GPU_BUSY["$finished_gpu"]=0
    unset PID_TO_GPU["$finished_pid"]
  fi
done

while ((${#PID_TO_GPU[@]} > 0)); do
  finished_pid=""
  wait -n -p finished_pid
  finished_gpu="${PID_TO_GPU[$finished_pid]}"
  GPU_BUSY["$finished_gpu"]=0
  unset PID_TO_GPU["$finished_pid"]
done

echo "All Caltech101 AUM pruning jobs finished."