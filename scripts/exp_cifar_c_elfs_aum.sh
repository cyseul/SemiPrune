#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="./data-model-0504/cifar100-c/elfs-dino-subset"
DATA_DIR="./data"
TRAIN_PY="train.py"

SCORE_SET_NAME="elfs-dino"
DATA_SCORE_PATH="data-model-0504/cifar100-c/elfs-dino/data-score-elfs-dino.pickle"

DATASET="cifar100-C"
NETWORK="resnet18"
EPOCHS=200
LR=0.1
LR_POLICY="cosine"

CORESET_RATIOS=(0.7 0.5 0.3 0.2 0.1)
MIS_RATIOS=(0.0 0.2 0.3 0.5 0.7)
BATCHES=(128 128 128 64 32)

RUNS=(1 2 3 4)
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

mkdir -p "${BASE_DIR}"

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

is_valid_mis_ratio() {
  local ratio="$1"
  local mis_ratio="$2"

  python - <<EOF
ratio = float("${ratio}")
mis_ratio = float("${mis_ratio}")

if mis_ratio <= 1.0 - ratio + 1e-12:
    exit(0)
else:
    exit(1)
EOF
}

launch_job() {
  local gpu="$1"
  local run="$2"
  local seed="$3"
  local ratio="$4"
  local mis_ratio="$5"
  local batch="$6"

  local task_name="${SCORE_SET_NAME}-aum-rn18-200ep-r$(ratio_tag "${ratio}")-m$(ratio_tag "${mis_ratio}")-run${run}"
  local log_path="${BASE_DIR}/${task_name}.launch.log"

  echo "[launch] gpu=${gpu} run=${run} seed=${seed} ratio=${ratio} mis_ratio=${mis_ratio} batch=${batch} task=${task_name}"

  (
    python "${TRAIN_PY}" \
      --dataset "${DATASET}" \
      --network "${NETWORK}" \
      --epochs "${EPOCHS}" \
      --batch-size "${batch}" \
      --lr "${LR}" \
      --data-dir "${DATA_DIR}" \
      --base-dir "${BASE_DIR}" \
      --task-name "${task_name}" \
      --coreset \
      --coreset-mode budget \
      --data-score-path "${DATA_SCORE_PATH}" \
      --coreset-ratio "${ratio}" \
      --coreset-key accumulated_margin \
      --mis-key accumulated_margin \
      --mis-ratio "${mis_ratio}" \
      --gpuid "${gpu}" \
      --seed "${seed}" \
      > "${log_path}" 2>&1

    status=$?
    if [[ ${status} -eq 0 ]]; then
      echo "[done] task=${task_name}"
    else
      echo "[failed] task=${task_name} status=${status} log=${log_path}"
    fi
  ) &
}

active_jobs=0
job_count=0

if [[ ${#CORESET_RATIOS[@]} -ne ${#MIS_RATIOS[@]} ]]; then
  echo "[error] CORESET_RATIOS and MIS_RATIOS must have the same length." >&2
  exit 1
fi

if [[ ${#CORESET_RATIOS[@]} -ne ${#BATCHES[@]} ]]; then
  echo "[error] CORESET_RATIOS and BATCHES must have the same length." >&2
  exit 1
fi

for run in "${RUNS[@]}"; do
  seed="${run}"

  for i in "${!CORESET_RATIOS[@]}"; do
    ratio="${CORESET_RATIOS[$i]}"
    mis_ratio="${MIS_RATIOS[$i]}"
    batch="${BATCHES[$i]}"

    if ! is_valid_mis_ratio "${ratio}" "${mis_ratio}"; then
      echo "[skip] ratio=${ratio} mis_ratio=${mis_ratio} because mis_ratio > 1 - coreset_ratio"
      continue
    fi

    gpu="${GPUS[$(( job_count % NUM_GPUS ))]}"

    launch_job "${gpu}" "${run}" "${seed}" "${ratio}" "${mis_ratio}" "${batch}"

    job_count=$((job_count + 1))
    active_jobs=$((active_jobs + 1))

    if (( active_jobs >= NUM_GPUS )); then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done
done

wait

echo "All CIFAR100-C AUM paired coreset-ratio/mis-ratio jobs finished."