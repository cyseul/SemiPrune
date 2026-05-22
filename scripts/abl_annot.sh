#!/usr/bin/env bash
set -euo pipefail

TRAIN_PY="${TRAIN_PY:-train.py}"
DATA_DIR="${DATA_DIR:-../data}"

NETWORK="${NETWORK:-resnet18}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"

CORESET_MODE="${CORESET_MODE:-abl}"
SEEDS=(0 1 2 3 4)

# additional-only coreset ratios
ADDITIONAL_CORESET_RATIOS=(0.1 0.2 0.3 0.5)

# Physical GPUs
GPUS=(0 1 2 3 4 5 6 7)

# GPU 1개당 동시에 올릴 job 수
JOBS_PER_GPU=1
MAX_PARALLEL=$(( ${#GPUS[@]} * JOBS_PER_GPU ))

MASK_ROOT="ablations/caltech101"

ADDITIONAL_BASE_DIR="data-model/caltech101/ablations/additional_labeled_only"
TOTAL_BASE_DIR="data-model/caltech101/ablations/initial_labeled_additional_labeled"

mkdir -p logs

get_batch_size() {
  case "$1" in
    0.2) echo 64 ;;
    0.1) echo 32 ;;
    *) echo 128 ;;
  esac
}

add_float() {
  python - "$1" "$2" <<'PY'
import sys
a = float(sys.argv[1])
b = float(sys.argv[2])
print(f"{a + b:.1f}")
PY
}

make_mis_ratios() {
  python - "$1" <<'PY'
import sys

additional_ratio = float(sys.argv[1])
max_mis = 0.9 - additional_ratio

vals = []
x = 0.0
while x <= max_mis + 1e-9:
    vals.append(f"{x:.1f}")
    x += 0.1

print(" ".join(vals))
PY
}

wait_for_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 5
  done
}

run_job() {
  local gpu_id="$1"
  local setting="$2"
  local coreset_ratio="$3"
  local mis_ratio="$4"
  local seed="$5"
  local data_score_path="$6"
  local base_dir="$7"

  local batch_size
  batch_size="$(get_batch_size "${coreset_ratio}")"

  local ratio_tag="${coreset_ratio/./p}"
  local mis_tag="${mis_ratio/./p}"

  local task_name="${setting}-r${ratio_tag}-mis${mis_tag}-s${seed}"
  local log_path="logs/${task_name}.log"

  if [ ! -f "${data_score_path}" ]; then
    echo "[skip] mask not found: ${data_score_path}"
    return 0
  fi

  echo "[launch] gpu=${gpu_id} setting=${setting} cr=${coreset_ratio} mis=${mis_ratio} bs=${batch_size} seed=${seed}"
  echo "[mask] ${data_score_path}"
  echo "[log] ${log_path}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python "${TRAIN_PY}" \
    --dataset caltech101 \
    --data-dir "${DATA_DIR}" \
    --caltech-split-path "data/embeddings/Caltech101-dino_vitb16/caltech101_split_seed0.pt" \
    --base-dir "${base_dir}" \
    --batch-size "${batch_size}" \
    --task-name "${task_name}" \
    --network "${NETWORK}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --coreset \
    --coreset-mode "${CORESET_MODE}" \
    --coreset-ratio "${coreset_ratio}" \
    --data-score-path "${data_score_path}" \
    --seed "${seed}" \
    --wandb-project "Label_Budget_Ablation_Caltech-101" \
    > "${log_path}" 2>&1 &
}

job_count=0

for seed in "${SEEDS[@]}"; do
  for additional_ratio in "${ADDITIONAL_CORESET_RATIOS[@]}"; do

    total_ratio="$(add_float "${additional_ratio}" 0.1)"

    read -ra MIS_RATIOS <<< "$(make_mis_ratios "${additional_ratio}")"

    for mis_ratio in "${MIS_RATIOS[@]}"; do

      # 1) additional labeled only
      additional_mask_path="${MASK_ROOT}/additional_labeled_only/coreset_${additional_ratio}_mis_${mis_ratio}.npy"

      wait_for_slot
      gpu_id="${GPUS[$((job_count % ${#GPUS[@]}))]}"
      run_job \
        "${gpu_id}" \
        "additional_only" \
        "${additional_ratio}" \
        "${mis_ratio}" \
        "${seed}" \
        "${additional_mask_path}" \
        "${ADDITIONAL_BASE_DIR}"

      job_count=$((job_count + 1))

      # 2) initial labeled + additional labeled
      total_mask_path="${MASK_ROOT}/initial_labeled_additional_labeled/coreset_${total_ratio}_mis_${mis_ratio}.npy"

      wait_for_slot
      gpu_id="${GPUS[$((job_count % ${#GPUS[@]}))]}"
      run_job \
        "${gpu_id}" \
        "initial_plus_additional" \
        "${total_ratio}" \
        "${mis_ratio}" \
        "${seed}" \
        "${total_mask_path}" \
        "${TOTAL_BASE_DIR}"

      job_count=$((job_count + 1))

    done
  done
done

wait

echo "All jobs finished."