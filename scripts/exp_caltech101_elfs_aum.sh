#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="data-model/caltech101"
DATA_DIR="../data"
SPLIT_PATH="data/embeddings/Caltech101-dino_vitb16/caltech101_split_seed0.pt"

SCORE_SET_NAMES=(
  "elfs-ps-rn18-200ep-pseudo-full"
)

RATIOS=(0.7 0.5 0.3 0.2 0.1)
MIS_RATIOS=(0.2 0.3 0.5 0.6 0.7)
BATCHES=(128 128 128 64 32)
RUNS=(2 3 4 5)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

job_count=0

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

for score_set_name in "${SCORE_SET_NAMES[@]}"; do
  SCORE_PATH="${BASE_DIR}/${score_set_name}/data-score-${score_set_name}.pickle"

  if [[ ! -f "$SCORE_PATH" ]]; then
    echo "Score file does not exist: $SCORE_PATH"
    exit 1
  fi

  for run in "${RUNS[@]}"; do
    seed=$((run - 1))

    for i in "${!RATIOS[@]}"; do
      ratio="${RATIOS[$i]}"
      mis_ratio="${MIS_RATIOS[$i]}"
      batch="${BATCHES[$i]}"

      gpu_idx=$(( job_count % NUM_GPUS ))
      gpu="${GPUS[$gpu_idx]}"

      task_name="${score_set_name}-aum-rn18-200ep-r$(ratio_tag "$ratio")-m$(ratio_tag "$mis_ratio")-run${run}"
      log_path="${BASE_DIR}/${task_name}.launch.log"

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
        --data-score-path "${SCORE_PATH}" \
        --coreset-ratio "${ratio}" \
        --mis-ratio "${mis_ratio}" \
        --gpuid "${gpu}" \
        --seed "${seed}" \
        > "${log_path}" 2>&1 &

      job_count=$((job_count + 1))

      if [[ $(( job_count % NUM_GPUS )) == 0 ]]; then
        echo "GPU ${NUM_GPUS}개 할당 완료. 현재 배치 작업들이 끝날 때까지 대기 중..."
        wait
      fi
    done
  done
done

wait
echo "All Caltech101 AUM pruning jobs finished."