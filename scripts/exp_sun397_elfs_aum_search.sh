#!/usr/bin/env bash
set -uo pipefail
export WANDB_MODE=offline

DATA_DIR="../data/sun397_full_split_resized_512"
BASE_DIR="./data-model-0504/sun397/elfs-aum"

RUNS=(0)
CORESET_RATIOS=(0.3 0.2 0.1)
MIS_RATIOS=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
BATCH_SIZES=(128 128 128 64 32)
DEVICES=(0 1 2 3 4 5 6 7)

num_gpus=${#DEVICES[@]}
active_jobs=0
job_id=0

run_job() {
  local coreset_ratio="$1"
  local mis_ratio="$2"
  local batch_size="$3"
  local run="$4"
  local device="$5"
  local task_name="$6"

  echo "[START] ratio=${coreset_ratio}, mis=${mis_ratio}, run=${run}, batch=${batch_size}, gpu=${device}, task=${task_name}"

  set +e

  python train.py \
    --epochs 200 \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --dataset sun397 \
    --batch-size "${batch_size}" \
    --ignore-td \
    --coreset \
    --coreset-mode budget \
    --data-score-path ./data-model-0504/sun397/elfs-ps/full-train/data-score-full-train.pickle \
    --coreset-key accumulated_margin \
    --mis-key accumulated_margin \
    --coreset-ratio "${coreset_ratio}" \
    --mis-ratio "${mis_ratio}" \
    --gpuid "${device}"

  status=$?

  if (( status == 0 )); then
    echo "[DONE]  ratio=${coreset_ratio}, mis=${mis_ratio}, run=${run}, gpu=${device}, task=${task_name}"
  else
    echo "[FAIL/SKIP] ratio=${coreset_ratio}, mis=${mis_ratio}, run=${run}, gpu=${device}, status=${status}, task=${task_name}"
  fi

  # 중요: job이 실패하거나 kill되어도 wait -n이 non-zero 때문에 마스터 스크립트를 죽이지 않게 함
  exit 0
}

for i in "${!CORESET_RATIOS[@]}"; do
  coreset_ratio="${CORESET_RATIOS[$i]}"
  batch_size="${BATCH_SIZES[$i]}"

  ratio_tag="${coreset_ratio/./p}"

  for mis_ratio in "${MIS_RATIOS[@]}"; do
    mis_tag="${mis_ratio/./p}"

    for run in "${RUNS[@]}"; do
      device="${DEVICES[$((job_id % num_gpus))]}"
      task_name="subset${ratio_tag}_mis${mis_tag}_run${run}"

      run_job "${coreset_ratio}" "${mis_ratio}" "${batch_size}" "${run}" "${device}" "${task_name}" &

      job_id=$((job_id + 1))
      active_jobs=$((active_jobs + 1))

      # 동시에 최대 GPU 개수만큼만 실행
      # 개별 job이 실패/kill되어도 다음 job이 올라가도록 wait 실패를 무시
      if (( active_jobs >= num_gpus )); then
        wait -n || true
        active_jobs=$((active_jobs - 1))
      fi
    done
  done
done

while (( active_jobs > 0 )); do
  wait -n || true
  active_jobs=$((active_jobs - 1))
done

echo "All jobs finished."