#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="data-model/caltech101"
SCORE_SET_NAME="gt-label-rn18-200ep-full"
DATA_DIR="../data"
SPLIT_PATH="data/embeddings/Caltech101-dino_vitb16/caltech101_split_seed0.pt"
TRAIN_PY="train.py"

SEED="${SEED:-0}"

# 사용할 GPU들
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# 고정값 / sweep 대상
BETA_CD=4
CORESET_RATIOS=(0.7 0.5 0.3 0.2 0.1)
RUNS=(1 2 3 4 5)

ratio_tag() {
  local x="$1"
  echo "${x//./p}"
}

get_batch_size() {
  local ratio="$1"
  case "$ratio" in
    0.7) echo 128 ;;
    0.5) echo 128 ;;
    0.3) echo 128 ;;
    0.2) echo 64 ;;
    0.1) echo 32 ;;
    *)
      echo "Unsupported coreset ratio: $ratio" >&2
      exit 1
      ;;
  esac
}

SCORE_PATH="${BASE_DIR}/${SCORE_SET_NAME}/data-score-${SCORE_SET_NAME}.pickle"

if [[ ! -f "$SCORE_PATH" ]]; then
  echo "Score file does not exist: $SCORE_PATH"
  exit 1
fi

if [[ ! -f "$SPLIT_PATH" ]]; then
  echo "Split file does not exist: $SPLIT_PATH"
  exit 1
fi

launch_job() {
  local gpu_id="$1"
  local ratio="$2"
  local run_id="$3"
  local batch_size="$4"

  local task_name="${SCORE_SET_NAME}-dual-beta-r$(ratio_tag "$ratio")-cd${BETA_CD}-run${run_id}"
  local log_path="${BASE_DIR}/${task_name}.launch.log"

  echo "[launch] gpu=${gpu_id} seed=${SEED} ratio=${ratio} run=${run_id} cd=${BETA_CD} batch=${batch_size} task=${task_name}"

  python "${TRAIN_PY}" \
    --dataset caltech101 \
    --network resnet18 \
    --epochs 200 \
    --batch-size "${batch_size}" \
    --lr 0.1 \
    --lr_policy cosine \
    --data-dir "${DATA_DIR}" \
    --base-dir "${BASE_DIR}" \
    --task-name "${task_name}" \
    --caltech-split-path "${SPLIT_PATH}" \
    --coreset \
    --coreset-mode dual_beta \
    --data-score-path "${SCORE_PATH}" \
    --coreset-ratio "${ratio}" \
    --pseudo-dual-p 1 \
    --pseudo-dual-T 50 \
    --beta-cd "${BETA_CD}" \
    --gpuid "${gpu_id}" \
    --seed "${SEED}" \
    > "${log_path}" 2>&1
}

# 모든 (ratio, run) 조합을 순서대로 job queue에 저장
job_ratios=()
job_runs=()
job_batches=()

for ratio in "${CORESET_RATIOS[@]}"; do
  batch_size="$(get_batch_size "$ratio")"
  for run_id in "${RUNS[@]}"; do
    job_ratios+=("$ratio")
    job_runs+=("$run_id")
    job_batches+=("$batch_size")
  done
done

TOTAL_JOBS=${#job_ratios[@]}
next_job_idx=0

pids=()
gpu_slots=()

# 먼저 GPU 수만큼 initial launch
for slot in "${!GPUS[@]}"; do
  if (( next_job_idx >= TOTAL_JOBS )); then
    break
  fi

  gpu_id="${GPUS[$slot]}"
  ratio="${job_ratios[$next_job_idx]}"
  run_id="${job_runs[$next_job_idx]}"
  batch_size="${job_batches[$next_job_idx]}"

  launch_job "${gpu_id}" "${ratio}" "${run_id}" "${batch_size}" &
  pids[$slot]=$!
  gpu_slots[$slot]="${gpu_id}"

  next_job_idx=$((next_job_idx + 1))
done

# 각 GPU 슬롯에서 job이 끝날 때마다 바로 다음 job 실행
while true; do
  active_found=0

  for slot in "${!GPUS[@]}"; do
    pid="${pids[$slot]:-}"
    gpu_id="${gpu_slots[$slot]:-${GPUS[$slot]}}"

    if [[ -z "${pid}" ]]; then
      continue
    fi

    active_found=1

    # 프로세스가 아직 살아있으면 넘어감
    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi

    # 종료 코드 회수
    wait "${pid}" || true

    # 다음 job이 남아 있으면 즉시 같은 GPU에 할당
    if (( next_job_idx < TOTAL_JOBS )); then
      ratio="${job_ratios[$next_job_idx]}"
      run_id="${job_runs[$next_job_idx]}"
      batch_size="${job_batches[$next_job_idx]}"

      echo "[done] gpu=${gpu_id} pid=${pid} finished, launching next job..."
      launch_job "${gpu_id}" "${ratio}" "${run_id}" "${batch_size}" &
      pids[$slot]=$!

      next_job_idx=$((next_job_idx + 1))
    else
      # 더 이상 할 job이 없으면 슬롯 비움
      unset 'pids[slot]'
    fi
  done

  # 더 이상 active pid도 없고, 남은 job도 없으면 종료
  if (( next_job_idx >= TOTAL_JOBS )) && [[ ${#pids[@]} -eq 0 ]]; then
    break
  fi

  # 아직 active process가 하나도 없으면 종료
  if (( active_found == 0 )) && (( next_job_idx >= TOTAL_JOBS )); then
    break
  fi

  sleep 5
done

echo "All Caltech101 dual_beta sweep jobs finished."