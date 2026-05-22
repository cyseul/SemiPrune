PYTHON_BIN=python
PYTHON_SCRIPT=train.py
DATA_DIR=../data
BASE_DIR=data-model/food101_swav
SCORE_PATH=${BASE_DIR}/data-score-food101_swav.pickle

GPUS=(0 1 2 4 5 6 7)
RATIOS=(0.2 0.1)
RUNS=(1 2 3 4 5)

get_batch_size() {
  case "$1" in
    0.7|0.5|0.3) echo 128 ;;
    0.2) echo 64 ;;
    0.1) echo 32 ;;
  esac
}

jobs=()
for ratio in "${RATIOS[@]}"; do
  for run in "${RUNS[@]}"; do
    jobs+=("${ratio} ${run}")
  done
done

declare -A PID_TO_GPU
declare -A GPU_BUSY
for gpu in "${GPUS[@]}"; do
  GPU_BUSY["$gpu"]=0
done

launch_job() {
  local gpu="$1"
  local ratio="$2"



  local run="$3"
  local seed=$((run - 1))
  local batch
  local task_name

  batch="$(get_batch_size "${ratio}")"
  task_name="food101-prototype-rn18-200ep-r${ratio/./p}-run${run}"

  echo "[launch] gpu=${gpu} run=${run} seed=${seed} ratio=${ratio} batch=${batch} task=${task_name}"

  ${PYTHON_BIN} ${PYTHON_SCRIPT} \
    --dataset food101 \
    --network resnet18 \
    --epochs 200 \
    --batch-size ${batch} \
    --lr 0.1 \
    --lr_policy cosine \
    --data-dir ${DATA_DIR} \
    --base-dir ${BASE_DIR} \
    --task-name ${task_name} \
    --coreset \
    --coreset-mode swav \
    --data-score-path ${SCORE_PATH} \
    --coreset-ratio ${ratio} \
    --gpuid ${gpu} \
    --seed ${seed} \
    > ${BASE_DIR}/${task_name}.launch.log 2>&1 &

  PID_TO_GPU[$!]="$gpu"
  GPU_BUSY["$gpu"]=1
}

release_finished_gpus() {
  local pid gpu
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu="${PID_TO_GPU[$pid]}"
      GPU_BUSY["$gpu"]=0
      unset PID_TO_GPU["$pid"]
    fi
  done
}

job_ptr=0
num_jobs=${#jobs[@]}

while (( job_ptr < num_jobs )); do
  for gpu in "${GPUS[@]}"; do
    if (( job_ptr >= num_jobs )); then
      break
    fi
    if [[ "${GPU_BUSY[$gpu]}" -eq 0 ]]; then
      read -r ratio run <<< "${jobs[$job_ptr]}"
      launch_job "${gpu}" "${ratio}" "${run}"
      ((job_ptr+=1))
    fi
  done
  if (( job_ptr < num_jobs )); then
    wait -n
    release_finished_gpus
  fi
done

while ((${#PID_TO_GPU[@]} > 0)); do
  wait -n
  release_finished_gpus
done
